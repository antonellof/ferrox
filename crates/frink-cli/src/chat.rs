//! Server-backed multi-turn chat REPL (`frink chat`).
//!
//! Talks to a running `frink-server` over HTTP so chat-template wrapping
//! and streaming stay on the validated serve path (no duplicated decode).

use std::io::{BufRead, Read, Write};

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Value};

use crate::http;

#[derive(Parser, Debug)]
pub struct ChatArgs {
    /// Base URL of frink-server (scheme + host:port).
    #[arg(long, default_value = "http://127.0.0.1:8383")]
    pub url: String,

    /// Model id echoed in requests (server uses the loaded GGUF).
    #[arg(long, default_value = "default")]
    pub model: String,

    /// Optional system prompt prepended once at the start of the session.
    #[arg(long)]
    pub system: Option<String>,

    #[arg(long, default_value_t = 256)]
    pub max_tokens: u32,

    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    /// Use SSE streaming (`stream: true`) and print tokens as they arrive.
    #[arg(long, default_value_t = true)]
    pub stream: bool,

    /// Disable SSE; wait for the full non-streaming JSON response.
    #[arg(long, default_value_t = false)]
    pub no_stream: bool,
}

pub fn run_chat(args: ChatArgs) -> Result<()> {
    let stream = args.stream && !args.no_stream;
    let base = args.url.trim_end_matches('/');
    let health = http::get(&format!("{base}/health"))
        .with_context(|| format!("health check failed for {base}"))?;
    if !(200..300).contains(&health.status) {
        anyhow::bail!("server /health returned HTTP {}", health.status);
    }
    install_sigint_handler();
    eprintln!("frink chat → {base}  (/help for commands, Ctrl-C stops a turn, /quit to leave)");
    if let Some(sys) = &args.system {
        eprintln!("system: {sys}");
    }

    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = &args.system {
        messages.push(json!({"role": "system", "content": sys}));
    }

    // The gears the SERVER advertises, read once at connect. Not a
    // hardcoded list: which efforts exist is a property of the served
    // checkpoint's own template, probed at load, so a client that
    // guesses offers gears the model does not grade -- and hides ones
    // it does. An older server that advertises none simply has no
    // `/think`.
    let mut gears = ThinkGears::fetch(base).unwrap_or_default();

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    loop {
        eprint!("> ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        let n = stdin.lock().read_line(&mut line)?;
        if n == 0 {
            eprintln!();
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "/quit" | "/exit" | "/q") {
            break;
        }
        if line == "/clear" {
            messages.clear();
            if let Some(sys) = &args.system {
                messages.push(json!({"role": "system", "content": sys}));
            }
            eprintln!("(history cleared)");
            continue;
        }
        if line == "/help" {
            eprint!("{HELP}");
            continue;
        }
        if line == "/stats" {
            match http::get(&format!("{base}/v1/stats")) {
                Ok(resp) => match serde_json::from_slice::<Value>(&resp.body) {
                    Ok(v) => eprintln!("{}", render_stats(&v)),
                    Err(e) => eprintln!("(could not read /v1/stats: {e})"),
                },
                Err(e) => eprintln!("(could not reach /v1/stats: {e:#})"),
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("/think") {
            let rest = rest.trim();
            match gears.select(rest) {
                Ok(chosen) => eprintln!("(thinking: {chosen})"),
                Err(why) => eprintln!("({why})"),
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("/cache") {
            let rest = rest.trim();
            eprintln!("{}", cache_command(base, rest));
            continue;
        }
        if line.starts_with('/') {
            // A mistyped command must not be sent to the model as a
            // prompt: the answer would be a plausible-looking
            // hallucination about a command that does not exist.
            eprintln!("(unknown command {line}; /help lists them)");
            continue;
        }

        messages.push(json!({"role": "user", "content": line}));
        let mut body = json!({
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stream": stream,
        });
        // Only when a gear is actually selected: an absent key lets the
        // checkpoint's own template default apply, which is not the
        // same as sending the default explicitly -- a template may
        // treat "unset" and "set to its default" differently.
        if let Some(kwargs) = gears.selected_kwargs() {
            body["chat_template_kwargs"] = kwargs;
        }

        let reply = if stream {
            print_streamed_completion(base, &body, &mut stdout)?
        } else {
            let text = post_completion(base, &body)?;
            print!("{text}");
            stdout.flush()?;
            println!();
            text
        };
        messages.push(json!({"role": "assistant", "content": reply}));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Ctrl-C: cancel the turn, not the session
// ---------------------------------------------------------------------

/// Set while a generation is streaming.
static GENERATING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Set by the signal handler when a turn should stop.
static INTERRUPTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Marks a generation in flight, and clears the mark however the
/// generation ends -- a `?` return or a panic included, which is why it
/// is a guard and not two calls.
struct Generating;

impl Generating {
    fn begin() -> Self {
        INTERRUPTED.store(false, std::sync::atomic::Ordering::SeqCst);
        GENERATING.store(true, std::sync::atomic::Ordering::SeqCst);
        Generating
    }
}

impl Drop for Generating {
    fn drop(&mut self) {
        GENERATING.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

fn interrupted() -> bool {
    INTERRUPTED.load(std::sync::atomic::Ordering::SeqCst)
}

/// What SIGINT does, and it depends on what the REPL is doing.
///
/// **Mid-generation**: set a flag and return. The stream loop notices
/// between chunks, cancels the generation server-side and hands back
/// what arrived, so Ctrl-C costs the turn rather than the session --
/// which is the whole point, since the alternative is losing a
/// conversation to stop one answer.
///
/// **At the prompt**: exit, because that is what Ctrl-C means at a
/// shell prompt and a REPL that cannot be quit with it is worse than
/// one with no handler at all.
///
/// Only two things happen in here, and both are async-signal-safe: an
/// atomic store and `_exit`. Anything else -- a `println!`, an
/// allocation, unwinding -- is undefined behaviour in a signal handler,
/// and the failure mode is a deadlock inside the allocator that looks
/// like a hang rather than a crash.
extern "C" fn on_sigint(_sig: libc::c_int) {
    if GENERATING.load(std::sync::atomic::Ordering::SeqCst) {
        INTERRUPTED.store(true, std::sync::atomic::Ordering::SeqCst);
    } else {
        // 130 is the shell convention for "killed by SIGINT".
        unsafe { libc::_exit(130) };
    }
}

/// Installs [`on_sigint`]. Best-effort: a platform that refuses simply
/// keeps the default behaviour, which is the one we started with.
fn install_sigint_handler() {
    unsafe {
        // Through a pointer rather than casting the function item
        // straight to an integer, which clippy rightly refuses: the
        // direct cast is a lint-level trap for the case where the
        // integer width and the pointer width differ.
        libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t);
    }
}

/// What `/help` prints.
const HELP: &str = "\
  /help            this list
  /clear           forget the conversation (keeps --system)
  /stats           what the server is doing right now
  /think [gear]    cycle, or set, the server's advertised thinking gear
  /cache [tokens]  show the KV pool, or resize it to N tokens
  /quit            leave
";

/// The thinking gears the SERVER advertises, and which one is selected.
///
/// Read from `/v1/models` rather than hardcoded, because which efforts
/// exist is a property of the served checkpoint's own template --
/// probed at load, not guessable. A client with a fixed list offers
/// gears the model does not grade and hides ones it does.
#[derive(Debug, Default)]
struct ThinkGears {
    supported: Vec<String>,
    /// Per-gear `chat_template_kwargs`, as the server derived them.
    /// Sent verbatim: the client's job is to pick a gear, not to know
    /// what selecting it means on this family.
    kwargs: std::collections::BTreeMap<String, Value>,
    selected: Option<String>,
}

impl ThinkGears {
    fn fetch(base: &str) -> Result<Self> {
        let resp = http::get(&format!("{base}/v1/models"))?;
        let body: Value = serde_json::from_slice(&resp.body).context("parse /v1/models")?;
        Ok(Self::from_models(&body))
    }

    fn from_models(body: &Value) -> Self {
        let first = body.pointer("/data/0").cloned().unwrap_or(Value::Null);
        let supported: Vec<String> = first
            .get("supported_reasoning_efforts")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let kwargs = first
            .get("reasoning_effort_kwargs")
            .and_then(Value::as_object)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        ThinkGears {
            supported,
            kwargs,
            selected: None,
        }
    }

    /// `/think` with no argument cycles; with one, sets that gear.
    ///
    /// Cycling starts from the FIRST gear rather than from the server's
    /// default, so the first press is deterministic whatever the
    /// checkpoint defaults to -- a user pressing it twice on two
    /// different models gets the same two gears both times.
    fn select(&mut self, want: &str) -> std::result::Result<String, String> {
        if self.supported.is_empty() {
            return Err("this server advertises no thinking gears".to_string());
        }
        let next = if want.is_empty() {
            let at = self
                .selected
                .as_ref()
                .and_then(|s| self.supported.iter().position(|g| g == s));
            match at {
                Some(i) => self.supported[(i + 1) % self.supported.len()].clone(),
                None => self.supported[0].clone(),
            }
        } else if self.supported.iter().any(|g| g == want) {
            want.to_string()
        } else {
            return Err(format!(
                "no gear {want:?}; this server has {}",
                self.supported.join(", ")
            ));
        };
        self.selected = Some(next.clone());
        Ok(next)
    }

    /// The kwargs for the selected gear, or `None` when nothing is
    /// selected.
    ///
    /// `None` and not an empty object: an absent key lets the
    /// checkpoint's template apply its own default, which a template
    /// may treat differently from being handed that default
    /// explicitly.
    fn selected_kwargs(&self) -> Option<Value> {
        let gear = self.selected.as_ref()?;
        // A gear with EMPTY kwargs is a real answer -- an
        // always-thinking family advertises one gear with nothing to
        // send -- so it is passed through rather than skipped.
        self.kwargs.get(gear).cloned()
    }
}

/// `/stats`, as one line per section.
///
/// A figure the server reported as `null` prints as `-`, never as `0`:
/// a p95 of zero would read as an instantaneous server and a memory
/// figure of zero as an idle one, and both are the shapes `/v1/stats`
/// deliberately avoids emitting.
fn render_stats(v: &Value) -> String {
    let num = |p: &str| -> String {
        match v.pointer(p) {
            Some(Value::Number(n)) => format!("{n}"),
            _ => "-".to_string(),
        }
    };
    let mut out = format!(
        "  model    {}  ({})\n",
        v.get("model").and_then(Value::as_str).unwrap_or("none"),
        v.get("state").and_then(Value::as_str).unwrap_or("?"),
    );
    out.push_str(&format!(
        "  tokens   decode {} tok/s   prefill {} tok/s\n",
        num("/throughput/decode_tps"),
        num("/throughput/prefill_tps"),
    ));
    out.push_str(&format!(
        "  requests {} active   {} done   p95 {} ms   ttft {} ms\n",
        num("/requests/active"),
        num("/requests/completed"),
        num("/requests/p95_ms"),
        num("/requests/ttft_mean_ms"),
    ));
    match v.pointer("/pools/kv_pages") {
        Some(kv) if !kv.is_null() => out.push_str(&format!(
            "  kv pool  {}/{} pages of {} tokens\n",
            kv.get("used").and_then(Value::as_u64).unwrap_or(0),
            kv.get("total").and_then(Value::as_u64).unwrap_or(0),
            kv.get("page_size").and_then(Value::as_u64).unwrap_or(0),
        )),
        // Absent, not zero: this deployment allocates privately per
        // request, which is a different thing from an empty pool.
        _ => out.push_str("  kv pool  none (every request allocates privately)\n"),
    }
    if let Some(mem) = v.get("memory").filter(|m| !m.is_null()) {
        out.push_str(&format!(
            "  memory   {:.2} GiB ({})\n",
            mem.get("bytes").and_then(Value::as_u64).unwrap_or(0) as f64 / (1 << 30) as f64,
            mem.get("kind").and_then(Value::as_str).unwrap_or("?"),
        ));
    }
    out.trim_end().to_string()
}

/// `/cache` with no argument shows the pool; with one, resizes it.
fn cache_command(base: &str, arg: &str) -> String {
    if arg.is_empty() {
        return match http::get(&format!("{base}/v1/cache/status")) {
            Ok(resp) => match serde_json::from_slice::<Value>(&resp.body) {
                Ok(v) => render_cache_status(&v),
                Err(e) => format!("(could not read /v1/cache/status: {e})"),
            },
            Err(e) => format!("(could not reach /v1/cache/status: {e:#})"),
        };
    }
    let Ok(tokens) = arg.parse::<u64>() else {
        return format!("(/cache takes a token count, not {arg:?})");
    };
    let body = json!({"kv": tokens});
    let bytes = match serde_json::to_vec(&body) {
        Ok(b) => b,
        Err(e) => return format!("(could not encode the request: {e})"),
    };
    match http::exchange("POST", &format!("{base}/v1/cache/rebuild"), Some(&bytes)) {
        Ok(resp) => {
            let v: Value = serde_json::from_slice(&resp.body).unwrap_or(Value::Null);
            // The server's own refusal text, verbatim: it names the
            // floor a shrink has to clear, which is the one thing a
            // user needs in order to retry with a number that works.
            match v.get("error").and_then(Value::as_str) {
                Some(err) => format!("(refused: {err})"),
                None => render_cache_status(&json!({"kv": v.get("kv")})),
            }
        }
        Err(e) => format!("(could not reach /v1/cache/rebuild: {e:#})"),
    }
}

fn render_cache_status(v: &Value) -> String {
    match v.get("kv").filter(|kv| !kv.is_null()) {
        Some(kv) => format!(
            "  kv pool  {} pages of {} tokens = {} tokens",
            kv.get("num_pages").and_then(Value::as_u64).unwrap_or(0),
            kv.get("page_size").and_then(Value::as_u64).unwrap_or(0),
            kv.get("num_tokens").and_then(Value::as_u64).unwrap_or(0),
        ),
        None => "  kv pool  none; this server allocates per request \
                 (start it with FRINK_KV_POOL_BLOCKS)"
            .to_string(),
    }
}

fn post_completion(base: &str, body: &Value) -> Result<String> {
    let bytes = serde_json::to_vec(body)?;
    let body = http::exchange("POST", &format!("{base}/v1/chat/completions"), Some(&bytes))?
        .ok_or_status()?;
    let v: Value = serde_json::from_slice(&body).context("parse chat response")?;
    extract_message_content(&v)
}

fn print_streamed_completion(base: &str, body: &Value, out: &mut impl Write) -> Result<String> {
    let bytes = serde_json::to_vec(body)?;
    let (status, mut reader) =
        http::open("POST", &format!("{base}/v1/chat/completions"), Some(&bytes))?;
    if !(200..300).contains(&status) {
        let mut rest = String::new();
        reader.read_to_string(&mut rest)?;
        anyhow::bail!("HTTP {status}: {rest}");
    }

    let mut render = StreamRender::default();
    let mut line = String::new();
    // The id `/v1/cancel` takes, stated by the server on the first
    // chunk. Captured rather than invented: a client cannot name a
    // generation the server has not told it about.
    let mut request_id: Option<String> = None;
    let _generating = Generating::begin();
    while reader.read_line(&mut line)? > 0 {
        let trimmed = line.trim_end();
        if let Some(data) = trimmed.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                if request_id.is_none() {
                    request_id = stated_request_id(&v);
                }
                render.push_chunk(&v, out)?;
            }
        }
        // Checked between chunks, which is the only place it can be:
        // the read above blocks, so an interrupt arriving mid-read is
        // noticed when the next chunk lands or the socket closes.
        if interrupted() {
            render.finish(out)?;
            match &request_id {
                Some(id) => match cancel_generation(base, id) {
                    Ok(()) => eprintln!("(interrupted; the server stopped generating)"),
                    // Worth saying: the local stream is over either
                    // way, but a server still decoding is burning the
                    // machine for an answer nobody will read.
                    Err(e) => eprintln!("(interrupted locally, but the cancel failed: {e:#})"),
                },
                // Before the first chunk there is no id to cancel with,
                // so dropping the socket is all there is -- which the
                // server treats as a disconnect and stops on.
                None => eprintln!("(interrupted before the server named the request)"),
            }
            return Ok(render.assembled);
        }
        line.clear();
    }
    render.finish(out)?;
    Ok(render.assembled)
}

/// The request id a streamed chunk states, if this is the chunk that
/// states it.
fn stated_request_id(chunk: &Value) -> Option<String> {
    chunk
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn cancel_generation(base: &str, request_id: &str) -> Result<()> {
    let body = serde_json::to_vec(&json!({"request_id": request_id}))?;
    let resp = http::exchange("POST", &format!("{base}/v1/cancel"), Some(&body))?;
    if !(200..300).contains(&resp.status) {
        anyhow::bail!(
            "HTTP {}: {}",
            resp.status,
            String::from_utf8_lossy(&resp.body)
        );
    }
    Ok(())
}

/// Turns one SSE chunk into terminal output, and keeps the answer.
///
/// Split out of the socket loop so its three rules can be tested
/// without one.
#[derive(Default)]
struct StreamRender {
    /// The ANSWER only. Deliberately not the chain of thought: this
    /// becomes the assistant turn in the next request's history, and a
    /// replayed chain of thought is both wrong to re-send and rejected
    /// outright by some templates.
    assembled: String,
    /// Whether the dim reasoning run is open, so the escape is written
    /// once per run rather than per delta -- and, more importantly, is
    /// always closed before the answer starts.
    thinking: bool,
}

impl StreamRender {
    fn push_chunk(&mut self, v: &Value, out: &mut impl Write) -> Result<()> {
        // Reading only `delta.content` -- which this did -- makes a
        // reasoning model look like it answered nothing for several
        // seconds and then blurted the answer, because its whole chain
        // of thought arrives on the other field.
        if let Some(thought) = v
            .pointer("/choices/0/delta/reasoning_content")
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
        {
            if !self.thinking {
                self.thinking = true;
                write!(out, "{DIM}")?;
            }
            write!(out, "{thought}")?;
            out.flush()?;
        }
        if let Some(delta) = v
            .pointer("/choices/0/delta/content")
            .and_then(|c| c.as_str())
        {
            // The first content delta closes the thought.
            if self.thinking && !delta.is_empty() {
                self.thinking = false;
                write!(out, "{RESET}")?;
            }
            write!(out, "{delta}")?;
            out.flush()?;
            self.assembled.push_str(delta);
        }
        // Non-streaming-shaped chunk with a full message.
        if self.assembled.is_empty() {
            if let Ok(full) = extract_message_content(v) {
                if self.thinking {
                    self.thinking = false;
                    write!(out, "{RESET}")?;
                }
                write!(out, "{full}")?;
                out.flush()?;
                self.assembled = full;
            }
        }
        Ok(())
    }

    /// A stream that ended mid-thought -- a cancel, a dropped
    /// connection -- must not leave the terminal dim.
    fn finish(&mut self, out: &mut impl Write) -> Result<()> {
        if self.thinking {
            self.thinking = false;
            write!(out, "{RESET}")?;
        }
        writeln!(out)?;
        Ok(())
    }
}

/// Dim, then back to normal. Written around the reasoning run only, so
/// a terminal that ignores them shows the thought as ordinary text
/// rather than as escape codes.
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

fn extract_message_content(v: &Value) -> Result<String> {
    if let Some(s) = v
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
    {
        return Ok(s.to_string());
    }
    if let Some(s) = v
        .pointer("/choices/0/delta/content")
        .and_then(|c| c.as_str())
    {
        return Ok(s.to_string());
    }
    anyhow::bail!("no choices[0].message.content in response: {v}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn thought(text: &str) -> Value {
        json!({"choices": [{"delta": {"reasoning_content": text}}]})
    }

    fn content(text: &str) -> Value {
        json!({"choices": [{"delta": {"content": text}}]})
    }

    fn render(chunks: &[Value]) -> (String, String) {
        let mut out: Vec<u8> = Vec::new();
        let mut r = StreamRender::default();
        for c in chunks {
            r.push_chunk(c, &mut out).expect("render");
        }
        r.finish(&mut out).expect("finish");
        (String::from_utf8(out).expect("utf-8"), r.assembled)
    }

    /// The bug this exists to fix: reading only `delta.content` shows
    /// nothing at all while a reasoning model thinks, so its answer
    /// looks like it arrived out of a long silence.
    #[test]
    fn a_reasoning_models_thoughts_are_shown_rather_than_dropped() {
        let (printed, _) = render(&[thought("weighing "), thought("it up"), content("Paris.")]);
        assert!(printed.contains("weighing it up"), "{printed:?}");
        assert!(printed.contains("Paris."));
    }

    /// The chain of thought is shown and NOT kept. `assembled` becomes
    /// the assistant turn in the next request's history, and replaying
    /// a chain of thought is both wrong to re-send and rejected by some
    /// templates.
    #[test]
    fn the_thought_is_never_part_of_the_answer_that_is_replayed() {
        let (_, assembled) = render(&[thought("weighing it up"), content("Paris.")]);
        assert_eq!(assembled, "Paris.");
    }

    /// One escape per run, not per delta, and the run is closed by the
    /// first content delta -- otherwise the answer itself renders dim.
    #[test]
    fn the_dim_run_is_opened_once_and_closed_before_the_answer() {
        let (printed, _) = render(&[thought("a"), thought("b"), content("X"), content("Y")]);
        assert_eq!(printed.matches(DIM).count(), 1, "{printed:?}");
        let dim_at = printed.find(DIM).expect("opened");
        let reset_at = printed.find(RESET).expect("closed");
        let answer_at = printed.find('X').expect("answered");
        assert!(dim_at < reset_at, "the run opens before it closes");
        assert!(
            reset_at < answer_at,
            "the answer must not be inside the dim run: {printed:?}"
        );
    }

    /// A stream cut off mid-thought -- a cancel, a dropped connection --
    /// must not leave the terminal dim for everything the user types
    /// afterwards.
    #[test]
    fn a_stream_that_ends_mid_thought_still_resets_the_terminal() {
        let (printed, assembled) = render(&[thought("weighing it up")]);
        assert!(printed.ends_with("\x1b[0m\n"), "{printed:?}");
        assert!(assembled.is_empty());
    }

    /// A server with no reasoning to report is unchanged: no escapes at
    /// all, so a terminal that does not understand them sees nothing new.
    #[test]
    fn a_plain_answer_is_printed_without_any_escapes() {
        let (printed, assembled) = render(&[content("hello "), content("world")]);
        assert_eq!(printed, "hello world\n");
        assert_eq!(assembled, "hello world");
    }

    /// The gears come from the SERVER, because which efforts exist is a
    /// property of the served checkpoint's own template. A client with
    /// a hardcoded list offers gears the model does not grade and hides
    /// ones it does.
    #[test]
    fn the_gears_are_whatever_this_server_advertises() {
        let body = json!({"data": [{
            "supported_reasoning_efforts": ["off", "low", "high"],
            "reasoning_effort_kwargs": {
                "off": {"enable_thinking": false},
                "low": {"enable_thinking": true, "reasoning_effort": "low"},
                "high": {"enable_thinking": true, "reasoning_effort": "high"},
            },
        }]});
        let mut gears = ThinkGears::from_models(&body);
        assert_eq!(gears.supported, vec!["off", "low", "high"]);

        // Nothing selected yet means the template's own default
        // applies, which is NOT the same as sending that default.
        assert!(gears.selected_kwargs().is_none());

        // Cycling starts at the first gear, so the first press is
        // deterministic whatever this checkpoint defaults to.
        assert_eq!(gears.select("").unwrap(), "off");
        assert_eq!(gears.select("").unwrap(), "low");
        assert_eq!(gears.select("").unwrap(), "high");
        assert_eq!(gears.select("").unwrap(), "off", "and it wraps");

        assert_eq!(gears.select("high").unwrap(), "high");
        assert_eq!(
            gears.selected_kwargs().unwrap()["reasoning_effort"],
            json!("high")
        );
    }

    /// A gear this server does not have is refused by name rather than
    /// sent: the server would either reject it or, worse, quietly
    /// render a prompt the user did not ask for.
    #[test]
    fn a_gear_this_server_does_not_have_is_refused_with_the_ones_it_does() {
        let mut gears = ThinkGears::from_models(&json!({"data": [{
            "supported_reasoning_efforts": ["on"],
            "reasoning_effort_kwargs": {"on": {}},
        }]}));
        let err = gears.select("turbo").unwrap_err();
        assert!(err.contains("turbo") && err.contains("on"), "{err}");

        // An always-thinking family advertises ONE gear with empty
        // kwargs -- there is nothing to send, and that is a real
        // answer, so it is passed through rather than skipped.
        assert_eq!(gears.select("").unwrap(), "on");
        assert_eq!(gears.selected_kwargs(), Some(json!({})));
    }

    /// A server too old to advertise gears simply has no `/think`,
    /// rather than the client inventing some.
    #[test]
    fn a_server_that_advertises_no_gears_has_no_think_command() {
        let mut gears = ThinkGears::from_models(&json!({"data": [{"id": "m"}]}));
        assert!(gears.select("").is_err());
        assert!(gears.selected_kwargs().is_none());
    }

    /// A figure the server reported as `null` prints as `-`, never as
    /// `0`. A p95 of zero reads as an instantaneous server and a memory
    /// figure of zero as an idle one -- both are exactly the shapes
    /// `/v1/stats` goes out of its way not to emit.
    #[test]
    fn a_statistic_the_server_could_not_state_renders_as_a_dash() {
        let rendered = render_stats(&json!({
            "model": "m",
            "state": "serving",
            "throughput": {"decode_tps": 12.5, "prefill_tps": 0.0},
            "requests": {
                "active": 0, "completed": 3,
                "p95_ms": Value::Null, "ttft_mean_ms": Value::Null,
            },
            "pools": {"kv_pages": Value::Null},
            "memory": Value::Null,
        }));
        assert!(rendered.contains("p95 - ms"), "{rendered}");
        assert!(rendered.contains("ttft - ms"), "{rendered}");
        assert!(rendered.contains("decode 12.5"), "{rendered}");
        assert!(
            rendered.contains("none (every request allocates privately)"),
            "an absent pool is not an empty one: {rendered}"
        );
        assert!(!rendered.contains("memory"), "absent memory prints nothing");
    }

    #[test]
    fn stats_renders_the_pool_and_memory_when_the_server_has_them() {
        let rendered = render_stats(&json!({
            "model": "m",
            "state": "serving",
            "throughput": {"decode_tps": 1.0, "prefill_tps": 2.0},
            "requests": {"active": 1, "completed": 2, "p95_ms": 30, "ttft_mean_ms": 10},
            "pools": {"kv_pages": {"used": 4, "total": 64, "page_size": 256}},
            "memory": {"bytes": 2u64 << 30, "kind": "pss"},
        }));
        assert!(rendered.contains("4/64 pages of 256 tokens"), "{rendered}");
        assert!(rendered.contains("2.00 GiB (pss)"), "{rendered}");
    }

    /// A client cannot cancel a generation the server has not named,
    /// so the id is taken from the stream rather than invented. Only
    /// the FIRST chunk carries it, which is why it is captured once and
    /// kept.
    #[test]
    fn the_request_id_is_read_from_the_chunk_that_states_it() {
        let first = json!({
            "id": "chatcmpl-1",
            "request_id": "chatcmpl-1",
            "choices": [{"delta": {"role": "assistant"}}],
        });
        assert_eq!(stated_request_id(&first).as_deref(), Some("chatcmpl-1"));

        // Later chunks carry no `request_id`, and must not be mistaken
        // for stating one -- `id` is the completion id, not the cancel
        // handle.
        let later = json!({"id": "chatcmpl-1", "choices": [{"delta": {"content": "hi"}}]});
        assert_eq!(stated_request_id(&later), None);
    }

    /// The guard exists so the flag clears however a turn ends -- an
    /// error return or a panic included. Left set, the next turn would
    /// cancel itself the moment it started.
    #[test]
    fn a_finished_turn_stops_being_interruptible_however_it_ended() {
        assert!(!GENERATING.load(std::sync::atomic::Ordering::SeqCst));
        {
            let _g = Generating::begin();
            assert!(GENERATING.load(std::sync::atomic::Ordering::SeqCst));
            INTERRUPTED.store(true, std::sync::atomic::Ordering::SeqCst);
            assert!(interrupted());
        }
        assert!(
            !GENERATING.load(std::sync::atomic::Ordering::SeqCst),
            "the guard must clear the mark on drop"
        );

        // And beginning the next turn clears a flag the last one left,
        // so it cannot cancel itself before its first chunk.
        let _g = Generating::begin();
        assert!(!interrupted());
    }
}
