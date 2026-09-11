//! Splitting a reasoning model's output into its chain of thought and
//! its answer.
//!
//! A reasoning model emits both on one token stream, separated by
//! markers its template taught it. The API contract is that they arrive
//! as two fields -- `reasoning_content` and `content` -- so something
//! has to cut the stream, and it has to do it *incrementally*, because
//! the marker that separates them can straddle any token boundary.
//!
//! # The three things that make this hard
//!
//! **A marker can arrive in pieces.** `</think>` may come as `</thi` +
//! `nk>`. So any trailing run of the buffer that is a proper prefix of
//! a marker is withheld until the next token settles it -- the same
//! rule as [`crate::policy::detokenize::stop_prefix_holdback`], for the same
//! reason: SSE cannot retract.
//!
//! **A tool call inside reasoning is not reasoning.** Some families
//! emit tool calls without ever closing the reasoning block. When a
//! tool marker appears mid-reasoning the text from the marker on is
//! *held*, not emitted, until enough of it has arrived to tell a real
//! tool call from the model quoting one in its own reasoning
//! ([`TOOL_HOLD_MAX`]). A complete `</think>` always wins over a tool
//! marker, which is what keeps a quoted marker inside a properly closed
//! reasoning block classified as reasoning.
//!
//! **Some templates open the block in the prompt.** A model whose
//! prompt already ends in `<think>` never emits the opening marker, so
//! the parser has to start *inside* the block. That is
//! `force_reasoning`, and it is read off the rendered prompt
//! ([`ReasoningFormat::prompt_opens_reasoning`]) rather than guessed
//! from the model name, because the same checkpoint does it or does not
//! depending on how the request was rendered. The same rule decides
//! muse-glimmer's `header_open`: its template ends *inside* an
//! assistant channel header, so the turn's first bytes are the tail of
//! a header nobody emitted.
//!
//! # Two shapes of grammar
//!
//! Most families delimit reasoning with a pair of markers. Two do not:
//! gpt-oss and muse-glimmer are *channel* grammars, where a header
//! names the recipient of everything up to the next terminator and the
//! recipient -- not a marker -- says which field the text belongs to.
//! Those two have their own scanners ([`HarmonyState`], [`AtemState`]);
//! the marker machinery is unused for them.
//!
//! Ported 1:1 from FreeToken's `server/reasoning_parser.py`
//! (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

/// How much text may be held after a mid-reasoning tool marker before
/// it is committed as a real tool call.
///
/// A model quoting `<tool_call>` inside its reasoning writes a few more
/// words and then closes the block; a model actually calling a tool
/// writes a long JSON payload. The bound turns "which is this?" into a
/// question that answers itself.
pub const TOOL_HOLD_MAX: usize = 512;

/// Which family's markers to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// `<think>` / `</think>`, with a DSML tool marker that can appear
    /// before the block closes. Its template always opens the block.
    DeepSeekV32,
    /// Plain `<think>` / `</think>`: Qwen3, GLM.
    Think,
    /// Plain `<think>` / `</think>`, but the template always opens the
    /// block.
    ThinkAlwaysOpen,
    /// MiniMax-M3's namespaced markers, plus its adaptive-thinking
    /// quirk.
    MiniMaxM3,
    /// Gemma-4's channel markers.
    Gemma4,
    /// The gpt-oss "harmony" channel format, which is a different shape
    /// of grammar entirely.
    GptOss,
    /// muse-glimmer's ATEM channel format: `<|start|>` headers naming a
    /// recipient, bodies closed by `<|eot|>` / `<|eom|>` /
    /// `<|end_of_text|>`.
    MuseGlimmer,
}

impl ReasoningFormat {
    /// The parser name a client or config would use.
    pub fn as_str(self) -> &'static str {
        match self {
            ReasoningFormat::DeepSeekV32 => "deepseekv32",
            ReasoningFormat::Think => "qwen3",
            ReasoningFormat::ThinkAlwaysOpen => "minimax",
            ReasoningFormat::MiniMaxM3 => "minimax_m3",
            ReasoningFormat::Gemma4 => "gemma4",
            ReasoningFormat::GptOss => "gpt_oss",
            ReasoningFormat::MuseGlimmer => "muse_glimmer",
        }
    }

    /// Resolve a configured parser name.
    pub fn parse(name: &str) -> Option<ReasoningFormat> {
        match name {
            "deepseekv32" => Some(ReasoningFormat::DeepSeekV32),
            "qwen3" | "glm" => Some(ReasoningFormat::Think),
            "minimax" => Some(ReasoningFormat::ThinkAlwaysOpen),
            "minimax_m3" => Some(ReasoningFormat::MiniMaxM3),
            "gemma4" => Some(ReasoningFormat::Gemma4),
            "gpt_oss" | "gpt-oss" => Some(ReasoningFormat::GptOss),
            "muse_glimmer" => Some(ReasoningFormat::MuseGlimmer),
            _ => None,
        }
    }

    /// Which parser a checkpoint's identity implies.
    ///
    /// Order matters: the specific arms have to be tested before the
    /// family arms they are a special case of, or MiniMax-M3 resolves
    /// to MiniMax and loses its namespaced markers.
    pub fn infer(marker: &str) -> Option<ReasoningFormat> {
        let marker = marker.to_ascii_lowercase();
        let has = |needle: &str| marker.contains(needle);
        if has("gpt_oss") || has("gpt-oss") || has("gptoss") {
            return Some(ReasoningFormat::GptOss);
        }
        // Written with every separator its checkpoints use; the two
        // words never occur together in another family's name.
        if has("muse") && has("glimmer") {
            return Some(ReasoningFormat::MuseGlimmer);
        }
        if has("deepseek") && (has("v4") || has("v3.2") || has("v32")) {
            return Some(ReasoningFormat::DeepSeekV32);
        }
        // R1 and every distill of it. The distills are named for the
        // model they were distilled INTO, so
        // `DeepSeek-R1-Distill-Qwen-1.5B` matched neither the DeepSeek
        // arm above (no v3.2/v4) nor the `qwen3` arm below, and got no
        // reasoning format at all. The result was `<think>` sitting in
        // `content` with `reasoning_content` empty, which is exactly
        // what this parser exists to prevent.
        if has("deepseek") && has("r1") {
            return Some(ReasoningFormat::Think);
        }
        if has("qwen3") {
            return Some(ReasoningFormat::Think);
        }
        if has("glm") {
            return Some(ReasoningFormat::Think);
        }
        if has("minimax_m3") || has("minimax-m3") || has("minimaxm3") {
            return Some(ReasoningFormat::MiniMaxM3);
        }
        if has("minimax") {
            return Some(ReasoningFormat::ThinkAlwaysOpen);
        }
        if has("gemma4") {
            return Some(ReasoningFormat::Gemma4);
        }
        None
    }

    /// Does a rendered prompt leave the model *inside* a reasoning
    /// block?
    ///
    /// This is the honest way to decide `force_reasoning`: some
    /// templates open the block in the prompt when thinking is enabled,
    /// so the model's first token is already reasoning and no opening
    /// marker will ever arrive. Which templates do that is not
    /// derivable from the family (Qwen3 does it on one revision and not
    /// the next) -- but it is plainly *observable* in the text that was
    /// actually rendered, so this reads that instead of assuming.
    ///
    /// A family whose block is always open needs no evidence, and the
    /// harmony format is not marker-delimited at all, so neither
    /// consults the prompt.
    pub fn prompt_opens_reasoning(self, prompt: &str) -> bool {
        let m = self.markers();
        if m.always_open {
            return true;
        }
        if m.start.is_empty() {
            return false;
        }
        match prompt.rfind(m.start) {
            None => false,
            Some(start) => prompt.rfind(m.end).is_none_or(|end| end < start),
        }
    }

    /// The opener and closer a chain of thought can be *written back*
    /// between, for a prompt that continues one.
    ///
    /// `None` for the channel grammars (harmony, ATEM): their reasoning
    /// is not a marker pair around text, so there is nothing a caller
    /// could concatenate a replayed thought into. A continuation for
    /// those families has to be refused rather than rendered, and this
    /// is the one place that says which families those are.
    pub fn continuation_markers(self) -> Option<(&'static str, &'static str)> {
        match self {
            // Harmony has no markers at all; ATEM's pair is a channel
            // HEADER (`<|start|>` ... `<|message|>`), which the parser
            // uses for `prompt_opens_reasoning` and which is not a pair
            // a thought can be written between.
            ReasoningFormat::GptOss | ReasoningFormat::MuseGlimmer => None,
            ReasoningFormat::DeepSeekV32
            | ReasoningFormat::Think
            | ReasoningFormat::ThinkAlwaysOpen
            | ReasoningFormat::MiniMaxM3
            | ReasoningFormat::Gemma4 => {
                let m = self.markers();
                Some((m.start, m.end))
            }
        }
    }

    /// Whether the parser reads every generation as starting INSIDE the
    /// block regardless of what the prompt says. A prompt that closes
    /// the block itself cannot tell such a parser so, which is why a
    /// content continuation is refused for these families.
    pub fn always_open(self) -> bool {
        self.markers().always_open
    }

    fn markers(self) -> Markers {
        match self {
            ReasoningFormat::DeepSeekV32 => Markers {
                start: "<think>",
                end: "</think>",
                // The tool grammar can open before the reasoning block
                // closes, so the reasoning parser has to know about it.
                tool_start: Some("<｜DSML｜"),
                always_open: true,
            },
            ReasoningFormat::Think => Markers {
                start: "<think>",
                end: "</think>",
                tool_start: None,
                always_open: false,
            },
            ReasoningFormat::ThinkAlwaysOpen => Markers {
                start: "<think>",
                end: "</think>",
                tool_start: None,
                always_open: true,
            },
            ReasoningFormat::MiniMaxM3 => Markers {
                start: "<mm:think>",
                end: "</mm:think>",
                tool_start: Some("]<]minimax[>[<tool_call>"),
                always_open: false,
            },
            ReasoningFormat::Gemma4 => Markers {
                start: "<|channel>thought\n",
                end: "<channel|>",
                tool_start: None,
                always_open: false,
            },
            // Not marker-delimited; see `HarmonyState`.
            ReasoningFormat::GptOss => Markers {
                start: "",
                end: "",
                tool_start: None,
                always_open: false,
            },
            // Not marker-delimited either; see `AtemState`. The pair is
            // still filled in, because for a channel grammar
            // `prompt_opens_reasoning` asks exactly the same question
            // the marker families ask -- "did the rendered prompt stop
            // between the opener and its closer?" -- with the channel
            // header's own two markers. That answer is muse-glimmer's
            // `header_open`.
            ReasoningFormat::MuseGlimmer => Markers {
                start: ATEM_START,
                end: ATEM_MESSAGE,
                tool_start: None,
                always_open: false,
            },
        }
    }
}

/// The ATEM control tokens, shared with the tool-call side.
///
/// muse-glimmer's turn is a sequence of channels: `<|start|>` opens a
/// header, `to=<recipient>` names who the body is for, `<|message|>`
/// ends the header, and one of [`ATEM_CLOSING_TOKENS`] ends the body.
/// The recipient is what classifies the text -- `self` is reasoning,
/// `user` is the answer, anything else is a tool.
pub const ATEM_START: &str = "<|start|>";
/// The token that ends a channel header and starts its body.
pub const ATEM_MESSAGE: &str = "<|message|>";
/// Everything that can end a channel body. `<|eom|>` separates channels
/// within one turn; the other two are stops.
pub const ATEM_CLOSING_TOKENS: [&str; 3] = ["<|eot|>", "<|eom|>", "<|end_of_text|>"];
/// Every ATEM control token, for the hold-back and the "a header never
/// contains a marker" test.
pub const ATEM_ALL_TOKENS: [&str; 5] = [
    ATEM_START,
    ATEM_MESSAGE,
    "<|eot|>",
    "<|eom|>",
    "<|end_of_text|>",
];
/// The longest a recipient name may be, in characters, in *every*
/// header shape.
///
/// One bound everywhere is what keeps streaming and one-shot agreeing
/// about which recipients are valid: a longer name degrades the same
/// way in the hold-back, the inline switch and the full header.
const ATEM_NAME_MAX: usize = 64;
/// How far past `<|start|>` a real header can run before its
/// `<|message|>` arrives (`assistant to=<name>` plus slack).
///
/// Past this a `<|start|>` cannot be opening a header any more, so it
/// is released as literal text instead of held forever -- a degenerate
/// wire must neither stall the stream nor eat the turn.
pub const ATEM_HEADER_SPAN: usize = 128;
/// How far back the hold-back looks for a headerless `to=…<|message|>`
/// switch that is still arriving.
const ATEM_HOLD_WINDOW: usize = 96;

/// Whether a complete ATEM control token lies inside `text[start..end]`.
///
/// A channel header can never contain one, so a `<|start|>` with a
/// marker before its `<|message|>` is not a header -- it is literal
/// text the model wrote. `<|message|>` itself is excluded: it is the
/// terminator being sought, not a byte of the header.
pub fn atem_marker_inside(text: &str, start: usize, end: usize) -> bool {
    if start > end || end > text.len() {
        return false;
    }
    let window = &text[start..end];
    ATEM_ALL_TOKENS
        .iter()
        .any(|token| *token != ATEM_MESSAGE && window.contains(token))
}

/// The streaming hold-back for ATEM output: the longest suffix that
/// could still grow into a control token *or* into a headerless
/// `to=<name><|message|>` channel switch.
///
/// The switch half is what the generic marker hold-back cannot do. A
/// switch has no fixed opener -- the model simply writes
/// `to=weather.get<|message|>` in the middle of a body -- so the only
/// way to avoid emitting text that later turns out to be a header is to
/// withhold a whole window of it ([`ATEM_HOLD_WINDOW`] characters)
/// while a `to=` is still unresolved. Emitted text can never shrink,
/// and SSE cannot retract, so the window is the price.
pub fn atem_hold_len(text: &str) -> usize {
    let owned: Vec<String> = ATEM_ALL_TOKENS.iter().map(|t| t.to_string()).collect();
    let mut best = crate::policy::detokenize::stop_prefix_holdback(text, &owned);

    let window_start = text
        .char_indices()
        .rev()
        .take(ATEM_HOLD_WINDOW)
        .last()
        .map(|(index, _)| index)
        .unwrap_or(0);
    let window = &text[window_start..];
    if let Some(index) = window.rfind("to=") {
        let tail = &window[index..];
        let name = &tail["to=".len()..];
        let run: String = name
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '<')
            .collect();
        if run.len() == name.len() && run.chars().count() <= ATEM_NAME_MAX {
            // The name (or the bare `to=`) is still streaming.
            best = best.max(tail.len());
        } else if run.chars().count() <= ATEM_NAME_MAX
            && !run.is_empty()
            && ATEM_MESSAGE.starts_with(&name[run.len()..])
            && name.len() > run.len()
        {
            // The name is complete and a partial `<|message|>` follows.
            best = best.max(tail.len());
        }
    }
    // Proper prefixes of `to=` itself: at most a one-chunk delay.
    for prefix in ["t", "to"] {
        if prefix.len() > best && text.ends_with(prefix) {
            best = prefix.len();
        }
    }
    best
}

/// A headerless channel switch: `to=<name>` abutting `<|message|>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtemSwitch {
    /// Where the `to=` begins.
    pub start: usize,
    /// Just past the `<|message|>`.
    pub end: usize,
    pub recipient: String,
}

/// The first complete headerless switch in `text`, if any.
///
/// The model leaves a channel without emitting `<|eom|>` more often
/// than the template suggests, and a complete match of this shape is a
/// channel boundary wherever it appears -- including inside a body.
pub fn find_atem_switch(text: &str) -> Option<AtemSwitch> {
    let mut from = 0usize;
    while let Some(index) = text[from..].find("to=").map(|i| i + from) {
        let name_start = index + "to=".len();
        let run: String = text[name_start..]
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '<')
            .collect();
        let count = run.chars().count();
        if (1..=ATEM_NAME_MAX).contains(&count)
            && text[name_start + run.len()..].starts_with(ATEM_MESSAGE)
        {
            return Some(AtemSwitch {
                start: index,
                end: name_start + run.len() + ATEM_MESSAGE.len(),
                recipient: run,
            });
        }
        from = index + "to=".len();
    }
    None
}

/// The recipient a header names, if it names one.
pub fn atem_recipient(header: &str) -> Option<String> {
    let index = header.find("to=")? + "to=".len();
    let run: String = header[index..]
        .chars()
        .take_while(|c| !c.is_whitespace() && *c != '<')
        .collect();
    let count = run.chars().count();
    (1..=ATEM_NAME_MAX).contains(&count).then_some(run)
}

/// What ends a channel body, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtemBoundary {
    /// One of [`ATEM_CLOSING_TOKENS`], which belongs to the body it
    /// closes.
    Closer { at: usize, token: &'static str },
    /// An abutting `<|start|>`: the next channel's header, which
    /// belongs to the next channel.
    Start { at: usize },
    /// A headerless switch.
    Switch(AtemSwitch),
}

impl AtemBoundary {
    pub fn at(&self) -> usize {
        match self {
            AtemBoundary::Closer { at, .. } | AtemBoundary::Start { at } => *at,
            AtemBoundary::Switch(switch) => switch.start,
        }
    }
}

/// The earliest channel boundary in `text`, if one is complete.
pub fn atem_boundary(text: &str) -> Option<AtemBoundary> {
    let mut best: Option<AtemBoundary> = None;
    let mut consider = |candidate: AtemBoundary| {
        if best.as_ref().is_none_or(|b| candidate.at() < b.at()) {
            best = Some(candidate);
        }
    };
    for token in ATEM_CLOSING_TOKENS {
        if let Some(at) = text.find(token) {
            consider(AtemBoundary::Closer { at, token });
        }
    }
    if let Some(at) = text.find(ATEM_START) {
        consider(AtemBoundary::Start { at });
    }
    if let Some(switch) = find_atem_switch(text) {
        consider(AtemBoundary::Switch(switch));
    }
    best
}

#[derive(Debug, Clone, Copy)]
struct Markers {
    start: &'static str,
    end: &'static str,
    tool_start: Option<&'static str>,
    /// Whether this family's template opens the reasoning block itself,
    /// so the model never emits the opening marker.
    always_open: bool,
}

/// One increment's worth of split output. Either side may be empty.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReasoningDelta {
    pub reasoning: String,
    pub content: String,
}

impl ReasoningDelta {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty()
    }
}

/// The gpt-oss channel format's incremental state.
#[derive(Debug, Default)]
struct HarmonyState {
    buffer: String,
    emitted_reasoning: usize,
    emitted_content: usize,
    /// Whether the last scan ran in passthrough mode -- output with no
    /// channel markers in it at all. The emitted counters index into
    /// the scan's own output, so they are meaningless across a mode
    /// change and are reset when one happens.
    passthrough: bool,
}

const HARMONY_CHANNEL: &str = "<|channel|>";
const HARMONY_MESSAGE: &str = "<|message|>";
/// Anything that can end a harmony message body.
const HARMONY_BOUNDARIES: [&str; 5] = [
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|start|>",
    "<|channel|>",
];
/// The subset that is a *closing* token, and so belongs to the message
/// it closes rather than to whatever follows.
const HARMONY_CLOSERS: [&str; 3] = ["<|end|>", "<|return|>", "<|call|>"];

/// Where in the ATEM channel grammar the stream currently is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtemMode {
    /// Between channels, looking for the next header.
    Seek,
    /// Streaming a channel body to [`AtemState::recipient`].
    Body,
}

/// The ATEM channel format's incremental state.
#[derive(Debug)]
struct AtemState {
    buffer: String,
    mode: AtemMode,
    recipient: String,
    /// Whether the `<|start|>` at the head of the buffer is the seed
    /// rather than something the model emitted.
    ///
    /// The seed exists because the template's prompt ends *inside* a
    /// header, so the turn's first bytes are a header continuation
    /// (` to=self<|message|>…`). Seeding the marker sends those bytes
    /// through the ordinary full-header machinery instead of a second,
    /// guessing code path -- but if the candidate is ever ruled *not* a
    /// header, the seed must be dropped rather than delivered: the
    /// model never wrote it.
    synthetic_open: bool,
    header_open: bool,
}

impl AtemState {
    fn new(header_open: bool) -> Self {
        AtemState {
            buffer: if header_open {
                ATEM_START.to_string()
            } else {
                String::new()
            },
            mode: AtemMode::Seek,
            recipient: "user".to_string(),
            synthetic_open: header_open,
            header_open,
        }
    }
}

/// Splits a model's output into reasoning and answer.
#[derive(Debug)]
pub struct ReasoningParser {
    format: ReasoningFormat,
    markers: Markers,
    /// Whether output is currently inside a reasoning block.
    in_reasoning: bool,
    /// Whether to stream reasoning as it arrives, or hold it until the
    /// block closes.
    stream_reasoning: bool,
    buffer: String,
    stripped_start: bool,
    harmony: HarmonyState,
    atem: AtemState,
    /// MiniMax-M3 only: the template may or may not have opened the
    /// block, and the model says which by whether it leads with a bare
    /// closer.
    leading_closer_pending: bool,
    head_buffer: String,
}

impl ReasoningParser {
    /// `force_reasoning` starts the parser inside the block, for a
    /// template that opened it in the prompt.
    ///
    /// For the ATEM channel format the same flag means "the prompt
    /// ended inside a channel header", which is the same fact read the
    /// same way ([`ReasoningFormat::prompt_opens_reasoning`]) and is
    /// what seeds [`AtemState::synthetic_open`].
    pub fn new(format: ReasoningFormat, force_reasoning: bool, stream_reasoning: bool) -> Self {
        let markers = format.markers();
        let force = force_reasoning || markers.always_open;
        ReasoningParser {
            format,
            markers,
            in_reasoning: force,
            stream_reasoning,
            buffer: String::new(),
            stripped_start: false,
            harmony: HarmonyState::default(),
            atem: AtemState::new(force),
            leading_closer_pending: format == ReasoningFormat::MiniMaxM3 && !force,
            head_buffer: String::new(),
        }
    }

    pub fn format(&self) -> ReasoningFormat {
        self.format
    }

    /// Split a complete response.
    ///
    /// Runs the streaming machinery on a fresh parser rather than
    /// having a second implementation: two implementations of a rule
    /// this fiddly disagree eventually, and the disagreement shows up
    /// as a streamed answer that differs from the non-streamed one.
    pub fn parse_complete(&self, text: &str) -> ReasoningDelta {
        let mut clone = ReasoningParser::new(self.format, self.in_reasoning, true);
        clone.leading_closer_pending = self.leading_closer_pending;
        let mut out = clone.push(text);
        let tail = clone.flush();
        out.reasoning.push_str(&tail.reasoning);
        out.content.push_str(&tail.content);
        ReasoningDelta {
            reasoning: out.reasoning.trim().to_string(),
            content: out.content.trim().to_string(),
        }
    }

    /// Feed one increment of model output.
    pub fn push(&mut self, chunk: &str) -> ReasoningDelta {
        if self.format == ReasoningFormat::GptOss {
            return self.push_harmony(chunk);
        }
        if self.format == ReasoningFormat::MuseGlimmer {
            self.atem.buffer.push_str(chunk);
            return self.drain_atem(false);
        }
        if self.leading_closer_pending {
            if let Some(delta) = self.push_leading_closer(chunk) {
                return delta;
            }
        }
        self.push_marker(chunk)
    }

    /// Whatever is still held when the stream ends.
    ///
    /// Released, not dropped: it was withheld against a marker that
    /// never arrived, so it is ordinary output, and discarding it would
    /// truncate every answer whose tail happens to look like the start
    /// of a marker.
    pub fn flush(&mut self) -> ReasoningDelta {
        if self.format == ReasoningFormat::GptOss {
            let (reasoning, content) = self.harmony_scan(false);
            let delta = self.harmony_delta(reasoning, content);
            self.harmony = HarmonyState::default();
            return delta;
        }
        if self.format == ReasoningFormat::MuseGlimmer {
            let delta = self.drain_atem(true);
            self.atem = AtemState::new(self.atem.header_open);
            return delta;
        }
        if self.leading_closer_pending && !self.head_buffer.is_empty() {
            // The model never produced the bare closer, so the head was
            // ordinary output all along.
            let head = std::mem::take(&mut self.head_buffer);
            self.leading_closer_pending = false;
            let mut delta = self.push_marker(&head);
            let tail = self.flush();
            delta.reasoning.push_str(&tail.reasoning);
            delta.content.push_str(&tail.content);
            return delta;
        }
        let held = std::mem::take(&mut self.buffer);
        if held.is_empty() {
            return ReasoningDelta::default();
        }
        // A held run that begins with a tool marker is a tool call,
        // whatever the block state said.
        if let Some(tool) = self.markers.tool_start {
            if held.trim_start().starts_with(tool) {
                self.in_reasoning = false;
                return ReasoningDelta {
                    content: held,
                    ..Default::default()
                };
            }
        }
        if self.in_reasoning {
            ReasoningDelta {
                reasoning: held,
                ..Default::default()
            }
        } else {
            ReasoningDelta {
                content: held,
                ..Default::default()
            }
        }
    }

    /// Consume as much of the ATEM buffer as is decided.
    ///
    /// Channel bodies are routed by recipient: `self` is reasoning,
    /// `user` (and a header naming nobody) is the answer, and anything
    /// else is a tool channel, which is passed through to `content`
    /// **verbatim, header and terminator included** -- the tool parser
    /// downstream needs the whole block, and a channel that ended
    /// without its own terminator (an abutting header, a headerless
    /// switch) is given a synthetic `<|eom|>` so that parser always
    /// receives a delimited block.
    ///
    /// The buffer is *consumed* as it emits rather than re-scanned from
    /// byte zero each chunk: this family's default chain of thought is
    /// long, and a per-chunk rescan is quadratic in it.
    ///
    /// `at_end` drops the hold-back: text withheld against a boundary
    /// that never arrived is ordinary output, and dropping it would
    /// truncate every reply whose tail happens to look like a header.
    fn drain_atem(&mut self, at_end: bool) -> ReasoningDelta {
        let mut out = ReasoningDelta::default();
        while !self.atem.buffer.is_empty() {
            let buf = std::mem::take(&mut self.atem.buffer);
            let progressed = match self.atem.mode {
                AtemMode::Seek => self.atem_seek(&buf, at_end, &mut out),
                AtemMode::Body => self.atem_body(&buf, at_end, &mut out),
            };
            if !progressed {
                break;
            }
        }
        out
    }

    /// Between channels: find the next header. Returns whether the scan
    /// made progress and should run again.
    fn atem_seek(&mut self, buf: &str, at_end: bool, out: &mut ReasoningDelta) -> bool {
        let start = buf.find(ATEM_START);
        if let Some(switch) = find_atem_switch(buf) {
            if start.is_none_or(|s| switch.start < s) {
                push_seek_text(&mut out.content, &buf[..switch.start]);
                self.atem.buffer = buf[switch.end..].to_string();
                let header = buf[switch.start..switch.end].to_string();
                self.atem_begin_body(&switch.recipient, &header, out);
                return true;
            }
        }
        let Some(start) = start else {
            // Nothing in sight: stream eagerly, holding only the tail
            // that could still grow into a boundary.
            let hold = if at_end { 0 } else { atem_hold_len(buf) };
            let split = crate::policy::detokenize::floor_char_boundary(buf, buf.len() - hold);
            push_seek_text(&mut out.content, &buf[..split]);
            self.atem.buffer = buf[split..].to_string();
            return false;
        };
        let synthetic = self.atem.synthetic_open;
        let header_start = start + ATEM_START.len();
        let message = buf[header_start..]
            .find(ATEM_MESSAGE)
            .map(|i| i + header_start);
        // A control token inside the candidate settles it immediately:
        // headers never contain one, so this `<|start|>` is literal
        // text. Deciding here rather than at the span bound is what
        // keeps a quoted marker from holding the stream.
        if atem_marker_inside(buf, header_start, message.unwrap_or(buf.len())) {
            push_seek_text(&mut out.content, &buf[..start]);
            self.atem_release_start(synthetic, out);
            self.atem.buffer = buf[header_start..].to_string();
            return true;
        }
        match message {
            // A `<|message|>` too far away belongs to the *next*
            // segment's real header, not to this marker: a stray
            // literal `<|start|>`, junk, then a genuine header. Parsing
            // the pair as one giant header would also let a `to=`
            // inside the junk hijack the recipient.
            Some(message) if message - header_start > ATEM_HEADER_SPAN => {
                push_seek_text(&mut out.content, &buf[..start]);
                self.atem_release_start(synthetic, out);
                self.atem.buffer = buf[header_start..].to_string();
                true
            }
            Some(message) => {
                push_seek_text(&mut out.content, &buf[..start]);
                let recipient =
                    atem_recipient(&buf[header_start..message]).unwrap_or_else(|| "user".into());
                let body_start = message + ATEM_MESSAGE.len();
                let header = buf[start..body_start].to_string();
                self.atem.synthetic_open = false;
                self.atem.buffer = buf[body_start..].to_string();
                self.atem_begin_body(&recipient, &header, out);
                true
            }
            None => {
                // A complete `<|start|>` whose `<|message|>` has not
                // arrived. Text before it is content now and is never
                // re-dropped; the candidate itself is held, with slack
                // for a `<|message|>` mid-arrival so a protocol-legal
                // long-name header is not cut at the nominal span.
                push_seek_text(&mut out.content, &buf[..start]);
                self.atem.buffer = buf[start..].to_string();
                let held = self.atem.buffer.len() - ATEM_START.len();
                if held > ATEM_HEADER_SPAN + ATEM_MESSAGE.len() {
                    self.atem_release_start(synthetic, out);
                    self.atem.buffer = self.atem.buffer[ATEM_START.len()..].to_string();
                    return true;
                }
                if at_end {
                    // At end of stream a candidate that never received
                    // its `<|message|>` is not a header: deliver the
                    // tail, drop only the marker.
                    let tail = self.atem.buffer[ATEM_START.len()..].to_string();
                    push_seek_text(&mut out.content, &tail);
                    self.atem.synthetic_open = false;
                    self.atem.buffer.clear();
                }
                false
            }
        }
    }

    /// Inside a channel body: emit up to the next boundary. Returns
    /// whether the scan made progress and should run again.
    fn atem_body(&mut self, buf: &str, at_end: bool, out: &mut ReasoningDelta) -> bool {
        let recipient = self.atem.recipient.clone();
        let Some(boundary) = atem_boundary(buf) else {
            let hold = if at_end { 0 } else { atem_hold_len(buf) };
            let split = crate::policy::detokenize::floor_char_boundary(buf, buf.len() - hold);
            atem_emit(out, &recipient, &buf[..split]);
            self.atem.buffer = buf[split..].to_string();
            return false;
        };
        let end = boundary.at();
        let mut body = buf[..end].to_string();
        if !matches!(recipient.as_str(), "self" | "user") {
            // A tool slice keeps its terminator; one that ended without
            // a terminator gets a synthetic `<|eom|>`, so the tool
            // parser always receives a delimited block.
            body.push_str(match &boundary {
                AtemBoundary::Closer { token, .. } => token,
                _ => "<|eom|>",
            });
        }
        atem_emit(out, &recipient, &body);
        match boundary {
            AtemBoundary::Closer { token, .. } => {
                self.atem.buffer = buf[end + token.len()..].to_string();
                self.atem.mode = AtemMode::Seek;
            }
            AtemBoundary::Start { .. } => {
                self.atem.buffer = buf[end..].to_string();
                self.atem.mode = AtemMode::Seek;
            }
            AtemBoundary::Switch(switch) => {
                self.atem.buffer = buf[switch.end..].to_string();
                let header = buf[switch.start..switch.end].to_string();
                self.atem_begin_body(&switch.recipient, &header, out);
            }
        }
        true
    }

    /// Enter a channel body. A tool channel's header is part of the
    /// slice the tool parser has to see, so it goes out with the body.
    fn atem_begin_body(&mut self, recipient: &str, header: &str, out: &mut ReasoningDelta) {
        self.atem.recipient = recipient.to_string();
        self.atem.mode = AtemMode::Body;
        if !matches!(recipient, "self" | "user") {
            atem_emit(out, recipient, header);
        }
    }

    /// Release a `<|start|>` that turned out not to open a header --
    /// unless it is the synthetic seed, which the model never emitted
    /// and which must therefore never be delivered.
    fn atem_release_start(&mut self, synthetic: bool, out: &mut ReasoningDelta) {
        if !synthetic {
            out.content.push_str(ATEM_START);
        }
        self.atem.synthetic_open = false;
    }

    /// MiniMax-M3's adaptive mode: the model signals "no thinking this
    /// turn" by leading with a bare closing marker, so the head of the
    /// stream is held until it is clear which it is.
    fn push_leading_closer(&mut self, chunk: &str) -> Option<ReasoningDelta> {
        self.head_buffer.push_str(chunk);
        let head = self.head_buffer.trim_start();
        if let Some(rest) = head.strip_prefix(self.markers.end) {
            let rest = rest.to_string();
            self.head_buffer.clear();
            self.leading_closer_pending = false;
            self.in_reasoning = false;
            return Some(self.push_marker(&rest));
        }
        if is_partial_prefix(head, self.markers.end) {
            // Still could become the bare closer; keep holding.
            return Some(ReasoningDelta::default());
        }
        // It diverged: replay the held head verbatim through the normal
        // machinery.
        let head = std::mem::take(&mut self.head_buffer);
        self.leading_closer_pending = false;
        Some(self.push_marker(&head))
    }

    fn push_marker(&mut self, chunk: &str) -> ReasoningDelta {
        self.buffer.push_str(chunk);

        if !self.stripped_start && !self.markers.start.is_empty() {
            if let Some(index) = self.buffer.find(self.markers.start) {
                let mut rebuilt = String::with_capacity(self.buffer.len());
                rebuilt.push_str(&self.buffer[..index]);
                rebuilt.push_str(&self.buffer[index + self.markers.start.len()..]);
                self.buffer = rebuilt;
                self.stripped_start = true;
                self.in_reasoning = true;
            }
        }

        // A complete closing marker beats everything: it is the only
        // thing that definitively ends the block, and honouring a tool
        // marker first would misclassify a tool marker the model merely
        // quoted inside its reasoning.
        if self.in_reasoning && !self.markers.end.is_empty() {
            if let Some(end) = self.buffer.find(self.markers.end) {
                let reasoning = self.buffer[..end].trim_end().to_string();
                let content = self.buffer[end + self.markers.end.len()..].to_string();
                self.buffer.clear();
                self.in_reasoning = false;
                return ReasoningDelta { reasoning, content };
            }
        }

        // A tool marker while still in reasoning: hold from the marker
        // on, and commit only once enough has arrived to be sure.
        if self.in_reasoning {
            if let Some(tool) = self.markers.tool_start {
                if let Some(index) = self.buffer.find(tool) {
                    let reasoning = self.buffer[..index].to_string();
                    let held = self.buffer[index..].to_string();
                    self.buffer = held.clone();
                    if held.len() > TOOL_HOLD_MAX {
                        self.buffer.clear();
                        self.in_reasoning = false;
                        return ReasoningDelta {
                            reasoning,
                            content: held,
                        };
                    }
                    return ReasoningDelta {
                        reasoning,
                        ..Default::default()
                    };
                }
            }
        }

        let hold = self.trailing_partial_len();
        let safe_end = self.buffer.len() - hold;
        if self.in_reasoning && !self.stream_reasoning {
            // Reasoning is being withheld wholesale; nothing goes out
            // until the block closes.
            return ReasoningDelta::default();
        }
        let safe: String = self.buffer.drain(..safe_end).collect();
        if safe.is_empty() {
            return ReasoningDelta::default();
        }
        if self.in_reasoning {
            ReasoningDelta {
                reasoning: safe,
                ..Default::default()
            }
        } else {
            ReasoningDelta {
                content: safe,
                ..Default::default()
            }
        }
    }

    /// The longest trailing run of the buffer that could still grow
    /// into one of the markers this parser tracks.
    fn trailing_partial_len(&self) -> usize {
        let mut candidates: Vec<&str> = vec![self.markers.end];
        if !self.stripped_start && !self.markers.start.is_empty() {
            candidates.push(self.markers.start);
        }
        if let Some(tool) = self.markers.tool_start {
            candidates.push(tool);
        }
        candidates.retain(|c| !c.is_empty());
        let owned: Vec<String> = candidates.iter().map(|c| c.to_string()).collect();
        let hold = crate::policy::detokenize::stop_prefix_holdback(&self.buffer, &owned);
        let split =
            crate::policy::detokenize::floor_char_boundary(&self.buffer, self.buffer.len() - hold);
        self.buffer.len() - split
    }

    fn push_harmony(&mut self, chunk: &str) -> ReasoningDelta {
        self.harmony.buffer.push_str(chunk);
        let (reasoning, content) = self.harmony_scan(true);
        self.harmony_delta(reasoning, content)
    }

    /// Emit only what this scan added beyond what has already gone out.
    fn harmony_delta(&mut self, reasoning: String, content: String) -> ReasoningDelta {
        let passthrough = !self.harmony.buffer.contains(HARMONY_CHANNEL);
        if passthrough != self.harmony.passthrough {
            // The scan's output is a different string now; offsets into
            // the old one would skip the head of the new one.
            self.harmony.emitted_reasoning = 0;
            self.harmony.emitted_content = 0;
            self.harmony.passthrough = passthrough;
        }
        let new_reasoning = reasoning
            .get(self.harmony.emitted_reasoning..)
            .unwrap_or("")
            .to_string();
        let new_content = content
            .get(self.harmony.emitted_content..)
            .unwrap_or("")
            .to_string();
        self.harmony.emitted_reasoning = reasoning.len();
        self.harmony.emitted_content = content.len();
        ReasoningDelta {
            reasoning: new_reasoning,
            content: new_content,
        }
    }

    /// Re-scan the whole harmony buffer into (reasoning, content).
    ///
    /// Re-scanning rather than advancing an index is what makes a
    /// header that arrives in pieces work: a `<|channel|>` with no
    /// `<|message|>` yet is simply not a message this time round, and
    /// becomes one when the rest lands.
    fn harmony_scan(&self, hold_partial: bool) -> (String, String) {
        let text = &self.harmony.buffer;
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut cursor = 0usize;

        if !text.contains(HARMONY_CHANNEL) {
            // Not harmony output at all -- a model that answered
            // plainly. Pass it through as content rather than
            // discarding it, holding back anything that could still
            // become a channel header.
            let mut body = text.as_str();
            if hold_partial {
                let markers = [HARMONY_CHANNEL.to_string()];
                let hold = crate::policy::detokenize::stop_prefix_holdback(body, &markers);
                body = &body
                    [..crate::policy::detokenize::floor_char_boundary(body, body.len() - hold)];
            }
            return (String::new(), body.to_string());
        }

        while let Some(channel) = text[cursor..].find(HARMONY_CHANNEL).map(|i| i + cursor) {
            let header_start = channel + HARMONY_CHANNEL.len();
            let Some(message) = text[header_start..]
                .find(HARMONY_MESSAGE)
                .map(|i| i + header_start)
            else {
                // The header is still streaming.
                break;
            };
            let header = &text[header_start..message];
            let body_start = message + HARMONY_MESSAGE.len();
            let (end, matched) = earliest_marker(text, body_start, &HARMONY_BOUNDARIES);
            let channel_name = header.split_whitespace().next().unwrap_or("");
            // A commentary channel addressed to a function is a tool
            // call, and belongs to the tool parser verbatim -- markers
            // and all -- not to either text field.
            let is_tool = channel_name == "commentary" && header.contains("to=functions");

            if is_tool {
                let slice_end = match matched {
                    Some(marker) if HARMONY_CLOSERS.contains(&marker) => end + marker.len(),
                    _ => end,
                };
                content.push_str(&text[channel..slice_end]);
            } else {
                let mut body = &text[body_start..end];
                if matched.is_none() && hold_partial {
                    let markers: Vec<String> =
                        HARMONY_BOUNDARIES.iter().map(|m| m.to_string()).collect();
                    let hold = crate::policy::detokenize::stop_prefix_holdback(body, &markers);
                    body = &body
                        [..crate::policy::detokenize::floor_char_boundary(body, body.len() - hold)];
                }
                if channel_name == "analysis" {
                    reasoning.push_str(body);
                } else {
                    content.push_str(body);
                }
            }
            if matched.is_none() {
                break;
            }
            cursor = end;
        }
        (reasoning, content)
    }
}

/// The earliest of `markers` at or after `from`, and which one it was.
fn earliest_marker<'a>(text: &str, from: usize, markers: &[&'a str]) -> (usize, Option<&'a str>) {
    let mut best = text.len();
    let mut which = None;
    for marker in markers {
        if let Some(index) = text[from..].find(marker).map(|i| i + from) {
            if index < best {
                best = index;
                which = Some(*marker);
            }
        }
    }
    (best, which)
}

/// Route one piece of a channel body by its recipient.
fn atem_emit(out: &mut ReasoningDelta, recipient: &str, piece: &str) {
    if piece.is_empty() {
        return;
    }
    if recipient == "self" {
        out.reasoning.push_str(piece);
    } else {
        // A `user` body is unwrapped; a tool slice is verbatim.
        out.content.push_str(piece);
    }
}

/// Deliver inter-channel text as content, stripping the protocol debris
/// a degenerate decode leaves in it.
///
/// Text between channels is still text -- delivering it is what keeps a
/// reply stranded after a stray closer from vanishing -- but a repeated
/// terminator in it is not something a client should ever render.
fn push_seek_text(content: &mut String, piece: &str) {
    if piece.is_empty() {
        return;
    }
    let mut piece = piece.to_string();
    for token in ATEM_CLOSING_TOKENS {
        if piece.contains(token) {
            piece = piece.replace(token, "");
        }
    }
    content.push_str(&piece);
}

/// Whether `text` is a non-empty proper prefix of `marker`.
fn is_partial_prefix(text: &str, marker: &str) -> bool {
    !text.is_empty() && text.len() < marker.len() && marker.starts_with(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(parser: &mut ReasoningParser, chunks: &[&str]) -> ReasoningDelta {
        let mut out = ReasoningDelta::default();
        for chunk in chunks {
            let delta = parser.push(chunk);
            out.reasoning.push_str(&delta.reasoning);
            out.content.push_str(&delta.content);
        }
        let tail = parser.flush();
        out.reasoning.push_str(&tail.reasoning);
        out.content.push_str(&tail.content);
        out
    }

    #[test]
    fn a_closed_block_splits_into_reasoning_and_answer() {
        let parser = ReasoningParser::new(ReasoningFormat::Think, false, true);
        let split = parser.parse_complete("<think>let me see</think>The answer is 4.");
        assert_eq!(split.reasoning, "let me see");
        assert_eq!(split.content, "The answer is 4.");
    }

    #[test]
    fn text_with_no_block_is_all_answer() {
        let parser = ReasoningParser::new(ReasoningFormat::Think, false, true);
        let split = parser.parse_complete("Just the answer.");
        assert_eq!(split.reasoning, "");
        assert_eq!(split.content, "Just the answer.");
    }

    /// A template that opened the block in the prompt means the model
    /// never emits `<think>`, so the parser must start inside.
    #[test]
    fn a_forced_parser_starts_inside_the_block() {
        let parser = ReasoningParser::new(ReasoningFormat::Think, true, true);
        let split = parser.parse_complete("thinking hard</think>done");
        assert_eq!(split.reasoning, "thinking hard");
        assert_eq!(split.content, "done");

        let unforced = ReasoningParser::new(ReasoningFormat::Think, false, true);
        let split = unforced.parse_complete("thinking hard</think>done");
        assert_eq!(split.reasoning, "");
        assert_eq!(split.content, "thinking hard</think>done");
    }

    /// The whole reason for the hold-back: a marker split across two
    /// chunks must not leak half of itself into the answer.
    #[test]
    fn a_marker_split_across_chunks_never_leaks() {
        let mut parser = ReasoningParser::new(ReasoningFormat::Think, true, true);
        let out = stream(&mut parser, &["thinking", "</thi", "nk>", "answer"]);
        assert_eq!(out.reasoning, "thinking");
        assert_eq!(out.content, "answer");
        assert!(!out.reasoning.contains("</"), "no partial marker leaked");
    }

    /// Streaming and one-shot must agree, whatever the chunking.
    #[test]
    fn streaming_agrees_with_one_shot() {
        let text = "<think>step one. step two.</think>The answer is 4.";
        let reference =
            ReasoningParser::new(ReasoningFormat::Think, false, true).parse_complete(text);
        for width in [1usize, 3, 7, 13, 64] {
            let mut parser = ReasoningParser::new(ReasoningFormat::Think, false, true);
            let chunks: Vec<String> = text
                .as_bytes()
                .chunks(width)
                .map(|c| String::from_utf8_lossy(c).into_owned())
                .collect();
            let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let out = stream(&mut parser, &refs);
            assert_eq!(out.reasoning.trim(), reference.reasoning, "width {width}");
            assert_eq!(out.content.trim(), reference.content, "width {width}");
        }
    }

    /// With `stream_reasoning` off, nothing goes out until the block
    /// closes -- but nothing is lost either.
    #[test]
    fn withheld_reasoning_still_arrives_in_full() {
        let mut parser = ReasoningParser::new(ReasoningFormat::Think, true, false);
        let first = parser.push("thinking ");
        assert!(first.is_empty(), "nothing yet");
        let second = parser.push("more</think>answer");
        assert_eq!(second.reasoning, "thinking more");
        assert_eq!(second.content, "answer");
    }

    /// A closing marker beats a tool marker, which is what keeps a tool
    /// marker the model merely quoted inside its reasoning classified
    /// as reasoning.
    #[test]
    fn a_closed_block_wins_over_a_quoted_tool_marker() {
        let parser = ReasoningParser::new(ReasoningFormat::DeepSeekV32, true, true);
        let split = parser.parse_complete(
            "I could call <｜DSML｜function_calls> here, but I won't.</think>No tool needed.",
        );
        assert!(split.reasoning.contains("but I won't"));
        assert_eq!(split.content, "No tool needed.");
    }

    /// ... while a tool marker followed by a real payload, with no
    /// closing marker, ends the reasoning there.
    #[test]
    fn a_real_tool_call_ends_reasoning_without_a_closing_marker() {
        let parser = ReasoningParser::new(ReasoningFormat::DeepSeekV32, true, true);
        let payload = "x".repeat(TOOL_HOLD_MAX + 1);
        let split = parser.parse_complete(&format!(
            "I should look it up.<｜DSML｜function_calls>{payload}"
        ));
        assert_eq!(split.reasoning, "I should look it up.");
        assert!(split.content.starts_with("<｜DSML｜function_calls>"));
    }

    /// A truncated stream releases what it was holding rather than
    /// dropping it.
    #[test]
    fn a_truncated_stream_releases_what_it_held() {
        let mut parser = ReasoningParser::new(ReasoningFormat::Think, true, true);
        let out = stream(&mut parser, &["thinking</thi"]);
        assert_eq!(out.reasoning, "thinking</thi");
        assert!(out.content.is_empty());
    }

    #[test]
    fn the_harmony_format_routes_analysis_to_reasoning() {
        let parser = ReasoningParser::new(ReasoningFormat::GptOss, false, true);
        let split = parser.parse_complete(
            "<|channel|>analysis<|message|>weighing it up<|end|>\
             <|channel|>final<|message|>The answer is 4.<|return|>",
        );
        assert_eq!(split.reasoning, "weighing it up");
        assert_eq!(split.content, "The answer is 4.");
    }

    /// A harmony tool channel is neither reasoning nor prose: it is
    /// handed on verbatim, markers included, for the tool parser.
    #[test]
    fn a_harmony_tool_channel_survives_verbatim() {
        let parser = ReasoningParser::new(ReasoningFormat::GptOss, false, true);
        let split = parser.parse_complete(
            "<|channel|>analysis<|message|>need the weather<|end|>\
             <|channel|>commentary to=functions.get_weather<|message|>{\"city\":\"Rome\"}<|call|>",
        );
        assert_eq!(split.reasoning, "need the weather");
        assert!(split.content.contains("to=functions.get_weather"));
        assert!(split.content.contains("{\"city\":\"Rome\"}"));
        assert!(split.content.ends_with("<|call|>"));
    }

    #[test]
    fn harmony_text_with_no_channels_passes_through() {
        let parser = ReasoningParser::new(ReasoningFormat::GptOss, false, true);
        let split = parser.parse_complete("plain answer");
        assert_eq!(split.content, "plain answer");
        assert_eq!(split.reasoning, "");
    }

    #[test]
    fn harmony_streams_incrementally_and_matches_one_shot() {
        let text = "<|channel|>analysis<|message|>step one<|end|><|channel|>final<|message|>done<|return|>";
        let reference =
            ReasoningParser::new(ReasoningFormat::GptOss, false, true).parse_complete(text);
        let mut parser = ReasoningParser::new(ReasoningFormat::GptOss, false, true);
        let chunks: Vec<String> = text
            .as_bytes()
            .chunks(5)
            .map(|c| String::from_utf8_lossy(c).into_owned())
            .collect();
        let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let out = stream(&mut parser, &refs);
        assert_eq!(out.reasoning.trim(), reference.reasoning);
        assert_eq!(out.content.trim(), reference.content);
    }

    /// MiniMax-M3's adaptive mode: a leading bare closer means the
    /// model chose not to think this turn.
    #[test]
    fn a_leading_bare_closer_means_no_thinking_this_turn() {
        let parser = ReasoningParser::new(ReasoningFormat::MiniMaxM3, false, true);
        let split = parser.parse_complete("</mm:think>Straight to the answer.");
        assert_eq!(split.reasoning, "");
        assert_eq!(split.content, "Straight to the answer.");
    }

    /// ... and it is only special at the head. A closer later in the
    /// stream is ordinary.
    #[test]
    fn a_later_closer_is_not_the_adaptive_signal() {
        let parser = ReasoningParser::new(ReasoningFormat::MiniMaxM3, false, true);
        let split = parser.parse_complete("<mm:think>weighing</mm:think>done");
        assert_eq!(split.reasoning, "weighing");
        assert_eq!(split.content, "done");
    }

    #[test]
    fn the_adaptive_head_is_replayed_when_it_is_not_a_closer() {
        let mut parser = ReasoningParser::new(ReasoningFormat::MiniMaxM3, false, true);
        let out = stream(&mut parser, &["</m", "m:th", "ought> hmm"]);
        assert!(
            out.content.starts_with("</mm:thought>"),
            "the held head came back verbatim: {out:?}"
        );
    }

    #[test]
    fn parser_names_round_trip_and_specific_families_win_inference() {
        assert_eq!(
            ReasoningFormat::parse("deepseekv32"),
            Some(ReasoningFormat::DeepSeekV32)
        );
        assert_eq!(ReasoningFormat::parse("nonsense"), None);
        assert_eq!(
            ReasoningFormat::infer("MiniMax-M3-Instruct"),
            Some(ReasoningFormat::MiniMaxM3),
            "the specific arm must beat the bare minimax arm"
        );
        assert_eq!(
            ReasoningFormat::infer("MiniMax-M2"),
            Some(ReasoningFormat::ThinkAlwaysOpen)
        );
        assert_eq!(
            ReasoningFormat::infer("DeepSeek-V4-Flash"),
            Some(ReasoningFormat::DeepSeekV32)
        );
        assert_eq!(ReasoningFormat::infer("llama-3.1-8b"), None);
    }

    #[test]
    fn gemma_channel_markers_split_the_same_way() {
        let parser = ReasoningParser::new(ReasoningFormat::Gemma4, false, true);
        let split = parser.parse_complete("<|channel>thought\nmulling<channel|>the answer");
        assert_eq!(split.reasoning, "mulling");
        assert_eq!(split.content, "the answer");
    }

    /// The whole point of reading the prompt rather than the family:
    /// the SAME format answers differently depending on what the
    /// template actually rendered.
    #[test]
    fn a_prompt_that_ends_inside_a_think_block_forces_reasoning() {
        let f = ReasoningFormat::Think;
        assert!(f.prompt_opens_reasoning("<|im_start|>assistant\n<think>\n"));
        assert!(!f.prompt_opens_reasoning("<|im_start|>assistant\n"));
        // A block opened and closed earlier in the conversation is a
        // replayed turn, not an open one.
        assert!(
            !f.prompt_opens_reasoning("<think>past</think>answer<|im_end|><|im_start|>assistant\n")
        );
    }

    #[test]
    fn an_always_open_family_needs_no_evidence_and_harmony_takes_none() {
        assert!(ReasoningFormat::ThinkAlwaysOpen.prompt_opens_reasoning("anything"));
        assert!(!ReasoningFormat::GptOss.prompt_opens_reasoning("<think>"));
    }

    // ---- muse-glimmer / ATEM ----

    fn atem(name: &str, params: &[(&str, &str)]) -> String {
        let body: String = params
            .iter()
            .map(|(k, v)| format!("<atem:parameter name=\"{k}\">{v}</atem:parameter>\n"))
            .collect();
        format!(
            "<atem:function_calls>\n<atem:invoke name=\"{name}\">\n{body}\
             </atem:invoke>\n</atem:function_calls>"
        )
    }

    /// Chunk by characters and run the whole stream, flush included.
    fn stream_chars(parser: &mut ReasoningParser, text: &str, width: usize) -> ReasoningDelta {
        let chars: Vec<char> = text.chars().collect();
        let mut out = ReasoningDelta::default();
        for chunk in chars.chunks(width) {
            let piece: String = chunk.iter().collect();
            let delta = parser.push(&piece);
            out.reasoning.push_str(&delta.reasoning);
            out.content.push_str(&delta.content);
        }
        let tail = parser.flush();
        out.reasoning.push_str(&tail.reasoning);
        out.content.push_str(&tail.content);
        out
    }

    fn muse(header_open: bool) -> ReasoningParser {
        ReasoningParser::new(ReasoningFormat::MuseGlimmer, header_open, true)
    }

    /// The turn begins *inside* a header, because the template's prompt
    /// ends there. A parser that assumes the header is closed has
    /// nothing to attach the turn's first bytes to, and a bare
    /// recipient-less header (`assistant<|message|>`) is then streamed
    /// to the client as literal markup -- this test fails outright
    /// without the synthetic seed.
    #[test]
    fn a_bare_first_header_is_completed_by_the_synthetic_start() {
        let split = muse(true).parse_complete("assistant<|message|>Plain.<|eot|>");
        assert_eq!(split.content, "Plain.");
        assert_eq!(split.reasoning, "");

        let closed = muse(false).parse_complete("assistant<|message|>Plain.<|eot|>");
        assert!(
            closed.content.contains("<|message|>"),
            "a header-closed parser has no choice but to leak: {closed:?}"
        );
    }

    /// The recipient, not a marker, decides which field a body lands in.
    #[test]
    fn channel_bodies_are_routed_by_their_recipient() {
        let split = muse(true).parse_complete(
            " to=self<|message|>Let me think about this.<|eom|>\
             <|start|>assistant to=user<|message|>The answer is 42.<|eot|>",
        );
        assert_eq!(split.reasoning, "Let me think about this.");
        assert_eq!(split.content, "The answer is 42.");
    }

    /// A tool channel is neither reasoning nor prose: it is handed on
    /// verbatim, header and terminator included, for the tool parser.
    #[test]
    fn a_tool_channel_reaches_content_verbatim_with_its_header() {
        let wire = format!(
            " to=self<|message|>check the weather<|eom|>\
             <|start|>assistant to=weather.get<|message|>{}<|eot|>",
            atem("weather.get", &[("city", "Paris")])
        );
        let split = muse(true).parse_complete(&wire);
        assert_eq!(split.reasoning, "check the weather");
        assert!(
            split
                .content
                .starts_with("<|start|>assistant to=weather.get<|message|>"),
            "{:?}",
            split.content
        );
        assert!(split.content.ends_with("<|eot|>"));
        assert!(split.content.contains("<atem:function_calls>"));
    }

    /// A channel the model left without a terminator still has to reach
    /// the tool parser as a *delimited* block, so one is supplied.
    #[test]
    fn a_tool_channel_cut_off_by_the_next_header_gets_a_synthetic_terminator() {
        let wire = format!(
            " to=weather.get<|message|>{}<|start|>assistant to=user<|message|>done<|eot|>",
            atem("weather.get", &[("city", "Rome")])
        );
        let split = muse(true).parse_complete(&wire);
        assert!(
            split.content.contains("</atem:function_calls><|eom|>"),
            "the slice must be delimited: {:?}",
            split.content
        );
        assert!(split.content.ends_with("done"));
    }

    /// The model leaves a channel without `<|eom|>` more often than the
    /// template suggests: a complete `to=X<|message|>` is a boundary
    /// wherever it appears.
    #[test]
    fn a_headerless_switch_ends_the_body_it_appears_in() {
        let split = muse(true).parse_complete(
            " to=self<|message|>quick thought to=user<|message|>Here you go.<|eot|>",
        );
        assert_eq!(split.reasoning, "quick thought");
        assert_eq!(split.content, "Here you go.");
    }

    /// The load-bearing hold-back: a headerless switch has no fixed
    /// opener, so the tail of a body is withheld while a `to=` in it is
    /// still unresolved. A hold-back that only knew the control tokens
    /// would emit `send it to=we` as content and then be unable to
    /// recognize the switch at all -- and SSE cannot retract.
    #[test]
    fn the_atem_hold_back_withholds_a_switch_that_is_still_arriving() {
        let mut parser = muse(true);
        let first = parser.push(" to=user<|message|>send it to=we");
        assert_eq!(first.content, "send it ");
        assert!(!first.content.contains("to=we"));

        let rest = parser.push(&format!(
            "ather.get<|message|>{}<|eot|>",
            atem("weather.get", &[("city", "Lima")])
        ));
        assert!(
            rest.content.starts_with("to=weather.get<|message|>"),
            "the switch reassembled into a tool slice: {:?}",
            rest.content
        );
        assert_eq!(atem_hold_len("send it to=we"), "to=we".len());
    }

    /// Streaming and one-shot must agree, at every chunk width -- one
    /// character per chunk included, which is where a hold-back bug
    /// hides.
    #[test]
    fn atem_streaming_agrees_with_one_shot_at_every_chunk_width() {
        let wire = format!(
            " to=self<|message|>step one\nstep two<|eom|>\
             <|start|>assistant to=weather.get<|message|>{}<|eot|>\
             <|start|>assistant to=user<|message|>Done: 42.<|eot|>",
            atem("weather.get", &[("city", "Paris")])
        );
        let reference = muse(true).parse_complete(&wire);
        assert_eq!(reference.reasoning, "step one\nstep two");
        for width in [1usize, 3, 17, 4096] {
            let mut parser = muse(true);
            let out = stream_chars(&mut parser, &wire, width);
            assert_eq!(out.reasoning.trim(), reference.reasoning, "width {width}");
            assert_eq!(out.content.trim(), reference.content, "width {width}");
        }
    }

    /// A `<|start|>` that never becomes a header is literal text, and a
    /// reply that merely looks like the start of one is delivered at the
    /// end of the stream rather than dropped.
    #[test]
    fn a_start_marker_that_opens_no_header_is_released_as_text() {
        let tail = "x".repeat(400);
        let mut parser = muse(true);
        let out = stream_chars(
            &mut parser,
            &format!(" to=user<|message|>The token <|start|> opens a segment. {tail}"),
            7,
        );
        assert!(out.content.contains("The token "), "{:?}", out.content);
        assert!(out.content.contains(&tail));

        let mut short = muse(true);
        let out = stream_chars(&mut short, "assistant", 3);
        assert_eq!(out.content, "assistant", "a short lookalike still arrives");
    }

    /// A stray terminator is protocol debris, but the prose around it is
    /// the answer.
    #[test]
    fn a_stray_terminator_is_stripped_without_losing_the_prose() {
        let mut parser = muse(true);
        let out = stream_chars(&mut parser, "The token <|eot|> ends a turn.", 5);
        assert!(out.content.contains("The token "), "{:?}", out.content);
        assert!(out.content.contains(" ends a turn."));
        assert!(!out.content.contains("<|eot|>"));
    }

    /// Output with no ATEM markers at all is somebody answering
    /// plainly, and passes straight through.
    #[test]
    fn atem_text_with_no_channels_passes_through() {
        let text = "Just a plain answer, no channels.";
        let split = muse(true).parse_complete(text);
        assert_eq!(split.content, text);
        assert_eq!(split.reasoning, "");
    }

    /// The same prompt-reading rule as the `<think>` families, asking
    /// the channel grammar's own question: did the render stop between
    /// `<|start|>` and its `<|message|>`?
    #[test]
    fn a_prompt_ending_inside_a_channel_header_opens_the_header() {
        let f = ReasoningFormat::MuseGlimmer;
        assert!(f.prompt_opens_reasoning("<|start|>user<|message|>hi<|eot|><|start|>assistant"));
        assert!(!f.prompt_opens_reasoning("<|start|>user<|message|>hi<|eot|>"));
        assert_eq!(
            ReasoningFormat::infer("Muse-Glimmer-40B-A3B"),
            Some(ReasoningFormat::MuseGlimmer)
        );
        assert_eq!(
            ReasoningFormat::parse("muse_glimmer"),
            Some(ReasoningFormat::MuseGlimmer)
        );
    }
}
