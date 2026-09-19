//! The differential turn-boundary analyzer: learn the string that ends
//! an assistant turn by RENDERING the template, never by matching its
//! source.
//!
//! # Why this is not a substring search
//!
//! The thing being looked for is "what does this template print after
//! the assistant's content". Searching the template's *source* for
//! `<|im_end|>` answers a different question, and answers it wrong in
//! both directions: a template that merely mentions the marker in a
//! branch it never takes claims one it does not emit, and a template
//! whose marker this project has never seen reports none at all. Every
//! family added meant another arm, and a family nobody had a checkpoint
//! for got nothing.
//!
//! Rendering answers the actual question. The same three renders derive
//! `<|im_end|>`, `<end_of_turn>`, `<turn|>`, `<|eot_id|>` and whatever a
//! checkpoint released next year terminates a turn with, because none
//! of them are written down here.
//!
//! # Why it matters that this is right
//!
//! The marker is a stop string, and a missing one does not degrade
//! output, it ruins it. Yi-1.5-6B ends every turn with `<|im_end|>`
//! while its `eos_token_id` is a different token entirely: nothing
//! stopped, so the marker was EMITTED AS TEXT and the model kept
//! talking past the end of its own answer. Gemma IT has the same shape
//! with `<end_of_turn>` before `<eos>`.
//!
//! # The method
//!
//! Three renders, all with `add_generation_prompt = false`:
//!
//! ```text
//! closed = [user U1, assistant ANSWER, user U2]
//! opener = [user U1]
//! ```
//!
//! In `closed`, the text between the end of `ANSWER` and the start of
//! `U2` is exactly the assistant turn's terminator followed by the next
//! turn's opener. The `opener` render splits that gap two ways, tried in
//! order:
//!
//! 1. Whatever it prints AFTER `U1` closes a turn with nothing following
//!    it to be confused with. When the gap starts with that string, the
//!    gap starts with the terminator, and that is the answer.
//! 2. Otherwise, whatever it prints BEFORE `U1` is the next turn's
//!    opener, minus the BOS only a first turn carries; removing it from
//!    the right of the gap leaves the terminator alone. This is the
//!    Mistral-Instruct case, where a user turn ends with `[/INST]` and
//!    an assistant turn with `</s>`.
//!
//! Step 1 is not an optimisation. Llama-3's template emits a default
//! system block, so its opener already contains an `<|eot_id|>`, and the
//! common suffix in step 2 runs back through the one that is the answer
//! and swallows the whole gap.
//!
//! The assistant message is deliberately NOT last. A template that
//! treats a trailing assistant message as a prefill to continue -- Qwen3
//! and DeepSeek both do -- prints no terminator after it at all, and a
//! two-message probe would conclude the model has none.
//!
//! # What it refuses
//!
//! Everything is derived, so everything is checked. The whole derivation
//! runs TWICE with different assistant content and the two results must
//! agree: a template that echoes content into its terminator has not
//! given us a delimiter, it has given us that render. A result is also
//! refused when it is empty, longer than [`MAX_MARKER`], carries a probe
//! string through, or is just the EOS token -- decoding already stops
//! there, and a stop string that duplicates EOS only adds withheld text
//! to every response.

use frink_models::chat_template::TemplateError;
use serde_json::{json, Value};

/// The longest string this will accept as a turn terminator.
///
/// A delimiter is a handful of special tokens. Anything longer is a
/// template printing a footer, a tool schema, or a system preamble into
/// the gap, and using it as a stop string would mean stopping on
/// something the model has no reason to ever emit.
const MAX_MARKER: usize = 64;

/// Probe strings, distinctive enough to locate in a render and dull
/// enough that no template treats them specially. The two answers differ
/// so the derivation can be run twice and compared.
///
/// They differ in LENGTH as well as in bytes, deliberately. A template
/// that prints `{{ content | length }}` into the gap would give the same
/// answer for two equal-length probes, and the content-dependence check
/// would pass on a gap that is not a delimiter at all.
const ANSWER_A: &str = "frinkprobeanswerone";
const ANSWER_B: &str = "frinkprobeanswerthealternateone";
const USER_ONE: &str = "frinkprobequestionone";
const USER_TWO: &str = "frinkprobequestiontwo";

/// Derive the string that ends an assistant turn, or `None` when the
/// template does not print one.
///
/// `render` renders a conversation with `add_generation_prompt = false`.
/// `bos` is the checkpoint's BOS text, which the first turn of a prompt
/// carries and later turns do not.
///
/// Whether the result is worth having is a separate question, and the
/// caller's: a terminator that IS the EOS adds nothing to a stop set,
/// because decoding stops there anyway.
pub(crate) fn probe_end_of_turn<R>(mut render: R, bos: Option<&str>) -> Option<String>
where
    R: FnMut(&[Value]) -> Result<String, TemplateError>,
{
    let first = derive(&mut render, ANSWER_A, bos)?;
    let second = derive(&mut render, ANSWER_B, bos)?;
    // Content-dependent, so it is this render rather than a delimiter.
    if first != second {
        return None;
    }
    if first.is_empty() || first.len() > MAX_MARKER {
        return None;
    }
    if [ANSWER_A, ANSWER_B, USER_ONE, USER_TWO]
        .iter()
        .any(|p| first.contains(p))
    {
        return None;
    }
    Some(first)
}

/// One pass of the derivation, for one assistant answer.
fn derive<R>(render: &mut R, answer: &str, bos: Option<&str>) -> Option<String>
where
    R: FnMut(&[Value]) -> Result<String, TemplateError>,
{
    let user = |text: &str| json!({"role": "user", "content": text});
    let assistant = |text: &str| json!({"role": "assistant", "content": text});

    let closed = render(&[user(USER_ONE), assistant(answer), user(USER_TWO)]).ok()?;
    let start = closed.find(answer)? + answer.len();
    let rest = &closed[start..];
    // Everything printed between the assistant's content and the next
    // user's: terminator, then the next turn's opener.
    let gap = &rest[..rest.find(USER_TWO)?];

    // A conversation of ONE user turn, which brackets that turn with
    // nothing after it. Both halves of the split come from here.
    let one = render(&[user(USER_ONE)]).ok()?;
    let at = one.find(USER_ONE)?;
    let opener = &one[..at];
    let user_terminator = one[at + USER_ONE.len()..].trim();

    // The cheap split, and the one that is right for every family whose
    // turns all end the same way: whatever closes the LAST user turn,
    // where nothing follows to be confused with it, also opens the gap
    // after an assistant turn.
    //
    // This is not merely an optimisation over the suffix strip below --
    // it is the only thing that gets Llama-3 right. Its template emits a
    // default system block, so `opener` is
    // `<|begin_of_text|>…system…<|eot_id|><|start_header_id|>user…`,
    // whose common suffix with the gap runs back THROUGH the `<|eot_id|>`
    // that is the answer and consumes the whole gap.
    if !user_terminator.is_empty() && gap.starts_with(user_terminator) {
        return Some(user_terminator.to_string());
    }

    // Families that close a user turn and an assistant turn differently
    // -- Mistral-Instruct ends the user turn with `[/INST]` and the
    // assistant turn with `</s>` -- have the next turn's opener removed
    // from the right instead.
    //
    // The BOS comes off first, because the first turn of a prompt
    // carries it and the second does not. It has to be an EXACT strip
    // after that: the longest common suffix looks like the more forgiving
    // choice and is the more dangerous one, because it keeps matching
    // past the opener into the terminator whenever the two end alike.
    // On Mistral it ate `</s>[INST] ` down to `</`, which is not a
    // string the model ever emits, so the stop never fires and the
    // request runs to `max_tokens`. A truncated marker is worse than no
    // marker: no marker leaves EOS to do the job, a truncated one claims
    // the job is done.
    let opener = bos.and_then(|b| opener.strip_prefix(b)).unwrap_or(opener);
    let terminator = gap.strip_suffix(opener)?.trim();
    if !terminator.is_empty() {
        return Some(terminator.to_string());
    }

    // No terminator at all: this template closes an assistant turn by
    // opening the next one, which is what the role-labeled builtin does
    // (`assistant: …` then a newline and `user: `). The boundary is then
    // the next turn's opener, and its LEADING whitespace is load-bearing
    // -- `\nuser:` stops at the start of a line, where `user:` alone
    // would also fire mid-sentence. Only the trailing end is trimmed.
    let boundary = gap.trim_end();
    (!boundary.is_empty() && boundary.len() <= MAX_MARKER).then(|| boundary.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render a conversation through a minimal `{prefix}{role}{infix}
    /// {content}{suffix}` template, which is the shape every ChatML-like
    /// family has.
    fn shaped(
        bos: &'static str,
        prefix: &'static str,
        infix: &'static str,
        suffix: &'static str,
    ) -> impl FnMut(&[Value]) -> Result<String, TemplateError> {
        move |messages: &[Value]| {
            let mut out = String::from(bos);
            for m in messages {
                out.push_str(prefix);
                out.push_str(m["role"].as_str().unwrap());
                out.push_str(infix);
                out.push_str(m["content"].as_str().unwrap());
                out.push_str(suffix);
            }
            Ok(out)
        }
    }

    #[test]
    fn chatml_terminator_is_derived_not_listed() {
        let r = shaped("", "<|im_start|>", "\n", "<|im_end|>\n");
        assert_eq!(
            probe_end_of_turn(r, Some("<bos>")).as_deref(),
            Some("<|im_end|>")
        );
    }

    /// The BOS on the first turn is why this cannot be `strip_suffix`:
    /// the opener render carries it and the second turn inside `closed`
    /// does not.
    #[test]
    fn a_leading_bos_does_not_confuse_the_opener() {
        let r = shaped("<bos>", "<start_of_turn>", "\n", "<end_of_turn>\n");
        assert_eq!(
            probe_end_of_turn(r, Some("<bos>")).as_deref(),
            Some("<end_of_turn>")
        );
    }

    /// gemma-4, whose marker is not ChatML's and is not Gemma-3's
    /// either. Nothing in this module names it.
    #[test]
    fn gemma4_turn_marker_needs_no_arm_of_its_own() {
        let r = shaped("<bos>", "<|turn>", "\n", "<turn|>\n");
        assert_eq!(
            probe_end_of_turn(r, Some("<bos>")).as_deref(),
            Some("<turn|>")
        );
    }

    /// A marker this project has never seen, to make the point that the
    /// list of families is not a list.
    #[test]
    fn an_unknown_family_is_derived_like_any_other() {
        let r = shaped("", "[[", "]]", "[[/turn]]\n");
        assert_eq!(
            probe_end_of_turn(r, Some("<bos>")).as_deref(),
            Some("[[/turn]]")
        );
    }

    /// Mistral-Instruct closes a user turn with `[/INST]` and an
    /// assistant turn with `</s>`, so the cheap split does not apply and
    /// the opener has to come off the right instead.
    ///
    /// This is also the truncation regression. Removing the two strings'
    /// LONGEST COMMON SUFFIX rather than the opener exactly matched
    /// `s>[INST] ` -- through the opener and into the `</s>` that is the
    /// answer -- and reported `</`. A stop string the model never emits
    /// is worse than none: no marker leaves EOS to stop the run, a
    /// truncated one claims something already stops it.
    #[test]
    fn a_family_whose_two_turns_end_differently_is_not_truncated() {
        let render = |messages: &[Value]| {
            let mut out = String::from("<s>");
            for m in messages {
                let content = m["content"].as_str().unwrap();
                match m["role"].as_str().unwrap() {
                    "user" => out.push_str(&format!("[INST] {content} [/INST]")),
                    _ => out.push_str(&format!("{content}</s>")),
                }
            }
            Ok(out)
        };
        assert_eq!(
            probe_end_of_turn(render, Some("<s>")).as_deref(),
            Some("</s>")
        );
    }

    /// The same shape with a marker between the turns, which is where
    /// the truncation was actually found: TinyLlama's `</s>\n<|user|>\n`
    /// came back as `</` for the same reason.
    #[test]
    fn a_zephyr_shaped_template_keeps_its_whole_terminator() {
        let render = |messages: &[Value]| {
            let mut out = String::new();
            for m in messages {
                out.push_str(&format!(
                    "<|{}|>\n{}</s>\n",
                    m["role"].as_str().unwrap(),
                    m["content"].as_str().unwrap()
                ));
            }
            Ok(out)
        };
        assert_eq!(probe_end_of_turn(render, None).as_deref(), Some("</s>"));
    }

    /// The role-labeled builtin prints no terminator at all, so the
    /// boundary is the next turn's opener -- with the newline that keeps
    /// it from matching mid-sentence.
    #[test]
    fn a_template_with_no_terminator_falls_back_to_the_next_turn() {
        let render = |messages: &[Value]| {
            let lines: Vec<String> = messages
                .iter()
                .map(|m| {
                    format!(
                        "{}: {}",
                        m["role"].as_str().unwrap(),
                        m["content"].as_str().unwrap()
                    )
                })
                .collect();
            Ok(lines.join("\n"))
        };
        assert_eq!(probe_end_of_turn(render, None).as_deref(), Some("\nuser:"));
    }

    /// A template that treats a TRAILING assistant message as a prefill
    /// prints no terminator after it. The probe puts a user turn last
    /// precisely so this template still reports its real marker.
    #[test]
    fn a_prefill_template_still_reports_its_terminator() {
        let render = |messages: &[Value]| {
            let mut out = String::new();
            for (i, m) in messages.iter().enumerate() {
                let last = i + 1 == messages.len();
                let role = m["role"].as_str().unwrap();
                out.push_str("<|im_start|>");
                out.push_str(role);
                out.push('\n');
                out.push_str(m["content"].as_str().unwrap());
                if !(last && role == "assistant") {
                    out.push_str("<|im_end|>\n");
                }
            }
            Ok(out)
        };
        assert_eq!(
            probe_end_of_turn(render, None).as_deref(),
            Some("<|im_end|>")
        );
    }

    /// A terminator that varies with the answer is not a delimiter.
    #[test]
    fn a_content_dependent_gap_is_refused() {
        let render = |messages: &[Value]| {
            let mut out = String::new();
            for m in messages {
                out.push_str(m["role"].as_str().unwrap());
                out.push(':');
                let content = m["content"].as_str().unwrap();
                out.push_str(content);
                // Echoes the content back into the gap.
                out.push_str(&format!("<end len={}>\n", content.len()));
            }
            Ok(out)
        };
        assert_eq!(probe_end_of_turn(render, None), None);
    }

    /// A template that errors on the probe conversation reports nothing,
    /// which is the safe answer: add no stop string rather than a
    /// guessed one.
    #[test]
    fn a_template_that_cannot_render_the_probe_reports_nothing() {
        let render = |_: &[Value]| Err(TemplateError::Render("probe".into()));
        assert_eq!(probe_end_of_turn(render, None), None);
    }

    /// A gap long enough to be a preamble rather than a delimiter is
    /// refused: stopping on it would mean never stopping.
    #[test]
    fn an_overlong_gap_is_not_a_delimiter() {
        let filler: &'static str = "x".repeat(MAX_MARKER + 1).leak();
        let r = shaped("", "<|s|>", "\n", filler);
        assert_eq!(probe_end_of_turn(r, Some("<bos>")), None);
    }
}

#[cfg(test)]
mod real_checkpoints {
    //! The derivation against the `tokenizer.chat_template` of real
    //! GGUFs, which is the only evidence that matters: every unit test
    //! above renders a template this file wrote.
    //!
    //! `#[ignore]` because it needs the checkpoints in `models/`.
    //! `cargo test -p frink-server --lib real_checkpoints -- --ignored`

    use super::probe_end_of_turn;
    use crate::chat_template::{turn_render, PromptTemplate};
    use frink_models::chat_template::ChatTemplate as JinjaTemplate;

    fn open(file: &str) -> frink_gguf::ShardedGguf {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/");
        frink_gguf::ShardedGguf::open(format!("{dir}{file}")).expect("open")
    }

    /// The terminator the template prints: what the derivation FOUND,
    /// before the EOS is folded out of it. Rendering still uses the
    /// checkpoint's real BOS and EOS, because a template that
    /// concatenates `eos_token` fails to render without one.
    fn derived(file: &str) -> Option<String> {
        let f = open(file);
        let template = JinjaTemplate::from_gguf_metadata(
            f.metadata_str("tokenizer.chat_template"),
            None,
            false,
            false,
        );
        let bos = f.token_text("tokenizer.ggml.bos_token_id");
        let eos = f.token_text("tokenizer.ggml.eos_token_id");
        probe_end_of_turn(
            turn_render(&template, bos.as_deref(), eos.as_deref()),
            bos.as_deref(),
        )
    }

    /// What actually reaches the stop set: the derivation minus anything
    /// decoding already stops on.
    fn stop_string(file: &str) -> Option<String> {
        let f = open(file);
        PromptTemplate::from_source(
            f.metadata_str("tokenizer.chat_template"),
            f.token_text("tokenizer.ggml.bos_token_id"),
            f.token_text("tokenizer.ggml.eos_token_id"),
        )
        .end_of_turn()
        .map(str::to_string)
    }

    /// Six families, six terminators, and not one of them is named in
    /// `policy::turn`. The old sniffer had an arm for three of these and
    /// reported nothing for the rest.
    /// Nine families, nine terminators, and not one of them is named in
    /// `policy::turn`. The sniffer this replaced had an arm for three of
    /// these and reported nothing at all for the other six -- including
    /// DeepSeek's, whose delimiters are FULLWIDTH pipes that no
    /// `<|...|>` pattern matches.
    #[test]
    #[ignore = "needs models/"]
    fn every_family_reports_the_terminator_its_template_prints() {
        for (file, want) in [
            ("Yi-1.5-6B-Chat-Q4_K_M.gguf", "<|im_end|>"),
            ("Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", "<|im_end|>"),
            ("Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf", "<|im_end|>"),
            ("gemma-2-2b-it-Q4_K_M.gguf", "<end_of_turn>"),
            ("gemma-4-E2B-it-Q4_K_M.gguf", "<turn|>"),
            ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "<|eot_id|>"),
            ("Mistral-7B-Instruct-v0.2-Q4_K_M.gguf", "</s>"),
            ("tinyllama-1.1b-chat-v1.0.Q8_0.gguf", "</s>"),
            ("Phi-4-mini-instruct-Q4_K_M.gguf", "<|end|>"),
            (
                "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
                "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
            ),
        ] {
            assert_eq!(derived(file).as_deref(), Some(want), "{file}");
        }
    }

    /// A checkpoint with no template at all, where the builtin prints
    /// `user:` / `assistant:` lines and a base model continues that
    /// pattern to the token cap because it has no EOS for a turn it
    /// never saw. OLMoE-1B-7B answered "Paris." correctly and then wrote
    /// the next `user:` line itself for 512 tokens.
    #[test]
    #[ignore = "needs models/olmoe-1b-7b-0924-q4_0.gguf"]
    fn a_checkpoint_with_no_template_stops_at_the_next_turn_label() {
        assert_eq!(
            derived("olmoe-1b-7b-0924-q4_0.gguf").as_deref(),
            Some("\nuser:")
        );
    }

    /// Yi-1.5 is why this exists. Its template ends every turn with
    /// `<|im_end|>` while `eos_token_id` is `<|endoftext|>`, so nothing
    /// stopped and the marker was emitted as literal text mid-answer.
    /// Gemma-2 has the same shape with `<end_of_turn>` before `<eos>`.
    #[test]
    #[ignore = "needs models/"]
    fn a_terminator_that_is_not_the_eos_reaches_the_stop_set() {
        assert_eq!(
            stop_string("Yi-1.5-6B-Chat-Q4_K_M.gguf").as_deref(),
            Some("<|im_end|>")
        );
        assert_eq!(
            stop_string("gemma-2-2b-it-Q4_K_M.gguf").as_deref(),
            Some("<end_of_turn>")
        );
    }

    /// Where the terminator IS the EOS, decoding already stops and the
    /// stop set gains nothing.
    ///
    /// gemma-4 is here rather than beside gemma-2 on purpose: the two
    /// look like one family and are not. gemma-4's `eos_token_id` is
    /// `<turn|>`, its own turn marker, where gemma-2's is a separate
    /// `<eos>`. A per-family arm would have to know that; the derivation
    /// does not.
    #[test]
    #[ignore = "needs models/"]
    fn a_terminator_that_is_the_eos_adds_no_stop_string() {
        for file in [
            "gemma-4-E2B-it-Q4_K_M.gguf",
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "Mistral-7B-Instruct-v0.2-Q4_K_M.gguf",
            "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
            "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        ] {
            assert_eq!(stop_string(file), None, "{file}");
        }
    }
}
