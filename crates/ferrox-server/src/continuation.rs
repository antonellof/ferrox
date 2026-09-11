//! Continuing a partial assistant turn: llama.cpp's
//! `continue_final_message`.
//!
//! A reasoning model that runs out of `max_tokens` inside its chain of
//! thought returns `finish_reason: "length"`, `reasoning_content` and
//! no content at all. The natural next request is "carry on from
//! there" -- and a trailing assistant message on this server rendered
//! as a CLOSED turn followed by a fresh generation prompt, so the model
//! started over instead of continuing. This module renders the
//! continuation llama.cpp renders (`common/chat-auto-parser-generator
//! .cpp:54-71`), under llama.cpp's own field and value set:
//!
//! - `true` -- **auto**: continue the reasoning when the message has a
//!   thought and no content, otherwise continue the content;
//! - `"reasoning_content"` -- continue INSIDE the reasoning block;
//! - `"content"` -- the thought is closed and the answer continues.
//!
//! The rendering is `messages[..n-1]` through the template with the
//! generation prompt, then the partial turn appended RAW: the family's
//! opener, the replayed thought, and for a content continuation the
//! closer and the text. An opener the template put into the generation
//! prompt itself (R1 distills end theirs with `<think>\n`) is cut off
//! first, so the block is opened exactly once. Nothing is re-tokenized
//! specially: the appended text is prompt like any other, and
//! `OutputPosture::resolve` reads the resulting prompt to learn whether
//! the model is inside the block, which is the same fact it reads for
//! every other request.
//!
//! **Default ON, llama.cpp's way.** Its server treats ANY trailing
//! assistant message as a continuation unless started with
//! `--no-prefill-assistant` (`tools/server/server-common.cpp:1046-1056`:
//! a body that does not set the field, on a server whose
//! `prefill_assistant` is true, with an assistant message last, gets
//! `AUTO`). Anthropic's API prefills a trailing assistant turn too.
//! That rule is [`ContinueFinalMessage::resolve`], and it is applied in
//! exactly ONE place -- `ChatCompletionRequest::render_prompt` -- which
//! `/v1/chat/completions`, `/v1/messages` and `/v1/responses` all render
//! through, so the three routes cannot hold three copies of the
//! default. The server flag is `--prefill-assistant` /
//! `--no-prefill-assistant`, llama.cpp's spelling, lowered to
//! [`PREFILL_ASSISTANT_ENV`].
//!
//! One deliberate difference from upstream: there, `false` parses to
//! the same thing as absence (`common/chat.cpp:567-581`), so a single
//! request cannot opt out of the server default. Here `false` is an
//! explicit "render it closed", because a client that spells the
//! opt-out meant it.
//!
//! **Refused by name, not approximated:** a channel grammar (harmony,
//! ATEM) has no marker pair to write a thought back between; an
//! always-open family (`deepseekv32`, `minimax`) cannot be told by the
//! prompt that its block is closed, so its content continuation would
//! be parsed as reasoning; a turn carrying tool calls is refused the
//! way llama.cpp refuses it.

use serde::Deserialize;

use crate::policy::parser::ReasoningFormat;
use crate::{chat_template, invalid_request, unsupported_feature, ApiError, ChatMessage, ToolDef};

/// A continuation mode: `true`, `"reasoning_content"` or `"content"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Continuation {
    Auto,
    Reasoning,
    Content,
}

/// The `continue_final_message` field in its three states.
///
/// `Unset` is the shape every route's lowering leaves it in when the
/// caller said nothing, and it is the ONLY state the server default
/// applies to; a route that wanted a different default would have to
/// spell a different variant, and there is none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ContinueFinalMessage {
    /// The caller said nothing: llama.cpp's server default decides.
    #[default]
    Unset,
    /// `false`: render the trailing assistant message as a closed turn.
    Off,
    /// A mode the caller asked for by name.
    Mode(Continuation),
}

/// Env var `--prefill-assistant` / `--no-prefill-assistant` lowers to.
/// Unset is on, as upstream's `prefill_assistant = true`
/// (`common/common.h:642`).
pub(crate) const PREFILL_ASSISTANT_ENV: &str = "FERROX_PREFILL_ASSISTANT";

fn prefill_assistant_enabled() -> bool {
    prefill_assistant_from_env(std::env::var(PREFILL_ASSISTANT_ENV).ok().as_deref())
}

fn prefill_assistant_from_env(value: Option<&str>) -> bool {
    !matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("false") | Some("off") | Some("no")
    )
}

impl ContinueFinalMessage {
    /// llama.cpp's rule (`server-common.cpp:1046-1056`): an explicit
    /// mode stands; an explicit `false` renders the turn closed; a
    /// request that said nothing continues a trailing assistant
    /// message when the server's prefill is on, and otherwise does not.
    ///
    /// The `bool` says whether the mode was IMPLIED by the default, so
    /// a refusal downstream can tell the caller what to turn off.
    pub(crate) fn resolve(self, history: &[crate::ChatMessage]) -> Option<(Continuation, bool)> {
        self.resolve_with(prefill_assistant_enabled(), history)
    }

    fn resolve_with(
        self,
        prefill_assistant: bool,
        history: &[crate::ChatMessage],
    ) -> Option<(Continuation, bool)> {
        match self {
            ContinueFinalMessage::Mode(mode) => Some((mode, false)),
            ContinueFinalMessage::Off => None,
            ContinueFinalMessage::Unset => {
                let trailing_assistant = history.last().is_some_and(|m| m.role == "assistant");
                (prefill_assistant && trailing_assistant).then_some((Continuation::Auto, true))
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ContinueWire {
    Flag(bool),
    Mode(String),
}

/// The field's deserializer: absent and `null` are `Unset`, `false` is
/// `Off`, everything else is a named mode or a parse error.
pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<ContinueFinalMessage, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(wire) = Option::<ContinueWire>::deserialize(deserializer)? else {
        return Ok(ContinueFinalMessage::Unset);
    };
    match wire {
        ContinueWire::Flag(false) => Ok(ContinueFinalMessage::Off),
        ContinueWire::Flag(true) => Ok(ContinueFinalMessage::Mode(Continuation::Auto)),
        ContinueWire::Mode(mode) => match mode.as_str() {
            "reasoning_content" => Ok(ContinueFinalMessage::Mode(Continuation::Reasoning)),
            "content" => Ok(ContinueFinalMessage::Mode(Continuation::Content)),
            other => Err(serde::de::Error::custom(format!(
                "continue_final_message must be true, false, \"reasoning_content\" or \
                 \"content\", not {other:?}"
            ))),
        },
    }
}

/// A refusal of a continuation nobody asked for by name, with the two
/// ways to make the request render the turn closed instead.
pub(crate) fn implied_by_default(error: ApiError) -> ApiError {
    let (status, mut body) = error;
    if let Some(message) = body.0["error"]["message"].as_str() {
        let message = format!(
            "{message} (This continuation was implied by the server default: a trailing \
             assistant message is continued, as llama.cpp's --prefill-assistant does. Send \
             `continue_final_message: false` to render it as a closed turn, or start the \
             server with --no-prefill-assistant.)"
        );
        body.0["error"]["message"] = serde_json::Value::String(message);
    }
    (status, body)
}

impl Continuation {
    /// llama.cpp's `auto` rule: a thought with nothing after it is
    /// continued as a thought; anything else continues the content.
    fn resolve(self, reasoning: &str, content: &str) -> Continuation {
        match self {
            Continuation::Auto if !reasoning.is_empty() && content.is_empty() => {
                Continuation::Reasoning
            }
            Continuation::Auto => Continuation::Content,
            explicit => explicit,
        }
    }
}

/// Render `messages` so the model continues the last one.
///
/// Every refusal is decided BEFORE the template runs, so a request that
/// cannot be continued costs no render. `format` is the served
/// checkpoint's reasoning family, the same value `OutputPosture` will
/// read the result with.
pub(crate) fn prompt_continuing_final_message(
    messages: &[ChatMessage],
    template: &chat_template::PromptTemplate,
    tools: &[ToolDef],
    extra: serde_json::Map<String, serde_json::Value>,
    format: Option<ReasoningFormat>,
    mode: Continuation,
) -> Result<String, ApiError> {
    let Some((last, head)) = messages.split_last() else {
        return Err(invalid_request(
            "continue_final_message needs a message to continue",
            "messages",
        ));
    };
    if last.role != "assistant" {
        return Err(invalid_request(
            "continue_final_message: the last message must be an assistant message",
            "messages",
        ));
    }
    if head.last().is_some_and(|m| m.role == "assistant") {
        // llama.cpp's rule too: a template may fold two adjacent
        // assistant turns into one, and which of them is being
        // continued would then depend on the template.
        return Err(invalid_request(
            "continue_final_message: cannot have 2 or more assistant messages at the end of \
             the list",
            "messages",
        ));
    }
    if last
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty())
    {
        return Err(unsupported_feature(
            "continue_final_message: continuing an assistant message that contains tool calls \
             is not implemented",
        ));
    }

    let reasoning = last.reasoning_content.as_deref().unwrap_or("");
    let content = last
        .content
        .as_ref()
        .map(crate::MessageContent::as_text)
        .unwrap_or_default();
    let mode = mode.resolve(reasoning, &content);

    // Whether the thought can be written back at all, decided from the
    // family alone: a plain checkpoint has no block and continues its
    // content; a marker family writes the block; a channel grammar
    // cannot be given text to continue by concatenation.
    let markers = match format {
        None => None,
        Some(format) => match format.continuation_markers() {
            Some(pair) => Some((format, pair)),
            None => {
                return Err(unsupported_feature(&format!(
                    "continue_final_message is not implemented for the {} reasoning format: \
                     its chain of thought is a channel grammar, not a marker pair a replayed \
                     thought can be written between",
                    format.as_str()
                )))
            }
        },
    };
    if markers.is_none() && !reasoning.is_empty() {
        return Err(unsupported_feature(
            "continue_final_message: the served model has no reasoning format, so a \
             reasoning_content on the message to continue cannot be rendered back into a prompt",
        ));
    }
    if let (Some((format, _)), Continuation::Content) = (markers, mode) {
        if format.always_open() {
            return Err(unsupported_feature(&format!(
                "continue_final_message: \"content\" is not implemented for the {} reasoning \
                 format, whose parser reads every generation as starting inside the thinking \
                 block; only a reasoning continuation can be rendered for it",
                format.as_str()
            )));
        }
    }

    let mut prompt = crate::prompt_from_messages(head, template, tools, extra)?;
    if let Some((format, (start, end))) = markers {
        // The template may already have opened the block in its
        // generation prompt. Cut it off and open it once, here, with the
        // thought inside -- what llama.cpp's generator does with its
        // `generation_prompt.find(reasoning.start)`.
        if format.prompt_opens_reasoning(&prompt) {
            if let Some(at) = prompt.rfind(start) {
                prompt.truncate(at);
            }
        }
        prompt.push_str(start);
        prompt.push_str(reasoning);
        if mode == Continuation::Content {
            prompt.push_str(end);
        }
    }
    if mode == Continuation::Content {
        prompt.push_str(&content);
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_template::PromptTemplate;
    use crate::output::OutputPosture;
    use crate::MessageContent;

    /// An R1-distill-shaped template: closes a replayed assistant turn,
    /// strips its thinking, and opens the block in the generation
    /// prompt -- the exact shape a naive trailing-assistant render gets
    /// wrong.
    const R1: &str = "{% for m in messages %}{% if m.role == 'user' %}<|User|>{{ m.content }}{% else %}<|Assistant|>{{ m.content }}<|end▁of▁sentence|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|Assistant|><think>\n{% endif %}";

    /// ChatML with no notion of thinking.
    const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    fn template(source: &str) -> PromptTemplate {
        PromptTemplate::from_gguf_metadata(Some(source), Some("llama"), false, true, None, None)
    }

    fn msg(role: &str, content: &str, reasoning: Option<&str>) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: reasoning.map(str::to_string),
        }
    }

    fn render(
        source: &str,
        messages: &[ChatMessage],
        format: Option<ReasoningFormat>,
        mode: Continuation,
    ) -> Result<String, ApiError> {
        prompt_continuing_final_message(
            messages,
            &template(source),
            &[],
            serde_json::Map::new(),
            format,
            mode,
        )
    }

    fn parse(value: serde_json::Value) -> Result<ContinueFinalMessage, String> {
        #[derive(Deserialize)]
        struct Body {
            #[serde(default, deserialize_with = "deserialize")]
            continue_final_message: ContinueFinalMessage,
        }
        serde_json::from_value::<Body>(serde_json::json!({ "continue_final_message": value }))
            .map(|b| b.continue_final_message)
            .map_err(|e| e.to_string())
    }

    /// The user's case: a thought cut off by `max_tokens`, sent back.
    /// The block is opened ONCE -- the template's own `<think>\n` is
    /// gone -- and left open, so the parser reads the continuation as
    /// reasoning.
    #[test]
    fn a_cut_off_thought_is_continued_inside_an_open_block() {
        let prompt = render(
            R1,
            &[
                msg("user", "why", None),
                msg("assistant", "", Some("Let me think about")),
            ],
            Some(ReasoningFormat::Think),
            Continuation::Auto,
        )
        .expect("renders");
        assert_eq!(
            prompt, "<|User|>why<|Assistant|><think>Let me think about",
            "the template's own opener must not be doubled"
        );
        // The parser the server will read the stream with is built off
        // this prompt: the continuation's first tokens are reasoning.
        let split = OutputPosture::resolve("DeepSeek-R1-Distill-Qwen-1.5B", &prompt)
            .reasoning_parser()
            .expect("R1 has a format")
            .parse_complete(" this.</think>Because.");
        assert_eq!(
            split.reasoning, "this.",
            "the parser must pick up inside the block"
        );
        assert_eq!(split.content, "Because.");
    }

    /// A finished thought with a cut-off answer: the block is closed and
    /// the answer text follows it, so the parser reads the rest as
    /// content.
    #[test]
    fn a_cut_off_answer_is_continued_after_a_closed_block() {
        let prompt = render(
            R1,
            &[
                msg("user", "why", None),
                msg("assistant", "Because", Some("thought")),
            ],
            Some(ReasoningFormat::Think),
            Continuation::Auto,
        )
        .expect("renders");
        assert_eq!(
            prompt,
            "<|User|>why<|Assistant|><think>thought</think>Because"
        );
        let split = OutputPosture::resolve("DeepSeek-R1-Distill-Qwen-1.5B", &prompt)
            .reasoning_parser()
            .expect("R1 has a format")
            .parse_complete(" it is so.");
        assert_eq!(
            split.reasoning, "",
            "the block is closed; nothing is reasoning"
        );
        assert_eq!(split.content, "it is so.");
    }

    /// The explicit modes override auto's rule.
    #[test]
    fn an_explicit_mode_wins_over_the_auto_rule() {
        let messages = [
            msg("user", "why", None),
            msg("assistant", "Because", Some("thought")),
        ];
        let reasoning = render(
            R1,
            &messages,
            Some(ReasoningFormat::Think),
            Continuation::Reasoning,
        )
        .expect("renders");
        assert_eq!(reasoning, "<|User|>why<|Assistant|><think>thought");
        let content = render(
            R1,
            &[
                msg("user", "why", None),
                msg("assistant", "", Some("thought")),
            ],
            Some(ReasoningFormat::Think),
            Continuation::Content,
        )
        .expect("renders");
        assert_eq!(content, "<|User|>why<|Assistant|><think>thought</think>");
    }

    /// A model with no reasoning format continues its text and nothing
    /// else is written.
    #[test]
    fn a_plain_model_continues_its_content() {
        let prompt = render(
            CHATML,
            &[msg("user", "hi", None), msg("assistant", "Hello, I", None)],
            None,
            Continuation::Auto,
        )
        .expect("renders");
        assert_eq!(
            prompt,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nHello, I"
        );
    }

    /// A thought for a model that cannot have written one is refused,
    /// not folded into the content.
    #[test]
    fn reasoning_for_a_model_with_no_format_is_refused() {
        let err = render(
            CHATML,
            &[
                msg("user", "hi", None),
                msg("assistant", "", Some("thought")),
            ],
            None,
            Continuation::Auto,
        )
        .expect_err("refused");
        assert_eq!(err.0, axum::http::StatusCode::NOT_IMPLEMENTED);
    }

    /// The families that cannot be continued by concatenation say so.
    #[test]
    fn channel_grammars_and_always_open_content_are_refused_by_name() {
        let messages = [msg("user", "hi", None), msg("assistant", "x", Some("t"))];
        for format in [ReasoningFormat::GptOss, ReasoningFormat::MuseGlimmer] {
            let err =
                render(CHATML, &messages, Some(format), Continuation::Auto).expect_err("refused");
            assert_eq!(err.0, axum::http::StatusCode::NOT_IMPLEMENTED, "{format:?}");
        }
        let err = render(
            CHATML,
            &messages,
            Some(ReasoningFormat::DeepSeekV32),
            Continuation::Content,
        )
        .expect_err("refused");
        assert_eq!(err.0, axum::http::StatusCode::NOT_IMPLEMENTED);
        // ...but the same family's THOUGHT can be continued.
        render(
            CHATML,
            &[msg("user", "hi", None), msg("assistant", "", Some("t"))],
            Some(ReasoningFormat::DeepSeekV32),
            Continuation::Auto,
        )
        .expect("a reasoning continuation renders");
    }

    #[test]
    fn the_last_message_must_be_a_lone_assistant_turn_without_tool_calls() {
        let bad_role = render(CHATML, &[msg("user", "hi", None)], None, Continuation::Auto)
            .expect_err("refused");
        assert_eq!(bad_role.0, axum::http::StatusCode::BAD_REQUEST);
        let two = render(
            CHATML,
            &[
                msg("user", "hi", None),
                msg("assistant", "a", None),
                msg("assistant", "b", None),
            ],
            None,
            Continuation::Auto,
        )
        .expect_err("refused");
        assert_eq!(two.0, axum::http::StatusCode::BAD_REQUEST);
        let empty = render(CHATML, &[], None, Continuation::Auto).expect_err("refused");
        assert_eq!(empty.0, axum::http::StatusCode::BAD_REQUEST);
    }

    /// llama.cpp's value set plus an explicit off: `false` is `Off`,
    /// absence and `null` are `Unset`, an unknown mode is a parse error
    /// rather than a guess.
    #[test]
    fn the_wire_value_set_is_llama_cpps_plus_an_explicit_off() {
        assert_eq!(
            parse(serde_json::json!(true)).unwrap(),
            ContinueFinalMessage::Mode(Continuation::Auto)
        );
        assert_eq!(
            parse(serde_json::json!(false)).unwrap(),
            ContinueFinalMessage::Off
        );
        assert_eq!(
            parse(serde_json::json!(null)).unwrap(),
            ContinueFinalMessage::Unset
        );
        assert_eq!(
            parse(serde_json::json!("reasoning_content")).unwrap(),
            ContinueFinalMessage::Mode(Continuation::Reasoning)
        );
        assert_eq!(
            parse(serde_json::json!("content")).unwrap(),
            ContinueFinalMessage::Mode(Continuation::Content)
        );
        assert!(parse(serde_json::json!("auto")).is_err());
        assert!(parse(serde_json::json!(1)).is_err());
        assert_eq!(ContinueFinalMessage::default(), ContinueFinalMessage::Unset);
    }

    /// llama.cpp's server default (`server-common.cpp:1046-1056`): a
    /// request that said nothing continues a trailing assistant
    /// message, and only one, and only while the server's prefill is
    /// on. An explicit mode or `false` is never overridden.
    #[test]
    fn a_trailing_assistant_message_is_continued_by_default() {
        let trailing = [msg("user", "why", None), msg("assistant", "Because", None)];
        let no_trailing = [msg("user", "why", None)];
        assert_eq!(
            ContinueFinalMessage::Unset.resolve_with(true, &trailing),
            Some((Continuation::Auto, true)),
            "implied, and marked as implied"
        );
        assert_eq!(
            ContinueFinalMessage::Unset.resolve_with(true, &no_trailing),
            None
        );
        assert_eq!(
            ContinueFinalMessage::Unset.resolve_with(false, &trailing),
            None,
            "--no-prefill-assistant"
        );
        assert_eq!(
            ContinueFinalMessage::Off.resolve_with(true, &trailing),
            None,
            "false opts out of the default"
        );
        assert_eq!(
            ContinueFinalMessage::Mode(Continuation::Content).resolve_with(false, &trailing),
            Some((Continuation::Content, false)),
            "a named mode stands whatever the server flag says"
        );
        assert!(prefill_assistant_from_env(None), "unset is on, as upstream");
        assert!(prefill_assistant_from_env(Some("1")));
        assert!(!prefill_assistant_from_env(Some("0")));
        assert!(!prefill_assistant_from_env(Some("false")));
    }

    /// The refusal a caller gets for a continuation they never asked
    /// for names both ways out.
    #[test]
    fn an_implied_refusal_says_how_to_turn_the_default_off() {
        let (status, body) = implied_by_default(unsupported_feature("not for this family"));
        assert_eq!(status, axum::http::StatusCode::NOT_IMPLEMENTED);
        let message = body.0["error"]["message"].as_str().unwrap();
        assert!(message.starts_with("not for this family"));
        assert!(message.contains("continue_final_message: false"));
        assert!(message.contains("--no-prefill-assistant"));
    }
}
