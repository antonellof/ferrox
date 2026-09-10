//! The grammar behind `tool_choice: "required"` and a named
//! `tool_choice`.
//!
//! Both used to be a 501 naming this file's absence: without a grammar,
//! "the model MUST call a tool" can only be asked for in the prompt, and
//! a request that asked to be forced and was merely asked would be served
//! a 200 whose answer may contain no call at all.
//!
//! # What is generated, and from what
//!
//! One grammar per request, from the request's own `tools`:
//!
//! ```text
//! root       ::= "<tool_call>" space call space "</tool_call>"
//! call       ::= tool-get-weather | tool-send-mail
//! tool-get-weather ::= "{" space "\"name\"" space ":" space "\"get_weather\"" space ","
//!                          space "\"arguments\"" space ":" space tool-get-weather-args
//!                          space "}"
//! ```
//!
//! `tool-<name>-args` is the tool's own `parameters` JSON Schema, run
//! through [`ferrox_models::grammar::json_schema`] -- the same converter
//! the CLI's `-j` / `--json-schema` uses, `pattern` included, and the same
//! one `response_format: {"type": "json_schema"}` goes through in
//! [`crate::grammar_request`]. So the arguments are not merely
//! well-formed JSON: a `required` property that
//! the schema declares is a property the model cannot omit, and an `enum`
//! is a choice it cannot invent a member of.
//!
//! `tool_choice: "required"` generates the union of every offered tool. A
//! NAMED `tool_choice` generates the same grammar with the union narrowed
//! to one alternative -- that is the whole difference, which is why they
//! are one function and not two.
//!
//! # Which wire format, and where its shape comes from
//!
//! [`wire`] holds one root rule per family, and every literal in it is
//! read off `ToolCallFormat::markers()` -- the SAME description
//! [`crate::policy::parser::tool_call`] reads a call with. There is no
//! second table of framings here: two of them, one to read a format and
//! one to write it, is this repo's dominant bug shape, and it decays into
//! a 200 whose forced call this server cannot parse.
//!
//! Ten of the eleven formats this server parses can be forced: the three
//! whose payload is a JSON object behind a marker (hermes/qwen2.5,
//! llama3, mistral), the five element grammars (qwen3_coder, glm47,
//! minimax, deepseekv32, minimax_m3), gemma4's pair list, and gpt-oss's
//! harmony channel. The remaining one -- muse_glimmer -- is refused BY
//! FORMAT NAME with the reason, in [`wire`]'s `shape`. A forced call
//! served with a 200 that does not parse is worse than the 501, because
//! the caller stops checking.
//!
//! # Lazy, and mandatory
//!
//! The grammar is LAZY (see [`ferrox_models::grammar::lazy`]), triggered
//! by the wire format's opening marker, and its trigger is MANDATORY.
//!
//! llama.cpp forces a tool call with an EAGER grammar instead
//! (`grammar_lazy = false` for `COMMON_CHAT_TOOL_CHOICE_REQUIRED`), so
//! the first token of the turn is already inside the call. That does not
//! survive this server: several families open a reasoning block in the
//! PROMPT ([`crate::policy::parser::reasoning`]'s `always_open`), so the
//! model's first token is inside `<think>`, and a call forced there is
//! read back as thinking by this server's own reasoning parser -- a
//! response with a `reasoning_content` and no tool call, from a request
//! that demanded one.
//!
//! Lazy plus mandatory keeps both halves of the promise without knowing
//! anything about the checkpoint's reasoning format: the prefix is free,
//! the turn cannot END until a call has begun, and from the marker onward
//! the call is forced to be complete and schema-valid.
//!
//! The cost, stated plainly: a model that writes forever without ever
//! opening a call runs to `max_tokens` and finishes with
//! `finish_reason: "length"`. For a caller who said `required`, a visible
//! failure is the honest outcome; the alternative is prose served as
//! though it were the call they asked for. The same cost is why the
//! element families are forced into the WRAPPED form their template
//! teaches (`<tool_call><function=…>`): triggering on a bare
//! `<function=` instead would fire only after the name it introduces had
//! already been sampled unconstrained.

pub(super) mod exclude;
mod wire;

#[cfg(test)]
mod format_tests;

use std::sync::Arc;

use axum::http::StatusCode;
use axum::Json;
use ferrox_models::grammar::json_schema::GrammarBuilder;
use ferrox_models::grammar::Grammar;

use crate::policy::parser::ToolCallFormat;
use crate::ApiError;

/// A `tool_choice` that forces a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Forced<'a> {
    /// `tool_choice: "required"`: any of the offered tools.
    Any,
    /// `tool_choice: {"type": "function", "function": {"name": …}}`.
    Named(&'a str),
}

/// One offered tool, reduced to what a grammar needs of it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ToolSpec<'a> {
    pub name: &'a str,
    pub parameters: Option<&'a serde_json::Value>,
}

/// Build the grammar that forces `forced` over `tools`, for a checkpoint
/// whose calls are spelled in `format`.
pub(crate) fn build(
    forced: Forced<'_>,
    tools: &[ToolSpec<'_>],
    format: ToolCallFormat,
) -> Result<Arc<Grammar>, ApiError> {
    let chosen = select(forced, tools)?;
    for tool in &chosen {
        check_name(tool.name)?;
    }

    let mut builder = GrammarBuilder::new();
    let (root, triggers) = wire::build_root(&mut builder, format, &chosen)?;
    builder.add_rule("root", &root);

    let text = builder.finish().map_err(|e| {
        // The builder re-parses its own output, so this is a defect in
        // this module rather than anything the caller sent.
        internal(format!("tool-call grammar failed to build: {e}"))
    })?;

    let grammar = Grammar::from_str_with_root(&text, "root")
        .map_err(|e| internal(format!("tool-call grammar does not compile: {e}")))?
        .into_lazy(triggers)
        .map_err(|e| internal(format!("tool-call grammar cannot be made lazy: {e}")))?;
    Ok(Arc::new(grammar))
}

/// The tools the grammar may choose between.
fn select<'a>(forced: Forced<'_>, tools: &[ToolSpec<'a>]) -> Result<Vec<ToolSpec<'a>>, ApiError> {
    if tools.is_empty() {
        return Err(invalid(
            "tool_choice forces a tool call, but no tools were offered; send \"tools\", or use \
             tool_choice \"none\"",
            "tool_choice",
        ));
    }
    match forced {
        Forced::Any => Ok(tools.to_vec()),
        Forced::Named(name) => match tools.iter().find(|t| t.name == name) {
            Some(t) => Ok(vec![*t]),
            None => Err(invalid(
                format!("tool_choice names {name:?}, which is not one of the tools offered"),
                "tool_choice",
            )),
        },
    }
}

/// Every character in a tool name reaches the grammar as a literal and
/// most of them reach a rule name too, so the name is held to OpenAI's
/// own rule for one rather than escaped into something unreadable.
fn check_name(name: &str) -> Result<(), ApiError> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.');
    if ok {
        return Ok(());
    }
    Err(invalid(
        format!(
            "tool name {name:?} cannot be forced: a forced tool call puts the name in a grammar, \
             and this server accepts only names of 1..=64 characters from [A-Za-z0-9_.-] there"
        ),
        "tools",
    ))
}

/// GBNF literal escaping, for the fixed markers above. They contain no
/// quotes or backslashes today; this is here so that adding one that
/// does cannot quietly emit a grammar that does not parse.
fn escape(literal: &str) -> String {
    literal.replace('\\', r"\\").replace('"', "\\\"")
}

fn schema_refused(tool: &str, err: &ferrox_models::grammar::SchemaError) -> ApiError {
    invalid(
        format!(
            "tool {tool:?} cannot be forced: its \"parameters\" schema does not convert to a \
             grammar: {err}"
        ),
        "tools",
    )
}

fn invalid(message: impl Into<String>, param: &str) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "param": param,
            }
        })),
    )
}

fn unsupported(message: impl Into<String>) -> ApiError {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "param": "tool_choice",
            }
        })),
    )
}

fn internal(message: String) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "server_error",
            }
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::format_tests::feed;
    use super::*;
    use crate::output::{parse_output, OutputPosture};
    use crate::{ToolDef, ToolFunctionDef};

    fn weather() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false,
        })
    }

    fn specs<'a>(defs: &'a [(&'a str, &'a serde_json::Value)]) -> Vec<ToolSpec<'a>> {
        defs.iter()
            .map(|(name, params)| ToolSpec {
                name,
                parameters: Some(params),
            })
            .collect()
    }

    /// The headline: the grammar accepts exactly the text this server's
    /// own parser turns back into a tool call.
    #[test]
    fn the_forced_grammar_accepts_what_the_parser_reads_back() {
        let params = weather();
        let offered = [("get_weather", &params)];
        let g = build(Forced::Any, &specs(&offered), ToolCallFormat::Qwen25)
            .expect("a grammar for one tool");

        let call =
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>"#;
        assert!(
            feed(&g, &["thinking about it... ", call]).expect("the grammar accepts the call"),
            "the parse should be complete after the closing marker"
        );

        let tools = vec![ToolDef {
            kind: "function".to_string(),
            function: ToolFunctionDef {
                name: "get_weather".to_string(),
                description: None,
                parameters: Some(params.clone()),
            },
        }];
        let parsed = parse_output(
            &format!("thinking about it... {call}"),
            &tools,
            OutputPosture::for_model("test-model"),
        );
        assert_eq!(
            parsed.calls.len(),
            1,
            "the grammar and the parser must agree on the wire format"
        );
        assert_eq!(parsed.calls[0].name, "get_weather");
    }

    /// The arguments are the tool's SCHEMA, not merely JSON: a required
    /// property cannot be dropped.
    #[test]
    fn a_required_property_cannot_be_omitted() {
        let params = weather();
        let offered = [("get_weather", &params)];
        let g = build(Forced::Any, &specs(&offered), ToolCallFormat::Qwen25).unwrap();
        let err = feed(
            &g,
            &[r#"<tool_call>{"name": "get_weather", "arguments": {}}"#],
        )
        .expect_err("\"city\" is required");
        assert!(err.contains("no grammar parse survives"), "{err}");
    }

    /// A named choice is the same grammar with one alternative: the other
    /// tool's name is then unreachable.
    #[test]
    fn a_named_choice_narrows_the_union_to_one_tool() {
        let params = weather();
        let offered = [("get_weather", &params), ("send_mail", &params)];
        let tools = specs(&offered);

        let any = build(Forced::Any, &tools, ToolCallFormat::Qwen25).unwrap();
        assert!(feed(
            &any,
            &[r#"<tool_call>{"name": "send_mail", "arguments": {"city": "Rome"}}</tool_call>"#]
        )
        .is_ok());

        let named = build(Forced::Named("get_weather"), &tools, ToolCallFormat::Qwen25).unwrap();
        assert!(
            feed(&named, &[r#"<tool_call>{"name": "send_mail""#]).is_err(),
            "a named tool_choice must make every other tool unreachable"
        );
        assert!(feed(
            &named,
            &[r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>"#]
        )
        .is_ok());
    }

    /// Free text before the call is not just tolerated, it is the point:
    /// a reasoning block has to be able to come first.
    #[test]
    fn a_reasoning_block_may_precede_the_call() {
        let params = weather();
        let offered = [("get_weather", &params)];
        let g = build(Forced::Any, &specs(&offered), ToolCallFormat::Qwen25).unwrap();
        assert!(g.is_awaiting_trigger());
        assert!(
            feed(
                &g,
                &[
                    "<think>",
                    "the user wants weather; I should call the tool.",
                    "</think>",
                    r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>"#,
                ]
            )
            .expect("thinking first is allowed"),
            "the call must still complete after a reasoning block"
        );
    }

    /// And the turn may not END before the call begins: that is what
    /// makes this `required` rather than a suggestion.
    #[test]
    fn the_turn_cannot_end_before_the_call_begins() {
        let params = weather();
        let offered = [("get_weather", &params)];
        let g = build(Forced::Any, &specs(&offered), ToolCallFormat::Qwen25).unwrap();
        assert!(!g.allows_eog(), "nothing has been called yet");
        let mut mid = (*g).clone();
        mid.accept_token(0, b"I think the answer is 4.").unwrap();
        assert!(
            !mid.allows_eog(),
            "prose must not be allowed to finish the turn"
        );
    }

    /// Each format gets its own markers, and a format with no grammar is
    /// refused by name rather than served a Hermes-shaped one.
    #[test]
    fn each_supported_format_uses_its_own_markers() {
        let params = weather();
        let offered = [("get_weather", &params)];
        let tools = specs(&offered);

        let llama = build(Forced::Any, &tools, ToolCallFormat::Llama3).unwrap();
        assert!(feed(
            &llama,
            &[r#"<|python_tag|>{"name": "get_weather", "arguments": {"city": "Rome"}}"#]
        )
        .unwrap());

        let mistral = build(Forced::Any, &tools, ToolCallFormat::Mistral).unwrap();
        assert!(feed(
            &mistral,
            &[r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Rome"}}]"#]
        )
        .unwrap());

        let gemma = build(Forced::Any, &tools, ToolCallFormat::Gemma4).unwrap();
        assert!(feed(
            &gemma,
            &[r#"<|tool_call>call:get_weather{city:<|"|>Rome<|"|>}<tool_call|>"#]
        )
        .unwrap());

        let (status, Json(body)) = build(Forced::Any, &tools, ToolCallFormat::MuseGlimmer)
            .expect_err("a muse_glimmer call's boundary is a channel this cannot write");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("muse_glimmer"),
            "the refusal must name the format: {body}"
        );
    }

    /// A tool with no `parameters` still needs a grammar, and it is the
    /// empty object rather than "any JSON".
    #[test]
    fn a_tool_without_parameters_takes_an_empty_object() {
        let g = build(
            Forced::Any,
            &[ToolSpec {
                name: "ping",
                parameters: None,
            }],
            ToolCallFormat::Qwen25,
        )
        .unwrap();
        assert!(feed(
            &g,
            &[r#"<tool_call>{"name": "ping", "arguments": {}}</tool_call>"#]
        )
        .unwrap());
        assert!(
            feed(&g, &[r#"<tool_call>{"name": "ping", "arguments": {"x""#]).is_err(),
            "a tool that declares no parameters must not accept invented ones"
        );
    }

    /// Refusals a caller can act on: no tools, an unknown name, a schema
    /// the converter will not honour.
    #[test]
    fn the_refusals_name_what_is_wrong() {
        let params = weather();
        let (status, _) =
            build(Forced::Any, &[], ToolCallFormat::Qwen25).expect_err("nothing to choose between");
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let offered = [("get_weather", &params)];
        let (status, Json(body)) = build(
            Forced::Named("nope"),
            &specs(&offered),
            ToolCallFormat::Qwen25,
        )
        .expect_err("no such tool");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"].as_str().unwrap().contains("nope"));

        // `allOf` is one of the keywords the converter refuses rather
        // than silently widening.
        let hard = serde_json::json!({"allOf": [{"type": "object"}]});
        let unconvertible = [("get_weather", &hard)];
        let (status, Json(body)) =
            build(Forced::Any, &specs(&unconvertible), ToolCallFormat::Qwen25)
                .expect_err("allOf has no grammar");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("get_weather"),
            "{body}"
        );
    }

    /// Two tools whose names collapse to the same rule name must stay
    /// distinct, not silently share one argument grammar.
    #[test]
    fn tools_with_colliding_rule_names_stay_distinct() {
        let a = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": false,
        });
        let b = serde_json::json!({
            "type": "object",
            "properties": {"b": {"type": "string"}},
            "required": ["b"],
            "additionalProperties": false,
        });
        // `collapse_invalid` maps both `_` and `.` to `-`.
        let offered = [("do_it", &a), ("do.it", &b)];
        let g = build(Forced::Any, &specs(&offered), ToolCallFormat::Qwen25).unwrap();
        assert!(feed(
            &g,
            &[r#"<tool_call>{"name": "do_it", "arguments": {"a": "x"}}</tool_call>"#]
        )
        .unwrap());
        assert!(
            feed(&g, &[r#"<tool_call>{"name": "do.it", "arguments": {"a""#]).is_err(),
            "the second tool must keep its own argument grammar"
        );
    }
}
