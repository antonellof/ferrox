//! One root rule per wire format, derived from the description the
//! PARSER reads that format with.
//!
//! # Why this is not a table of eleven framings
//!
//! `policy::parser::tool_call` already knows how every family spells a
//! call, as `Markers`: the block markers, the invoke and parameter
//! tags, how a value is trimmed. A second table here -- one to read a
//! framing and one to write it -- is this repo's dominant bug shape
//! spelled out in full, and it would decay the usual way: a marker
//! corrected on the reading side, a forced call still emitted in the old
//! spelling, and a 200 whose tool call this server cannot read back.
//!
//! So there is one description. [`shape`] says what KIND of root rule a
//! format needs, and everything else -- every literal in the grammar --
//! comes from `format.markers()`. The formats whose framing cannot be
//! written as a root rule are refused BY NAME with the reason, because a
//! forced call served with a 200 that does not parse is worse than the
//! 501: the caller stops checking.
//!
//! # What each shape is
//!
//! | Shape | Formats | Root |
//! |---|---|---|
//! | [`Shape::Json`] | hermes/qwen2.5, llama3, mistral | a marker, a JSON object naming the tool, a closing marker |
//! | [`Shape::Elements`] | qwen3_coder, glm47, minimax, deepseekv32 | an invoke element holding one element per argument |
//! | [`Shape::Harmony`] | gpt_oss | a channel header addressed to `functions.<name>`, then JSON |
//!
//! One module per shape, beside this one. What each does about a
//! value's TYPE is written where that shape is built.
//!
mod elements;
mod harmony;
mod json;

use serde_json::Value;

use ferrox_models::grammar::json_schema::GrammarBuilder;
use ferrox_models::grammar::LazyTriggers;

use super::{escape, internal, invalid, unsupported, ToolSpec};
use crate::policy::parser::ToolCallFormat;
use crate::ApiError;

/// What kind of root rule a format's calls need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// A marker, a JSON object naming the tool, and whatever closes it.
    /// `array` is Mistral's one-element list around the object.
    Json { array: bool },
    /// An element grammar: an invoke tag naming the call, one parameter
    /// tag per argument. Entirely described by `Markers`.
    Elements,
    /// gpt-oss's harmony channel addressed to a function.
    Harmony,
}

/// The shape a format's root rule takes, or the refusal naming it.
///
/// Exhaustive on purpose: a twelfth wire format must decide what a
/// forced call in it looks like, or say why it cannot, before this
/// compiles.
fn shape(format: ToolCallFormat) -> Result<Shape, ApiError> {
    match format {
        ToolCallFormat::Qwen25 | ToolCallFormat::Llama3 => Ok(Shape::Json { array: false }),
        ToolCallFormat::Mistral => Ok(Shape::Json { array: true }),
        ToolCallFormat::Qwen3Coder
        | ToolCallFormat::Glm47
        | ToolCallFormat::MiniMax
        | ToolCallFormat::DeepSeekV32 => Ok(Shape::Elements),
        ToolCallFormat::GptOss => Ok(Shape::Harmony),
        // The three that stay refused, each for a reason about the
        // format rather than about effort.
        ToolCallFormat::Gemma4 => Err(refused(
            format,
            "a gemma4 call's arguments are a comma-separated list in gemma's own quoting rather \
             than a JSON object, so which of them are required cannot be expressed by the object \
             rule every other format here shares; writing a second one beside the JSON Schema \
             converter is the drift this refusal exists to avoid",
        )),
        ToolCallFormat::MiniMaxM3 => Err(refused(
            format,
            "a minimax_m3 call names each argument with an ELEMENT of its own, and what a \
             repeated element means -- an array rather than a value -- depends on siblings that \
             have not been written yet, so no root rule can force a call whose arguments this \
             server would read back the way the schema declares them",
        )),
        ToolCallFormat::MuseGlimmer => Err(refused(
            format,
            "a muse_glimmer call's boundary is not syntactic: the same <atem:function_calls> \
             block is a call inside a channel addressed to a tool and prose inside one addressed \
             to the user, so a grammar over the block alone would force text this server reads \
             back as content",
        )),
    }
}

/// Build the body of `root` for `format`, and the trigger that switches
/// the lazy grammar on.
///
/// Every rule this adds to `builder` is added in dependency order, and
/// the exclusion automata come FIRST: their states reference each other
/// by name, so they must be added while nothing but the builtins is
/// bound. See [`text_excluding`].
pub(super) fn build_root(
    builder: &mut GrammarBuilder,
    format: ToolCallFormat,
    tools: &[ToolSpec<'_>],
) -> Result<(String, LazyTriggers), ApiError> {
    match shape(format)? {
        Shape::Json { array } => json::json_root(builder, format, tools, array),
        Shape::Elements => elements::elements_root(builder, format, tools),
        Shape::Harmony => harmony::harmony_root(builder, tools),
    }
}

// ---- shared ----

/// `OPEN body CLOSE`, with the close omitted for the formats that end at
/// end-of-text.
pub(super) fn block(open: &str, body: &str, close: &str) -> String {
    let mut root = format!(r#""{}" {body}"#, escape(open));
    if !close.is_empty() {
        root.push_str(&format!(r#" "{}""#, escape(close)));
    }
    root
}

/// The lazy trigger for a format whose calls open with one marker.
pub(super) fn trigger(open: &str) -> Result<LazyTriggers, ApiError> {
    LazyTriggers::new()
        .with_word(open)
        .map_err(|e| internal(format!("tool-call trigger does not compile: {e}")))
        .map(LazyTriggers::mandatory)
}

/// The tool's `parameters`, or the schema of a call that takes none.
pub(super) fn parameters<'a>(tool: &ToolSpec<'a>) -> &'a Value {
    static EMPTY: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    tool.parameters.unwrap_or_else(|| {
        EMPTY.get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
            })
        })
    })
}

/// Every character of an argument's name reaches the grammar as a
/// literal, and several formats end the name at a `>` or a `"`. Held to
/// the same rule as a tool name rather than escaped into something the
/// checkpoint was never trained to write.
pub(super) fn check_key(tool: &str, key: &str) -> Result<(), ApiError> {
    let ok = !key.is_empty()
        && key.len() <= 64
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.');
    if ok {
        return Ok(());
    }
    Err(invalid(
        format!(
            "tool {tool:?} cannot be forced: its argument {key:?} is written into the wire format \
             as a bare name, and this server accepts only names of 1..=64 characters from \
             [A-Za-z0-9_.-] there"
        ),
        "tools",
    ))
}

pub(super) fn object_expected(tool: &str) -> ApiError {
    invalid(
        format!(
            "tool {tool:?} cannot be forced: this checkpoint's wire format writes a call's \
             arguments as named members, so its \"parameters\" must be an object schema"
        ),
        "tools",
    )
}

pub(super) fn untyped(tool: &str, key: &str, why: &str) -> ApiError {
    invalid(
        format!(
            "tool {tool:?} cannot be forced: its argument {key:?} cannot be given a value \
                 rule, because {why}"
        ),
        "tools",
    )
}

fn refused(format: ToolCallFormat, why: &str) -> ApiError {
    unsupported(format!(
        "tool_choice cannot be enforced for a {} checkpoint: {why}. Use tool_choice \"auto\", \
         which asks for a call in the prompt instead of forcing one.",
        format.as_str()
    ))
}
