//! gpt-oss's harmony channel, addressed to a function.
//!
//! Every literal here comes from
//! [`crate::policy::parser::tool_call::harmony`], which `parse_harmony`
//! reads the same call with.

use serde_json::Value;

use ferrox_models::grammar::json_schema::GrammarBuilder;
use ferrox_models::grammar::LazyTriggers;

use super::{object_expected, parameters};
use crate::policy::parser::tool_call::harmony;
use crate::tool_grammar::{escape, internal, schema_refused, ToolSpec};
use crate::ApiError;

/// `<|channel|>commentary to=functions.NAME<|message|>{…}<|call|>`.
///
/// Every literal comes from [`harmony`], which `parse_harmony` reads the
/// same call with.
pub(super) fn harmony_root(
    builder: &mut GrammarBuilder,
    tools: &[ToolSpec<'_>],
) -> Result<(String, LazyTriggers), ApiError> {
    // The `<|constrain|>json` hint the harmony spec allows between the
    // recipient and the message. Optional: the header is read by
    // splitting on whitespace, so it changes nothing about the call.
    let constrain = builder.add_rule(
        "harmony-constrain",
        &format!(r#"| " {}json""#, escape(harmony::CONSTRAIN)),
    );
    let channel = builder.add_rule(
        "harmony-channel",
        &harmony::CHANNELS
            .iter()
            .map(|name| format!("\"{}\"", escape(name)))
            .collect::<Vec<_>>()
            .join(" | "),
    );

    let mut alternatives = Vec::with_capacity(tools.len());
    for tool in tools {
        let schema = parameters(tool);
        match schema.get("type").and_then(Value::as_str) {
            Some("object") | None => {}
            // A harmony message body that is not a JSON object is read
            // back as `{}` (`normalize_arguments`), so forcing one would
            // serve a call with its arguments thrown away.
            Some(_) => return Err(object_expected(tool.name)),
        }
        let args = builder
            .add_schema_value(&format!("tool-{}-args", tool.name), schema)
            .map_err(|e| schema_refused(tool.name, &e))?;
        let body = format!(
            r#""{name}" {constrain} "{message}" {args} "{call}""#,
            name = escape(tool.name),
            message = escape(harmony::MESSAGE_OPEN),
            call = escape(harmony::CALL_CLOSE),
        );
        alternatives.push(builder.add_rule(&format!("tool-{}-call", tool.name), &body));
    }
    let call = builder.add_rule("tool-call", &alternatives.join(" | "));

    let root = format!(
        r#""{open}" {channel} " {key}{namespace}" {call}"#,
        open = escape(harmony::CHANNEL_OPEN),
        key = escape(harmony::RECIPIENT_KEY),
        namespace = escape(harmony::FUNCTION_NAMESPACE),
    );

    // One trigger per channel a call may be written on, each ending at
    // `to=` rather than at the whole recipient: the name that follows is
    // then still ahead of the sampler, and so still constrained. A
    // trigger on `<|channel|>` alone would fire on the reasoning channel
    // and force a call inside the model's own thinking.
    let mut triggers = LazyTriggers::new().mandatory();
    for name in harmony::CHANNELS {
        triggers = triggers
            .with_word(&format!(
                "{}{name} {}",
                harmony::CHANNEL_OPEN,
                harmony::RECIPIENT_KEY
            ))
            .map_err(|e| internal(format!("tool-call trigger does not compile: {e}")))?;
    }
    Ok((root, triggers))
}
