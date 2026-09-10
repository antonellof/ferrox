//! The families whose payload is a JSON object behind a marker:
//! hermes/qwen2.5, llama 3 and mistral.
//!
//! Their arguments are the tool's `parameters` schema run straight
//! through the JSON Schema converter, because the wire spelling of a
//! value and the schema's own spelling of it are the same thing here.

use ferrox_models::grammar::json_schema::GrammarBuilder;
use ferrox_models::grammar::LazyTriggers;

use super::{block, parameters, trigger};
use crate::policy::parser::ToolCallFormat;
use crate::tool_grammar::{schema_refused, ToolSpec};
use crate::ApiError;

/// `OPEN {"name": …, "arguments": …} CLOSE`, the shape whose payload is
/// a JSON object.
pub(super) fn json_root(
    builder: &mut GrammarBuilder,
    format: ToolCallFormat,
    tools: &[ToolSpec<'_>],
    array: bool,
) -> Result<(String, LazyTriggers), ApiError> {
    let markers = format.markers();
    let mut alternatives = Vec::with_capacity(tools.len());
    for tool in tools {
        let args = builder
            .add_schema_value(&format!("tool-{}-args", tool.name), parameters(tool))
            .map_err(|e| schema_refused(tool.name, &e))?;
        let body = format!(
            r#""{{" space "\"name\"" space ":" space "\"{name}\"" space "," space "\"arguments\"" space ":" space {args} space "}}""#,
            name = tool.name,
        );
        alternatives.push(builder.add_rule(&format!("tool-{}-call", tool.name), &body));
    }

    let call = builder.add_rule("tool-call", &alternatives.join(" | "));
    let payload = if array {
        builder.add_rule("tool-call-list", &format!(r#""[" space {call} space "]""#))
    } else {
        call
    };
    Ok((
        block(
            markers.open,
            &format!("space {payload} space"),
            markers.close,
        ),
        trigger(markers.open)?,
    ))
}
