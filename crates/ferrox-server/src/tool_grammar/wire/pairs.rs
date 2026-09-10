//! Gemma 4's own call syntax: a name, then a comma-separated list of
//! `key:value` pairs in gemma's quoting.
//!
//! ```text
//! <|tool_call>call:get_weather{city:<|"|>Rome<|"|>,days:3}<tool_call|>
//! ```
//!
//! Every literal comes from [`gemma`], which `parse_gemma` reads the
//! same call with.
//!
//! # What this does about types
//!
//! The list is not a JSON object, so the schema converter -- which
//! writes one -- is the wrong instrument for the WHOLE argument list,
//! and it is the right one for some of the values inside it. Which is
//! which follows from `gemma_arguments`, the function that reads the
//! list back:
//!
//! * `string` -- wrapped in [`gemma::QUOTE`], which is the only form
//!   that reaches the tool as a string. The text between the quotes may
//!   hold anything except the quote itself and the marker that ends the
//!   call, because the reader stops at either regardless of the other.
//! * `integer` / `number` / `boolean` / `null` -- the property's own
//!   JSON. Gemma's template writes these exactly as JSON does
//!   (`format_argument` emits `true`, `null` and a bare number), and
//!   `parse_loose` reads them back with `serde_json`, so the two agree.
//! * `object` / `array` -- REFUSED, naming the property. Gemma's
//!   template writes a composite in its own DSL, with bare keys and
//!   `<|"|>`-quoted strings inside it, and `parse_loose` parses the
//!   value with `serde_json`. So gemma's own spelling of a nested object
//!   comes back as a STRING, and JSON's spelling is not what the
//!   checkpoint was trained to write. Neither is a forced call worth
//!   serving, and writing one anyway is the "approximately the format"
//!   this whole module exists to avoid.
//! * anything else, including a property with no declared `type` --
//!   refused, naming the property, exactly as the element shape does.
//!
//! # Why the pairs are ordered, and each one optional
//!
//! The template renders a call's arguments `| dictsort`ed, so the
//! checkpoint writes them sorted by key, and this writes them in the
//! same order. Each pair carries its own separator rather than being an
//! independent `?`: a comma belongs between two pairs and nowhere else,
//! so what may be skipped is the pair TOGETHER WITH the comma before it,
//! and the first pair written is the one with no comma. That is what
//! [`arg_list`] builds, and it is why a tool whose properties are all
//! optional can still be called with none of them.

use serde_json::Value;

use ferrox_models::grammar::json_schema::GrammarBuilder;
use ferrox_models::grammar::LazyTriggers;

use super::{block, check_key, object_expected, parameters, trigger, untyped};
use crate::policy::parser::tool_call::gemma;
use crate::policy::parser::ToolCallFormat;
use crate::tool_grammar::exclude::text_excluding;
use crate::tool_grammar::{escape, schema_refused, ToolSpec};
use crate::ApiError;

/// `<|tool_call>call:NAME{k:v,k:v}<tool_call|>`.
pub(super) fn pairs_root(
    builder: &mut GrammarBuilder,
    format: ToolCallFormat,
    tools: &[ToolSpec<'_>],
) -> Result<(String, LazyTriggers), ApiError> {
    let markers = format.markers();
    // A quoted value stops at either of these, whichever the model
    // writes first, so a value that may hold one is a value this server
    // reads back wrong. See `exclude`.
    let text = text_excluding(builder, "arg-text", &[gemma::QUOTE, markers.close])?;

    let mut alternatives = Vec::with_capacity(tools.len());
    for tool in tools {
        let args = arg_list(builder, tool, &text)?;
        let body = format!(
            r#""{key}{name}{open}" {args} "{close}""#,
            key = escape(gemma::CALL_KEY),
            name = escape(tool.name),
            open = escape(gemma::ARGS_OPEN),
            close = escape(gemma::ARGS_CLOSE),
        );
        alternatives.push(builder.add_rule(&format!("tool-{}-call", tool.name), &body));
    }

    let call = builder.add_rule("tool-call", &alternatives.join(" | "));
    Ok((
        block(markers.open, &call, markers.close),
        trigger(markers.open)?,
    ))
}

/// The rule for one tool's whole argument list: the pairs in the order
/// the template writes them, each optional one skippable together with
/// the separator in front of it.
fn arg_list(
    builder: &mut GrammarBuilder,
    tool: &ToolSpec<'_>,
    text: &str,
) -> Result<String, ApiError> {
    let schema = parameters(tool);
    let Some(object) = schema.as_object() else {
        return Err(object_expected(tool.name));
    };
    match object.get("type").and_then(Value::as_str) {
        Some("object") | None => {}
        Some(_) => return Err(object_expected(tool.name)),
    }
    let properties = match object.get("properties") {
        None => return Ok(builder.add_rule(&format!("tool-{}-args", tool.name), r#""""#)),
        Some(Value::Object(map)) => map,
        Some(_) => return Err(object_expected(tool.name)),
    };
    let required: Vec<&str> = object
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    // `properties` is a `serde_json::Map`, whose iteration order is the
    // key order -- the same `| dictsort` the template renders with.
    let mut pairs = Vec::with_capacity(properties.len());
    for (key, property) in properties {
        pairs.push((
            pair_rule(builder, tool, key, property, text)?,
            required.contains(&key.as_str()),
        ));
    }
    for name in &required {
        if !properties.contains_key(*name) {
            return Err(crate::tool_grammar::invalid(
                format!(
                    "tool {:?} cannot be forced: it requires the argument {name:?}, which its \
                     \"parameters\" schema does not declare",
                    tool.name
                ),
                "tools",
            ));
        }
    }

    // Built from the end, so each rule can name the one after it.
    // `rest` is the tail once SOMETHING has been written -- so every
    // pair in it carries a separator -- and `first` is the tail while
    // nothing has been, so the pair that opens the list has none.
    let mut rest = String::from(r#""""#);
    let mut first = String::from(r#""""#);
    for (index, (rule, is_required)) in pairs.iter().enumerate().rev() {
        let separator = escape(gemma::PAIR_SEPARATOR);
        let with_separator = format!(r#""{separator}" {rule} {rest}"#);
        let without = format!("{rule} {rest}");
        let (rest_body, first_body) = if *is_required {
            (with_separator, without)
        } else {
            (
                format!("{with_separator} | {rest}"),
                format!("{without} | {first}"),
            )
        };
        rest = builder.add_rule(&format!("tool-{}-rest-{index}", tool.name), &rest_body);
        first = builder.add_rule(&format!("tool-{}-from-{index}", tool.name), &first_body);
    }
    Ok(first)
}

/// One `key:value` pair.
fn pair_rule(
    builder: &mut GrammarBuilder,
    tool: &ToolSpec<'_>,
    key: &str,
    property: &Value,
    text: &str,
) -> Result<String, ApiError> {
    check_key(tool.name, key)?;
    let value = value_rule(builder, tool, key, property, text)?;
    let body = format!(
        r#""{key}{separator}" {value}"#,
        key = escape(key),
        separator = escape(gemma::KEY_SEPARATOR),
    );
    Ok(builder.add_rule(&format!("tool-{}-pair-{key}", tool.name), &body))
}

/// How one value must be written so that `gemma_arguments` reads it back
/// as the schema declares it.
fn value_rule(
    builder: &mut GrammarBuilder,
    tool: &ToolSpec<'_>,
    key: &str,
    property: &Value,
    text: &str,
) -> Result<String, ApiError> {
    let Some(object) = property.as_object() else {
        return Err(untyped(tool.name, key, "it is not a schema object"));
    };
    let Some(declared) = object.get("type").and_then(Value::as_str) else {
        return Err(untyped(
            tool.name,
            key,
            "it declares no \"type\", and this server would have to GUESS whether the text the \
             model writes there is a string, a number or JSON",
        ));
    };
    let quote = escape(gemma::QUOTE);
    match declared {
        "string" => {
            // A quoted run, or the `enum` members inside the same quotes:
            // the quotes are what make `gemma_arguments` produce a
            // string rather than hand the text to `parse_loose`.
            let inner = match object.get("enum").or_else(|| object.get("const")) {
                None => text.to_string(),
                Some(members) => {
                    let members = match members {
                        Value::Array(members) => members.clone(),
                        single => vec![single.clone()],
                    };
                    if members.is_empty() {
                        return Err(untyped(tool.name, key, "its \"enum\" lists no members"));
                    }
                    let mut alternatives = Vec::with_capacity(members.len());
                    for member in &members {
                        let Some(member) = member.as_str() else {
                            return Err(untyped(
                                tool.name,
                                key,
                                "it is a string whose \"enum\" holds a member that is not a string",
                            ));
                        };
                        if member.contains(gemma::QUOTE) || member.contains(gemma::BLOCK_CLOSE) {
                            return Err(untyped(
                                tool.name,
                                key,
                                "one of its \"enum\" members contains the markup that ends an \
                                 argument, so writing it would end the argument early",
                            ));
                        }
                        alternatives.push(format!("\"{}\"", escape(member)));
                    }
                    builder.add_rule(
                        &format!("tool-{}-enum-{key}", tool.name),
                        &alternatives.join(" | "),
                    )
                }
            };
            Ok(builder.add_rule(
                &format!("tool-{}-arg-{key}", tool.name),
                &format!(r#""{quote}" {inner} "{quote}""#),
            ))
        }
        // Gemma writes these the way JSON does, and `parse_loose` reads
        // them with `serde_json`.
        "integer" | "number" | "boolean" | "null" => builder
            .add_schema_value(&format!("tool-{}-arg-{key}", tool.name), property)
            .map_err(|e| schema_refused(tool.name, &e)),
        "object" | "array" => Err(untyped(
            tool.name,
            key,
            &format!(
                "it is declared {declared:?}, and gemma writes a composite value in its own DSL -- \
                 bare keys, and strings wrapped in gemma's quote -- while this server reads a \
                 value back with `serde_json`. So the spelling the checkpoint was trained to \
                 write is read back as a string, and the spelling that reads back correctly is \
                 not one this family emits"
            ),
        )),
        other => Err(untyped(
            tool.name,
            key,
            &format!("this server does not map the declared type {other:?} onto gemma's syntax"),
        )),
    }
}
