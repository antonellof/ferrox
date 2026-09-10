//! The XML-ish families: an invoke element naming the call, one
//! parameter element per argument.
//!
//! # What this does about types
//!
//! An XML-ish family writes every value as TEXT, and this server decides
//! what it MEANS from the tool's own schema
//! (`ToolCallParser::convert_value`). The grammar has to agree with that
//! decision or the forced call arrives with arguments the schema does
//! not describe, so the value rule is chosen from the same declared
//! `type`:
//!
//! * `string` -- free text, everything up to the tag that ends the value
//!   ([`crate::tool_grammar::exclude`]), or the `enum` members verbatim
//!   when the schema lists them. Not JSON: a `"…"`-quoted string reaches
//!   `convert_declared` as a string WITH its quotes.
//! * everything else -- the property's own JSON, through the same
//!   converter the JSON families use, because `convert_declared` parses
//!   an `integer` / `number` / `boolean` / `object` / `array` / `null`
//!   value as JSON.
//! * a property with no declared `type`, or one this server does not map
//!   -- refused, naming the property. `parse_loose` would guess, and a
//!   guess is the thing a forced call may not be built on.

use serde_json::Value;

use ferrox_models::grammar::json_schema::GrammarBuilder;
use ferrox_models::grammar::LazyTriggers;

use super::{block, check_key, object_expected, parameters, trigger, untyped};
use crate::policy::parser::tool_call::{Markers, NameStyle, TagGrammar};
use crate::policy::parser::ToolCallFormat;
use crate::tool_grammar::exclude::text_excluding;
use crate::tool_grammar::{escape, internal, invalid, schema_refused, ToolSpec};
use crate::ApiError;

/// `OPEN <invoke name> <param>value</param> … </invoke> CLOSE`.
pub(super) fn elements_root(
    builder: &mut GrammarBuilder,
    format: ToolCallFormat,
    tools: &[ToolSpec<'_>],
) -> Result<(String, LazyTriggers), ApiError> {
    // No `..`: a new field of `Markers` must be looked at here before
    // this compiles again, which is the only thing that keeps a reader
    // and a writer of one framing honest.
    let Markers {
        open,
        close,
        invoke,
        param,
        // Both of these are about a value this grammar does not write.
        // `trim_newlines` strips whitespace from around a value, and the
        // only values here that may carry any are the JSON ones, which
        // `convert_declared` trims before it parses whatever this leaves.
        // `undeclared` decides what an argument the schema never
        // mentioned is worth, and this grammar can only write arguments
        // the schema declares.
        trim_newlines: _,
        undeclared: _,
    } = format.markers();

    let Some(param) = param else {
        return Err(internal(format!(
            "{} was given the element shape but its framing declares no parameter tag",
            format.as_str()
        )));
    };
    // The literals a value ends at are the ones it may not contain.
    let forbidden = value_forbidden(param);
    let text = text_excluding(builder, "arg-text", &forbidden)?;

    let mut alternatives = Vec::with_capacity(tools.len());
    for tool in tools {
        let mut body = invoke_open(invoke, tool.name)?;
        for arg in element_args(builder, param, tool, &text)? {
            body.push_str(" space ");
            body.push_str(&arg);
        }
        if let Some(tag) = invoke {
            body.push_str(&format!(r#" space "{}""#, escape(tag.close)));
        }
        alternatives.push(builder.add_rule(&format!("tool-{}-call", tool.name), &body));
    }

    let call = builder.add_rule("tool-call", &alternatives.join(" | "));
    // A format that names its call in a TAG may have whitespace before
    // it. One that names it in bare text (GLM) may not: its name ends at
    // the first newline, so a newline before it is an empty name.
    let lead = if invoke.is_some() { "space " } else { "" };
    Ok((
        block(open, &format!("{lead}{call} space"), close),
        trigger(open)?,
    ))
}

/// The rules for one tool's arguments, in the order a call writes them:
/// the required ones, then each optional one on its own.
fn element_args(
    builder: &mut GrammarBuilder,
    param: TagGrammar,
    tool: &ToolSpec<'_>,
    text: &str,
) -> Result<Vec<String>, ApiError> {
    let schema = parameters(tool);
    let Some(object) = schema.as_object() else {
        return Err(object_expected(tool.name));
    };
    match object.get("type").and_then(Value::as_str) {
        Some("object") | None => {}
        Some(_) => return Err(object_expected(tool.name)),
    }
    let properties = match object.get("properties") {
        None => return Ok(Vec::new()),
        Some(Value::Object(map)) => map,
        Some(_) => return Err(object_expected(tool.name)),
    };
    let required: Vec<&str> = object
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let mut args = Vec::new();
    for key in required.iter().copied() {
        let Some(property) = properties.get(key) else {
            return Err(invalid(
                format!(
                    "tool {:?} cannot be forced: it requires the argument {key:?}, which its \
                     \"parameters\" schema does not declare",
                    tool.name
                ),
                "tools",
            ));
        };
        args.push(param_rule(builder, param, tool, key, property, text)?);
    }
    for (key, property) in properties {
        if required.contains(&key.as_str()) {
            continue;
        }
        let rule = param_rule(builder, param, tool, key, property, text)?;
        args.push(format!("{rule}?"));
    }
    Ok(args)
}

/// One argument: the parameter tag, its value, and the tag that closes
/// it -- all four spellings taken from [`TagGrammar`].
fn param_rule(
    builder: &mut GrammarBuilder,
    param: TagGrammar,
    tool: &ToolSpec<'_>,
    key: &str,
    property: &Value,
    text: &str,
) -> Result<String, ApiError> {
    check_key(tool.name, key)?;
    // A JSON value may be surrounded by whitespace -- the template's own
    // layout puts a newline there, and `convert_declared` trims before it
    // parses. A text or `enum` value may NOT: it reaches the tool
    // verbatim, so whitespace around it would be part of it.
    let value = match value_shape(tool.name, key, property, &value_forbidden(param))? {
        ValueShape::Text => text.to_string(),
        ValueShape::Literals(body) => {
            builder.add_rule(&format!("tool-{}-enum-{key}", tool.name), &body)
        }
        ValueShape::Json => format!(
            "space {} space",
            builder
                .add_schema_value(&format!("tool-{}-arg-{key}", tool.name), property)
                .map_err(|e| schema_refused(tool.name, &e))?
        ),
    };
    let (head, close) = param_tags(param, key);
    let body = format!(r#""{}" {value} "{}""#, escape(&head), escape(&close));
    Ok(builder.add_rule(&format!("tool-{}-param-{key}", tool.name), &body))
}

/// The two literals that wrap one argument's value: what opens it, and
/// what closes it, both as they reach the wire.
///
/// A pair rather than "the head, and `TagGrammar::close`", because
/// [`NameStyle::Element`] is the style whose CLOSING tag repeats the
/// argument's name -- MiniMax-M3 writes
/// `]<]minimax[>[<city>Rome]<]minimax[>[</city>` -- so there is no one
/// closing literal to read off the framing.
fn param_tags(param: TagGrammar, key: &str) -> (String, String) {
    let close = param.close.to_string();
    match param.name {
        NameStyle::Bare => (format!("{}{key}>", param.open), close),
        NameStyle::Attribute => (format!("{} name=\"{key}\">", param.open), close),
        NameStyle::Paired {
            key_close,
            value_open,
        } => (format!("{}{key}{key_close}{value_open}", param.open), close),
        NameStyle::Element => (
            format!("{}{key}>", param.open),
            format!("{}{key}>", param.close),
        ),
    }
}

/// The literals a value written as bare TEXT may not contain, because
/// the reader would stop or restart at one.
fn value_forbidden(param: TagGrammar) -> [&'static str; 1] {
    match param.name {
        // The value runs to one fixed closing tag, and the reader looks
        // for exactly that.
        NameStyle::Bare | NameStyle::Attribute | NameStyle::Paired { .. } => [param.close],
        // M3's closing tag depends on the argument, so there is no
        // single one to forbid -- but every structural tag it has (the
        // wrapper's, the invoke's, an element's, and every closer)
        // begins with this prefix, and `m3_scan_elements` re-reads a
        // value as STRUCTURE from any occurrence of it. Forbidding the
        // prefix forbids all of them at once.
        NameStyle::Element => [param.open],
    }
}

/// How one argument's value must be written so that
/// `ToolCallParser::convert_value` reads it back as the schema declares
/// it.
enum ValueShape {
    /// Free text up to the closing tag.
    Text,
    /// The `enum` members, written as they are.
    Literals(String),
    /// The property's own JSON.
    Json,
}

/// Keywords that constrain nothing, so a value rule may ignore them.
/// The same list `json_schema` ignores.
const ANNOTATIONS: [&str; 10] = [
    "title",
    "description",
    "default",
    "examples",
    "$schema",
    "$id",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
];

fn value_shape(
    tool: &str,
    key: &str,
    property: &Value,
    forbidden: &[&str],
) -> Result<ValueShape, ApiError> {
    let Some(object) = property.as_object() else {
        return Err(untyped(tool, key, "it is not a schema object"));
    };
    let declared = object.get("type").and_then(Value::as_str);
    let Some(declared) = declared else {
        return Err(untyped(
            tool,
            key,
            "it declares no \"type\", and this server would have to GUESS whether the text the \
             model writes there is a string, a number or JSON",
        ));
    };
    if declared != "string" {
        // Every other declared type reaches `convert_declared` as JSON,
        // which is exactly what the schema converter emits.
        return Ok(ValueShape::Json);
    }

    // A declared string is handed to the tool verbatim, so its value is
    // TEXT and the schema converter -- which would quote it -- is the
    // wrong instrument.
    if let Some(members) = object.get("enum").or_else(|| object.get("const")) {
        let members = match members {
            Value::Array(members) => members.clone(),
            single => vec![single.clone()],
        };
        if members.is_empty() {
            return Err(untyped(tool, key, "its \"enum\" lists no members"));
        }
        let mut alternatives = Vec::with_capacity(members.len());
        for member in &members {
            let Some(member) = member.as_str() else {
                return Err(untyped(
                    tool,
                    key,
                    "it is a string whose \"enum\" holds a member that is not a string",
                ));
            };
            if forbidden.iter().any(|literal| member.contains(literal)) {
                return Err(untyped(
                    tool,
                    key,
                    "one of its \"enum\" members contains the markup that ends an argument, so \
                     writing it would end the argument early",
                ));
            }
            alternatives.push(format!("\"{}\"", escape(member)));
        }
        return Ok(ValueShape::Literals(alternatives.join(" | ")));
    }

    for keyword in object.keys() {
        if keyword == "type" || ANNOTATIONS.contains(&keyword.as_str()) {
            continue;
        }
        return Err(untyped(
            tool,
            key,
            &format!(
                "it is a string carrying {keyword:?}, which this server cannot honour in a value \
                 that is written as bare text rather than as JSON"
            ),
        ));
    }
    Ok(ValueShape::Text)
}

/// The invoke element that names the call.
fn invoke_open(invoke: Option<TagGrammar>, name: &str) -> Result<String, ApiError> {
    match invoke {
        Some(tag) => match tag.name {
            NameStyle::Bare => Ok(format!(r#""{}{}>""#, escape(tag.open), escape(name))),
            NameStyle::Attribute => Ok(format!(
                r#""{} name=\"{}\">""#,
                escape(tag.open),
                escape(name)
            )),
            NameStyle::Paired { .. } | NameStyle::Element => Err(internal(format!(
                "the invoke tag {:?} is named the way a parameter is, which has no reader",
                tag.open
            ))),
        },
        // GLM has no invoke tag: the name is bare text right after the
        // block opener, and it ends at the first newline.
        None => Ok(format!(r#""{}\n""#, escape(name))),
    }
}
