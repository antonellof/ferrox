//! Which of `visit`'s branches a schema takes, and which keywords that
//! branch therefore consumes.
//!
//! llama.cpp's `visit` is a chain of `if`s, each testing the keywords it
//! is about to use. That works, but it means every keyword no branch
//! tested is silently discarded. Selecting the branch *as a value* first
//! lets the same conditions serve twice: once to dispatch, and once to
//! ask which keywords were left over. Anything left over is refused by
//! name in [`super::error::SchemaError::UnsupportedKeyword`], so the
//! chain has no silent exit.
//!
//! The conditions and their order are transcribed from upstream. The
//! branches upstream has that this port does not -- `allOf`, `pattern`,
//! and the integer `minimum`/`maximum` range builder -- are simply absent,
//! so their keywords fall out as unconsumed and are refused.

use super::error::{at, SchemaError};
use super::primitives::{builtin, primitive};
use super::value::JsonValue;

/// The branch `visit` takes, carrying the parts of the schema it needs.
pub(super) enum Branch<'a> {
    /// `$ref`.
    Ref(&'a str),
    /// `oneOf` / `anyOf`.
    Union(&'a [JsonValue]),
    /// `"type"` given as an array: one copy of the schema per type.
    TypeUnion(Vec<JsonValue>),
    /// `const`.
    Const(&'a JsonValue),
    /// `enum`.
    Enum(&'a [JsonValue]),
    /// An object with declared properties and/or a closed/typed tail.
    Object {
        properties: &'a [(String, JsonValue)],
        required: Vec<String>,
        additional: Option<&'a JsonValue>,
    },
    /// `items` / `prefixItems` given as an array: a fixed-length tuple.
    Tuple(&'a [JsonValue]),
    /// `items` given as a single schema: a homogeneous list.
    List {
        items: &'a JsonValue,
        min: u64,
        max: Option<u64>,
    },
    /// `pattern`: an anchored ECMA-262 regular expression.
    Pattern(&'a str),
    /// `"format": "uuid"` (or `uuid1`..`uuid5`).
    Uuid { format: &'a str },
    /// `"format"` of `date`, `time` or `date-time`.
    StringFormat { format: &'a str },
    /// `"type": "string"` with `minLength` / `maxLength`.
    BoundedString { min: u64, max: Option<u64> },
    /// The unconstrained-object fallback: `{}` or a bare `"type": "object"`.
    ObjectPrimitive,
    /// The unconstrained-anything fallback: annotations only, no `"type"`.
    ValuePrimitive,
    /// A named primitive type.
    Primitive(&'a str),
}

/// Keywords that carry no constraint, so ignoring one cannot widen the
/// grammar. `$defs` and `definitions` are here because they are containers
/// for `$ref` targets, visited only when something points at them.
const ANNOTATIONS: &[&str] = &[
    "$anchor",
    "$comment",
    "$defs",
    "$id",
    "$schema",
    "default",
    "definitions",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
];

/// The JSON types a keyword says anything about. A keyword outside the
/// instance's declared type is vacuously satisfied -- JSON Schema scopes
/// `items` to arrays, `properties` to objects, `pattern` to strings and so
/// on -- so ignoring it there is correct rather than a silent widening.
/// `None` means the keyword applies to every instance.
fn scope(keyword: &str) -> Option<&'static [&'static str]> {
    const ARRAY: &[&str] = &["array"];
    const OBJECT: &[&str] = &["object"];
    const STRING: &[&str] = &["string"];
    const NUMERIC: &[&str] = &["number", "integer"];
    match keyword {
        "items" | "prefixItems" | "minItems" | "maxItems" | "additionalItems"
        | "unevaluatedItems" | "uniqueItems" | "contains" | "minContains" | "maxContains" => {
            Some(ARRAY)
        }
        "properties"
        | "required"
        | "additionalProperties"
        | "patternProperties"
        | "propertyNames"
        | "minProperties"
        | "maxProperties"
        | "dependentSchemas"
        | "dependentRequired"
        | "dependencies"
        | "unevaluatedProperties" => Some(OBJECT),
        "pattern" | "minLength" | "maxLength" | "format" => Some(STRING),
        "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" | "multipleOf" => {
            Some(NUMERIC)
        }
        _ => None,
    }
}

/// The chosen branch plus the keywords it acted on.
pub(super) struct Selection<'a> {
    pub branch: Branch<'a>,
    consumed: Vec<&'static str>,
    /// The schema's `"type"`, when it is a single type name. A keyword
    /// outside that type's scope is vacuous, not ignored.
    type_str: Option<&'a str>,
    /// A `TypeUnion` hands every remaining keyword to its per-type copies,
    /// which run this same check; refusing here would refuse twice and
    /// name the wrong schema.
    delegates: bool,
}

impl Selection<'_> {
    /// Refuse every keyword the branch did not act on and that the
    /// schema's declared type does not make vacuous.
    pub(super) fn check_unconsumed(
        &self,
        obj: &[(String, JsonValue)],
        name: &str,
    ) -> Result<(), SchemaError> {
        if self.delegates {
            return Ok(());
        }
        for (key, value) in obj {
            let k = key.as_str();
            if self.consumed.contains(&k) || ANNOTATIONS.contains(&k) {
                continue;
            }
            // `"additionalProperties": true` asserts nothing anywhere.
            if k == "additionalProperties" && value.as_bool() == Some(true) {
                continue;
            }
            if let (Some(types), Some(t)) = (scope(k), self.type_str) {
                if !types.contains(&t) {
                    continue;
                }
            }
            if k == "format" {
                // A format this port has no rule for gets a clearer
                // message than "unsupported keyword".
                let f = value.as_str().unwrap_or_default();
                if !is_uuid_format(f) && builtin(&format!("{f}-string")).is_none() {
                    return Err(SchemaError::UnsupportedFormat {
                        format: f.to_string(),
                        at: at(name),
                    });
                }
            }
            return Err(SchemaError::UnsupportedKeyword {
                keyword: key.clone(),
                at: at(name),
                why: why(k),
            });
        }
        Ok(())
    }
}

/// `^uuid[1-5]?$`, the regex upstream matches `format` against.
fn is_uuid_format(format: &str) -> bool {
    match format.as_bytes() {
        b"uuid" => true,
        [b'u', b'u', b'i', b'd', v] => v.is_ascii_digit() && (b'1'..=b'5').contains(v),
        _ => false,
    }
}

/// Why a keyword is refused. These are the strings a client sees, so they
/// say what would go wrong rather than "not implemented".
fn why(keyword: &str) -> &'static str {
    match keyword {
        "allOf" => {
            "schema intersection is not ported: llama.cpp merges the components' properties and \
             intersects their enums, and a component it does not recognise is dropped"
        }
        "not" | "if" | "then" | "else" => {
            "negated and conditional subschemas have no GBNF form; a grammar cannot express them"
        }
        "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" => {
            "numeric range constraints are not ported"
        }
        "multipleOf" => "divisibility has no GBNF form",
        "uniqueItems" | "contains" | "minContains" | "maxContains" => {
            "an element-set constraint has no GBNF form; a grammar cannot compare two elements"
        }
        "minItems" | "maxItems" => {
            "item counts apply only beside an \"items\" that is a single schema, not a tuple"
        }
        "minLength" | "maxLength" => {
            "string lengths apply only beside an explicit \"type\": \"string\""
        }
        "prefixItems" => "llama.cpp ignores \"prefixItems\" whenever \"items\" is also present",
        "items" | "additionalItems" | "unevaluatedItems" => {
            "array item schemas apply only to \"type\": \"array\""
        }
        "properties" | "required" | "additionalProperties" | "unevaluatedProperties" => {
            "object member schemas apply only to \"type\": \"object\""
        }
        "patternProperties" | "propertyNames" => {
            "constraining which property *names* may appear, rather than what a declared one \
             holds, is not ported"
        }
        "minProperties" | "maxProperties" => "property counts have no GBNF form",
        "dependencies" | "dependentSchemas" | "dependentRequired" => {
            "a constraint conditioned on another member being present has no GBNF form"
        }
        "type" => {
            "\"type\" is not intersected with the branch this schema selected; the branch alone \
             would decide what is accepted"
        }
        "format" => "this format has no grammar in the branch this schema selected",
        "enum" | "const" => "the branch this schema selected does not narrow to a fixed value set",
        _ => "this port does not recognise the keyword, so it cannot honour it",
    }
}

/// Choose the branch, in llama.cpp's `visit` order.
pub(super) fn select<'a>(
    obj: &'a [(String, JsonValue)],
    schema: &'a JsonValue,
    name: &str,
) -> Result<Selection<'a>, SchemaError> {
    let get = |k: &str| obj.iter().find(|(n, _)| n == k).map(|(_, v)| v);
    let schema_type = get("type");
    let type_str = schema_type.and_then(JsonValue::as_str);
    let type_absent = schema_type.is_none();
    let format = match get("format") {
        None => None,
        Some(v) => Some(v.as_str().ok_or_else(|| SchemaError::BadValue {
            keyword: "format".into(),
            at: at(name),
            why: format!("expected a string, found {}", v.kind()),
        })?),
    };

    let sel = |branch, consumed: &[&'static str]| {
        Ok(Selection {
            branch,
            consumed: consumed.to_vec(),
            type_str,
            delegates: false,
        })
    };

    if let Some(v) = get("$ref") {
        let r = v.as_str().ok_or_else(|| SchemaError::BadValue {
            keyword: "$ref".into(),
            at: at(name),
            why: format!("expected a string, found {}", v.kind()),
        })?;
        return sel(Branch::Ref(r), &["$ref"]);
    }

    for key in ["oneOf", "anyOf"] {
        if let Some(v) = get(key) {
            let alts = v.as_array().ok_or_else(|| SchemaError::BadValue {
                keyword: key.into(),
                at: at(name),
                why: format!("expected an array, found {}", v.kind()),
            })?;
            let consumed: &[&'static str] = if key == "oneOf" {
                &["oneOf"]
            } else {
                &["anyOf"]
            };
            return sel(Branch::Union(alts), consumed);
        }
    }

    if let Some(types) = schema_type.and_then(JsonValue::as_array) {
        let copies = types.iter().map(|t| with_type(obj, t)).collect();
        return Ok(Selection {
            branch: Branch::TypeUnion(copies),
            consumed: vec!["type"],
            type_str,
            delegates: true,
        });
    }

    // `const` and `enum` fix the accepted value set outright. A sibling
    // `"type"` can only remove members from that set, never add one, so
    // dropping it cannot widen the grammar -- the one place this port
    // treats an unused keyword as consumed.
    if let Some(v) = get("const") {
        return sel(Branch::Const(v), &["const", "type"]);
    }
    if let Some(v) = get("enum") {
        let values = v.as_array().ok_or_else(|| SchemaError::BadValue {
            keyword: "enum".into(),
            at: at(name),
            why: format!("expected an array, found {}", v.kind()),
        })?;
        return sel(Branch::Enum(values), &["enum", "type"]);
    }

    let object_typed = type_absent || type_str == Some("object");
    let additional = get("additionalProperties");
    let closed_or_typed_tail = additional.is_some_and(|v| v.as_bool() != Some(true));
    if object_typed && (get("properties").is_some() || closed_or_typed_tail) {
        let properties = match get("properties") {
            None => &[][..],
            Some(v) => v.as_object().ok_or_else(|| SchemaError::BadValue {
                keyword: "properties".into(),
                at: at(name),
                why: format!("expected an object, found {}", v.kind()),
            })?,
        };
        let required = required_names(get("required"), properties, name)?;
        return sel(
            Branch::Object {
                properties,
                required,
                additional,
            },
            &["type", "properties", "required", "additionalProperties"],
        );
    }

    let array_typed = type_absent || type_str == Some("array");
    let items_kv = match (get("items"), get("prefixItems")) {
        // Upstream prefers `items` and drops `prefixItems`. Dropping a
        // positional schema widens the grammar, so refuse the pair.
        (Some(_), Some(_)) if array_typed => {
            return Err(SchemaError::UnsupportedKeyword {
                keyword: "prefixItems".into(),
                at: at(name),
                why: why("prefixItems"),
            })
        }
        (Some(v), None) => Some(("items", v)),
        (None, Some(v)) => Some(("prefixItems", v)),
        _ => None,
    };
    if let Some((key, items)) = items_kv.filter(|_| array_typed) {
        let is_items = key == "items";
        if let Some(tuple) = items.as_array() {
            let consumed: &[&'static str] = if is_items {
                &["type", "items"]
            } else {
                &["type", "prefixItems"]
            };
            return sel(Branch::Tuple(tuple), consumed);
        }
        let min = count(get("minItems"), "minItems", name)?.unwrap_or(0);
        let max = count(get("maxItems"), "maxItems", name)?;
        let consumed: &[&'static str] = if is_items {
            &["type", "items", "minItems", "maxItems"]
        } else {
            &["type", "prefixItems", "minItems", "maxItems"]
        };
        return sel(Branch::List { items, min, max }, consumed);
    }

    let string_typed = type_absent || type_str == Some("string");
    if string_typed {
        if let Some(v) = get("pattern") {
            let p = v.as_str().ok_or_else(|| SchemaError::BadValue {
                keyword: "pattern".into(),
                at: at(name),
                why: format!("expected a string, found {}", v.kind()),
            })?;
            return sel(Branch::Pattern(p), &["type", "pattern"]);
        }
    }
    if string_typed {
        if let Some(f) = format {
            if is_uuid_format(f) {
                return sel(Branch::Uuid { format: f }, &["type", "format"]);
            }
            if builtin(&format!("{f}-string")).is_some() {
                return sel(Branch::StringFormat { format: f }, &["type", "format"]);
            }
        }
    }

    if type_str == Some("string") && (get("minLength").is_some() || get("maxLength").is_some()) {
        let min = count(get("minLength"), "minLength", name)?.unwrap_or(0);
        let max = count(get("maxLength"), "maxLength", name)?;
        return sel(
            Branch::BoundedString { min, max },
            &["type", "minLength", "maxLength"],
        );
    }

    if obj.is_empty() || type_str == Some("object") {
        return sel(Branch::ObjectPrimitive, &["type", "additionalProperties"]);
    }

    if type_absent {
        // No type and no structural keyword: equivalent to `{}`.
        return sel(Branch::ValuePrimitive, &[]);
    }

    match type_str {
        Some(t) if primitive(t).is_some() => sel(Branch::Primitive(t), &["type"]),
        _ => Err(SchemaError::UnknownType {
            at: at(name),
            found: schema.get("type").map(JsonValue::dump).unwrap_or_default(),
        }),
    }
}

/// A copy of the schema with `"type"` replaced, keeping key order.
fn with_type(obj: &[(String, JsonValue)], ty: &JsonValue) -> JsonValue {
    let mut entries = obj.to_vec();
    for slot in entries.iter_mut() {
        if slot.0 == "type" {
            slot.1 = ty.clone();
        }
    }
    JsonValue::Object(entries)
}

/// `required`, validated. llama.cpp skips non-string entries and never
/// checks that a required name is declared; a name it cannot find is
/// simply never emitted, so the grammar stops requiring it.
fn required_names(
    value: Option<&JsonValue>,
    properties: &[(String, JsonValue)],
    name: &str,
) -> Result<Vec<String>, SchemaError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let items = value.as_array().ok_or_else(|| SchemaError::BadValue {
        keyword: "required".into(),
        at: at(name),
        why: format!("expected an array, found {}", value.kind()),
    })?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let s = item.as_str().ok_or_else(|| SchemaError::BadValue {
            keyword: "required".into(),
            at: at(name),
            why: format!("expected an array of strings, found a {}", item.kind()),
        })?;
        if !properties.iter().any(|(k, _)| k == s) {
            return Err(SchemaError::BadValue {
                keyword: "required".into(),
                at: at(name),
                why: format!(
                    "{s:?} is required but is not declared in \"properties\", so no grammar can \
                     insist on it"
                ),
            });
        }
        out.push(s.to_string());
    }
    Ok(out)
}

/// A non-negative integer keyword (`minItems`, `maxLength`, …).
fn count(value: Option<&JsonValue>, keyword: &str, name: &str) -> Result<Option<u64>, SchemaError> {
    match value {
        None => Ok(None),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| SchemaError::BadValue {
            keyword: keyword.into(),
            at: at(name),
            why: format!("expected a non-negative integer, found {}", v.dump()),
        }),
    }
}
