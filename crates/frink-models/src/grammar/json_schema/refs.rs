//! `$ref` collection, llama.cpp's `common_schema_converter::resolve_refs`.
//!
//! Upstream walks the whole document once, and for each `$ref` it has not
//! seen, resolves the JSON pointer against the root and files the target
//! under the ref string. Because the walk covers `$defs` and `definitions`
//! like any other member, a ref chain inside them is found by the same
//! pass. Recursion is not a problem for *this* half: nothing is expanded
//! here, only recorded. It is `Converter::resolve_ref` that has to break
//! the cycle, and it does so with a "currently resolving" set.
//!
//! Two upstream behaviours are deliberately not carried over:
//!
//! - remote refs (`https://…`) are fetched. This port refuses them; the
//!   build llama.cpp ships hands `_fetch_json` a stub that returns null,
//!   so the fetch path resolves against nothing anyway.
//! - a pointer token is used verbatim. This port also decodes the JSON
//!   Pointer escapes `~1` and `~0`, which upstream cannot address at all.

use super::error::SchemaError;
use super::value::JsonValue;
use std::collections::HashMap;

/// Record every `$ref` in `root` against the subschema it points at.
pub(super) fn collect(root: &JsonValue) -> Result<HashMap<String, JsonValue>, SchemaError> {
    let mut refs = HashMap::new();
    walk(root, root, &mut refs)?;
    Ok(refs)
}

fn walk(
    root: &JsonValue,
    node: &JsonValue,
    refs: &mut HashMap<String, JsonValue>,
) -> Result<(), SchemaError> {
    match node {
        JsonValue::Array(items) => {
            for item in items {
                walk(root, item, refs)?;
            }
        }
        JsonValue::Object(entries) => {
            // Upstream stops descending at a node that has a `$ref`: its
            // siblings are, per JSON Schema before 2019-09, ignored.
            if let Some(reference) = node.get("$ref") {
                let reference = reference
                    .as_str()
                    .ok_or_else(|| SchemaError::UnsupportedRef {
                        reference: reference.dump(),
                    })?;
                if !refs.contains_key(reference) {
                    let target = resolve_pointer(root, reference)?;
                    refs.insert(reference.to_string(), target);
                }
            } else {
                for (_, value) in entries {
                    walk(root, value, refs)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Follow a `#/`-rooted JSON pointer into `root`.
fn resolve_pointer(root: &JsonValue, reference: &str) -> Result<JsonValue, SchemaError> {
    if !reference.starts_with("#/") {
        return Err(SchemaError::UnsupportedRef {
            reference: reference.to_string(),
        });
    }
    let mut target = root;
    for raw in reference[1..].split('/').skip(1) {
        let token = unescape_pointer_token(raw);
        let next = match target {
            JsonValue::Object(_) => target.get(&token),
            JsonValue::Array(items) => token.parse::<usize>().ok().and_then(|i| items.get(i)),
            _ => None,
        };
        target = next.ok_or_else(|| SchemaError::RefNotFound {
            reference: reference.to_string(),
            token,
        })?;
    }
    Ok(target.clone())
}

/// RFC 6901: `~1` is `/`, `~0` is `~`, in that order.
fn unescape_pointer_token(token: &str) -> String {
    if token.contains('~') {
        token.replace("~1", "/").replace("~0", "~")
    } else {
        token.to_string()
    }
}

/// `_resolve_ref`'s rule name: `ref` followed by the fragment with every
/// run of non-`[a-zA-Z0-9-]` collapsed to a single `-`.
pub(super) fn ref_rule_name(reference: &str) -> String {
    let fragment = match reference.find('#') {
        Some(i) => &reference[i + 1..],
        None => reference,
    };
    let mut out = String::from("ref");
    out.push_str(&super::converter::collapse_invalid(fragment));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(text: &str) -> JsonValue {
        JsonValue::parse(text).expect("test schema parses")
    }

    #[test]
    fn ref_names_match_upstream() {
        assert_eq!(ref_rule_name("#/definitions/foo"), "ref-definitions-foo");
        assert_eq!(ref_rule_name("#/$defs/Node"), "ref-defs-Node");
        assert_eq!(
            ref_rule_name("#/properties/a/anyOf/0"),
            "ref-properties-a-anyOf-0"
        );
    }

    #[test]
    fn collects_nested_and_recursive_refs() {
        let s = schema(
            r##"{"$ref":"#/$defs/a","$defs":{"a":{"properties":{"n":{"$ref":"#/$defs/a"}}}}}"##,
        );
        let refs = collect(&s).expect("refs resolve");
        assert_eq!(refs.len(), 1);
        assert!(refs["#/$defs/a"].contains_key("properties"));
    }

    #[test]
    fn refuses_remote_and_dangling_refs() {
        let remote = schema(r##"{"$ref":"https://example.com/s.json#/x"}"##);
        assert!(matches!(
            collect(&remote),
            Err(SchemaError::UnsupportedRef { .. })
        ));
        let dangling = schema(r##"{"$ref":"#/$defs/missing","$defs":{}}"##);
        assert!(matches!(
            collect(&dangling),
            Err(SchemaError::RefNotFound { .. })
        ));
    }

    #[test]
    fn pointer_escapes_are_decoded() {
        let s = schema(r##"{"$ref":"#/$defs/a~1b","$defs":{"a/b":{"type":"string"}}}"##);
        let refs = collect(&s).expect("escaped pointer resolves");
        assert_eq!(
            refs["#/$defs/a~1b"].get("type").unwrap().as_str(),
            Some("string")
        );
    }
}
