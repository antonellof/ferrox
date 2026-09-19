//! `common_schema_converter`: the rule table, and `visit`'s emission half.
//!
//! [`super::branch`] decides *which* branch a schema takes; this module
//! emits the GBNF for it. Splitting the two is what lets the branch
//! conditions be read twice -- once to dispatch, once to refuse the
//! keywords no branch used -- without a second copy of the conditions.

use super::branch::{select, Branch};
use super::error::{at, SchemaError};
use super::pattern::PatternCompiler;
use super::primitives::{
    build_repetition, builtin, escape_in_range, format_literal, is_reserved_name, primitive,
    BuiltinRule, SPACE_RULE,
};
use super::refs::ref_rule_name;
use super::value::JsonValue;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

/// `INVALID_RULE_CHARS_RE`: every run of characters a GBNF rule name
/// cannot contain becomes one `-`.
pub(super) fn collapse_invalid(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut in_run = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' {
            out.push(c);
            in_run = false;
        } else if !in_run {
            out.push('-');
            in_run = true;
        }
    }
    out
}

/// The rule table under construction.
pub(super) struct Converter {
    /// A `BTreeMap` because upstream's is a `std::map`, and
    /// `format_grammar` emits it in iteration order: the grammar text is
    /// sorted by rule name.
    rules: BTreeMap<String, String>,
    refs: HashMap<String, JsonValue>,
    resolving: HashSet<String>,
}

impl Converter {
    pub(super) fn new(refs: HashMap<String, JsonValue>) -> Self {
        let mut rules = BTreeMap::new();
        rules.insert("space".to_string(), SPACE_RULE.to_string());
        Converter {
            rules,
            refs,
            resolving: HashSet::new(),
        }
    }

    /// Fold another document's `$ref` table in, for a builder that
    /// compiles several schemas into ONE grammar
    /// ([`super::GrammarBuilder`]).
    ///
    /// A pointer that two documents spell the same way and mean
    /// differently is refused: the rule name is derived from the pointer,
    /// so keeping the first silently compiles the second tool's arguments
    /// against the first tool's definition.
    pub(super) fn add_refs(&mut self, refs: HashMap<String, JsonValue>) -> Result<(), SchemaError> {
        for (pointer, target) in refs {
            match self.refs.get(&pointer) {
                Some(existing) if *existing != target => {
                    return Err(SchemaError::RefCollision { pointer });
                }
                Some(_) => {}
                None => {
                    self.refs.insert(pointer, target);
                }
            }
        }
        Ok(())
    }

    /// Whether a rule of this exact name has been emitted.
    pub(super) fn has_rule(&self, name: &str) -> bool {
        self.rules.contains_key(name)
    }

    /// `_add_rule`. A name already bound to a *different* body gets the
    /// lowest numeric suffix that is free or already holds this same body.
    pub(super) fn add_rule(&mut self, name: &str, rule: &str) -> String {
        let esc = collapse_invalid(name);
        match self.rules.get(&esc) {
            None => {
                self.rules.insert(esc.clone(), rule.to_string());
                return esc;
            }
            Some(existing) if existing == rule => return esc,
            Some(_) => {}
        }
        let mut i = 0u32;
        let key = loop {
            let candidate = format!("{esc}{i}");
            match self.rules.get(&candidate) {
                Some(existing) if existing != rule => i += 1,
                _ => break candidate,
            }
        };
        self.rules.insert(key.clone(), rule.to_string());
        key
    }

    /// `_add_primitive`: add the rule, then every builtin its body names.
    fn add_primitive(&mut self, name: &str, rule: &BuiltinRule) -> String {
        let n = self.add_rule(name, rule.content);
        for dep in rule.deps {
            if !self.rules.contains_key(*dep) {
                // Every name in a `deps` list is a key of one of the two
                // builtin tables; `primitives::builtin` is total over them.
                if let Some(dep_rule) = builtin(dep) {
                    self.add_primitive(dep, dep_rule);
                }
            }
        }
        n
    }

    /// `format_grammar`.
    pub(super) fn finish(self) -> String {
        let mut out = String::new();
        for (name, rule) in &self.rules {
            let _ = writeln!(out, "{name} ::= {rule}");
        }
        out
    }

    /// `visit`.
    pub(super) fn visit(&mut self, schema: &JsonValue, name: &str) -> Result<String, SchemaError> {
        let obj = schema.as_object().ok_or_else(|| SchemaError::NotAnObject {
            at: at(name),
            kind: schema.kind(),
        })?;
        let selection = select(obj, schema, name)?;
        selection.check_unconsumed(obj, name)?;

        let rule_name = if is_reserved_name(name) {
            format!("{name}-")
        } else if name.is_empty() {
            "root".to_string()
        } else {
            name.to_string()
        };

        match selection.branch {
            Branch::Ref(reference) => {
                let target = self.resolve_ref(reference)?;
                Ok(self.add_rule(&rule_name, &target))
            }
            Branch::Union(alts) => {
                let body = self.union_rule(name, alts)?;
                Ok(self.add_rule(&rule_name, &body))
            }
            Branch::TypeUnion(copies) => {
                let body = self.union_rule(name, &copies)?;
                Ok(self.add_rule(&rule_name, &body))
            }
            Branch::Const(value) => {
                let body = format_literal(&value.dump());
                Ok(self.add_rule(&rule_name, &body))
            }
            Branch::Enum(values) => {
                let alts: Vec<String> = values.iter().map(|v| format_literal(&v.dump())).collect();
                let body = format!("({})", alts.join(" | "));
                Ok(self.add_rule(&rule_name, &body))
            }
            Branch::Object {
                properties,
                required,
                additional,
            } => {
                let body = self.object_rule(properties, &required, name, additional)?;
                Ok(self.add_rule(&rule_name, &body))
            }
            Branch::Tuple(items) => {
                let mut body = String::from("\"[\" space ");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        body.push_str(" \",\" space ");
                    }
                    let sub = self.visit(item, &child_name(name, &format!("tuple-{i}")))?;
                    body.push_str(&sub);
                }
                body.push_str(" space \"]\"");
                Ok(self.add_rule(&rule_name, &body))
            }
            Branch::List { items, min, max } => {
                let item_rule = self.visit(items, &child_name(name, "item"))?;
                let body = format!(
                    "\"[\" space {} space \"]\"",
                    build_repetition(&item_rule, min, max, "\",\" space")
                );
                Ok(self.add_rule(&rule_name, &body))
            }
            Branch::Pattern(pattern) => PatternCompiler::compile(self, pattern, &rule_name),
            Branch::Uuid { format } => {
                let target = if rule_name == "root" { "root" } else { format };
                // `uuid` is a primitive with no dependencies, so the rule
                // name is free to be `uuid3` while the body stays the same.
                let uuid = primitive("uuid").ok_or(SchemaError::Internal(
                    "a builtin rule vanished from its table",
                ))?;
                Ok(self.add_primitive(target, uuid))
            }
            Branch::StringFormat { format } => {
                let prim_name = format!("{format}-string");
                let rule = builtin(&prim_name).ok_or(SchemaError::Internal(
                    "a builtin rule vanished from its table",
                ))?;
                let target = self.add_primitive(&prim_name, rule);
                Ok(self.add_rule(&rule_name, &target))
            }
            Branch::BoundedString { min, max } => {
                let char_rule = self.add_primitive_named("char");
                let body = format!(
                    "\"\\\"\" {} \"\\\"\"",
                    build_repetition(&char_rule, min, max, "")
                );
                Ok(self.add_rule(&rule_name, &body))
            }
            Branch::ObjectPrimitive => {
                let target = self.add_primitive_named("object");
                Ok(self.add_rule(&rule_name, &target))
            }
            Branch::ValuePrimitive => {
                let target = self.add_primitive_named("value");
                Ok(self.add_rule(&rule_name, &target))
            }
            Branch::Primitive(type_name) => {
                let target = if rule_name == "root" {
                    "root"
                } else {
                    type_name
                };
                let rule = primitive(type_name).ok_or(SchemaError::Internal(
                    "a builtin rule vanished from its table",
                ))?;
                Ok(self.add_primitive(target, rule))
            }
        }
    }

    /// `_add_primitive` for a builtin referred to by its own name.
    fn add_primitive_named(&mut self, name: &str) -> String {
        match primitive(name) {
            Some(rule) => self.add_primitive(name, rule),
            // Unreachable: every caller passes a `PRIMITIVE_RULES` key.
            None => name.to_string(),
        }
    }

    /// `_generate_union_rule`.
    fn union_rule(&mut self, name: &str, alts: &[JsonValue]) -> Result<String, SchemaError> {
        let mut rules = Vec::with_capacity(alts.len());
        for (i, alt) in alts.iter().enumerate() {
            let sub_name = if name.is_empty() {
                format!("alternative-{i}")
            } else {
                format!("{name}-{i}")
            };
            rules.push(self.visit(alt, &sub_name)?);
        }
        Ok(rules.join(" | "))
    }

    /// `_resolve_ref`. The "currently resolving" set is what stops a
    /// recursive schema from expanding forever: the inner sighting of the
    /// ref returns the name the outer one is about to define.
    fn resolve_ref(&mut self, reference: &str) -> Result<String, SchemaError> {
        let mut ref_name = ref_rule_name(reference);
        if !self.rules.contains_key(&ref_name) && !self.resolving.contains(reference) {
            self.resolving.insert(reference.to_string());
            let resolved =
                self.refs
                    .get(reference)
                    .cloned()
                    .ok_or_else(|| SchemaError::UnsupportedRef {
                        reference: reference.to_string(),
                    })?;
            let visited = self.visit(&resolved, &ref_name);
            self.resolving.remove(reference);
            ref_name = visited?;
        }
        Ok(ref_name)
    }

    /// `_build_object_rule`.
    fn object_rule(
        &mut self,
        properties: &[(String, JsonValue)],
        required: &[String],
        name: &str,
        additional: Option<&JsonValue>,
    ) -> Result<String, SchemaError> {
        let mut required_props: Vec<String> = Vec::new();
        let mut optional_props: Vec<String> = Vec::new();
        let mut kv_rule_names: HashMap<String, String> = HashMap::new();
        let mut prop_names: Vec<String> = Vec::new();

        for (prop_name, prop_schema) in properties {
            let prop_rule = self.visit(prop_schema, &child_name(name, prop_name))?;
            let kv = self.add_rule(
                &child_name(name, &format!("{prop_name}-kv")),
                &format!(
                    "{} space \":\" space {prop_rule}",
                    format_literal(&JsonValue::String(prop_name.clone()).dump())
                ),
            );
            kv_rule_names.insert(prop_name.clone(), kv);
            if required.iter().any(|r| r == prop_name) {
                required_props.push(prop_name.clone());
            } else {
                optional_props.push(prop_name.clone());
            }
            prop_names.push(prop_name.clone());
        }

        let open_tail =
            matches!(additional, Some(v) if v.as_bool() == Some(true) || v.as_object().is_some());
        if open_tail {
            let sub_name = child_name(name, "additional");
            let value_rule = match additional {
                Some(v) if v.as_object().is_some() => {
                    self.visit(v, &format!("{sub_name}-value"))?
                }
                _ => self.add_primitive_named("value"),
            };
            let key_rule = if prop_names.is_empty() {
                self.add_primitive_named("string")
            } else {
                let body = self.not_strings(&prop_names);
                self.add_rule(&format!("{sub_name}-k"), &body)
            };
            let kv_rule = self.add_rule(
                &format!("{sub_name}-kv"),
                &format!("{key_rule} \":\" space {value_rule}"),
            );
            kv_rule_names.insert("*".to_string(), kv_rule);
            optional_props.push("*".to_string());
        }

        let mut rule = String::from("\"{\" space ");
        for (i, key) in required_props.iter().enumerate() {
            if i > 0 {
                rule.push_str(" \",\" space ");
            }
            rule.push_str(kv_rule_names.get(key).map(String::as_str).unwrap_or(""));
        }

        if !optional_props.is_empty() {
            rule.push_str(" (");
            if !required_props.is_empty() {
                rule.push_str(" \",\" space ( ");
            }
            for i in 0..optional_props.len() {
                if i > 0 {
                    rule.push_str(" | ");
                }
                let tail = self.recursive_refs(name, &optional_props[i..], &kv_rule_names, false);
                rule.push_str(&tail);
            }
            if !required_props.is_empty() {
                rule.push_str(" )");
            }
            rule.push_str(" )?");
        }

        rule.push_str(" space \"}\"");
        Ok(rule)
    }

    /// `get_recursive_refs`: the chain of `<k>-rest` rules that lets the
    /// optional properties appear in declaration order, each one optional.
    fn recursive_refs(
        &mut self,
        name: &str,
        ks: &[String],
        kv_rule_names: &HashMap<String, String>,
        first_is_optional: bool,
    ) -> String {
        let Some(k) = ks.first() else {
            return String::new();
        };
        let kv_rule_name = kv_rule_names.get(k).map(String::as_str).unwrap_or("");
        let comma_ref = format!("( \",\" space {kv_rule_name} )");
        let mut res = if first_is_optional {
            format!("{comma_ref}{}", if k == "*" { "*" } else { "?" })
        } else if k == "*" {
            format!("{kv_rule_name} {comma_ref}*")
        } else {
            kv_rule_name.to_string()
        };
        if ks.len() > 1 {
            let rest = self.recursive_refs(name, &ks[1..], kv_rule_names, true);
            let rest_name = self.add_rule(&child_name(name, &format!("{k}-rest")), &rest);
            res.push(' ');
            res.push_str(&rest_name);
        }
        res
    }

    /// `_not_strings`: a JSON string that is none of `strings`, used as
    /// the key rule for the additional-properties tail so a declared
    /// property cannot be smuggled in through it.
    ///
    /// The trie is keyed on `char`, not on `u8` as upstream's is, so a
    /// non-ASCII property name yields codepoint classes rather than lone
    /// UTF-8 bytes.
    fn not_strings(&mut self, strings: &[String]) -> String {
        #[derive(Default)]
        struct TrieNode {
            children: BTreeMap<char, TrieNode>,
            is_end: bool,
        }

        fn insert(node: &mut TrieNode, s: &str) {
            let mut cur = node;
            for c in s.chars() {
                cur = cur.children.entry(c).or_default();
            }
            cur.is_end = true;
        }

        fn visit(node: &TrieNode, char_rule: &str, out: &mut String) {
            let mut rejects = String::new();
            let mut first = true;
            for (c, child) in &node.children {
                escape_in_range(*c, &mut rejects);
                if first {
                    first = false;
                } else {
                    out.push_str(" | ");
                }
                out.push('[');
                escape_in_range(*c, out);
                out.push(']');
                if !child.children.is_empty() {
                    out.push_str(" (");
                    visit(child, char_rule, out);
                    out.push(')');
                } else if child.is_end {
                    let _ = write!(out, " {char_rule}+");
                }
            }
            if !node.children.is_empty() {
                if !first {
                    out.push_str(" | ");
                }
                let _ = write!(out, "[^\"{rejects}] {char_rule}*");
            }
        }

        let mut trie = TrieNode::default();
        for s in strings {
            insert(&mut trie, s);
        }
        let char_rule = self.add_primitive_named("char");

        let mut out = String::from("[\"] ( ");
        visit(&trie, &char_rule, &mut out);
        out.push_str(" )");
        if !trie.is_end {
            out.push('?');
        }
        out.push_str(" [\"]");
        out
    }
}

/// `name + (name.empty() ? "" : "-") + suffix`.
fn child_name(name: &str, suffix: &str) -> String {
    if name.is_empty() {
        suffix.to_string()
    } else {
        format!("{name}-{suffix}")
    }
}
