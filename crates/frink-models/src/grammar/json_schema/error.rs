//! Every way this converter refuses a schema, naming what is missing.
//!
//! llama.cpp collects strings into `_errors` and throws one
//! `std::invalid_argument` at the end. Worse, several keywords it does not
//! implement are not errors at all: they fall through the `visit` chain
//! and vanish. A `pattern` it cannot compile, a `minLength` on a schema
//! with no explicit `"type": "string"`, `minItems` beside a tuple
//! `items` -- each of those produces a grammar that *permits documents the
//! schema forbids*, which is a wrong answer wearing the costume of a
//! working one.
//!
//! Per `CLAUDE.md` ("return `Result` and name what is missing"), this port
//! refuses instead. [`SchemaError::UnsupportedKeyword`] is raised for any
//! keyword the branch that was chosen does not consume, so silence is not
//! a reachable outcome.

use crate::grammar::GrammarError;
use std::fmt;

/// What the JSON-schema converter refused, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    /// The schema text was not valid JSON.
    NotJson(String),
    /// A schema, or a subschema, was not a JSON object. JSON Schema's
    /// boolean form (`true` / `false` as a whole schema) is not ported.
    NotAnObject { at: String, kind: &'static str },
    /// A keyword is present that the chosen branch does not act on.
    /// Ignoring it would widen the grammar past what the schema allows.
    UnsupportedKeyword {
        keyword: String,
        at: String,
        why: &'static str,
    },
    /// A `pattern` this port cannot compile to GBNF. llama.cpp's
    /// `_visit_pattern` handles a subset of ECMA-262; the cases named here
    /// are the ones it mishandles rather than the ones it skips.
    UnsupportedPattern { pattern: String, why: String },
    /// `"format"` names something outside the six llama.cpp special-cases
    /// (`date`, `time`, `date-time`, `uuid`, `uuid1`..`uuid5`).
    UnsupportedFormat { format: String, at: String },
    /// `"type"` is not a string, or names no JSON type.
    UnknownType { at: String, found: String },
    /// A `$ref` this port will not follow: a remote URL, or anything that
    /// is not a `#/`-rooted JSON pointer into the same document.
    UnsupportedRef { reference: String },
    /// A `#/`-rooted `$ref` whose pointer does not land on anything.
    RefNotFound { reference: String, token: String },
    /// Two schemas compiled into ONE grammar ([`super::GrammarBuilder`])
    /// spell the same `$ref` pointer and mean different subschemas. The
    /// rule name is derived from the pointer, so honouring the first
    /// would compile the second against the wrong definition.
    RefCollision { pointer: String },
    /// A keyword's value has the wrong JSON type for what it means.
    BadValue {
        keyword: String,
        at: String,
        why: String,
    },
    /// The top-level schema did not end up owning the rule named `root`,
    /// so the grammar's entry point would be some subschema's rule. This
    /// is reachable upstream (a property named `""` takes `root` first)
    /// and silently produces a grammar for the wrong document.
    RootDisplaced { got: String },
    /// The converter emitted a grammar the engine in this repo cannot
    /// parse. That is a bug in this module, never in the caller's schema.
    Emitted(GrammarError),
    /// An invariant of this module was violated -- a builtin table lookup
    /// that cannot fail did. A bug here, not in the caller's schema.
    Internal(&'static str),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaError::NotJson(msg) => write!(f, "JSON schema is not valid JSON: {msg}"),
            SchemaError::NotAnObject { at, kind } => write!(
                f,
                "JSON schema at {at} is a {kind}, not an object; boolean schemas are not supported"
            ),
            SchemaError::UnsupportedKeyword { keyword, at, why } => write!(
                f,
                "JSON schema keyword {keyword:?} at {at} is not supported ({why}); it would be \
                 ignored, and the grammar would then accept documents the schema rejects"
            ),
            SchemaError::UnsupportedPattern { pattern, why } => write!(
                f,
                "JSON schema \"pattern\" {pattern:?} cannot be compiled to a grammar: {why}"
            ),
            SchemaError::UnsupportedFormat { format, at } => write!(
                f,
                "JSON schema format {format:?} at {at} is not supported; only \"date\", \"time\", \
                 \"date-time\" and \"uuid\" (or \"uuid1\"..\"uuid5\") have grammars"
            ),
            SchemaError::UnknownType { at, found } => {
                write!(
                    f,
                    "JSON schema \"type\" at {at} is {found}, which names no JSON type"
                )
            }
            SchemaError::UnsupportedRef { reference } => write!(
                f,
                "JSON schema $ref {reference:?} is not supported; only same-document refs of the \
                 form \"#/...\" are resolved, and nothing is fetched over the network"
            ),
            SchemaError::RefNotFound { reference, token } => write!(
                f,
                "JSON schema $ref {reference:?} does not resolve: {token:?} is not in the document"
            ),
            SchemaError::RefCollision { pointer } => write!(
                f,
                "two schemas compiled into one grammar both define {pointer:?} and disagree about \
                 what it is; rename one of them"
            ),
            SchemaError::BadValue { keyword, at, why } => {
                write!(
                    f,
                    "JSON schema keyword {keyword:?} at {at} is invalid: {why}"
                )
            }
            SchemaError::RootDisplaced { got } => write!(
                f,
                "the top-level schema compiled to rule {got:?} rather than \"root\", so the \
                 grammar would start from a subschema; rename the colliding property"
            ),
            SchemaError::Internal(what) => {
                write!(f, "internal error in the JSON schema converter: {what}")
            }
            SchemaError::Emitted(e) => write!(
                f,
                "the grammar generated from this schema does not parse, which is a bug in the \
                 converter: {e}"
            ),
        }
    }
}

impl std::error::Error for SchemaError {}

impl From<GrammarError> for SchemaError {
    fn from(e: GrammarError) -> Self {
        SchemaError::Emitted(e)
    }
}

/// The name of a subschema as it appears in an error, with the empty
/// top-level name spelled out.
pub(super) fn at(name: &str) -> String {
    if name.is_empty() {
        "the top level".to_string()
    } else {
        format!("{name:?}")
    }
}
