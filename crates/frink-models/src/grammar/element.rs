//! The compiled form of a GBNF grammar: elements, rules, and a cursor
//! into them.
//!
//! Ported from `llama_gretype` / `llama_grammar_element` in llama.cpp's
//! `src/llama-grammar.h`. The one representational change is the cursor:
//! llama.cpp walks a grammar with `const llama_grammar_element *` raw
//! pointers into the rule vectors, which is why `llama_grammar_clone_impl`
//! has to rewrite every stack entry after a copy. [`RulePos`] is the same
//! cursor expressed as `(rule, index)`, so a [`Grammar`](super::Grammar)
//! clones for free.

/// Element kinds, matching `enum llama_gretype` one for one.
///
/// The discriminants are llama.cpp's, so a serialized grammar could be
/// compared against upstream without a translation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum GreType {
    /// End of a rule definition.
    End = 0,
    /// Start of an alternate definition for a rule.
    Alt = 1,
    /// Non-terminal: a reference to another rule, by rule id.
    RuleRef = 2,
    /// Terminal: a single Unicode code point.
    Char = 3,
    /// Inverted character set: `[^a]`, `[^a-b]`, `[^abc]`.
    CharNot = 4,
    /// Modifies a preceding `Char` or `CharAlt` into an inclusive range
    /// (`[a-z]`).
    CharRngUpper = 5,
    /// Modifies a preceding `Char` or `CharRngUpper` by adding another
    /// alternative to match (`[ab]`, `[a-zA]`).
    CharAlt = 6,
    /// Any character (`.`).
    CharAny = 7,
    /// Terminal: a token id (`<[42]>`).
    Token = 8,
    /// Inverted token (`!<[42]>`).
    TokenNot = 9,
}

impl GreType {
    /// True for the element kinds that consume a character.
    ///
    /// `llama_grammar_is_char_element`, used only by the printer and by
    /// rule validation.
    pub fn is_char_element(self) -> bool {
        matches!(
            self,
            GreType::Char
                | GreType::CharNot
                | GreType::CharAlt
                | GreType::CharRngUpper
                | GreType::CharAny
        )
    }

    /// True for the element kinds a parse stack may legally rest on.
    ///
    /// `llama_grammar_advance_stack` aborts if a stack top is anything
    /// else; we return an error instead (see [`super::GrammarError`]).
    pub fn is_stack_terminal(self) -> bool {
        matches!(
            self,
            GreType::Char
                | GreType::CharNot
                | GreType::CharAny
                | GreType::Token
                | GreType::TokenNot
        )
    }
}

/// One grammar element: a kind plus a code point, rule id, or token id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GrammarElement {
    pub gtype: GreType,
    pub value: u32,
}

impl GrammarElement {
    pub const fn new(gtype: GreType, value: u32) -> Self {
        Self { gtype, value }
    }

    /// `llama_grammar_is_end_of_sequence`: true iff this position ends one
    /// of the alternate definitions of a rule.
    pub fn is_end_of_sequence(self) -> bool {
        matches!(self.gtype, GreType::End | GreType::Alt)
    }
}

/// One rule: a flat element sequence, alternates separated by [`GreType::Alt`],
/// always terminated by [`GreType::End`].
pub type GrammarRule = Vec<GrammarElement>;

/// A cursor into the compiled rule table: `rules[rule][index]`.
///
/// Ordering is `(rule, index)` lexicographic, which stands in for
/// llama.cpp's pointer comparison in `llama_grammar_advance_stack`'s
/// `seen` set. Any total order does the job there; only distinctness
/// matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RulePos {
    pub rule: u32,
    pub index: u32,
}

impl RulePos {
    pub const fn new(rule: u32, index: u32) -> Self {
        Self { rule, index }
    }

    /// `pos + 1`: the next element in the same rule.
    pub const fn next(self) -> Self {
        Self {
            rule: self.rule,
            index: self.index + 1,
        }
    }

    /// `pos + n`.
    pub const fn advance(self, n: u32) -> Self {
        Self {
            rule: self.rule,
            index: self.index + n,
        }
    }
}

/// A pushdown stack: cursors, innermost last. `stack.last()` is the
/// position the next character has to satisfy.
pub type GrammarStack = Vec<RulePos>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_and_alt_terminate_a_sequence() {
        assert!(GrammarElement::new(GreType::End, 0).is_end_of_sequence());
        assert!(GrammarElement::new(GreType::Alt, 0).is_end_of_sequence());
        assert!(!GrammarElement::new(GreType::Char, b'a' as u32).is_end_of_sequence());
        assert!(!GrammarElement::new(GreType::RuleRef, 3).is_end_of_sequence());
    }

    #[test]
    fn char_range_modifiers_are_char_elements_but_not_stack_terminals() {
        // llama.cpp's `advance_stack` aborts on a stack resting on
        // CHAR_ALT or CHAR_RNG_UPPER; both are still "char elements" to
        // the printer. The two predicates genuinely differ.
        assert!(GreType::CharAlt.is_char_element());
        assert!(GreType::CharRngUpper.is_char_element());
        assert!(!GreType::CharAlt.is_stack_terminal());
        assert!(!GreType::CharRngUpper.is_stack_terminal());
        assert!(GreType::CharAny.is_char_element());
        assert!(GreType::CharAny.is_stack_terminal());
        assert!(!GreType::Token.is_char_element());
        assert!(GreType::Token.is_stack_terminal());
    }

    #[test]
    fn discriminants_match_llama_cpp() {
        assert_eq!(GreType::End as u8, 0);
        assert_eq!(GreType::Alt as u8, 1);
        assert_eq!(GreType::RuleRef as u8, 2);
        assert_eq!(GreType::Char as u8, 3);
        assert_eq!(GreType::CharNot as u8, 4);
        assert_eq!(GreType::CharRngUpper as u8, 5);
        assert_eq!(GreType::CharAlt as u8, 6);
        assert_eq!(GreType::CharAny as u8, 7);
        assert_eq!(GreType::Token as u8, 8);
        assert_eq!(GreType::TokenNot as u8, 9);
    }

    #[test]
    fn cursor_orders_by_rule_then_index() {
        assert!(RulePos::new(0, 5) < RulePos::new(1, 0));
        assert!(RulePos::new(1, 0) < RulePos::new(1, 1));
        assert_eq!(RulePos::new(2, 3).next(), RulePos::new(2, 4));
        assert_eq!(RulePos::new(2, 3).advance(2), RulePos::new(2, 5));
    }
}
