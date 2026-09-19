//! The grammars llama.cpp ships in its `grammars/` directory, verbatim.
//!
//! These are the cases upstream considered worth keeping, and between them
//! they exercise negated classes, multibyte ranges, nested alternation,
//! all four repetition operators and the `{m,n}` form. They are embedded
//! rather than read from disk so the tests need no checkout of llama.cpp.
//!
//! llama.cpp is MIT licensed; see `docs/THIRD_PARTY_NOTICES.md`.

/// Every grammar in the corpus: file name, source, and the minimum number
/// of rules it must compile to.
pub(super) const SHIPPED_GRAMMARS: &[(&str, &str, usize)] = &[
    ("arithmetic.gbnf", ARITHMETIC_GBNF, 13),
    ("c.gbnf", C_GBNF, 20),
    ("chess.gbnf", CHESS_GBNF, 5),
    ("english.gbnf", ENGLISH_GBNF, 5),
    ("japanese.gbnf", JAPANESE_GBNF, 5),
    ("json.gbnf", JSON_GBNF, 7),
    ("json_arr.gbnf", JSON_ARR_GBNF, 7),
    ("list.gbnf", LIST_GBNF, 2),
];

pub(super) const ARITHMETIC_GBNF: &str = r#"root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
"#;

pub(super) const C_GBNF: &str = r#"root ::= (declaration)*

declaration ::= dataType identifier "(" parameter? ")" "{" statement* "}"

dataType  ::= "int" ws | "float" ws | "char" ws
identifier ::= [a-zA-Z_] [a-zA-Z_0-9]*

parameter ::= dataType identifier

statement ::=
    ( dataType identifier ws "=" ws expression ";" ) |
    ( identifier ws "=" ws expression ";" ) |
    ( identifier ws "(" argList? ")" ";" ) |
    ( "return" ws expression ";" ) |
    ( "while" "(" condition ")" "{" statement* "}" ) |
    ( "for" "(" forInit ";" ws condition ";" ws forUpdate ")" "{" statement* "}" ) |
    ( "if" "(" condition ")" "{" statement* "}" ("else" "{" statement* "}")? ) |
    ( singleLineComment ) |
    ( multiLineComment )

forInit ::= dataType identifier ws "=" ws expression | identifier ws "=" ws expression
forUpdate ::= identifier ws "=" ws expression

condition ::= expression relationOperator expression
relationOperator ::= ("<=" | "<" | "==" | "!=" | ">=" | ">")

expression ::= term (("+" | "-") term)*
term ::= factor(("*" | "/") factor)*

factor ::= identifier | number | unaryTerm | funcCall | parenExpression
unaryTerm ::= "-" factor
funcCall ::= identifier "(" argList? ")"
parenExpression ::= "(" ws expression ws ")"

argList ::= expression ("," ws expression)*

number ::= [0-9]+

singleLineComment ::= "//" [^\n]* "\n"
multiLineComment ::= "/*" ( [^*] | ("*" [^/]) )* "*/"

ws ::= ([ \t\n]+)
"#;

pub(super) const CHESS_GBNF: &str = r#"# Specifies chess moves as a list in algebraic notation, using PGN conventions

# Force first move to "1. ", then any 1-2 digit number after, relying on model to follow the pattern
root    ::= "1. " move " " move "\n" ([1-9] [0-9]? ". " move " " move "\n")+
move    ::= (pawn | nonpawn | castle) [+#]?

# piece type, optional file/rank, optional capture, dest file & rank
nonpawn ::= [NBKQR] [a-h]? [1-8]? "x"? [a-h] [1-8]

# optional file & capture, dest file & rank, optional promotion
pawn    ::= ([a-h] "x")? [a-h] [1-8] ("=" [NBKQR])?

castle  ::= "O-O" "-O"?
"#;

pub(super) const ENGLISH_GBNF: &str = r##"# note: this might be incomplete, mostly an example
root        ::= en-char+ ([ \t\n] en-char+)*
en-char     ::= letter | digit | punctuation
letter      ::= [a-zA-Z]
digit       ::= [0-9]
punctuation ::= [!"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~]
"##;

pub(super) const JAPANESE_GBNF: &str = r#"# A probably incorrect grammar for Japanese
root        ::= jp-char+ ([ \t\n] jp-char+)*
jp-char     ::= hiragana | katakana | punctuation | cjk
hiragana    ::= [ぁ-ゟ]
katakana    ::= [ァ-ヿ]
punctuation ::= [、-〾]
cjk         ::= [一-鿿]
"#;

pub(super) const JSON_GBNF: &str = r#"root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
"#;

pub(super) const JSON_ARR_GBNF: &str = r#"# This is the same as json.gbnf but we restrict whitespaces at the end of the root array
# Useful for generating JSON arrays

root   ::= arr
value  ::= object | array | string | number | ("true" | "false" | "null") ws

arr  ::=
  "[\n" ws (
            value
    (",\n" ws value)*
  )? "]"

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [1-9] [0-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
"#;

pub(super) const LIST_GBNF: &str = r#"root ::= item+

# Excludes various line break characters
item ::= "- " [^\r\n\x0b\x0c\x85\u2028\u2029]+ "\n"
"#;
