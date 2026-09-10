//! One table, walked by every test here: for each of the eleven wire
//! formats this server parses, either a forced call it can WRITE and
//! read back, or the refusal that names the format.
//!
//! The table is the third structure on purpose. [`super::wire`] derives
//! the grammar from `ToolCallFormat::markers()`, and the parser reads by
//! the same values, so the two cannot disagree with each other -- but
//! they could agree and both be wrong about what a checkpoint actually
//! emits. The `call` strings below are written the way the family's own
//! template writes one, and each says where it was checked.

use ferrox_models::grammar::Grammar;

use super::{build, Forced, ToolSpec};
use crate::output::{parse_output, OutputPosture};
use crate::policy::parser::tool_call::{ToolCallParser, ToolSchema};
use crate::policy::parser::ToolCallFormat;
use crate::{ToolDef, ToolFunctionDef};

/// Drive `text` through the grammar the way a decode loop would, one
/// piece at a time, and report whether the parse is complete.
pub(super) fn feed(grammar: &Grammar, pieces: &[&str]) -> Result<bool, String> {
    let mut g = grammar.clone();
    for (i, piece) in pieces.iter().enumerate() {
        g.accept_token(i as u32, piece.as_bytes())
            .map_err(|e| format!("piece {piece:?}: {e}"))?;
    }
    Ok(g.allows_eog())
}

/// The tool every sample calls: one required string, one optional
/// integer. The integer is what catches a format whose values are text
/// -- `3` has to arrive as a number, not as `"3"`.
fn weather_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
        },
        "required": ["city"],
        "additionalProperties": false,
    })
}

/// What every sample's call must be read back as, in every format.
const EXPECTED_ARGUMENTS: &str = r#"{"city":"Rome","days":3}"#;

/// One format's canonical call.
struct Sample {
    /// A served-model name that [`ToolCallFormat::infer`] resolves to
    /// this format, so the test walks the same path a request does.
    model: &'static str,
    /// What a real turn of this family writes before the call.
    ///
    /// A lazy trigger exists so the model may talk first, and for a
    /// family whose reasoning block is opened by the PROMPT that talk
    /// ends the way the checkpoint ends it. MiniMax-M2 is the one that
    /// has to say so: its reasoning is always-open and it declares no
    /// tool marker that closes it, so a call written before `</think>`
    /// is thinking as far as this server is concerned -- for a forced
    /// call exactly as for an unforced one.
    prose: &'static str,
    /// A complete `get_weather` call, written the way this family's own
    /// template writes one.
    call: &'static str,
    /// A call that is nearly right -- a required argument dropped, or a
    /// tool that was never offered -- and must be refused.
    near_miss: &'static str,
}

/// The eleven formats, and what a forced call in each looks like.
///
/// Exhaustive with no `_` arm: a twelfth wire format has to be given a
/// sample here, or declared unforceable, before this compiles.
fn sample(format: ToolCallFormat) -> Option<Sample> {
    match format {
        // Hermes/Qwen2.5, Llama 3 and Mistral: the three whose payload
        // is a JSON object, forced since this module existed.
        ToolCallFormat::Qwen25 => Some(Sample {
            model: "qwen2.5-7b-instruct",
            prose: "Let me look that up. ",
            call: r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Rome", "days": 3}}</tool_call>"#,
            near_miss: r#"<tool_call>{"name": "get_weather", "arguments": {"days""#,
        }),
        ToolCallFormat::Llama3 => Some(Sample {
            model: "llama-3.1-8b-instruct",
            prose: "Let me look that up. ",
            call: r#"<|python_tag|>{"name": "get_weather", "arguments": {"city": "Rome", "days": 3}}"#,
            near_miss: r#"<|python_tag|>{"name": "get_weather", "arguments": {"days""#,
        }),
        ToolCallFormat::Mistral => Some(Sample {
            model: "mistral-7b-instruct",
            prose: "Let me look that up. ",
            call: r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Rome", "days": 3}}]"#,
            near_miss: r#"[TOOL_CALLS] {"name""#,
        }),

        // The element grammars. Every one of these was checked against
        // llama.cpp's `common/chat.cpp` builder for the same family:
        // `common_chat_params_init_qwen3_coder` writes
        // `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n`,
        // and the ferrox parser's own `wire_for` fixture in
        // `policy::parser::tool_call` agrees.
        ToolCallFormat::Qwen3Coder => Some(Sample {
            model: "qwen3-coder-30b",
            prose: "Let me look that up. ",
            call: "<tool_call>\n<function=get_weather>\n\
                   <parameter=city>\nRome\n</parameter>\n\
                   <parameter=days>\n3\n</parameter>\n\
                   </function>\n</tool_call>",
            near_miss: "<tool_call>\n<function=get_weather>\n<parameter=days>",
        }),
        ToolCallFormat::Glm47 => Some(Sample {
            model: "glm-4.7",
            prose: "Let me look that up. ",
            call: "<tool_call>get_weather\n\
                   <arg_key>city</arg_key><arg_value>Rome</arg_value>\n\
                   <arg_key>days</arg_key><arg_value>3</arg_value>\n\
                   </tool_call>",
            near_miss: "<tool_call>get_weather\n<arg_key>days</arg_key>",
        }),
        ToolCallFormat::MiniMax => Some(Sample {
            model: "minimax-m2",
            // MiniMax-M2's reasoning block is opened by the prompt, so
            // a turn closes it before it calls anything. See `prose`.
            prose: "Let me look that up.</think>",
            call: "<minimax:tool_call><invoke name=\"get_weather\">\
                   <parameter name=\"city\">Rome</parameter>\
                   <parameter name=\"days\">3</parameter>\
                   </invoke></minimax:tool_call>",
            near_miss: "<minimax:tool_call><invoke name=\"get_weather\">\
                        <parameter name=\"days\"",
        }),
        ToolCallFormat::DeepSeekV32 => Some(Sample {
            model: "deepseek-v3.2",
            prose: "Let me look that up. ",
            call: "<｜DSML｜function_calls><｜DSML｜invoke name=\"get_weather\">\
                   <｜DSML｜parameter name=\"city\">Rome</｜DSML｜parameter>\
                   <｜DSML｜parameter name=\"days\">3</｜DSML｜parameter>\
                   </｜DSML｜invoke></｜DSML｜function_calls>",
            near_miss: "<｜DSML｜function_calls><｜DSML｜invoke name=\"get_weather\">\
                        <｜DSML｜parameter name=\"days\"",
        }),

        // gpt-oss. `common_chat_params_init_gpt_oss` writes the same
        // header -- `<|channel|>(commentary|analysis) to=functions.NAME
        // [<|constrain|>TYPE] <|message|>ARGS` -- and ends the call with
        // `<|call|>`.
        ToolCallFormat::GptOss => Some(Sample {
            model: "gpt-oss-20b",
            prose: "Let me look that up. ",
            call: "<|channel|>commentary to=functions.get_weather <|constrain|>json\
                   <|message|>{\"city\": \"Rome\", \"days\": 3}<|call|>",
            near_miss: "<|channel|>commentary to=functions.get_weather<|message|>{\"days\"",
        }),

        // MiniMax-M3, whose element grammar is M2's with every tag
        // behind the `]<]minimax[>[` namespace. The shape is the
        // parser's own `M3_*` constants, and its fixture in
        // `policy::parser::tool_call` writes a call the same way.
        ToolCallFormat::MiniMaxM3 => Some(Sample {
            model: "minimax-m3",
            prose: "Let me look that up. ",
            call: "]<]minimax[>[<tool_call>\
                   ]<]minimax[>[<invoke name=\"get_weather\">\
                   ]<]minimax[>[<city>Rome]<]minimax[>[</city>\
                   ]<]minimax[>[<days>3]<]minimax[>[</days>\
                   ]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
            near_miss: "]<]minimax[>[<tool_call>\
                        ]<]minimax[>[<invoke name=\"get_weather\">\
                        ]<]minimax[>[<days>",
        }),

        // Gemma 4. The call is the chat template's own
        // `'<|tool_call>call:' + name + '{'`, then
        // `key:format_argument(value, escape_keys=False)` per argument
        // `| dictsort`ed, then `'}<tool_call|>'` -- read off
        // `gemma-4-E2B-it`'s embedded template, and exercised against
        // that checkpoint through the server.
        ToolCallFormat::Gemma4 => Some(Sample {
            model: "gemma-4-e2b-it",
            prose: "Let me look that up. ",
            call: "<|tool_call>call:get_weather{city:<|\"|>Rome<|\"|>,days:3}<tool_call|>",
            near_miss: "<|tool_call>call:get_weather{days:",
        }),

        // The one that refuses. See `wire::shape` for the reason.
        ToolCallFormat::MuseGlimmer => None,
    }
}

/// Every variant of [`ToolCallFormat`], so the tests below walk all of
/// them. `sample` is the exhaustive match that makes a new one visible;
/// this list is what makes it TESTED.
const EVERY_FORMAT: [ToolCallFormat; 11] = [
    ToolCallFormat::Qwen25,
    ToolCallFormat::Llama3,
    ToolCallFormat::Mistral,
    ToolCallFormat::Qwen3Coder,
    ToolCallFormat::Glm47,
    ToolCallFormat::DeepSeekV32,
    ToolCallFormat::MiniMax,
    ToolCallFormat::MiniMaxM3,
    ToolCallFormat::GptOss,
    ToolCallFormat::Gemma4,
    ToolCallFormat::MuseGlimmer,
];

fn tool_defs() -> Vec<ToolDef> {
    ["get_weather", "send_mail"]
        .iter()
        .map(|name| ToolDef {
            kind: "function".to_string(),
            function: ToolFunctionDef {
                name: (*name).to_string(),
                description: None,
                parameters: Some(weather_schema()),
            },
        })
        .collect()
}

/// The headline, and the whole point of the issue this closes: for every
/// format a forced `tool_choice` is served for, the grammar accepts a
/// real call in that family's framing AND this server's own parser reads
/// that same text back as the call the schema declares. A format it is
/// not served for refuses BY NAME.
#[test]
fn every_format_either_forces_a_call_this_server_reads_back_or_refuses_by_name() {
    let schema = weather_schema();
    let offered = [
        ToolSpec {
            name: "get_weather",
            parameters: Some(&schema),
        },
        ToolSpec {
            name: "send_mail",
            parameters: Some(&schema),
        },
    ];
    let defs = tool_defs();

    for format in EVERY_FORMAT {
        let Some(sample) = sample(format) else {
            let (status, axum::Json(body)) = build(Forced::Any, &offered, format)
                .expect_err("this format has no root rule and must refuse");
            assert_eq!(
                status,
                axum::http::StatusCode::NOT_IMPLEMENTED,
                "{format:?}"
            );
            let message = body["error"]["message"].as_str().unwrap_or_default();
            assert!(
                message.contains(format.as_str()),
                "a refusal must name the format: {message}"
            );
            continue;
        };

        // The served model name is what picks the format, exactly as
        // `generation_params_for_template` picks it.
        assert_eq!(
            ToolCallFormat::infer(sample.model),
            format,
            "the sample's model name must resolve to its own format"
        );

        let grammar = build(Forced::Any, &offered, format)
            .unwrap_or_else(|e| panic!("{format:?} should build a grammar: {e:?}"));

        // Written after free prose, because that is what a lazy trigger
        // is for.
        assert!(
            feed(&grammar, &[sample.prose, sample.call])
                .unwrap_or_else(|e| panic!("{format:?} should accept its own call: {e}")),
            "{format:?}: the parse must be complete once the call is written"
        );

        // The format's OWN parser -- not `parse_output`'s Hermes
        // fallback -- reads it back.
        let native = ToolCallParser::new(
            format,
            vec![ToolSchema::with_parameters("get_weather", schema.clone())],
        );
        let (_, calls) = native.parse_complete(sample.call);
        assert_eq!(calls.len(), 1, "{format:?} should read back one call");
        assert_eq!(calls[0].name, "get_weather", "{format:?}");
        assert_eq!(calls[0].arguments, EXPECTED_ARGUMENTS, "{format:?}");

        // And so does the whole response path, from the served model's
        // name onward.
        let parsed = parse_output(
            &format!("{}{}", sample.prose, sample.call),
            &defs,
            OutputPosture::for_model(sample.model),
        );
        assert_eq!(parsed.calls.len(), 1, "{format:?} through parse_output");
        assert_eq!(parsed.calls[0].name, "get_weather", "{format:?}");
        assert_eq!(parsed.calls[0].arguments, EXPECTED_ARGUMENTS, "{format:?}");

        // A near miss is refused rather than served.
        assert!(
            feed(&grammar, &[sample.near_miss]).is_err(),
            "{format:?}: {:?} must not be a legal forced call",
            sample.near_miss
        );
    }
}

/// A forced `tool_choice` has to actually FORCE: the turn may not end
/// before a call has begun, prose may not finish it, and a named choice
/// makes every other offered tool unreachable in every format.
#[test]
fn a_forced_choice_constrains_every_format_it_is_served_for() {
    let schema = weather_schema();
    let offered = [
        ToolSpec {
            name: "get_weather",
            parameters: Some(&schema),
        },
        ToolSpec {
            name: "send_mail",
            parameters: Some(&schema),
        },
    ];

    for format in EVERY_FORMAT {
        let Some(sample) = sample(format) else {
            continue;
        };

        let any = build(Forced::Any, &offered, format).expect("a grammar");
        assert!(any.is_awaiting_trigger(), "{format:?} must be lazy");
        assert!(!any.allows_eog(), "{format:?}: nothing has been called yet");
        let mut prose = (*any).clone();
        prose
            .accept_token(0, b"I do not think a tool is needed here.")
            .expect("prose before the trigger is free");
        assert!(
            !prose.allows_eog(),
            "{format:?}: prose must not be allowed to finish a forced turn"
        );

        // The same grammar with the union narrowed to the OTHER tool:
        // the sample call must now be unreachable.
        let named = build(Forced::Named("send_mail"), &offered, format).expect("a grammar");
        assert!(
            feed(&named, &[sample.call]).is_err(),
            "{format:?}: a named tool_choice must make every other tool unreachable"
        );
    }
}

/// An argument whose value is written as bare text is typed from the
/// tool's own schema, so a property the schema does not type is refused
/// rather than guessed at. `parse_loose` would decide `018956` is a
/// number, and a forced call may not be built on a guess.
#[test]
fn an_element_format_refuses_an_argument_it_cannot_type() {
    let untyped = serde_json::json!({
        "type": "object",
        "properties": {"city": {"description": "where"}},
        "required": ["city"],
    });
    let offered = [ToolSpec {
        name: "get_weather",
        parameters: Some(&untyped),
    }];
    for format in [ToolCallFormat::Qwen3Coder, ToolCallFormat::MiniMax] {
        let (status, axum::Json(body)) = build(Forced::Any, &offered, format)
            .expect_err("an untyped argument has no value rule");
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{format:?}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("city"),
            "the refusal must name the argument: {body}"
        );
    }
}

/// A declared `enum` of strings is written as its members, not as JSON:
/// a quoted `"celsius"` would reach the tool WITH its quotes, because
/// `convert_declared` hands a declared string over verbatim.
#[test]
fn an_enum_argument_is_written_as_its_member_and_not_as_json() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["unit"],
    });
    let offered = [ToolSpec {
        name: "get_weather",
        parameters: Some(&schema),
    }];
    let grammar = build(Forced::Any, &offered, ToolCallFormat::MiniMax).expect("a grammar");
    let call = "<minimax:tool_call><invoke name=\"get_weather\">\
                <parameter name=\"unit\">celsius</parameter>\
                </invoke></minimax:tool_call>";
    assert!(feed(&grammar, &[call]).expect("the member is legal"));
    assert!(
        feed(
            &grammar,
            &["<minimax:tool_call><invoke name=\"get_weather\">\
               <parameter name=\"unit\">kelvin"]
        )
        .is_err(),
        "a member the enum does not list must be unreachable"
    );

    let parser = ToolCallParser::new(
        ToolCallFormat::MiniMax,
        vec![ToolSchema::with_parameters("get_weather", schema)],
    );
    let (_, calls) = parser.parse_complete(call);
    assert_eq!(calls[0].arguments, r#"{"unit":"celsius"}"#);
}

/// A value may contain anything but the tag that ends it -- which is the
/// difference between forcing a call a coding agent can make and one it
/// cannot, because its arguments are whole files.
#[test]
fn an_argument_may_hold_markup_that_is_not_its_closing_tag() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"patch": {"type": "string"}},
        "required": ["patch"],
    });
    let offered = [ToolSpec {
        name: "write_file",
        parameters: Some(&schema),
    }];
    let grammar = build(Forced::Any, &offered, ToolCallFormat::Qwen3Coder).expect("a grammar");
    let call = "<tool_call>\n<function=write_file>\n<parameter=patch>\n\
                <html><body>a < b && c > d</body></html>\n\
                </parameter>\n</function>\n</tool_call>";
    assert!(feed(&grammar, &[call]).expect("markup is legal in a value"));

    let parser = ToolCallParser::new(
        ToolCallFormat::Qwen3Coder,
        vec![ToolSchema::with_parameters("write_file", schema)],
    );
    let (_, calls) = parser.parse_complete(call);
    assert_eq!(
        calls[0].arguments,
        r#"{"patch":"<html><body>a < b && c > d</body></html>"}"#
    );
}

/// The grammar lets a JSON-valued argument sit on its own line, because
/// that is where a template's layout puts it -- so the reader has to
/// tolerate the same whitespace the writer allows.
///
/// DeepSeek is the format that proves it: its values are handed over
/// with `TrimStyle::None`, because a declared STRING's spaces are the
/// model's. A declared integer's are not, and `"\n3\n".parse::<i64>()`
/// fails, so before `convert_declared` trimmed them a `3` written the
/// way the template writes it arrived at the tool as the string
/// `"\n3\n"` for a property the schema calls an integer.
#[test]
fn a_json_argument_may_sit_on_its_own_line_and_still_arrive_typed() {
    let schema = weather_schema();
    let offered = [ToolSpec {
        name: "get_weather",
        parameters: Some(&schema),
    }];
    let grammar = build(Forced::Any, &offered, ToolCallFormat::DeepSeekV32).expect("a grammar");
    let call = "<｜DSML｜function_calls><｜DSML｜invoke name=\"get_weather\">\
                <｜DSML｜parameter name=\"city\">Rome</｜DSML｜parameter>\
                <｜DSML｜parameter name=\"days\">\n3\n</｜DSML｜parameter>\
                </｜DSML｜invoke></｜DSML｜function_calls>";
    assert!(feed(&grammar, &[call]).expect("a newline around a JSON value is legal"));

    let parser = ToolCallParser::new(
        ToolCallFormat::DeepSeekV32,
        vec![ToolSchema::with_parameters("get_weather", schema)],
    );
    let (_, calls) = parser.parse_complete(call);
    assert_eq!(calls[0].arguments, EXPECTED_ARGUMENTS);
}

/// Gemma's arguments are separated by a comma, so an optional one is not
/// an independent `?`: what may be skipped is the pair TOGETHER WITH the
/// comma in front of it, and whichever pair is written first has none.
///
/// A grammar that made each pair `?` on its own would accept `{,b:1}`
/// and `{a:…,}`, and `gemma_arguments` reads a pair with no `:` in it as
/// nothing at all -- so a call with a stray comma loses an argument
/// silently.
#[test]
fn a_gemma_call_may_skip_any_optional_argument_and_still_be_comma_separated() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "integer"},
        },
        "additionalProperties": false,
    });
    let offered = [ToolSpec {
        name: "ping",
        parameters: Some(&schema),
    }];
    let grammar = build(Forced::Any, &offered, ToolCallFormat::Gemma4).expect("a grammar");

    for legal in [
        "<|tool_call>call:ping{}<tool_call|>",
        "<|tool_call>call:ping{a:<|\"|>x<|\"|>}<tool_call|>",
        "<|tool_call>call:ping{b:1}<tool_call|>",
        "<|tool_call>call:ping{a:<|\"|>x<|\"|>,b:1}<tool_call|>",
    ] {
        assert!(
            feed(&grammar, &[legal]).unwrap_or_else(|e| panic!("{legal:?}: {e}")),
            "{legal:?} should be a complete call"
        );
    }
    for illegal in [
        // A separator with nothing before it, and one with nothing
        // after it.
        "<|tool_call>call:ping{,b:1}",
        "<|tool_call>call:ping{a:<|\"|>x<|\"|>,}",
        // The template writes the pairs `| dictsort`ed, so this order
        // is not one the checkpoint emits.
        "<|tool_call>call:ping{b:1,a:",
        // An argument the schema never declared.
        "<|tool_call>call:ping{c:",
    ] {
        assert!(
            feed(&grammar, &[illegal]).is_err(),
            "{illegal:?} must not be a legal forced call"
        );
    }
}

/// A gemma string value is wrapped in gemma's own quote, and the reader
/// toggles quoting at every one of those while ending the whole call at
/// the first `<tool_call|>` regardless. So a value may hold any markup
/// except those two -- which is what makes a forced call usable by a
/// coding agent, whose arguments are whole files.
#[test]
fn a_gemma_string_holds_markup_but_not_the_two_markers_that_end_it() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"patch": {"type": "string"}},
        "required": ["patch"],
    });
    let offered = [ToolSpec {
        name: "write_file",
        parameters: Some(&schema),
    }];
    let grammar = build(Forced::Any, &offered, ToolCallFormat::Gemma4).expect("a grammar");

    let call = "<|tool_call>call:write_file{patch:<|\"|>\
                <html>a < b, {\"json\": true} & <|tool_call> too\
                <|\"|>}<tool_call|>";
    assert!(feed(&grammar, &[call]).expect("markup is legal in a value"));

    let parser = ToolCallParser::new(
        ToolCallFormat::Gemma4,
        vec![ToolSchema::with_parameters("write_file", schema)],
    );
    let (_, calls) = parser.parse_complete(call);
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].arguments,
        r#"{"patch":"<html>a < b, {\"json\": true} & <|tool_call> too"}"#
    );

    for stops_the_reader in [
        "<|tool_call>call:write_file{patch:<|\"|>a<|\"|>b",
        "<|tool_call>call:write_file{patch:<|\"|>a<tool_call|>",
    ] {
        assert!(
            feed(&grammar, &[stops_the_reader]).is_err(),
            "{stops_the_reader:?} is text this server would read back wrong"
        );
    }
}

/// A gemma argument the schema calls an object or an array is REFUSED by
/// name, because the writer and the reader disagree about it: gemma's
/// template spells a composite in its own DSL -- bare keys, gemma-quoted
/// strings -- and `parse_loose` reads a value with `serde_json`, so the
/// spelling the checkpoint writes comes back as a string.
#[test]
fn a_gemma_composite_argument_is_refused_by_name() {
    for declared in ["object", "array"] {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"where": {"type": declared}},
            "required": ["where"],
        });
        let offered = [ToolSpec {
            name: "get_weather",
            parameters: Some(&schema),
        }];
        let (status, axum::Json(body)) = build(Forced::Any, &offered, ToolCallFormat::Gemma4)
            .expect_err("gemma writes a composite in a DSL this server reads as a string");
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains("where"), "{message}");
        assert!(message.contains(declared), "{message}");
    }
}

/// An M3 value is bare text between two elements, and
/// `m3_scan_elements` re-reads it as STRUCTURE from any occurrence of
/// the namespace prefix -- not just from the tag that closes this
/// argument. So the prefix is what a value may not contain.
#[test]
fn an_m3_value_may_not_contain_the_namespace_that_starts_every_tag() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"patch": {"type": "string"}},
        "required": ["patch"],
    });
    let offered = [ToolSpec {
        name: "write_file",
        parameters: Some(&schema),
    }];
    let grammar = build(Forced::Any, &offered, ToolCallFormat::MiniMaxM3).expect("a grammar");

    let call = "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"write_file\">\
                ]<]minimax[>[<patch><html>a < b</html> and ]<]minimax[ too\
                ]<]minimax[>[</patch>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>";
    assert!(feed(&grammar, &[call]).expect("markup that is not a tag is legal"));

    let parser = ToolCallParser::new(
        ToolCallFormat::MiniMaxM3,
        vec![ToolSchema::with_parameters("write_file", schema)],
    );
    let (_, calls) = parser.parse_complete(call);
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].arguments,
        r#"{"patch":"<html>a < b</html> and ]<]minimax[ too"}"#
    );

    // The namespace prefix is legal at the value's END, because that is
    // how its own closing tag begins -- so the text that must be
    // unreachable is a tag that is NOT this argument's closer.
    for opens_a_tag in [
        "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"write_file\">\
         ]<]minimax[>[<patch>a]<]minimax[>[<x",
        "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"write_file\">\
         ]<]minimax[>[<patch>a]<]minimax[>[</other>",
    ] {
        assert!(
            feed(&grammar, &[opens_a_tag]).is_err(),
            "a value that opens a tag of its own would be read back as structure"
        );
    }
}

/// gpt-oss opens its reasoning on the same marker its calls use, so the
/// trigger must be the RECIPIENT and not the channel: a grammar switched
/// on by `<|channel|>` would force a tool call inside the model's own
/// analysis.
#[test]
fn the_harmony_trigger_does_not_fire_on_a_reasoning_channel() {
    let schema = weather_schema();
    let offered = [ToolSpec {
        name: "get_weather",
        parameters: Some(&schema),
    }];
    let grammar = build(Forced::Any, &offered, ToolCallFormat::GptOss).expect("a grammar");

    let mut g = (*grammar).clone();
    g.accept_token(
        0,
        "<|channel|>analysis<|message|>The user wants weather.<|end|>".as_bytes(),
    )
    .expect("an analysis channel must stay unconstrained");
    assert!(
        g.is_awaiting_trigger(),
        "a reasoning channel must not switch the grammar on"
    );
    assert!(
        !g.allows_eog(),
        "and the turn still may not end without a call"
    );
    g.accept_token(
        1,
        "<|start|>assistant<|channel|>commentary to=functions.get_weather\
         <|message|>{\"city\": \"Rome\"}<|call|>"
            .as_bytes(),
    )
    .expect("the call that follows is legal");
    assert!(g.allows_eog(), "the call completes the turn");
}
