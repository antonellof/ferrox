//! Who owns BOS: the chat template, or the loader?
//!
//! Measured, not reasoned about. `sweep_local_gguf_bos_policy` opens
//! every GGUF under `models/` (or `$FERROX_TEST_MODELS_DIR`), renders that
//! checkpoint's own `tokenizer.chat_template` through the minijinja
//! evaluator, encodes the result with that checkpoint's own tokenizer,
//! and counts how many BOS ids come out at the front under
//! [`ferrox_models::tokenizer::prepend_bos`].
//!
//! Run it with:
//!
//! ```text
//! cargo test -p ferrox-models --test bos_policy -- --ignored --nocapture
//! ```

mod common;
use common::collect_gguf;
use ferrox_models::chat_template::{ChatTemplate, RenderOptions};
use ferrox_models::tokenizer::{
    prepend_bos, should_add_bos_token, GgufBpeTokenizer, GgufSpmTokenizer, GgufUnigramTokenizer,
    SpecialTokens,
};
use serde_json::json;

enum Tok {
    Bpe(Box<GgufBpeTokenizer>),
    Spm(GgufSpmTokenizer),
    Unigram(GgufUnigramTokenizer),
}

impl Tok {
    fn open(file: &ferrox_gguf::ShardedGguf) -> Option<Self> {
        match file.metadata_str("tokenizer.ggml.model") {
            Some("gpt2" | "gemma4") => {
                Some(Tok::Bpe(Box::new(GgufBpeTokenizer::from_gguf(file).ok()?)))
            }
            Some("llama") => Some(Tok::Spm(GgufSpmTokenizer::from_gguf(file).ok()?)),
            Some("t5") => Some(Tok::Unigram(GgufUnigramTokenizer::from_gguf(file).ok()?)),
            _ => None,
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Tok::Bpe(t) => t.encode(text, SpecialTokens::Parse),
            Tok::Spm(t) => t.encode(text, SpecialTokens::Parse),
            Tok::Unigram(t) => t.encode(text, SpecialTokens::Parse),
        }
    }
}

#[test]
#[ignore = "needs the real GGUF checkpoints in models/"]
fn sweep_local_gguf_bos_policy() {
    let root = std::env::var("FERROX_TEST_MODELS_DIR").unwrap_or_else(|_| "models".to_string());
    let mut files = Vec::new();
    collect_gguf(std::path::Path::new(&root), &mut files);
    files.sort();
    assert!(!files.is_empty(), "no GGUFs under {root}");

    let messages = vec![json!({"role": "user", "content": "hi"})];
    let mut doubled = Vec::new();
    let mut checked = 0usize;

    for path in &files {
        let Ok(file) = ferrox_gguf::ShardedGguf::open(path.to_str().unwrap()) else {
            continue;
        };
        let Some(tok) = Tok::open(&file) else {
            continue;
        };
        let Some(src) = file
            .metadata_str("tokenizer.chat_template")
            .filter(|t| !t.trim().is_empty())
        else {
            continue;
        };
        let bos_text = file.token_text("tokenizer.ggml.bos_token_id");
        let eos_text = file.token_text("tokenizer.ggml.eos_token_id");
        let bos_id = file
            .metadata_u64("tokenizer.ggml.bos_token_id")
            .map(|v| v as u32);
        let add_bos = should_add_bos_token(&file);

        let opts = RenderOptions {
            add_generation_prompt: true,
            bos_token: bos_text.clone(),
            eos_token: eos_text,
            ..Default::default()
        };
        let Ok(tmpl) = ChatTemplate::from_jinja(src) else {
            continue;
        };
        let Ok(rendered) = tmpl.render(&messages, &opts) else {
            continue;
        };

        let mut ids = tok.encode(&rendered);
        let template_emitted_bos = bos_id.is_some() && ids.first() == bos_id.as_ref();
        prepend_bos(&mut ids, bos_id.filter(|_| add_bos));
        let leading = match bos_id {
            Some(b) => ids.iter().take_while(|&&t| t == b).count(),
            None => 0,
        };
        checked += 1;
        println!(
            "{:<52} add_bos={add_bos:<5} bos={:?} template_emits_bos={template_emitted_bos:<5} \
             leading_bos={leading}",
            path.file_name().unwrap().to_string_lossy(),
            bos_text.as_deref().unwrap_or("-"),
        );
        if leading > 1 {
            doubled.push(format!("{}: {leading} leading BOS ids", path.display()));
        }
    }

    println!("checked {checked} checkpoints, {} doubled", doubled.len());
    assert!(
        checked > 0,
        "no checkpoint carried both a tokenizer and a template"
    );
    assert!(doubled.is_empty(), "{doubled:#?}");
}
