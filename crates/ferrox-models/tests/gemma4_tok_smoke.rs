//! Ignored smoke: Gemma-4 SPM-BPE vs sanity checks on the local GGUF.
use ferrox_gguf::GgufFile;
use ferrox_models::tokenizer::{should_add_bos_token, GgufBpeTokenizer, SpecialTokens};

fn gemma4_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-4-E2B-it-Q4_K_M.gguf")
}

#[test]
#[ignore = "needs models/gemma-4-E2B-it-Q4_K_M.gguf"]
fn gemma4_bpe_roundtrip_and_counts() {
    let path = gemma4_path();
    if !path.exists() {
        eprintln!("skip: missing {path:?}");
        return;
    }
    let file = GgufFile::open(&path).expect("open");
    assert_eq!(file.metadata_str("tokenizer.ggml.model"), Some("gemma4"));
    let tok = GgufBpeTokenizer::from_gguf(&file).expect("tok");
    assert!(tok.has_merges());
    assert_eq!(tok.vocab_size(), 262144);

    let how = tok.encode("How are you", SpecialTokens::AsText);
    let paris = tok.encode("The capital of France is", SpecialTokens::AsText);
    assert!(
        how.len() >= 3,
        "expected multi-token for How are you, got {how:?}"
    );
    assert_eq!(tok.decode(&how), "How are you");
    assert_eq!(tok.decode(&paris), "The capital of France is");
    assert!(should_add_bos_token(&file));

    let eos = file.metadata_u64("tokenizer.ggml.eos_token_id").unwrap() as u32;
    assert_eq!(tok.decode(&[eos]), "<turn|>");
}
