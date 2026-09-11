use ferrox_gguf::GgufFile;
use ferrox_models::decoder::Decoder;
use ferrox_models::tokenizer::{GgufSpmTokenizer, SpecialTokens};
use ferrox_models::ModelConfig;

fn main() {
    let path = "models/hf_test/gemma-3-1b-it-Q8_0.gguf";
    let file = GgufFile::open(path).unwrap();
    let tok = GgufSpmTokenizer::from_gguf(&file).unwrap();
    let prompt = "<start_of_turn>user\nWhat is the capital of France? Answer with one word.<end_of_turn>\n<start_of_turn>model\n";
    let mut tokens: Vec<usize> = tok
        .encode(prompt, SpecialTokens::Parse)
        .into_iter()
        .map(|t| t as usize)
        .collect();
    let bos = 2usize;
    if tokens.first() != Some(&bos) {
        tokens.insert(0, bos);
    }
    println!("tokens={tokens:?}");
    let config = ModelConfig::from_gguf(&file).unwrap();
    let decoder = Decoder::from_gguf(path, config).unwrap();
    let mut caches: Vec<_> = decoder.config.new_kv_caches();
    let mut logits = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        logits = decoder.forward_token(t, i, &mut caches);
    }
    let mut out = Vec::new();
    for pos in (tokens.len()..).take(16) {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        out.push(next);
        if next == 106 || next == 1 {
            break;
        }
        logits = decoder.forward_token(next, pos, &mut caches);
    }
    println!("gen={out:?}");
    let decoded = tok.decode(&out.iter().map(|x| *x as u32).collect::<Vec<_>>());
    println!("decoded={decoded:?}");
}
