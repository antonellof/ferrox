use ferrox_gguf::GgufFile;
use ferrox_models::tokenizer::{GgufSpmTokenizer, SpecialTokens};
use std::env;

fn main() {
    let path = env::args().nth(1).unwrap();
    let file = GgufFile::open(&path).unwrap();
    // Try SPM first
    let tok = GgufSpmTokenizer::from_gguf(&file).expect("spm");
    let prompt = "<start_of_turn>user\nWhat is the capital of France? Answer with one word.<end_of_turn>\n<start_of_turn>model\n";
    let ids = tok.encode(prompt, SpecialTokens::Parse);
    println!("n={}", ids.len());
    println!("{:?}", &ids[..ids.len().min(40)]);
}
