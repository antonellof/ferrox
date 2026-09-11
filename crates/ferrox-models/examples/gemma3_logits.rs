use ferrox_gguf::GgufFile;
use ferrox_models::decoder::Decoder;
use ferrox_models::ModelConfig;
use std::env;

fn main() {
    let path = env::args().nth(1).expect("gguf path");
    let file = GgufFile::open(&path).expect("gguf");
    let config = ModelConfig::from_gguf(&file).expect("config");
    eprintln!(
        "swa={:?} pattern={:?} emb_scale={:?} rope={} rope_swa={:?} ffn={:?} qk={:?}",
        config.sliding_window,
        config.swa_pattern,
        config.embedding_scale,
        config.rope_theta,
        config.rope_theta_swa,
        config.ffn_activation,
        config.qk_norm_style,
    );
    let decoder = Decoder::from_gguf(&path, config).expect("decoder");
    let mut tokens: Vec<usize> = env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    if tokens.is_empty() {
        tokens = vec![2, 3689, 563, 506, 5279, 529, 7001, 236881];
    }
    let n_gen: usize = env::var("N_GEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let mut caches: Vec<_> = decoder.config.new_kv_caches();
    let mut logits = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        logits = decoder.forward_token(tok, i, &mut caches);
    }
    print!("FIRST10");
    for v in &logits[..10.min(logits.len())] {
        print!(" {v:.6}");
    }
    println!();
    print!("GEN");
    for pos in (tokens.len()..).take(n_gen) {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        print!(" {next}");
        logits = decoder.forward_token(next, pos, &mut caches);
    }
    println!();
}
