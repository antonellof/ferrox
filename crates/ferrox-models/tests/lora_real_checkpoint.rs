//! A LoRA adapter on a REAL checkpoint, against libllama's logits for
//! the same adapter: the measurement behind the number in the PR, kept
//! runnable rather than pasted.
//!
//! The tiny-fixture suite (`lora_graphs.rs`) is the gate; this one is
//! the evidence that the same seam holds on a real Q8_0 model whose
//! adapter came out of `convert_lora_to_gguf.py` from a PEFT directory
//! -- rank 8 over q/k/v/o/gate/up/down of Llama-3.2-1B-Instruct, when
//! it was measured (KL is printed; the numbers are in the assertion's
//! comment below).
//!
//! Skipped unless all of these are set:
//!
//! ```text
//! FERROX_LORA_MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf
//! FERROX_LORA_ADAPTERS=adapter.gguf:1.0[,other.gguf:0.5]
//! FERROX_LORA_TOKENS="128000 791 6864 315 9822 374"   # "The capital of France is"
//! FERROX_LORA_GOLDEN=/tmp/ref.txt   # `ref_logits --lora adapter.gguf:1.0 model.gguf 1 504 ...`
//! cargo test -p ferrox-models --test lora_real_checkpoint -- --nocapture
//! ```
//!
//! `FERROX_LORA_ADAPTERS=` (empty) measures the base alone, which is
//! the floor the adapted number is read against.

mod common;
use common::{kl_vs_golden, worst_vs};
use ferrox_gguf::ShardedGguf;
use ferrox_models::lora_attach::LoraSpec;
use ferrox_models::{Decoder, ModelConfig};

#[test]
fn a_real_checkpoint_with_an_adapter_matches_libllama() {
    let Ok(model) = std::env::var("FERROX_LORA_MODEL") else {
        eprintln!("FERROX_LORA_MODEL unset; skipping");
        return;
    };
    let golden_path = std::env::var("FERROX_LORA_GOLDEN").expect("FERROX_LORA_GOLDEN");
    let tokens: Vec<usize> = std::env::var("FERROX_LORA_TOKENS")
        .expect("FERROX_LORA_TOKENS")
        .split_whitespace()
        .map(|t| t.parse().expect("token id"))
        .collect();
    let adapters = std::env::var("FERROX_LORA_ADAPTERS").unwrap_or_default();
    let specs: Vec<LoraSpec> = adapters
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| LoraSpec::parse_scaled(s.trim()).expect("FNAME:SCALE"))
        .collect();

    let golden: Vec<f32> = std::fs::read_to_string(&golden_path)
        .expect("golden file")
        .lines()
        .map(|l| l.trim().parse().expect("a float per line"))
        .collect();

    let file = ShardedGguf::open(std::path::Path::new(&model)).expect("model opens");
    let config = ModelConfig::from_gguf(&file).expect("config");
    let mut d = Decoder::from_gguf(&model, config).expect("model loads");
    d.attach_lora_specs(&file, &specs).expect("adapters attach");

    let mut kv = d.config.new_kv_caches();
    let got = d.forward_batch_last(&tokens, 0, &mut kv);
    assert_eq!(got.len(), golden.len(), "vocab");
    let kl = kl_vs_golden(&got, &golden);
    let worst = worst_vs(&got, &golden);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    println!(
        "| `{}` + [{}] | KL {kl:.2e} | max abs {worst:.2e} | argmax {} vs {} |",
        std::path::Path::new(&model)
            .file_name()
            .unwrap()
            .to_string_lossy(),
        adapters,
        argmax(&got),
        argmax(&golden)
    );
    assert_eq!(argmax(&got), argmax(&golden), "greedy token");
    // `ferrox parity`'s MATCH line (`parity.rs` `KL_NOISE`): below it the
    // two engines are doing the same arithmetic to within accumulation
    // order. Measured on Llama-3.2-1B-Instruct Q8_0: base 1.89e-4, the
    // rank-8 adapter at 1.0 5.23e-4, at 0.5 3.87e-4, its f16 export
    // 5.69e-4 -- while the adapter moves libllama's own distribution by
    // 1.47e-1.
    assert!(kl < 1e-3, "KL {kl}");
}
