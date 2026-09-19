//! The library example from the repository README, compiled.
//!
//! Documentation examples rot silently: nothing fails when a signature
//! changes under them, so the first person to notice is a reader who
//! pastes it and gets a compile error. This file is the same snippet,
//! built by `cargo check --workspace --all-targets`, so that stops
//! being possible.
//!
//! It is never RUN in CI -- it wants a real checkpoint on disk -- so
//! keep it to API shape rather than behaviour. If you edit the README
//! example, edit this too; they are meant to be the same text.

#![allow(unused)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use frink_inference::gguf::ShardedGguf;
    use frink_inference::models::{Decoder, ModelConfig};

    let path = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

    // Read metadata without loading a single weight.
    let file = ShardedGguf::open(path)?;
    println!("{} tensors", file.tensor_count());

    // Hyperparameters come from the file. Anything that had to be guessed
    // is listed in `config.best_effort_fields`.
    let config = ModelConfig::from_gguf(&file)?;

    // Weights stay quantized and mmap-resident; dequant happens inside the
    // matmul. This REFUSES a checkpoint carrying tensors this build never
    // reads, rather than quietly computing a different graph.
    let decoder = Decoder::from_gguf(path, config)?;

    Ok(())
}
