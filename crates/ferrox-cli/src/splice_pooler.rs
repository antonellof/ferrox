//! `ferrox splice-pooler`: write a reranker GGUF that carries the
//! pooler llama.cpp's converter dropped, so `/v1/rerank` scores on the
//! range the checkpoint was trained to produce (issue #82). The work
//! and every refusal are in `ferrox_models::rerank_pooler`; this is
//! argument parsing and the report.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use ferrox_models::splice_pooler;

#[derive(Parser, Debug)]
pub struct SplicePoolerArgs {
    /// The reranker GGUF as the converter left it: a `bert` checkpoint
    /// carrying `cls.output` and no `cls`.
    #[arg(short = 'm', long = "model")]
    pub gguf: PathBuf,

    /// The HuggingFace `model.safetensors` of the SAME checkpoint,
    /// carrying `bert.pooler.dense.*` and `classifier.*`. The
    /// classifier in both files must agree to within the GGUF's own
    /// storage precision, or nothing is written.
    #[arg(long)]
    pub safetensors: PathBuf,

    /// Where to write the pooled GGUF. Never the input path.
    #[arg(short = 'o', long)]
    pub output: PathBuf,
}

pub fn run(args: SplicePoolerArgs) -> Result<()> {
    if args.output == args.gguf {
        anyhow::bail!(
            "--output must differ from --model: the input is read while the output is written"
        );
    }
    let done = splice_pooler(&args.gguf, &args.safetensors, &args.output)
        .with_context(|| format!("splicing a pooler into {}", args.gguf.display()))?;
    println!(
        "wrote {}: cls.weight [{n}, {n}] + cls.bias [{n}] (F32) from {}",
        done.output.display(),
        args.safetensors.display(),
        n = done.n_embd,
    );
    println!(
        "identity: cls.output ({:?}, {} output(s)) agrees with classifier.* to |diff| <= {:.3e} \
         (bound {:.3e})",
        done.head_dtype, done.n_out, done.classifier_max_abs_diff, done.classifier_allowed,
    );
    println!(
        "the head is classifier(tanh(pooler(cls))) now; /v1/rerank reports it as \
         ferrox_score_head on every response"
    );
    Ok(())
}
