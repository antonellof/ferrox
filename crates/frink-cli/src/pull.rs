//! Download GGUF checkpoints from Hugging Face Hub via the `hf` CLI.

use std::path::PathBuf;

use clap::Args;
use frink_models::hf_pull;

#[derive(Args, Debug, Clone)]
pub struct PullArgs {
    /// Hugging Face repo id (`org/model`).
    pub repo: String,

    /// Glob of files to fetch (default: all `*.gguf` shards).
    #[arg(long = "file", default_value = "*.gguf")]
    pub file_pattern: String,

    /// Directory to store downloads (passed to `hf download --local-dir`).
    #[arg(long = "dir")]
    pub local_dir: Option<PathBuf>,
}

pub fn run_pull(args: PullArgs) -> anyhow::Result<()> {
    let path = hf_pull::pull_hf_gguf(&args.repo, &args.file_pattern, args.local_dir)?;
    println!("{}", path.display());
    Ok(())
}

pub fn resolve_model_path(model: &str) -> anyhow::Result<String> {
    hf_pull::resolve_model_path(model)
}
