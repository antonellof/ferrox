//! Hugging Face Hub download helper.
//!
//! This used to shell out to the `hf` CLI, which meant a Rust engine
//! could not fetch its own weights without a Python install and
//! `pip install huggingface_hub`. The transport is native now: see
//! [`crate::hub`], which resolves IPv4 first, reads `HF_TOKEN`, honours
//! `HF_ENDPOINT`, and asks for a byte range so an interrupted download
//! resumes rather than starting from zero.
//!
//! `hf` is still used when it is on PATH and the native path is not
//! compiled in, so an existing setup keeps working.

use std::path::{Path, PathBuf};
#[cfg(not(feature = "hub"))]
use std::process::Command;

/// True when `path` looks like a Hub repo id rather than a local file.
pub fn looks_like_hf_repo(path: &str) -> bool {
    let p = Path::new(path);
    path.contains('/')
        && !path.contains('\\')
        && !p.exists()
        && !path.ends_with(".gguf")
        && !path.starts_with('.')
}

#[cfg(not(feature = "hub"))]
fn hf_available() -> bool {
    Command::new("hf")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn default_cache_dir(repo: &str) -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/frink/hf")
        .join(repo.replace('/', "--"))
}

fn resolve_gguf_in_dir(dir: &Path) -> anyhow::Result<PathBuf> {
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .collect();
    ggufs.sort();
    ggufs.into_iter().next().ok_or_else(|| {
        anyhow::anyhow!(
            "no .gguf file found under {} after hf download",
            dir.display()
        )
    })
}

/// Download `repo` and return a local `.gguf` path.
pub fn pull_hf_gguf(
    repo: &str,
    file_pattern: &str,
    local_dir: Option<PathBuf>,
) -> anyhow::Result<PathBuf> {
    let local_dir = local_dir.unwrap_or_else(|| default_cache_dir(repo));
    std::fs::create_dir_all(&local_dir)?;

    #[cfg(feature = "hub")]
    {
        crate::hub::fetch_to_dir(repo, file_pattern, &local_dir)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        resolve_gguf_in_dir(&local_dir)
    }

    #[cfg(not(feature = "hub"))]
    {
        if !hf_available() {
            anyhow::bail!(
                "this build has no native downloader (feature `hub` is off) and the \
                 Hugging Face CLI `hf` is not on PATH. Either install it with \
                 `pip install huggingface_hub`, or use a frink build with `hub` on."
            );
        }
        let status = Command::new("hf")
            .arg("download")
            .arg(repo)
            .arg(file_pattern)
            .arg("--local-dir")
            .arg(&local_dir)
            .status()?;
        if !status.success() {
            anyhow::bail!("hf download failed for {repo}");
        }
        resolve_gguf_in_dir(&local_dir)
    }
}

/// If `model` is a Hub repo id, download and return the local GGUF path.
pub fn resolve_model_path(model: &str) -> anyhow::Result<String> {
    if !looks_like_hf_repo(model) {
        return Ok(model.to_string());
    }
    let path = pull_hf_gguf(model, "*.gguf", None)?;
    Ok(path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_repo_heuristic() {
        assert!(looks_like_hf_repo("org/model"));
        assert!(!looks_like_hf_repo("./local/model.gguf"));
        assert!(!looks_like_hf_repo("model.gguf"));
    }
}
