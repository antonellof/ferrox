//! Cross-backend output agreement.
//!
//! The benchmark harness measures throughput and never looks at what the
//! engine produced. That gap has hidden real bugs more than once: a d=64
//! prefill kernel applied softcap twice through a cross-simdgroup race
//! while reporting healthy tok/s; SmolLM2 once generated word salad on
//! Metal at full speed; Gemma-2 still emits a bare `:` on Metal. Every
//! one of those rows looked green.
//!
//! So: generate the same greedy continuation on two backends and compare
//! the token sequences. Greedy decoding is deterministic, and a kernel
//! that computes the wrong thing diverges within a few tokens. The CPU
//! path is the reference because it is the one cross-validated against
//! NumPy in `gguf_roundtrip.rs`.
//!
//! Backend choice is a process-lifetime `OnceLock` read from the
//! environment, so this cannot switch backends in-process — it re-invokes
//! itself once per backend, exactly as `bench --suite` does.

use anyhow::Context;
use std::path::Path;
use std::process::Command;

/// How many tokens to compare. Divergence from a real kernel bug shows up
/// almost immediately; the cost of going further is wall time on the
/// slowest backend.
const N_TOKENS: usize = 24;

/// Prompt is fixed so runs are comparable across invocations and models.
const PROMPT: &str = "The capital of France is";

/// Batch size at which the batched-prefill attention kernels turn on
/// (`fa_ext` / the simdgroup-MMA variants gate on `n_q >= 8`). Below it
/// the run only exercises single-token decode, so a green verdict says
/// nothing about the prefill kernels this tool exists to catch — hence
/// the warning rather than a silent pass.
const PREFILL_MIN_TOKENS: usize = 8;

pub struct VerifyArgs {
    pub model: String,
    /// Backend to check against the CPU reference: `metal` or `cuda`.
    pub backend: String,
    /// Emit the token ids rather than a verdict (used by the child).
    pub emit: bool,
    /// Stretch the prompt to this many tokens before prefill, so the
    /// batched-prefill kernels are actually reached.
    pub prompt_tokens: Option<usize>,
    /// Override the fixed prompt.
    pub prompt: Option<String>,
    /// The global `--allow-multiple-instances` flag. Forwarded to the
    /// children, which are the processes that actually load a model: a
    /// clap flag, unlike an environment variable, is not inherited.
    pub allow_multiple_instances: bool,
}

/// Marker the child prints so the parent can find the payload even if the
/// engine writes other things to stdout.
const TAG: &str = "FERROX_VERIFY_TOKENS ";

/// Second marker: how many tokens the prompt actually became. Only the
/// child tokenizes, so this is the one number that says whether the run
/// reached the batched-prefill kernels — `--prompt-tokens` is optional
/// and a long `--prompt` is just as good a way to get there.
const LEN_TAG: &str = "FERROX_VERIFY_PROMPT_LEN ";

pub fn run(args: VerifyArgs) -> anyhow::Result<()> {
    let prompt = args.prompt.clone().unwrap_or_else(|| PROMPT.to_string());
    if args.emit {
        return emit_tokens(&args.model, &prompt, args.prompt_tokens);
    }

    let (reference, prompt_len) = child_tokens(&args, "cpu", &prompt)?;
    let (candidate, _) = child_tokens(&args, &args.backend, &prompt)?;

    if prompt_len < PREFILL_MIN_TOKENS {
        eprintln!(
            "verify: prompt is {prompt_len} tokens, under the {PREFILL_MIN_TOKENS} at which the \
             batched-prefill attention kernels turn on — this run checks decode only. \
             Pass --prompt-tokens {PREFILL_MIN_TOKENS} or more (or a longer --prompt) to \
             cover prefill."
        );
    }

    if reference.is_empty() {
        anyhow::bail!("CPU reference produced no tokens; cannot verify");
    }

    let diverge = reference
        .iter()
        .zip(&candidate)
        .position(|(a, b)| a != b)
        .or_else(|| {
            (reference.len() != candidate.len()).then_some(reference.len().min(candidate.len()))
        });

    match diverge {
        None => {
            println!(
                "verify {}: OK — {} tokens identical on cpu and {} ({})",
                short(&args.model),
                reference.len(),
                args.backend,
                prompt_desc(prompt_len)
            );
            Ok(())
        }
        Some(i) => {
            // Report where and with what, not just that it failed: the
            // index says whether the kernel is wrong from the first token
            // (dense path) or drifts (accumulating / attention path),
            // which is the first thing anyone debugging this needs.
            println!(
                "verify {}: DIVERGED at token {i} ({}) — cpu={:?} {}={:?}",
                short(&args.model),
                prompt_desc(prompt_len),
                &reference[i.saturating_sub(2)..reference.len().min(i + 3)],
                args.backend,
                &candidate[i.saturating_sub(2)..candidate.len().min(i + 3)],
            );
            anyhow::bail!(
                "{} disagrees with the CPU reference from token {i}",
                args.backend
            )
        }
    }
}

/// Says which path the verdict covers, so a decode-only pass is never
/// read as a prefill pass.
fn prompt_desc(prompt_len: usize) -> String {
    if prompt_len >= PREFILL_MIN_TOKENS {
        format!("{prompt_len}-token prompt, prefill covered")
    } else {
        format!("{prompt_len}-token prompt, decode only")
    }
}

fn short(model: &str) -> String {
    Path::new(model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.to_string())
}

/// Runs one backend in a child and parses back the token ids.
fn child_tokens(
    args: &VerifyArgs,
    backend: &str,
    prompt: &str,
) -> anyhow::Result<(Vec<u32>, usize)> {
    let exe = std::env::current_exe()?;
    let mut cmd = Command::new(&exe);
    cmd.arg("verify")
        .args(["-m", &args.model])
        .args(["--backend", backend])
        .args(["--prompt", prompt])
        .arg("--emit");
    if let Some(n) = args.prompt_tokens {
        cmd.args(["--prompt-tokens", &n.to_string()]);
    }
    if args.allow_multiple_instances {
        cmd.env("FERROX_ALLOW_MULTIPLE_INSTANCES", "1");
    }
    // `FERROX_METAL` alone only turns on the Metal *matvecs*: the fused
    // attention block, its RoPE kernels and the resident KV are behind
    // `FERROX_METAL_ATTN`, which `run --ngl 99` sets and this harness
    // did not. A `verify --backend metal` that leaves it unset checks a
    // graph nobody runs, and reports OK for kernels it never reached.
    // Match `run`: default it on for the accelerated child, but honour a
    // pre-set value so `FERROX_METAL_ATTN=0 ferrox verify …` still
    // ablates.
    let metal_attn = std::env::var("FERROX_METAL_ATTN").unwrap_or_else(|_| "1".to_string());
    let out = cmd
        // `-ngl` is what actually selects the backend for `run`; the env
        // var alone is overridden by the CLI default.
        .env("FERROX_METAL", if backend == "cpu" { "0" } else { "1" })
        .env(
            "FERROX_METAL_ATTN",
            if backend == "cpu" { "0" } else { &metal_attn },
        )
        .env("FERROX_CUDA", if backend == "cuda" { "1" } else { "0" })
        .output()
        .with_context(|| format!("spawning verify child for {backend}"))?;
    if !out.status.success() {
        anyhow::bail!(
            "verify child for {backend} failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("(no stderr)")
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix(TAG))
        .with_context(|| format!("verify child for {backend} printed no token line"))?;
    let prompt_len = text
        .lines()
        .find_map(|l| l.strip_prefix(LEN_TAG))
        .and_then(|l| l.trim().parse::<usize>().ok())
        .with_context(|| format!("verify child for {backend} printed no prompt-length line"))?;
    Ok((
        line.split_whitespace()
            .filter_map(|t| t.parse::<u32>().ok())
            .collect(),
        prompt_len,
    ))
}

/// Child side: greedy-decode `N_TOKENS` and print the ids.
fn emit_tokens(model: &str, prompt: &str, prompt_tokens: Option<usize>) -> anyhow::Result<()> {
    let path = crate::pull::resolve_model_path(model)?;
    let (ids, prompt_len) = crate::verify_engine::greedy_token_ids(
        Path::new(&path),
        prompt,
        ferrox_models::tokenizer::SpecialTokens::Parse,
        N_TOKENS,
        prompt_tokens,
    )?;
    let joined: Vec<String> = ids.iter().map(|t| t.to_string()).collect();
    println!("{LEN_TAG}{prompt_len}");
    println!("{TAG}{}", joined.join(" "));
    Ok(())
}
