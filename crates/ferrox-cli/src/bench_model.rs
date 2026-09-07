//! `ferrox bench -m model.gguf` — a `llama-bench` work-alike.
//!
//! Same workload definition as `llama-bench` (`pp<N>` = batched prefill
//! of N synthetic tokens, `tg<N>` = N single-token decode steps after a
//! one-token prime), same reporting (median ± population stddev over
//! `-r` repetitions, after one discarded warmup), same flag names.
//! Output is directly comparable to
//! `llama-bench -m <same gguf> -p <N> -n <N> -t <T> -ngl <L>`.
//!
//! Deliberately *not* here: HTTP, chat template, real tokenizer,
//! sampling. Synthetic token ids exercise the same weights and the same
//! KV growth without making the number depend on a tokenizer's behavior.
//!
//! Engines: generic [`Decoder`] (batched `forward_batch_last` for `pp*`)
//! and dedicated [`Gemma4Engine`] (sequential `forward_token` for both
//! `pp*` and `tg*` until a batched Gemma-4 prefill lands).

use crate::bench_guard::{self, CacheProbe, WorkloadDigest};
use crate::host_state::{self, ThermalReading};
use anyhow::Context;
use ferrox_core::cache::KvCache;
use ferrox_models::engine::Engine;
use ferrox_models::{
    load_gemma4_engine_from_path, select_engine_kind, Decoder, Gemma4Engine, ModelConfig,
    SelectedEngineKind, ServedEngine,
};
use std::path::Path;
use std::time::Instant;

/// One `llama-bench` row: a named workload and its per-rep tok/s.
struct Row {
    test: String,
    samples: Vec<f64>,
    /// Digest of the token stream every timed repetition fed. Written
    /// into the receipt so two rows for the same workload can be shown
    /// to have measured the same work, across sessions.
    digest: WorkloadDigest,
}

impl Row {
    fn median(&self) -> f64 {
        let mut s = self.samples.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        if n == 0 {
            return 0.0;
        }
        if n % 2 == 1 {
            s[n / 2]
        } else {
            0.5 * (s[n / 2 - 1] + s[n / 2])
        }
    }

    /// Population stddev, matching what `llama-bench` prints after `±`.
    fn stddev(&self) -> f64 {
        let n = self.samples.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.samples.iter().sum::<f64>() / n as f64;
        (self.samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
    }
}

/// Applies the same backend/threading env `ferrox run` does, before any
/// worker thread exists. Kept separate from [`run`] so the caller can
/// order it ahead of everything else in `main`.
///
/// # Safety
/// Must be called while the process is still single-threaded.
/// Applies the backend/threading env, then CHECKS that it took.
///
/// The check is the point. Setting `FERROX_METAL=0` is not the same as
/// running on the CPU: the backend decision is cached in a `OnceLock`
/// the first time anything reads it, so a caller that ran too late set
/// a variable nobody would ever consult again. That is exactly what
/// happened to every `cpu` row in `benchmarks/RESULTS.md` (#126) --
/// each receipt recorded `backend: "cpu"` beside `backend_active:
/// "Metal"`, and nothing compared them.
///
/// So this asks `active_backend()` immediately, which both forces the
/// cache while the env is still fresh and reveals a disagreement, and
/// it fails loudly rather than measuring the wrong device.
pub fn apply_env(threads: usize, n_gpu_layers: usize) -> anyhow::Result<()> {
    // SAFETY: called from `main` before rayon/Metal workers spawn.
    unsafe {
        if threads > 0 {
            std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
            std::env::set_var("FERROX_CPU_THREADS", threads.to_string());
        }
        if n_gpu_layers == 0 {
            std::env::set_var("FERROX_METAL", "0");
            std::env::set_var("FERROX_METAL_ATTN", "0");
            std::env::set_var("FERROX_CUDA", "0");
        } else {
            std::env::set_var("FERROX_METAL", "auto");
            if std::env::var_os("FERROX_METAL_ATTN").is_none() {
                std::env::set_var("FERROX_METAL_ATTN", "1");
            }
            std::env::set_var("FERROX_CUDA", "auto");
        }
        ferrox_core::weight_matrix::default_cpu_int_dot_on();
    }
    ferrox_core::threads::init_cpu_pool();

    // Reads (and therefore fixes) the cached backend while the env
    // above is still the most recent word on it.
    let active = active_backend();
    let wanted_cpu = n_gpu_layers == 0;
    if wanted_cpu && active != "CPU" {
        anyhow::bail!(
            "--n-gpu-layers 0 asks for CPU but this process resolved to {active}. \
             The backend is decided once per process and something fixed it before \
             this call, so the run would measure {active} and label the receipt `cpu`. \
             Refusing rather than publishing a mislabelled row (#126)."
        );
    }
    Ok(())
}

pub struct BenchArgs {
    pub model: String,
    pub n_prompt: usize,
    pub n_gen: usize,
    pub reps: usize,
    pub ctx_size: usize,
    /// Also run `llama-bench` on the same GGUF with matching flags and
    /// report the gap, instead of leaving the comparison to the reader.
    pub compare: bool,
    /// Backend label recorded in the receipt (`cpu` / `metal` / `cuda`).
    pub backend: String,
    /// Where to write the JSON receipt, if anywhere.
    pub receipt: Option<std::path::PathBuf>,
    /// Suite id this run belongs to, for the receipt.
    pub id: Option<String>,
    /// 1-minute load average above which the run refuses to start.
    /// `0.0` disables the check. See `host_state`.
    pub max_load: f64,
}

pub fn run(args: BenchArgs) -> anyhow::Result<()> {
    // Before anything is loaded: a timed run on a busy or hot host
    // produces a number that looks like a measurement and is not one.
    bench_guard::check_repetitions(args.reps)?;
    let load_start = crate::host_state::ensure_quiet_enough(args.max_load)?;
    let thermal_start = crate::host_state::thermal_reading();
    // `--max-load 0` is already the documented "measure anyway, not
    // publishable" escape; the thermal bar shares it rather than
    // growing a second flag that has to be remembered separately.
    crate::host_state::ensure_cool_enough(&thermal_start, args.max_load > 0.0)?;
    let model = crate::pull::resolve_model_path(&args.model)?;
    let path = Path::new(&model);
    if !path.exists() {
        anyhow::bail!("model not found: {model}");
    }

    // A busy host and a hot host are already refused above. A FULL host
    // is the same failure with a different cause: the weights page to
    // disk and the run times the page file. The file size is the floor
    // on the footprint, and `--max-load 0` waives this the same way it
    // waives the other two.
    if args.max_load > 0.0 {
        if let Ok(meta) = std::fs::metadata(path) {
            let weights_gb = meta.len() as f64 / 1024.0 / 1024.0 / 1024.0;
            crate::host_state::ensure_fits_in_ram(weights_gb, 2.0)?;
        }
    }

    let file = ferrox_gguf::ShardedGguf::open(path)?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    let kind = select_engine_kind(&arch).map_err(|e| anyhow::anyhow!("{e}"))?;

    let ctx_needed = args.n_prompt.max(1) + args.n_gen + 2;
    if args.ctx_size > 0 && ctx_needed > args.ctx_size {
        anyhow::bail!(
            "workload needs {ctx_needed} positions but -c is {}",
            args.ctx_size
        );
    }

    let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let (params_b, load_s, rows) = match kind {
        SelectedEngineKind::GenericDecoder => {
            let config = ModelConfig::from_gguf(&file)
                .with_context(|| format!("reading model config for arch {arch}"))?;
            let params_b = estimate_params(&config);
            let load_t = Instant::now();
            let decoder = Decoder::from_gguf(path, config)?;
            let load_s = load_t.elapsed().as_secs_f64();
            (params_b, load_s, measure_decoder(&decoder, &args)?)
        }
        SelectedEngineKind::Gemma4 => {
            let load_t = Instant::now();
            let served = load_gemma4_engine_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
            let ServedEngine::Gemma4(engine) = served else {
                anyhow::bail!("expected ServedEngine::Gemma4 for arch {arch}");
            };
            let engine = *engine;
            let load_s = load_t.elapsed().as_secs_f64();
            let params_b = estimate_gemma4_params(&engine);
            // Gemma-4 has no batched prefill yet: pp* is sequential
            // forward_token (same as `ferrox run`). llama-bench still
            // batches, so the pp gap is partly that asymmetry.
            eprintln!("ferrox bench: gemma4 uses sequential prefill (no forward_batch_last yet)");
            (params_b, load_s, measure_gemma4(&engine, &args)?)
        }
        other => anyhow::bail!(
            "`ferrox bench` does not cover {other:?} yet (arch {arch}); \
             use the dedicated engine path via `ferrox run` for inference"
        ),
    };

    let backend = active_backend();
    let threads = ferrox_core::threads::resolve_cpu_threads();
    println!(
        "| {:<30} | {:>10} | {:>10} | {:<10} | {:>7} | {:>15} | {:>20} |",
        "model", "size", "params", "backend", "threads", "test", "t/s"
    );
    println!(
        "| {:-<30} | {:->10} | {:->10} | {:-<10} | {:->7} | {:->15} | {:->20} |",
        "", "", "", "", "", "", ""
    );
    for row in &rows {
        println!(
            "| {:<30} | {:>10} | {:>10} | {:<10} | {:>7} | {:>15} | {:>13.2} ± {:>4.2} |",
            truncate(&format!("{arch} {}", quant_label(&file)), 30),
            human_bytes(size_bytes),
            human_params(params_b),
            backend,
            threads,
            row.test,
            row.median(),
            row.stddev(),
        );
    }
    let load_end = crate::host_state::load_average_1min();
    let thermal_end = crate::host_state::thermal_reading();
    eprintln!(
        "\nferrox bench: load {load_s:.2}s, {} reps + {} discarded warmup each",
        args.reps,
        bench_guard::WARMUP_REPS
    );
    eprintln!(
        "ferrox bench: host 1-min load {} -> {}, {} -> {}",
        fmt_load(load_start),
        fmt_load(load_end),
        thermal_start.describe(),
        thermal_end.describe(),
    );
    // A run that started cool and finished hot produced its later
    // repetitions under different physics than its first ones.
    if !thermal_start.is_degraded() && thermal_end.is_degraded() {
        eprintln!(
            "ferrox bench: WARNING -- the host became thermally limited during this \
             run ({}); the later repetitions did not run under the same conditions \
             as the first",
            thermal_end.describe()
        );
    }
    let engine_env = bench_guard::nondefault_engine_env(std::env::vars());
    if !engine_env.is_empty() {
        eprintln!(
            "ferrox bench: non-default engine env in effect: {}",
            engine_env
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    let ngl = if backend == "CPU" { 0 } else { 99 };
    let llama = if args.compare {
        match run_llama_bench(&model, args.n_prompt, args.n_gen, args.reps, ngl) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("ferrox bench: llama-bench comparison unavailable: {e}");
                None
            }
        }
    } else {
        eprintln!(
            "compare: llama-bench -m {model} -p {} -n {} -ngl {ngl}   (or pass --compare)",
            args.n_prompt, args.n_gen,
        );
        None
    };

    if let Some(llama) = &llama {
        println!();
        println!(
            "| {:<15} | {:>12} | {:>12} | {:>8} |",
            "test", "ferrox", "llama.cpp", "gap"
        );
        println!("| {:-<15} | {:->12} | {:->12} | {:->8} |", "", "", "", "");
        for row in &rows {
            let l = llama.get(&row.test).copied();
            let gap = l.map(|l| l / row.median());
            println!(
                "| {:<15} | {:>12.2} | {:>12} | {:>8} |",
                row.test,
                row.median(),
                l.map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into()),
                gap.map(|g| format!("{g:.2}×"))
                    .unwrap_or_else(|| "—".into()),
            );
        }
        eprintln!("gap = llama / ferrox; <1 means ferrox is faster");
    }

    if let Some(dest) = &args.receipt {
        write_receipt(
            dest,
            &args,
            &model,
            &arch,
            &file,
            size_bytes,
            params_b,
            threads,
            backend,
            &rows,
            llama.as_ref(),
            load_s,
            HostState {
                load_start,
                load_end,
                thermal_start,
                thermal_end,
            },
            &engine_env,
        )?;
        eprintln!("ferrox bench: receipt written to {}", dest.display());
    }
    Ok(())
}

/// `pp<N>` / `tg<N>` for the generic [`Decoder`].
fn measure_decoder(decoder: &Decoder, args: &BenchArgs) -> anyhow::Result<Vec<Row>> {
    let mut rows: Vec<Row> = Vec::new();
    if args.n_prompt > 0 {
        rows.push(bench_prefill(decoder, args.n_prompt, args.reps)?);
    }
    if args.n_gen > 0 {
        rows.push(bench_decode(decoder, args.n_gen, args.reps)?);
    }
    for row in &rows {
        bench_guard::check_timed_samples(&row.test, args.reps, row.samples.len())?;
        bench_guard::check_sample_rates(&row.test, &row.samples)?;
    }
    Ok(rows)
}

/// `pp<N>` / `tg<N>` for [`Gemma4Engine`] (sequential tokens only).
fn measure_gemma4(engine: &Gemma4Engine, args: &BenchArgs) -> anyhow::Result<Vec<Row>> {
    let mut rows: Vec<Row> = Vec::new();
    if args.n_prompt > 0 {
        rows.push(bench_prefill_gemma4(engine, args.n_prompt, args.reps)?);
    }
    if args.n_gen > 0 {
        rows.push(bench_decode_gemma4(engine, args.n_gen, args.reps)?);
    }
    for row in &rows {
        bench_guard::check_timed_samples(&row.test, args.reps, row.samples.len())?;
        bench_guard::check_sample_rates(&row.test, &row.samples)?;
    }
    Ok(rows)
}

/// The engine-side half of the determinism assertion, threaded through
/// one row's repetitions.
///
/// Also where an empty logit vector is caught: a forward pass that
/// returned nothing cannot be shown to have computed anything, so the
/// duration it took is not a throughput.
fn check_result(
    test: &str,
    rep: usize,
    logits: &[f32],
    first: &mut Option<(usize, f32)>,
) -> anyhow::Result<()> {
    let pick = bench_guard::greedy_pick(logits)?.ok_or_else(|| {
        anyhow::anyhow!(
            "{test} rep {rep}: the forward pass returned no logits, so it cannot be \
             shown to have computed anything and its duration is not a throughput"
        )
    })?;
    let seen = *first.get_or_insert(pick);
    bench_guard::check_same_result(test, rep, seen, pick)
}

/// Snapshots every layer's KV cache for the guards in `bench_guard`.
fn probe(caches: &[KvCache]) -> Vec<CacheProbe> {
    caches
        .iter()
        .map(|c| CacheProbe {
            // ROWS: this probe reports buffer state, beside the two
            // lengths it is derived from.
            seq_len: c.rows(),
            k_len: c.k.len(),
            v_len: c.v.len(),
        })
        .collect()
}

/// Same, for Gemma-4's sparse `Vec<Option<KvCache>>` (shared-KV layers
/// leave holes, which are not caches and are not checked).
fn probe_gemma4(state: &ferrox_models::gemma4_engine::Gemma4DecodeState) -> Vec<CacheProbe> {
    state
        .kv
        .iter()
        .flatten()
        .map(|c| CacheProbe {
            // ROWS: this probe reports buffer state, beside the two
            // lengths it is derived from.
            seq_len: c.rows(),
            k_len: c.k.len(),
            v_len: c.v.len(),
        })
        .collect()
}

/// `pp<N>`: one batched forward over N tokens into fresh KV caches.
/// Reported as prompt tokens per second, as `llama-bench` does.
fn bench_prefill(decoder: &Decoder, n_prompt: usize, reps: usize) -> anyhow::Result<Row> {
    let test = format!("pp{n_prompt}");
    let tokens = synthetic_tokens(decoder.config.vocab_size, n_prompt);
    // Asserted BEFORE the run: the stream about to be fed is exactly as
    // long as the row's label promises, and every id is in vocabulary.
    bench_guard::check_prompt_before(&test, n_prompt, &tokens, decoder.config.vocab_size)?;
    let mut out = Vec::with_capacity(reps);
    let mut first_digest: Option<WorkloadDigest> = None;
    let mut first_result: Option<(usize, f32)> = None;
    for rep in 0..reps + bench_guard::WARMUP_REPS {
        let mut caches = fresh_caches(decoder);
        // Every rep starts from an empty KV: a prefill that reused a
        // warm cache would report the engine's speed at work it did
        // not do. Asserted rather than assumed, so a later change to
        // `fresh_caches` cannot silently turn this into a cache-hit
        // benchmark.
        bench_guard::check_caches_cold(&test, rep, &probe(&caches))?;
        let mut digest = WorkloadDigest::new();
        digest.feed_all(&tokens);
        let t = Instant::now();
        // `forward_batch_last`, not `forward_batch`: llama-bench's
        // `pp512` asks for logits at the final position only, so
        // projecting all 512 rows through the vocabulary would be work
        // the reference engine never does.
        let logits = decoder.forward_batch_last(&tokens, 0, &mut caches);
        let dt = t.elapsed().as_secs_f64();
        // The prompt length is asserted AFTER the run too, not only
        // before: an engine that silently truncated the batch would
        // otherwise divide the full token count by a partial run's
        // time and report a speedup for doing less work.
        bench_guard::check_prefill_after(&test, n_prompt, &probe(&caches))?;
        let first = *first_digest.get_or_insert(digest);
        bench_guard::check_same_workload(&test, rep, first, digest)?;
        check_result(&test, rep, &logits, &mut first_result)?;
        // Rep 0 is the warmup: first touch of every weight page, and on
        // Metal the first pipeline compile. Timing it would measure the
        // OS and the shader cache, not the engine.
        if rep >= bench_guard::WARMUP_REPS {
            out.push(n_prompt as f64 / dt);
        }
    }
    Ok(Row {
        test,
        samples: out,
        digest: first_digest.unwrap_or_default(),
    })
}

/// Gemma-4 `pp<N>`: sequential `forward_token` over N synthetic ids.
/// Keeps only the final logits (same work llama-bench reports), but
/// without batched matmuls — honest for today's engine.
fn bench_prefill_gemma4(
    engine: &Gemma4Engine,
    n_prompt: usize,
    reps: usize,
) -> anyhow::Result<Row> {
    let test = format!("pp{n_prompt}");
    let vocab = Engine::vocab_size(engine);
    let tokens = synthetic_tokens(vocab, n_prompt);
    bench_guard::check_prompt_before(&test, n_prompt, &tokens, vocab)?;
    let mut out = Vec::with_capacity(reps);
    let mut first_digest: Option<WorkloadDigest> = None;
    let mut first_result: Option<(usize, f32)> = None;
    for rep in 0..reps + bench_guard::WARMUP_REPS {
        let mut state = Engine::new_state(engine);
        bench_guard::check_caches_cold(&test, rep, &probe_gemma4(&state))?;
        let mut digest = WorkloadDigest::new();
        let t = Instant::now();
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            digest.feed(tok);
            logits = Engine::forward_token(engine, tok, i, &mut state);
        }
        let dt = t.elapsed().as_secs_f64();
        bench_guard::check_prefill_after(&test, n_prompt, &probe_gemma4(&state))?;
        let first = *first_digest.get_or_insert(digest);
        bench_guard::check_same_workload(&test, rep, first, digest)?;
        check_result(&test, rep, &logits, &mut first_result)?;
        if rep >= bench_guard::WARMUP_REPS {
            out.push(n_prompt as f64 / dt);
        }
    }
    Ok(Row {
        test,
        samples: out,
        digest: first_digest.unwrap_or_default(),
    })
}

/// `tg<N>`: N single-token decode steps after a one-token prime, so KV
/// length grows exactly as it does in real generation.
/// Whether the host-side `KvCache` is where the KV actually lands.
///
/// With GPU offload it is not: the device holds the KV and the host
/// struct stays empty unless `FERROX_CPU_KV_OFFLOAD=1` syncs it back.
/// Any assertion that counts host cache positions has to know that, or
/// it refuses every GPU run for doing nothing wrong.
fn host_kv_is_the_record() -> bool {
    let off = |k: &str| std::env::var(k).map(|v| v == "0").unwrap_or(false);
    let synced = std::env::var("FERROX_CPU_KV_OFFLOAD").as_deref() == Ok("1");
    synced || (off("FERROX_METAL") && off("FERROX_CUDA"))
}

fn bench_decode(decoder: &Decoder, n_gen: usize, reps: usize) -> anyhow::Result<Row> {
    let test = format!("tg{n_gen}");
    let vocab = decoder.config.vocab_size;
    let tokens = decode_tokens(vocab, n_gen);
    // The decode stream gets the same BEFORE check the prompt does. It
    // is built up front for exactly that reason: a stream generated
    // inside the timed loop can only be checked once it is too late to
    // refuse.
    bench_guard::check_prompt_before(&test, n_gen, &tokens, vocab)?;
    let mut out = Vec::with_capacity(reps);
    let mut first_digest: Option<WorkloadDigest> = None;
    let mut first_result: Option<(usize, f32)> = None;
    for rep in 0..reps + bench_guard::WARMUP_REPS {
        let mut caches = fresh_caches(decoder);
        bench_guard::check_caches_cold(&test, rep, &probe(&caches))?;
        let _ = decoder.forward_token(0, 0, &mut caches);
        let mut digest = WorkloadDigest::new();
        let t = Instant::now();
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            digest.feed(tok);
            logits = decoder.forward_token(tok, i + 1, &mut caches);
        }
        let dt = t.elapsed().as_secs_f64();
        // One priming token plus n_gen decode steps. A short cache
        // means steps were skipped, which would inflate the rate.
        let kv_checked = bench_guard::check_decode_after(
            &test,
            n_gen,
            &probe(&caches),
            host_kv_is_the_record(),
        )?;
        let _ = kv_checked;
        let first = *first_digest.get_or_insert(digest);
        bench_guard::check_same_workload(&test, rep, first, digest)?;
        check_result(&test, rep, &logits, &mut first_result)?;
        if rep >= bench_guard::WARMUP_REPS {
            out.push(n_gen as f64 / dt);
        }
    }
    Ok(Row {
        test,
        samples: out,
        digest: first_digest.unwrap_or_default(),
    })
}

fn bench_decode_gemma4(engine: &Gemma4Engine, n_gen: usize, reps: usize) -> anyhow::Result<Row> {
    let test = format!("tg{n_gen}");
    let vocab = Engine::vocab_size(engine);
    let tokens = decode_tokens(vocab, n_gen);
    bench_guard::check_prompt_before(&test, n_gen, &tokens, vocab)?;
    let mut out = Vec::with_capacity(reps);
    let mut first_digest: Option<WorkloadDigest> = None;
    let mut first_result: Option<(usize, f32)> = None;
    for rep in 0..reps + bench_guard::WARMUP_REPS {
        let mut state = Engine::new_state(engine);
        bench_guard::check_caches_cold(&test, rep, &probe_gemma4(&state))?;
        let _ = Engine::forward_token(engine, 0, 0, &mut state);
        let mut digest = WorkloadDigest::new();
        let t = Instant::now();
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            digest.feed(tok);
            logits = Engine::forward_token(engine, tok, i + 1, &mut state);
        }
        let dt = t.elapsed().as_secs_f64();
        let kv_checked = bench_guard::check_decode_after(
            &test,
            n_gen,
            &probe_gemma4(&state),
            host_kv_is_the_record(),
        )?;
        let _ = kv_checked;
        let first = *first_digest.get_or_insert(digest);
        bench_guard::check_same_workload(&test, rep, first, digest)?;
        check_result(&test, rep, &logits, &mut first_result)?;
        if rep >= bench_guard::WARMUP_REPS {
            out.push(n_gen as f64 / dt);
        }
    }
    Ok(Row {
        test,
        samples: out,
        digest: first_digest.unwrap_or_default(),
    })
}

fn fresh_caches(decoder: &Decoder) -> Vec<KvCache> {
    decoder
        .layers
        .iter()
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect()
}

/// Token ids that exist in the vocabulary but carry no linguistic
/// meaning: the point is to move the same bytes through the same
/// kernels, not to generate text.
fn synthetic_tokens(vocab: usize, n: usize) -> Vec<usize> {
    let vocab = vocab.max(1);
    (0..n).map(|i| (i * 7 + 1) % vocab).collect()
}

/// The token stream a `tg<N>` row feeds after its priming token, built
/// before the clock starts so its length can be asserted BEFORE the run
/// rather than only reconstructed from the KV cache afterwards.
fn decode_tokens(vocab: usize, n_gen: usize) -> Vec<usize> {
    let vocab = vocab.max(1);
    (0..n_gen).map(|i| (i + 1) % vocab).collect()
}

/// Whether a receipt's `backend` label describes the backend that ran.
///
/// The label is lowercase and comes from the suite (`cpu`, `metal`,
/// `cuda`); the active backend is what [`active_backend`] resolved.
/// Compared case-insensitively and nothing else: a mismatch is always
/// a defect, never a naming convention.
fn backend_label_agrees(label: &str, active: &str) -> bool {
    label.eq_ignore_ascii_case(active)
}

pub fn active_backend() -> &'static str {
    #[cfg(feature = "metal")]
    {
        if ferrox_core::weight_matrix::metal_dense_enabled() {
            return "Metal";
        }
    }
    #[cfg(feature = "cuda")]
    {
        if ferrox_core::weight_matrix::cuda_dense_enabled() {
            return "CUDA";
        }
    }
    "CPU"
}

/// Rough total for the report column only — embeddings plus the
/// per-layer estimate `ModelConfig` already computes. Not a substitute
/// for the GGUF's own tensor accounting, and not used in any tok/s math.
fn estimate_params(config: &ModelConfig) -> u64 {
    let embeddings = 2 * config.vocab_size as u64 * config.hidden_dim as u64;
    embeddings + config.approx_active_params_per_token() as u64
}

/// Display-only param estimate for Gemma-4 (emb + head + per-layer Q/O/FFN).
fn estimate_gemma4_params(engine: &Gemma4Engine) -> u64 {
    let hp = &engine.hp;
    let v = Engine::vocab_size(engine) as u64;
    let h = hp.hidden_dim as u64;
    let mut n = 2 * v * h;
    for (il, &ffn) in hp.ffn_dims.iter().enumerate() {
        let hd = hp.head_dim(il) as u64;
        let nh = hp.n_heads as u64;
        let nkv = hp.n_kv_heads as u64;
        n += nh * hd * h; // q
        if hp.has_kv(il) {
            n += 2 * nkv * hd * h; // k, v
        }
        n += nh * hd * h; // o
        let f = ffn as u64;
        n += 3 * f * h; // gate, up, down
    }
    n
}

fn quant_label(file: &ferrox_gguf::ShardedGguf) -> String {
    file.metadata_str("general.file_type")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "quantized".to_string())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

fn human_bytes(b: u64) -> String {
    let mib = b as f64 / (1024.0 * 1024.0);
    if mib >= 1024.0 {
        format!("{:.2} GiB", mib / 1024.0)
    } else {
        format!("{mib:.2} MiB")
    }
}

fn human_params(p: u64) -> String {
    let b = p as f64;
    if b >= 1e9 {
        format!("{:.2} B", b / 1e9)
    } else {
        format!("{:.2} M", b / 1e6)
    }
}

/// Runs `llama-bench` on the same GGUF with matching workload flags and
/// parses its markdown table into `{test -> tok/s}`.
///
/// Thread count is deliberately *not* forced. Each engine picking its
/// own default is the comparison that means something: llama.cpp
/// defaults to performance cores and degrades sharply above them, so
/// pinning both engines to the same oversubscribed count (as the old
/// suite did with `-t 10`) handicaps llama by 2-4x on Apple Silicon and
/// flatters ferrox.
fn run_llama_bench(
    model: &str,
    n_prompt: usize,
    n_gen: usize,
    reps: usize,
    ngl: usize,
) -> anyhow::Result<std::collections::BTreeMap<String, f64>> {
    let out = std::process::Command::new("llama-bench")
        .args([
            "-m",
            model,
            "-p",
            &n_prompt.to_string(),
            "-n",
            &n_gen.to_string(),
            "-r",
            &reps.to_string(),
            "-ngl",
            &ngl.to_string(),
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("could not run llama-bench (is it on PATH?): {e}"))?;
    if !out.status.success() {
        anyhow::bail!("llama-bench exited with {}", out.status);
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(parse_llama_bench_table(&text))
}

/// Extracts `{test -> tok/s}` from `llama-bench`'s markdown table.
///
/// Its rows look like `| model | size | params | backend | threads |
/// test | t/s |`, where the `t/s` cell is `123.45 ± 6.78`. Anything that
/// does not have that shape (header, separator, the trailing `build:`
/// line) is skipped rather than guessed at.
fn parse_llama_bench_table(text: &str) -> std::collections::BTreeMap<String, f64> {
    let mut out = std::collections::BTreeMap::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 8 {
            continue;
        }
        let test = cells[6];
        if test.is_empty() || test == "test" || test.starts_with('-') {
            continue;
        }
        let Some(value) = cells[7].split('±').next() else {
            continue;
        };
        if let Ok(v) = value.trim().parse::<f64>() {
            out.insert(test.to_string(), v);
        }
    }
    out
}

/// What the host was doing around this run, for the receipt.
#[derive(Clone, Copy)]
pub struct HostState {
    pub load_start: Option<f64>,
    pub load_end: Option<f64>,
    pub thermal_start: ThermalReading,
    pub thermal_end: ThermalReading,
}

/// Serializes a thermal reading so that "we did not measure" and
/// "we measured, and it was nominal" can never be read as the same
/// thing. `measured` is stated outright rather than left to be
/// inferred from a null, because a field that is always null while
/// implying it was measured is exactly the lie this receipt exists to
/// prevent.
fn thermal_json(r: &ThermalReading) -> serde_json::Value {
    serde_json::json!({
        "measured": r.measured(),
        "pressure": r.pressure.map(|p| p.as_str()),
        "source": r.source,
        "cpu_speed_limit_percent": r.cpu_speed_limit_percent,
        "degraded": r.measured().then(|| r.is_degraded()),
    })
}

/// A load average we could not read prints as `?`, never as `0.00`.
fn fmt_load(l: Option<f64>) -> String {
    l.map(|l| format!("{l:.2}"))
        .unwrap_or_else(|| "?".to_string())
}

#[allow(clippy::too_many_arguments)] // one call site; every field is reported
fn write_receipt(
    dest: &Path,
    args: &BenchArgs,
    model: &str,
    arch: &str,
    file: &ferrox_gguf::ShardedGguf,
    size_bytes: u64,
    params: u64,
    threads: usize,
    backend: &str,
    rows: &[Row],
    llama: Option<&std::collections::BTreeMap<String, f64>>,
    load_s: f64,
    host: HostState,
    engine_env: &[(String, String)],
) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tests = Vec::new();
    for row in rows {
        let l = llama.and_then(|m| m.get(&row.test)).copied();
        tests.push(serde_json::json!({
            "test": row.test,
            "ferrox_tps": row.median(),
            "ferrox_stddev": row.stddev(),
            "ferrox_samples": row.samples,
            "llama_tps": l,
            "gap": l.map(|l| l / row.median()),
            // Same workload, same digest -- across models, sessions and
            // machines. Two rows that claim to measure `pp512` on the
            // same checkpoint and disagree here did not measure the
            // same work.
            "workload_digest": row.digest.hex(),
        }));
    }
    // WHAT the machine was, never WHICH machine it was. A receipt is
    // public, and `--render` reads every receipt in the directory:
    // without this, rows from two hosts merge into one table with no
    // column saying so.
    let host_spec_json = {
        let spec = host_state::host_spec();
        serde_json::json!({
            "label": host_state::host_label(&spec),
            "cpu": spec.cpu,
            "arch": spec.arch,
            "cores": spec.cores,
            "perf_cores": spec.perf_cores,
            "ram_gb": spec.ram_gb.map(|g| (g * 10.0).round() / 10.0),
            "os": spec.os,
        })
    };
    // `host_spec` names the CPU. For a `cpu` row that is the whole
    // machine; for a `cuda` or `metal` row it is the least interesting
    // part. Ten published CUDA rows carried a Xeon's model number and
    // NOTHING about the card, in a file whose own header says rows are
    // never compared across machines -- there was no field to compare
    // them by, so a Pascal row and an Ampere row grouped as one host.
    let accelerator = accelerator_name(backend);
    if backend != "CPU" && accelerator.is_none() {
        anyhow::bail!(
            "refusing to write a `{}` receipt that does not name the accelerator it ran on. \
             A GPU gap is meaningless without the card: the ledger groups rows by host, and \
             two different GPUs in one host section are two different machines.",
            args.backend
        );
    }

    // A receipt may not claim one backend and record another. Every
    // `cpu` row in the ledger did exactly that (#126), and the
    // contradiction sat in the file: `backend: "cpu"` beside
    // `backend_active: "Metal"`. Two fields that must agree, with
    // nothing comparing them, which is this repo's oldest bug shape.
    if !backend_label_agrees(&args.backend, backend) {
        anyhow::bail!(
            "refusing to write a receipt labelled `{}` for a run that executed on {}. \
             The label is what the ledger publishes; the active backend is what ran (#126).",
            args.backend,
            backend
        );
    }
    let receipt = serde_json::json!({
        "schema": 2,
        "kind": "engine",
        "id": args.id,
        "model_path": model,
        "arch": arch,
        "quant": quant_label(file),
        "size_bytes": size_bytes,
        "approx_params": params,
        "backend": args.backend,
        "backend_active": backend,
        "threads": threads,
        "reps": args.reps,
        "warmup_reps": bench_guard::WARMUP_REPS,
        "load_s": load_s,
        // Null, not zero, when the platform would not say -- see host_state.
        "host_load_1min_start": host.load_start,
        "host_load_1min_end": host.load_end,
        "host_thermal_start": thermal_json(&host.thermal_start),
        "host_thermal_end": thermal_json(&host.thermal_end),
        // The single field an auditor reads first: was this row taken
        // under the measurement contract's quiet-host bar at all?
        "quiet_host": host.load_start.map(|l| l < host_state::DEFAULT_MAX_LOAD),
        // WHAT the machine was, never WHICH machine it was. A receipt
        // is public, and `--render` reads every receipt in the
        // directory: without this, rows from two hosts merge into one
        // table with no column that says so.
        "host_spec": host_spec_json,
        // The card, for a row that ran on one. `null` for `cpu` rows,
        // where `host_spec` already is the machine.
        "accelerator": accelerator,
        // Non-default `FERROX_*` knobs in effect. Some of them (MoE
        // stage ablation, the fail-closed loader override) change how
        // much work the engine does, so a row taken under one is not
        // comparable to a row taken without it.
        "engine_env": engine_env
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect::<serde_json::Map<_, _>>(),
        "ferrox_version": env!("CARGO_PKG_VERSION"),
        "tests": tests,
    });
    std::fs::write(dest, serde_json::to_string_pretty(&receipt)? + "\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::backend_label_agrees;

    /// The receipt guard that #126 needed and did not have.
    #[test]
    fn a_cpu_label_does_not_describe_a_metal_run() {
        assert!(backend_label_agrees("cpu", "CPU"));
        assert!(backend_label_agrees("metal", "Metal"));
        assert!(backend_label_agrees("cuda", "CUDA"));
        assert!(
            !backend_label_agrees("cpu", "Metal"),
            "this exact pair is what all 13 published cpu receipts recorded"
        );
        assert!(!backend_label_agrees("metal", "CPU"));
        assert!(!backend_label_agrees("cuda", "Metal"));
    }

    use super::*;

    fn row(samples: &[f64]) -> Row {
        Row {
            test: "tg128".to_string(),
            samples: samples.to_vec(),
            digest: WorkloadDigest::new(),
        }
    }

    #[test]
    fn median_of_an_odd_sample_count_is_the_middle_value_regardless_of_order() {
        assert_eq!(row(&[30.0, 10.0, 20.0]).median(), 20.0);
    }

    #[test]
    fn median_of_an_even_sample_count_averages_the_two_middle_values() {
        assert_eq!(row(&[10.0, 20.0, 30.0, 40.0]).median(), 25.0);
    }

    #[test]
    fn stddev_is_zero_for_identical_samples_and_undefined_counts_report_zero() {
        assert_eq!(row(&[5.0, 5.0, 5.0]).stddev(), 0.0);
        assert_eq!(row(&[5.0]).stddev(), 0.0);
        assert_eq!(row(&[]).stddev(), 0.0);
    }

    #[test]
    fn stddev_is_the_population_form_that_llama_bench_prints() {
        // mean 20, deviations -10/0/+10 -> sqrt(200/3), not the sample
        // form sqrt(200/2). Matching llama-bench matters because these
        // numbers are printed side by side.
        let got = row(&[10.0, 20.0, 30.0]).stddev();
        assert!(
            (got - (200.0f64 / 3.0).sqrt()).abs() < 1e-9,
            "population stddev expected, got {got}"
        );
    }

    #[test]
    fn both_token_streams_satisfy_the_before_check_they_are_handed_to() {
        // The generators and the guard have to agree, or the BEFORE
        // check would be a refusal of every run rather than of a bad
        // one. `vocab` deliberately includes values smaller than the
        // stream, where the modulo wraps.
        for &(vocab, n) in &[(49152usize, 512usize), (7, 512), (2, 4), (49152, 1)] {
            bench_guard::check_prompt_before("pp", n, &synthetic_tokens(vocab, n), vocab).unwrap();
            bench_guard::check_prompt_before("tg", n, &decode_tokens(vocab, n), vocab).unwrap();
        }
    }

    #[test]
    fn the_decode_stream_is_the_one_the_loop_used_to_generate_inline() {
        // Hoisting it out of the timed loop must not have changed the
        // workload: same ids, same order, so digests published before
        // and after this change still describe the same work.
        let vocab = 32000;
        let inline: Vec<usize> = (0..128).map(|i| (i + 1) % vocab).collect();
        assert_eq!(decode_tokens(vocab, 128), inline);
    }

    #[test]
    fn a_repetition_that_returned_no_logits_is_refused_rather_than_timed() {
        let mut first = None;
        let err = check_result("pp512", 1, &[], &mut first)
            .unwrap_err()
            .to_string();
        assert!(err.contains("returned no logits"), "{err}");
    }

    #[test]
    fn the_first_repetitions_answer_becomes_the_one_the_rest_must_match() {
        let mut first = None;
        check_result("pp512", 0, &[0.1, 0.9, 0.2], &mut first).unwrap();
        assert_eq!(first, Some((1, 0.9)));
        // Same answer, different magnitudes: still the same work.
        check_result("pp512", 1, &[0.2, 0.7, 0.1], &mut first).unwrap();
        // A different answer from the same token stream is not.
        assert!(check_result("pp512", 2, &[0.9, 0.1, 0.2], &mut first).is_err());
    }

    #[test]
    fn human_readable_sizes_switch_units_at_a_gibibyte() {
        assert_eq!(human_bytes(512 * 1024 * 1024), "512.00 MiB");
        assert_eq!(human_bytes(2 * 1024 * 1024 * 1024), "2.00 GiB");
    }

    #[test]
    fn human_readable_params_switch_units_at_a_billion() {
        assert_eq!(human_params(135_000_000), "135.00 M");
        assert_eq!(human_params(8_000_000_000), "8.00 B");
    }

    #[test]
    fn truncate_leaves_short_strings_alone_and_clips_long_ones() {
        assert_eq!(truncate("llama Q8_0", 30), "llama Q8_0");
        assert_eq!(truncate(&"x".repeat(40), 30), "x".repeat(30));
    }
}

/// The accelerator a non-CPU run executed on, as the backend itself
/// reports it.
///
/// Returns `None` on CPU, where `host_spec` already describes the
/// machine, and `None` when a GPU backend cannot name its device --
/// which the receipt writer treats as a refusal rather than a blank,
/// because an unnamed card is what made ten CUDA rows unattributable.
fn accelerator_name(backend: &str) -> Option<String> {
    match backend {
        #[cfg(feature = "cuda")]
        "CUDA" => ferrox_cuda::gpu::probe().and_then(|i| i.first_device_name),
        #[cfg(feature = "metal")]
        "Metal" => ferrox_metal::gpu::probe(),
        _ => None,
    }
}

#[cfg(test)]
mod accelerator_tests {
    use super::*;

    /// A `cpu` row has no accelerator and must not be refused for it.
    #[test]
    fn the_cpu_backend_names_no_accelerator() {
        assert_eq!(accelerator_name("CPU"), None);
    }

    /// The guard is keyed on the ACTIVE backend, not the label, for the
    /// same reason `backend_label_agrees` is: the label is what the
    /// ledger publishes and the active backend is what ran.
    #[test]
    fn only_a_gpu_backend_is_required_to_name_a_device() {
        for (backend, required) in [("CPU", false), ("CUDA", true), ("Metal", true)] {
            assert_eq!(backend != "CPU", required, "{backend}");
        }
    }
}
