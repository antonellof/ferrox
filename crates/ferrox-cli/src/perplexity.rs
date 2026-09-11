//! `ferrox perplexity` — corpus perplexity, llama.cpp's `perplexity` tool.
//!
//! # Why this exists
//!
//! Every quality claim in this repo was, until this command, either a
//! token-for-token comparison (`ferrox verify`) or a single-position
//! distribution comparison (`ferrox parity`). Neither answers the
//! question a quantization change actually raises: *is this file worse,
//! and by how much*. A K-quant encoder that rounds badly produces a GGUF
//! that loads, tokenizes, and generates fluent text; the only cheap
//! signal that it is worse is a corpus perplexity that moved.
//!
//! So the number here is not for bragging. It is an acceptance test with
//! a scale.
//!
//! # The method, and why it is copied rather than invented
//!
//! A perplexity is only comparable to another perplexity computed the
//! same way. Change the stride, the scored fraction, or whether a BOS is
//! injected, and the number still looks like a perplexity while being
//! incomparable to every published figure — which is worse than having
//! no command, because someone will compare it anyway.
//!
//! This follows `.scratch/llama.cpp/tools/perplexity/perplexity.cpp`,
//! function `perplexity()` (the `ppl_stride == 0` path, which is the
//! default), read at ggml commit present in that checkout:
//!
//! - **Tokenize the whole file once**, with the checkpoint's BOS added
//!   at the front if the checkpoint adds one
//!   (`common_tokenize(ctx, prompt, /*add_special=*/true)`).
//! - **Refuse below `2 * n_ctx` tokens.** Upstream errors out with
//!   exactly that bound.
//! - **Non-overlapping windows of `n_ctx`.** The stride *is* `n_ctx`;
//!   `n_chunk_max = n_tokens / n_ctx`. (Upstream's `--ppl-stride` picks
//!   a different, overlapping algorithm — `perplexity_v2` — and is not
//!   implemented here. See "Deviations".)
//! - **The first token of every window is overwritten with BOS**, when
//!   the checkpoint adds one. The KV cache is cleared between windows,
//!   so each window starts from nothing.
//! - **Only the second half is scored.** `first = n_ctx / 2`, and the
//!   logits at window positions `first ..= n_ctx - 2` are scored against
//!   the tokens at `first + 1 ..= n_ctx - 1`. That is `n_ctx - first - 1`
//!   scored tokens per window — 255 at the default `n_ctx = 512`, not
//!   256, because the last position has no successor inside the window.
//!   The first half exists only to give the model context.
//! - **Natural log throughout.** `nll` sums `-log_softmax(logits)[target]`
//!   in nats and the report is `exp(nll / count)` — an *unweighted mean
//!   over scored tokens*, pooled across windows, not a mean of per-window
//!   perplexities.
//! - **`+/-` is the standard error of that mean, pushed through `exp`**:
//!   `sqrt((E[v²] - E[v]²) / (count - 1)) * ppl`. Upstream computes it
//!   from the same two running sums.
//!
//! `n_ctx` defaults to 512 here because upstream's `main` sets
//! `params.n_ctx = 512` before parsing argv, overriding the 4096 that
//! every other llama.cpp tool defaults to. A perplexity quoted without
//! its context length is meaningless, so the value is printed.
//!
//! # Deviations from upstream, stated because they are the whole risk
//!
//! 1. **No `n_batch` split and no parallel sequences.** Upstream cuts a
//!    window into `n_batch`-sized decodes, and packs `n_batch / n_ctx`
//!    windows into one batch as separate sequences. Here one window is
//!    one [`Decoder::forward_batch`] call. Attention is causal and the
//!    KV cache is cleared per window either way, so every scored
//!    position sees exactly the same context; only the grouping of f32
//!    reductions differs, at the same scale as the accumulation-order
//!    noise `ferrox parity` already calls a match.
//! 2. **The output head runs at every position, not only the scored
//!    ones.** Upstream sets `batch.logits[idx] = pos >= first`, so it
//!    projects roughly half as many rows to vocabulary. That is a cost
//!    difference, not a numeric one — and it is why this command holds
//!    `n_ctx * n_vocab` f32s at once (100 MB at SmolLM2's 49k vocab,
//!    260 MB at Llama-3's 128k).
//! 3. **`--ppl-stride` / `perplexity_v2`, HellaSwag, WinoGrande,
//!    multiple-choice and KL-divergence are not implemented.** None of
//!    them shares this code path upstream either.
//! 4. **Tokenization is ferrox's.** Upstream tokenizes with libllama.
//!    The evidence that these agree is `ferrox parity`'s tokenizer half,
//!    which compares the two on every local checkpoint libllama can
//!    load. Special-token markup in the corpus is *not* parsed on either
//!    side (upstream leaves `parse_special` false).

use anyhow::Context;
use ferrox_core::cache::KvCache;
use ferrox_gguf::ShardedGguf;
use std::path::Path;

/// llama.cpp's `perplexity` sets `params.n_ctx = 512` before argv is
/// parsed, unlike every other tool in that repo. Matching it is what
/// makes a ferrox number comparable to a published one without both
/// sides having to name the context length.
pub const DEFAULT_CTX: usize = 512;

pub struct PerplexityArgs {
    pub model: String,
    /// Plain-text corpus, read whole. `-f` upstream.
    pub file: String,
    pub ctx_size: usize,
    /// Stop after this many windows. `--chunks` upstream.
    pub chunks: Option<usize>,
}

/// How the corpus is cut up. Computed from the token count alone, before
/// any weights are touched, so an impossible request fails in
/// milliseconds instead of after a model load.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Window length. Upstream's `n_ctx`.
    n_ctx: usize,
    /// First window position whose logits are scored. Upstream's `first`.
    first: usize,
    /// Scored tokens per window: `n_ctx - first - 1`.
    per_window: usize,
    /// Windows actually run.
    n_window: usize,
}

impl Plan {
    fn scored_total(&self) -> usize {
        self.per_window * self.n_window
    }

    /// Where window `i` starts in the token stream.
    ///
    /// This is the STRIDE, and it lives here rather than inline in
    /// [`run`] because a stride is invisible to a test that only checks
    /// how many windows there are: halving it doubles the overlap,
    /// scores every token twice, and still produces the same window
    /// count from `plan`. Sabotaging the inline version turned nothing
    /// red, which is how it ended up as a method with a test of its own.
    ///
    /// Upstream: `const int start = i * n_ctx;`.
    pub(crate) fn window_start(&self, window: usize) -> usize {
        window * self.n_ctx
    }

    /// The window positions whose logits are scored, as an inclusive
    /// range over `first ..= n_ctx - 2`. The target for position `p` is
    /// the token at `p + 1`, which is why the last position is excluded.
    pub(crate) fn scored_positions(&self) -> std::ops::Range<usize> {
        self.first..self.first + self.per_window
    }
}

/// Cut `n_tokens` into llama.cpp's windows, or say why it cannot be done.
///
/// Every refusal here names both numbers, because "not enough tokens" on
/// its own sends the user off to guess which of the two to change.
pub(crate) fn plan(n_tokens: usize, n_ctx: usize, chunks: Option<usize>) -> anyhow::Result<Plan> {
    // Upstream refuses `n_ctx <= 0`; the extra bound is `per_window >= 1`,
    // which `n_ctx = 2` violates (first = 1, and position 1 has no
    // successor inside the window). Without it the mean divides by zero
    // and reports NaN as if it were a measurement.
    let first = n_ctx / 2;
    let per_window = n_ctx.saturating_sub(first).saturating_sub(1);
    if per_window == 0 {
        anyhow::bail!(
            "--ctx-size {n_ctx} scores no tokens: llama.cpp scores window positions \
             {first}..{} against their successors, which is empty below --ctx-size 3",
            n_ctx.saturating_sub(1)
        );
    }
    // The `2 * n_ctx` floor is upstream's, verbatim: with one window the
    // "second half only" rule has nothing to average over and the
    // standard error is undefined.
    if n_tokens < 2 * n_ctx {
        anyhow::bail!(
            "corpus is {n_tokens} tokens; llama.cpp needs at least 2 * --ctx-size = {} \
             to evaluate perplexity at --ctx-size {n_ctx}. Use a longer corpus or a \
             smaller --ctx-size",
            2 * n_ctx
        );
    }
    let n_window_max = n_tokens / n_ctx;
    let n_window = match chunks {
        Some(0) => anyhow::bail!("--chunks 0 would score nothing"),
        Some(want) => want.min(n_window_max),
        None => n_window_max,
    };
    Ok(Plan {
        n_ctx,
        first,
        per_window,
        n_window,
    })
}

/// `-log softmax(logits)[token]`, in nats.
///
/// Deliberately mirrors llama.cpp's `log_softmax()` term by term,
/// including where it narrows to f32 and where it widens to f64: the
/// exponentials are `expf` on f32 differences and are summed in a
/// double, and `logits[tok] - max_logit` is an f32 subtraction before it
/// meets the f64 `log(sum_exp)`. Doing the whole thing in f64 is more
/// accurate and would put a systematic bias between the two engines'
/// numbers, which is the one thing this command exists not to have.
pub(crate) fn nll_nats(logits: &[f32], token: usize) -> anyhow::Result<f64> {
    let logit = *logits.get(token).with_context(|| {
        format!(
            "token id {token} is outside this model's {} logits",
            logits.len()
        )
    })?;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f64;
    for &l in logits {
        sum_exp += (l - max).exp() as f64;
    }
    Ok(-((logit - max) as f64 - sum_exp.ln()))
}

/// The running estimate: llama.cpp's `nll`, `nll2` and `count`.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Estimate {
    nll: f64,
    nll2: f64,
    count: usize,
}

impl Estimate {
    fn observe(&mut self, nll: f64) {
        self.nll += nll;
        self.nll2 += nll * nll;
        self.count += 1;
    }

    /// `exp(mean nll)`. `None` before anything has been scored, so a
    /// caller cannot print `NaN` and have it read as a measurement.
    pub(crate) fn ppl(&self) -> Option<f64> {
        (self.count > 0).then(|| (self.nll / self.count as f64).exp())
    }

    /// llama.cpp's `+/-`: the standard error of the mean nll, pushed
    /// through `exp` by multiplying by the perplexity itself.
    ///
    /// `None` when upstream also declines to print one — fewer than two
    /// observations, or a variance that came out non-positive (which
    /// upstream reports as "Unexpected negative standard deviation").
    pub(crate) fn stderr(&self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        let n = self.count as f64;
        let mean = self.nll / n;
        let var = self.nll2 / n - mean * mean;
        if var <= 0.0 {
            return None;
        }
        Some((var / (n - 1.0)).sqrt() * mean.exp())
    }
}

pub fn run(args: PerplexityArgs) -> anyhow::Result<()> {
    let path = crate::pull::resolve_model_path(&args.model)?;
    let text = std::fs::read_to_string(&args.file)
        .with_context(|| format!("reading corpus {}", args.file))?;

    // Loading and tokenizing go through the same helper `ferrox verify`
    // and `ferrox parity` use, so the tokenizer this number is computed
    // over is the one `parity` proves against llama.cpp. A second
    // spelling of "encode, then add BOS if the checkpoint says to" is
    // exactly the drift that makes two numbers incomparable.
    let (decoder, tokens, _eos) = crate::verify_engine::load_and_tokenize(
        Path::new(&path),
        &text,
        // llama-perplexity tokenizes its corpus with
        // `common_tokenize(ctx, params.prompt, true)`, whose
        // `parse_special` defaults to false.
        ferrox_models::tokenizer::SpecialTokens::AsText,
        None,
    )
    .context("loading the model and tokenizing the corpus")?;

    // The per-window BOS reset needs the id itself, and needs to know
    // whether this checkpoint uses one at all. Same predicate as the
    // tokenizer above — called, not restated.
    let file = ShardedGguf::open(&path)?;
    let bos = if ferrox_models::tokenizer::should_add_bos_token(&file) {
        file.metadata_u64("tokenizer.ggml.bos_token_id")
            .map(|v| v as usize)
    } else {
        None
    };
    drop(file);

    let plan = plan(tokens.len(), args.ctx_size, args.chunks)?;

    println!(
        "perplexity: {} tokens, n_ctx = {}, {} windows, scoring {} tokens each ({} total)",
        tokens.len(),
        plan.n_ctx,
        plan.n_window,
        plan.per_window,
        plan.scored_total()
    );
    println!(
        "perplexity: window positions {}..{} are scored; BOS reset per window: {}",
        plan.first,
        plan.n_ctx - 1,
        match bos {
            Some(b) => format!("yes (id {b})"),
            None => "no (this checkpoint does not add BOS)".to_string(),
        }
    );

    let mut est = Estimate::default();
    for window in 0..plan.n_window {
        let start = plan.window_start(window);
        let mut ids = tokens[start..start + plan.n_ctx].to_vec();
        if let Some(b) = bos {
            ids[0] = b;
        }

        // A fresh cache per window is llama.cpp's `llama_memory_clear`:
        // windows are independent, and a carried-over cache would let
        // window N attend over window N-1 and quietly lower the number.
        let mut caches: Vec<KvCache> = (0..decoder.layers.len())
            .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
            .collect();
        let logits = decoder.forward_batch(&ids, 0, &mut caches);
        anyhow::ensure!(
            logits.len() == plan.n_ctx,
            "forward_batch returned {} logit rows for a {}-token window",
            logits.len(),
            plan.n_ctx
        );

        for pos in plan.scored_positions() {
            // The target comes from the ORIGINAL token stream, not from
            // the BOS-substituted copy — upstream restores the token it
            // overwrote before scoring for the same reason.
            est.observe(nll_nats(&logits[pos], tokens[start + pos + 1])?);
        }

        // llama.cpp prints `[n]ppl,` per chunk. Same shape, so a run of
        // both can be diffed line for line.
        match est.ppl() {
            Some(p) => println!("[{}]{p:.4}", window + 1),
            None => unreachable!("a window always scores at least one token"),
        }
    }

    let ppl = est
        .ppl()
        .context("no tokens were scored, so there is no perplexity to report")?;
    match est.stderr() {
        Some(se) => println!("Final estimate: PPL = {ppl:.4} +/- {se:.5}"),
        None => println!("Final estimate: PPL = {ppl:.4} (no standard error: too few tokens)"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{nll_nats, plan, Estimate, DEFAULT_CTX};

    /// The corpus fixture has to survive in the tree AND stay long
    /// enough for the default context: `plan` refuses below `2 * n_ctx`
    /// tokens, and a fixture trimmed below that would turn every
    /// perplexity run into a refusal with nothing to say it used to
    /// work. Bytes are a proxy for tokens here, and a deliberately
    /// generous one: English averages well under 5 bytes per token on
    /// every tokenizer in this repo, so 2 * 512 * 5 bytes is a floor no
    /// real tokenizer can fall through.
    #[test]
    fn the_committed_corpus_is_long_enough_for_the_default_context() {
        const CORPUS: &str = include_str!("../tests/corpus/alice-ch1-2.txt");
        let floor = 2 * DEFAULT_CTX * 5;
        assert!(
            CORPUS.len() >= floor,
            "corpus is {} bytes, below the {floor} that guarantees 2 * {DEFAULT_CTX} tokens",
            CORPUS.len()
        );
    }

    /// The scored window is llama.cpp's, and getting it wrong is the
    /// failure this whole command is written to avoid: a number that
    /// looks like a perplexity and is not comparable to any published
    /// one. 512 scores 255 tokens starting at position 256 — not 256,
    /// because the last position in the window has no successor in it.
    #[test]
    fn the_default_window_scores_the_second_half_minus_one() {
        let p = plan(4096, DEFAULT_CTX, None).unwrap();
        assert_eq!(p.first, 256);
        assert_eq!(p.per_window, 255);
        assert_eq!(p.n_window, 8);
        assert_eq!(p.scored_total(), 2040);
    }

    /// Windows do not overlap: the stride is the full context. A stride
    /// of half the context would score every token twice and report a
    /// different, lower-variance number under the same name.
    ///
    /// The window COUNT is checked here and the window OFFSETS in the
    /// test below, because they fail independently: this assertion
    /// survived a sabotage that halved the stride, since halving a
    /// stride does not change how many windows `plan` reports.
    #[test]
    fn windows_tile_the_corpus_without_overlap() {
        // 5 * 64 = 320 tokens, 63 spare: 5 windows, not 6 and not 9.
        let p = plan(383, 64, None).unwrap();
        assert_eq!(p.n_window, 5);
        assert_eq!(p.n_ctx, 64);
        assert_eq!(p.per_window, 31);
    }

    /// The stride itself: window `i` starts at `i * n_ctx`, so
    /// consecutive windows abut exactly — no gap, no overlap, and the
    /// last one ends within the corpus.
    #[test]
    fn consecutive_windows_abut_exactly_and_stay_inside_the_corpus() {
        let n_tokens = 383;
        let p = plan(n_tokens, 64, None).unwrap();
        assert_eq!(p.window_start(0), 0);
        for w in 0..p.n_window {
            let start = p.window_start(w);
            assert_eq!(
                start % p.n_ctx,
                0,
                "window {w} starts mid-stride at {start}"
            );
            if w > 0 {
                // Abutting: the previous window's end IS this start.
                assert_eq!(p.window_start(w - 1) + p.n_ctx, start);
            }
            assert!(
                start + p.n_ctx <= n_tokens,
                "window {w} runs past the corpus"
            );
        }
    }

    /// The scored positions, spelled out, because "the second half" is
    /// off by one from what the phrase suggests: the last position in a
    /// window has no successor inside it, so it cannot be scored, and
    /// every target sits strictly after its own position.
    #[test]
    fn every_scored_position_has_its_successor_inside_the_window() {
        let p = plan(4096, DEFAULT_CTX, None).unwrap();
        let scored: Vec<usize> = p.scored_positions().collect();
        assert_eq!(scored.first().copied(), Some(256));
        assert_eq!(scored.last().copied(), Some(510));
        assert_eq!(scored.len(), p.per_window);
        for pos in scored {
            assert!(pos + 1 < p.n_ctx, "position {pos} has no target");
        }
    }

    #[test]
    fn odd_context_sizes_round_the_scored_half_down() {
        // first = 7/2 = 3, so positions 3..=5 are scored: 3 tokens.
        let p = plan(64, 7, None).unwrap();
        assert_eq!(p.first, 3);
        assert_eq!(p.per_window, 3);
    }

    /// Upstream's floor, kept verbatim. Below it there is one window,
    /// and a single window's "second half" has no second window to
    /// average against.
    #[test]
    fn a_corpus_shorter_than_two_contexts_is_refused() {
        let err = plan(1023, 512, None).unwrap_err().to_string();
        assert!(err.contains("1023"), "{err}");
        assert!(err.contains("1024"), "{err}");
        assert!(plan(1024, 512, None).is_ok());
    }

    /// `--ctx-size 2` scores nothing: `first` is 1 and position 1 has no
    /// successor. Without this refusal the mean divides by zero and the
    /// command prints NaN as though it had measured something.
    #[test]
    fn a_context_that_scores_nothing_is_refused_rather_than_reported_as_nan() {
        assert!(plan(100_000, 2, None).is_err());
        assert!(plan(100_000, 1, None).is_err());
        assert!(plan(100_000, 3, None).is_ok());
    }

    #[test]
    fn chunks_clamps_down_but_never_up_and_zero_is_refused() {
        assert_eq!(plan(4096, 512, Some(3)).unwrap().n_window, 3);
        // Asking for more windows than the corpus holds must not invent
        // them: upstream takes min(n_chunks, n_chunk_max).
        assert_eq!(plan(4096, 512, Some(99)).unwrap().n_window, 8);
        assert!(plan(4096, 512, Some(0)).is_err());
    }

    /// A uniform distribution over `v` tokens has `-log p = ln(v)` for
    /// every target, so the perplexity of a corpus the model is
    /// completely ignorant of is the vocabulary size. This pins the log
    /// BASE: in base 2 the same input would give 2.0 for a 4-way tie.
    #[test]
    fn a_uniform_distribution_costs_the_natural_log_of_the_vocabulary() {
        let flat = vec![0.0f32; 4];
        for token in 0..4 {
            let v = nll_nats(&flat, token).unwrap();
            assert!((v - 4.0f64.ln()).abs() < 1e-9, "{v}");
        }
        let mut est = Estimate::default();
        for token in 0..4 {
            est.observe(nll_nats(&flat, token).unwrap());
        }
        assert!((est.ppl().unwrap() - 4.0).abs() < 1e-9);
    }

    /// The softmax is shift-invariant, and llama.cpp leans on that by
    /// subtracting the max logit before exponentiating. A constant
    /// offset must not move the answer, or a model whose logits happen
    /// to sit high would score differently from the same model shifted
    /// down.
    ///
    /// The offset is +200 on purpose. `expf(200)` is `+inf` in f32, so
    /// dropping the max subtraction makes this NaN — at +40 the
    /// unstable version still lands on a finite, correct-looking answer
    /// and the test passes while the guard is gone.
    #[test]
    fn adding_a_constant_to_every_logit_does_not_move_the_score() {
        let a = [1.0f32, -2.0, 0.5, 3.25, -0.75];
        let b: Vec<f32> = a.iter().map(|v| v + 200.0).collect();
        for token in 0..a.len() {
            let (x, y) = (nll_nats(&a, token).unwrap(), nll_nats(&b, token).unwrap());
            assert!(y.is_finite(), "token {token}: shifted score is {y}");
            assert!((x - y).abs() < 1e-6, "token {token}: {x} vs {y}");
        }
    }

    /// A confident, correct prediction costs almost nothing; a confident
    /// wrong one costs a lot. Directionality is worth a test because a
    /// dropped minus sign still produces a plausible-looking positive
    /// perplexity.
    #[test]
    fn confidence_in_the_right_token_costs_less_than_confidence_in_the_wrong_one() {
        let logits = [10.0f32, 0.0, 0.0];
        let right = nll_nats(&logits, 0).unwrap();
        let wrong = nll_nats(&logits, 1).unwrap();
        assert!(right > 0.0 && right < 0.01, "{right}");
        assert!(wrong > 9.0, "{wrong}");
    }

    /// The report is `exp` of the mean of the per-token nll, NOT the
    /// mean of the per-token perplexities and NOT the mean of per-window
    /// perplexities. Those three differ by Jensen's inequality, and only
    /// the first is what "perplexity" means.
    #[test]
    fn the_estimate_exponentiates_the_mean_rather_than_averaging_exponentials() {
        let mut est = Estimate::default();
        est.observe(0.0);
        est.observe(4.0);
        let geometric = (2.0f64).exp();
        assert!((est.ppl().unwrap() - geometric).abs() < 1e-12);
        let arithmetic = (0.0f64.exp() + 4.0f64.exp()) / 2.0;
        assert!((est.ppl().unwrap() - arithmetic).abs() > 1.0);
    }

    /// llama.cpp's `+/-`, term for term: the standard error of the mean
    /// nll times the perplexity. Checked against the closed form on a
    /// three-sample set rather than against a recomputation of the same
    /// expression.
    #[test]
    fn the_standard_error_is_the_error_of_the_mean_pushed_through_exp() {
        let mut est = Estimate::default();
        for v in [1.0, 2.0, 3.0] {
            est.observe(v);
        }
        // mean 2, population variance 2/3, /(n-1) = 1/3, sqrt = 0.57735…
        let expected = (2.0f64 / 3.0 / 2.0).sqrt() * 2.0f64.exp();
        assert!((est.stderr().unwrap() - expected).abs() < 1e-12, "{est:?}");
    }

    /// Zero and one observations have no standard error, and a variance
    /// that lands at exactly zero (every token equally surprising) has
    /// none either. Upstream declines to print one in all three cases;
    /// printing 0.00000 instead would read as a perfectly precise
    /// measurement.
    #[test]
    fn too_few_or_identical_observations_report_no_standard_error() {
        assert_eq!(Estimate::default().ppl(), None);
        assert_eq!(Estimate::default().stderr(), None);
        let mut one = Estimate::default();
        one.observe(1.5);
        assert!(one.ppl().is_some());
        assert_eq!(one.stderr(), None);
        let mut flat = Estimate::default();
        flat.observe(1.5);
        flat.observe(1.5);
        assert_eq!(flat.stderr(), None);
    }

    /// A target id past the end of the logit row is a corpus/model
    /// mismatch, not something to index into. It must name the id, since
    /// the only way it happens is a tokenizer emitting ids the head does
    /// not cover.
    #[test]
    fn a_target_outside_the_logit_row_is_refused_by_name() {
        let err = nll_nats(&[0.0, 1.0], 7).unwrap_err().to_string();
        assert!(err.contains('7'), "{err}");
    }
}
