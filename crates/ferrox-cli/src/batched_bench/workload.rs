//! One `llama-batched-bench` row: the workload of
//! `tools/batched-bench/batched-bench.cpp:145-235`, driven through the
//! seams the continuous batcher uses, with every `bench_guard` check
//! `ferrox bench` applies to its own rows.
//!
//! Upstream runs each combination once after a global 16-token warmup
//! (`:110-122`). Here each combination runs `WARMUP_REPS` untimed
//! passes and then one timed pass, and the two must feed the same
//! tokens and produce the same greedy picks. That costs a second pass
//! per row and buys the determinism assertion `ferrox bench` has: a
//! single pass has nothing to disagree with.

use crate::bench_guard::{self, WorkloadDigest};
use crate::bench_model::{check_result, decode_tokens, fresh_caches, probe, synthetic_tokens};
use ferrox_core::cache::KvCache;
use ferrox_models::Decoder;
use std::time::Instant;

/// One `(PP, TG, B)` cell of the sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Combo {
    pub pp: usize,
    pub tg: usize,
    pub pl: usize,
}

impl Combo {
    /// `batched-bench.cpp:139` with `kv_unified = false` (the `-kvu`
    /// arm is refused): a shared prompt is copied into every sequence,
    /// so both branches come to `pl * (pp + tg)`. Written as upstream
    /// writes it so the two can be read line for line.
    pub fn n_kv(&self, pp_shared: bool) -> usize {
        let Combo { pp, tg, pl } = *self;
        if pp_shared {
            pl * pp + pl * tg
        } else {
            pl * (pp + tg)
        }
    }

    pub fn label(&self) -> String {
        format!("pp{} tg{} pl{}", self.pp, self.tg, self.pl)
    }
}

/// The run-wide switches that change what a row does.
#[derive(Debug, Clone, Copy)]
pub(super) struct Shape {
    /// `-pps`: prefill one prompt and copy its KV to every sequence
    /// (`:147,168-171`).
    pub pp_shared: bool,
    /// `-tgs`: `0 0 0 ... 1 1 1 ...` instead of `0123 0123 ...`
    /// (`:189-223`).
    pub tg_separate: bool,
    /// Prompt tokens per forward call (`-ub`).
    pub ubatch: usize,
}

/// A measured row, in the units upstream prints (`:229-235`).
#[derive(Debug, Clone)]
pub(super) struct Measured {
    pub combo: Combo,
    pub n_kv: usize,
    pub t_pp: f64,
    pub speed_pp: f64,
    pub t_tg: f64,
    pub speed_tg: f64,
    pub t: f64,
    pub speed: f64,
    /// Digest of every token fed inside the two timed regions, prompt
    /// then decode, so two receipts for the same row can be shown to
    /// have measured the same work.
    pub digest: WorkloadDigest,
}

/// Runs one combination: `WARMUP_REPS` untimed passes, one timed.
pub(super) fn measure(decoder: &Decoder, combo: Combo, shape: &Shape) -> anyhow::Result<Measured> {
    let Combo { pp, tg, pl } = combo;
    let label = combo.label();
    let vocab = decoder.config.vocab_size;

    // `:147`: one prompt when shared, else one per sequence. Built and
    // checked BEFORE the clock, like every stream `ferrox bench` feeds.
    let n_prompts = if shape.pp_shared { 1 } else { pl };
    let prompts: Vec<Vec<usize>> = (0..n_prompts)
        .map(|s| synthetic_tokens(vocab, pp, s))
        .collect();
    for (s, prompt) in prompts.iter().enumerate() {
        bench_guard::check_prompt_before(&format!("{label} prompt {s}"), pp, prompt, vocab)?;
    }
    let gens: Vec<Vec<usize>> = (0..pl).map(|s| decode_tokens(vocab, tg, s)).collect();
    for (s, gen) in gens.iter().enumerate() {
        bench_guard::check_prompt_before(&format!("{label} decode {s}"), tg, gen, vocab)?;
    }

    let mut first_digest: Option<WorkloadDigest> = None;
    let mut first_pp: Vec<Option<(usize, f32)>> = vec![None; n_prompts];
    let mut first_tg: Vec<Option<(usize, f32)>> = vec![None; pl];
    let mut samples_pp = Vec::with_capacity(1);
    let mut samples_tg = Vec::with_capacity(1);

    for rep in 0..1 + bench_guard::WARMUP_REPS {
        // `:153` clears the memory; here every sequence starts from a
        // cache that is asserted cold rather than assumed so.
        let mut caches: Vec<Vec<KvCache>> = (0..n_prompts).map(|_| fresh_caches(decoder)).collect();
        for c in &caches {
            bench_guard::check_caches_cold(&label, rep, &probe(c))?;
        }
        let mut digest = WorkloadDigest::new();

        // `:155-166`: the prompt phase. Upstream puts every sequence's
        // prompt in one batch and lets `n_ubatch` slice it; ferrox has
        // no cross-sequence prefill, so each prompt is fed in `-ub`
        // chunks through the call the batcher makes for an admitted
        // row (`serving/batch/prefill.rs:165-171`). The host-KV
        // variant is deliberate: `forward_multi_seq` attends over host
        // rows, and on Metal the plain variant leaves them zeroed.
        let mut pp_logits: Vec<Vec<f32>> = Vec::with_capacity(n_prompts);
        let t = Instant::now();
        for (s, prompt) in prompts.iter().enumerate() {
            let mut logits = Vec::new();
            let mut pos = 0;
            for chunk in prompt.chunks(shape.ubatch) {
                digest.feed_all(chunk);
                logits = decoder.forward_batch_last_host_kv(chunk, pos, &mut caches[s]);
                pos += chunk.len();
            }
            pp_logits.push(logits);
        }
        let dt_pp = t.elapsed().as_secs_f64();

        for c in &caches {
            bench_guard::check_prefill_after(&label, pp, &probe(c))?;
        }
        for (s, logits) in pp_logits.iter().enumerate() {
            check_result(
                &format!("{label} prompt {s}"),
                rep,
                logits,
                &mut first_pp[s],
            )?;
        }

        // `:168-185`: a shared prompt is copied to the other sequences
        // between the two timers. Upstream's `llama_memory_seq_cp` is a
        // KV clone here, and the clones are re-checked: a copy that
        // came back short would make every decode step attend over
        // less than the prompt.
        if shape.pp_shared {
            let clones: Vec<Vec<KvCache>> = (1..pl).map(|_| caches[0].clone()).collect();
            caches.extend(clones);
            for c in &caches {
                bench_guard::check_prefill_after(&label, pp, &probe(c))?;
            }
        }
        anyhow::ensure!(
            caches.len() == pl,
            "{label}: {} sequences hold KV for a {pl}-sequence row",
            caches.len()
        );

        // `:187-225`: the decode phase, through the batcher's step.
        let mut tg_logits: Vec<Vec<f32>> = vec![Vec::new(); pl];
        let t = Instant::now();
        if shape.tg_separate {
            // `:189-205`: 0 0 0 ... 1 1 1 ... -- a one-sequence batch
            // per call, through the same entry point, so the two
            // patterns differ only in how the batch is grouped.
            for (j, gen) in gens.iter().enumerate() {
                for (i, &tok) in gen.iter().enumerate() {
                    digest.feed(tok);
                    let out = decoder.forward_multi_seq(&[tok], &[pp + i], &mut caches[j..=j]);
                    tg_logits[j] = out.into_iter().next().unwrap_or_default();
                }
            }
        } else {
            // `:207-223`: 0123 0123 ... -- one call per step across
            // every sequence, at each sequence's own position.
            let mut toks = vec![0usize; pl];
            let mut positions = vec![0usize; pl];
            for i in 0..tg {
                for (j, gen) in gens.iter().enumerate() {
                    toks[j] = gen[i];
                    positions[j] = pp + i;
                    digest.feed(gen[i]);
                }
                let out = decoder.forward_multi_seq(&toks, &positions, &mut caches);
                anyhow::ensure!(
                    out.len() == pl,
                    "{label} step {i}: forward_multi_seq returned {} logit rows for {pl} \
                     sequences",
                    out.len()
                );
                tg_logits = out;
            }
        }
        let dt_tg = t.elapsed().as_secs_f64();

        // The batched step pushes to and attends from the host cache on
        // every backend (`decoder/attn_block.rs`, `KvStep::Batched`), so
        // unlike `ferrox bench` the host cache is always the record and
        // the length check always runs.
        for c in &caches {
            bench_guard::check_decode_after(&label, pp, tg, &probe(c), true)?;
        }
        let first = *first_digest.get_or_insert(digest);
        bench_guard::check_same_workload(&label, rep, first, digest)?;
        for (s, logits) in tg_logits.iter().enumerate() {
            check_result(
                &format!("{label} decode {s}"),
                rep,
                logits,
                &mut first_tg[s],
            )?;
        }
        if rep >= bench_guard::WARMUP_REPS {
            samples_pp.push(dt_pp);
            samples_tg.push(dt_tg);
        }
    }

    // One timed pass per row is what upstream reports; the warmup
    // accounting guard says exactly one landed.
    bench_guard::check_timed_samples(&label, 1, samples_pp.len())?;
    bench_guard::check_timed_samples(&label, 1, samples_tg.len())?;
    let t_pp = samples_pp[0];
    let t_tg = samples_tg[0];
    let rates = rates(combo, shape.pp_shared, t_pp, t_tg);
    bench_guard::check_sample_rates(&label, &[rates.speed_pp, rates.speed_tg, rates.speed])?;

    Ok(Measured {
        combo,
        n_kv: combo.n_kv(shape.pp_shared),
        t_pp,
        speed_pp: rates.speed_pp,
        t_tg,
        speed_tg: rates.speed_tg,
        t: rates.t,
        speed: rates.speed,
        digest: first_digest.unwrap_or_default(),
    })
}

/// The three rates of `batched-bench.cpp:229-235`, separated from the
/// timing so the arithmetic can be pinned without a model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Rates {
    pub speed_pp: f64,
    pub speed_tg: f64,
    pub t: f64,
    pub speed: f64,
}

pub(super) fn rates(combo: Combo, pp_shared: bool, t_pp: f64, t_tg: f64) -> Rates {
    let Combo { pp, tg, pl } = combo;
    let t = t_pp + t_tg;
    // `:233`: a shared prompt is `pp` tokens of work, not `pl*pp`.
    let prompt_tokens = if pp_shared { pp } else { pl * pp };
    Rates {
        speed_pp: prompt_tokens as f64 / t_pp,
        // `:234`
        speed_tg: (pl * tg) as f64 / t_tg,
        t,
        // `:235`
        speed: (prompt_tokens + pl * tg) as f64 / t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `batched-bench.cpp:139`, both arms, with `-kvu` off: the two
    /// spellings agree, and the README's `N_KV = B*(PP+TG)` holds.
    #[test]
    fn n_kv_is_sequences_times_prompt_plus_generation_in_both_modes() {
        let c = Combo {
            pp: 128,
            tg: 64,
            pl: 4,
        };
        assert_eq!(c.n_kv(false), 4 * (128 + 64));
        assert_eq!(c.n_kv(true), 4 * 128 + 4 * 64);
    }

    /// `:233-235`: a shared prompt counts once in S_PP and S, and the
    /// decode rate is always over every sequence's tokens.
    #[test]
    fn rates_credit_a_shared_prompt_once_and_decode_across_every_sequence() {
        let c = Combo {
            pp: 100,
            tg: 10,
            pl: 4,
        };
        let separate = rates(c, false, 2.0, 1.0);
        assert_eq!(separate.speed_pp, 400.0 / 2.0);
        assert_eq!(separate.speed_tg, 40.0 / 1.0);
        assert_eq!(separate.t, 3.0);
        assert_eq!(separate.speed, 440.0 / 3.0);

        let shared = rates(c, true, 2.0, 1.0);
        assert_eq!(shared.speed_pp, 100.0 / 2.0);
        assert_eq!(shared.speed_tg, 40.0 / 1.0);
        assert_eq!(shared.speed, 140.0 / 3.0);
    }

    #[test]
    fn the_row_label_names_all_three_dimensions() {
        assert_eq!(
            Combo {
                pp: 8,
                tg: 4,
                pl: 2
            }
            .label(),
            "pp8 tg4 pl2"
        );
    }
}
