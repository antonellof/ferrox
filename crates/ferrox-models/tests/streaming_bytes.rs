//! Does a streamed expert see the same bytes as a resident one?
//!
//! WRITTEN AGAINST A LIVE BUG, KEPT AS THE REGRESSION TEST FOR IT.
//!
//! Expert streaming used to answer " amongst amongst, and of" where
//! resident answered " Paris." on OLMoE-1B-7B Q4_0, deterministically,
//! while the synthetic-fixture test pinned the two as bit-identical. So
//! this asks the narrower question against a REAL checkpoint: are the
//! weight bytes themselves the same, or is the difference downstream of
//! them?
//!
//! It passes now, three consecutive runs, and the fixture test that
//! missed it still passes too. That is the point of keeping it: a
//! synthetic fixture agreed while a real checkpoint did not, so the
//! fixture alone was never enough evidence for this path.
//!
//! `#[ignore]`d because it needs a multi-gigabyte model on disk.
//! `cargo test -p ferrox-models --test streaming_bytes -- --ignored`
use ferrox_models::{config::ModelConfig, Decoder};

/// Workspace-relative: `cargo test -p` runs with the CRATE as cwd, so
/// a bare "models/..." path silently skips instead of running.
const MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/olmoe-1b-7b-0924-q4_0.gguf"
);

#[test]
#[ignore]
fn a_streamed_expert_sees_the_same_bytes_as_a_resident_one() {
    let path = std::path::Path::new(MODEL);
    if !path.exists() {
        eprintln!("{MODEL} not present, skipping");
        return;
    }
    let file = ferrox_gguf::ShardedGguf::open(MODEL).expect("open");
    let cfg = ModelConfig::from_gguf(&file).expect("config");
    let resident = Decoder::from_gguf(MODEL, cfg.clone()).expect("resident load");
    let streamed =
        Decoder::from_gguf_with_expert_cache(MODEL, cfg, Some(128 << 20)).expect("streamed load");

    let mut checked = 0usize;
    for li in [0usize, 1, 7] {
        for e in [0usize, 1, 5] {
            let (rg, ru, rd) = resident.layers[li].moe.with_expert(e, |x| {
                (x.gate.bytes_len(), x.up.bytes_len(), x.down.bytes_len())
            });
            let (sg, su, sd) = streamed.layers[li].moe.with_expert(e, |x| {
                (x.gate.bytes_len(), x.up.bytes_len(), x.down.bytes_len())
            });
            assert_eq!(
                (rg, ru, rd),
                (sg, su, sd),
                "layer {li} expert {e}: byte LENGTHS differ"
            );

            let same = resident.layers[li].moe.with_expert(e, |r| {
                streamed.layers[li].moe.with_expert(e, |s| {
                    r.gate.bytes_eq(&s.gate) && r.up.bytes_eq(&s.up) && r.down.bytes_eq(&s.down)
                })
            });
            assert!(
                same,
                "layer {li} expert {e}: weight BYTES differ between backings"
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}

/// Streamed and resident must produce the SAME TOKENS on a real
/// checkpoint, with the int-dot path on, which is the default.
///
/// This is the test that was missing. The synthetic-fixture test pins
/// streaming as bit-identical and PASSES even at a 1-byte budget, but
/// its buffers never collide in the repack cache, so it could not see
/// the bug: the cache keys on a buffer ADDRESS, and an expert store
/// recycles buffers, so two experts landed at one address and one was
/// served the other's repacked bytes.
///
/// The symptom was not subtle. On OLMoE-1B-7B Q4_0, "The capital of
/// France is" answered " Paris." resident and " amongst amongst, and
/// of" streamed, deterministically at temperature 0.
#[test]
#[ignore]
fn streamed_and_resident_agree_on_a_real_checkpoint() {
    let path = std::path::Path::new(MODEL);
    if !path.exists() {
        eprintln!("{MODEL} not present, skipping");
        return;
    }
    let file = ferrox_gguf::ShardedGguf::open(MODEL).expect("open");
    let cfg = ModelConfig::from_gguf(&file).expect("config");
    let resident = Decoder::from_gguf(MODEL, cfg.clone()).expect("resident load");

    // FOUR prompts, not one. The bug this test was written for was
    // deterministic, so a single passing prompt proves very little: the
    // synthetic fixture test also passed while the real checkpoint
    // answered " amongst amongst, and of" instead of " Paris.". Diverse
    // shapes (prose, a story opener, code, a list) exercise different
    // expert routes, which is what a streamed cache actually varies.
    for prompt in [
        vec![791usize, 6864, 315, 9822, 374],
        vec![12805usize, 5304, 264, 892],
        vec![755usize, 16178, 41160, 1471],
        vec![791usize, 2380, 6156, 8146, 527],
    ] {
        let ids_of = |d: &Decoder| -> Vec<usize> {
            let mut caches: Vec<ferrox_core::cache::KvCache> = d.config.new_kv_caches();
            // PREFILL the prompt as a batch, then decode. The CLI does
            // this, and it matters: prefill and decode take different MoE
            // paths, and a decode-only test does not reproduce the repack
            // cache collision at all.
            let mut out = Vec::new();
            let batch = d.forward_batch(&prompt, 0, &mut caches);
            out.push(argmax(
                batch.last().expect("prefill returns one row per token"),
            ));
            for pos in (prompt.len()..).take(4) {
                let last = *out.last().unwrap();
                let logits = d.forward_token(last, pos, &mut caches);
                out.push(argmax(&logits));
            }
            out
        };

        let want = ids_of(&resident);
        // A generous budget and a 1-byte one: the second forces every
        // acquire to miss, which is when buffers are recycled hardest.
        for budget in [128u64 << 20, 1] {
            let streamed = Decoder::from_gguf_with_expert_cache(MODEL, cfg.clone(), Some(budget))
                .expect("streamed load");
            assert_eq!(
                ids_of(&streamed),
                want,
                "budget={budget}: streamed experts must produce the same tokens as resident"
            );
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}
