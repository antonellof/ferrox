//! The identity a device-resident activation carries, and the one rule
//! that decides whether a chain may consume it.
//!
//! This module holds no CUDA types on purpose. It is compiled on every
//! build, exactly like [`crate::mul_mm`]'s kernel text and scalar twin,
//! so the rule that keeps two activations from aliasing is exercised by
//! the plain `cargo test --workspace` run on a host with no GPU and no
//! `cuda` feature. The device half is [`crate::act_chain`].
//!
//! # Why this is not a length comparison
//!
//! `ferrox-metal`'s `take_resident_activation_if_matches` decides that
//! an incoming `&[f32]` is "the buffer already on the device" by
//! comparing LENGTHS. That is sound there only because exactly one site
//! ever sets the thread-local and it is cleared aggressively; it is a
//! discipline, not a guarantee, and it does not survive being copied.
//!
//! In a decoder every layer produces activations of the SAME length:
//! `hidden_dim` for every residual and every `o_proj` output,
//! `ffn_dim` for every gate and every up. A length comparison cannot
//! tell layer 3's residual from layer 11's, or this token's from the
//! previous token's. Two of them alias, the model answers WRONG, and
//! nothing reports anything -- which is this repo's worst failure mode,
//! strictly worse than being slow.
//!
//! llama.cpp does not have this problem because a tensor's device
//! buffer IS its identity, permanently: every intermediate is allocated
//! in a device buffer up front and the host never sees one. That is a
//! stronger guarantee than any comparison, and it is the shape this
//! module approximates: an activation is stamped, at the moment it is
//! created, with the process-unique id of the chain that created it,
//! and a chain refuses anything it did not stamp.
//!
//! The stamp is the whole scheme. It cannot be got wrong by a caller
//! because a caller cannot mint one: [`ChainId::mint`] is the only
//! constructor, `ferrox_cuda::act_chain::ActChain` is the only caller
//! of it, and `DeviceAct`'s device pointer is unreachable except
//! through [`check_same_chain`].

use std::sync::atomic::{AtomicU64, Ordering};

/// Process-unique identity of one activation chain.
///
/// Opaque and `Copy`. There is no `Default`, no `From<u64>` and no
/// arithmetic: the only way to obtain one is [`ChainId::mint`], so an
/// id that exists was minted by a chain that exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChainId(u64);

/// Ids start at 1 so that a zero can never be a valid id, which is what
/// makes the wrap assertion below meaningful.
static NEXT_CHAIN_ID: AtomicU64 = AtomicU64::new(1);

impl ChainId {
    /// Mints the next id. Monotonic, never reused within a process, and
    /// safe to call from any thread.
    ///
    /// A `u64` counter incremented once per chain cannot realistically
    /// wrap -- a chain is opened a few times per decoded token, so
    /// reaching 2^64 would take longer than the age of the universe --
    /// but "cannot realistically" is how reuse gets introduced, and a
    /// reused id is precisely the aliasing this type exists to prevent.
    /// So the impossible case is asserted rather than assumed.
    pub fn mint() -> ChainId {
        let raw = NEXT_CHAIN_ID.fetch_add(1, Ordering::Relaxed);
        assert_ne!(
            raw, 0,
            "ChainId counter wrapped: ids would be reused and two activations could alias"
        );
        ChainId(raw)
    }

    /// The raw counter value, for error messages and diagnostics only.
    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ChainId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// A device-resident activation was handed to a chain that did not
/// produce it.
///
/// This is a REFUSAL, not a fallback: the caller asked to compute
/// against a buffer whose contents this chain cannot vouch for, and the
/// only correct answer is to stop. Naming both ids makes the mistake
/// legible -- a stale activation from an earlier token shows up as a
/// much smaller `act` than `chain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "device activation belongs to chain #{act}, but was passed to chain #{chain}: \
     a device-resident activation may only be consumed by the chain that produced it"
)]
pub struct ForeignActivation {
    /// The chain that was asked to consume the activation.
    pub chain: u64,
    /// The chain that actually produced it.
    pub act: u64,
}

/// The entire identity rule, in one place so there is one of it.
///
/// Every operation that reads a device activation routes through this
/// (structurally -- see `DeviceAct::slice_for`, which is the only way
/// to reach the device pointer and takes a [`ChainId`] to do it).
pub fn check_same_chain(chain: ChainId, act: ChainId) -> Result<(), ForeignActivation> {
    if chain == act {
        Ok(())
    } else {
        Err(ForeignActivation {
            chain: chain.get(),
            act: act.get(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minted_ids_are_distinct() {
        let a = ChainId::mint();
        let b = ChainId::mint();
        let c = ChainId::mint();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn a_chain_accepts_its_own_activation() {
        let a = ChainId::mint();
        assert_eq!(check_same_chain(a, a), Ok(()));
    }

    /// The load-bearing one. Two chains are distinguished by identity
    /// and by NOTHING else: there is no length here, no pointer, no
    /// shape. A stale activation cannot pass by happening to be the
    /// same size, which is exactly how Metal's length-only match would
    /// have failed if it were copied here.
    #[test]
    fn two_chains_are_refused_even_though_nothing_else_distinguishes_them() {
        let older = ChainId::mint();
        let newer = ChainId::mint();
        assert_eq!(
            check_same_chain(newer, older),
            Err(ForeignActivation {
                chain: newer.get(),
                act: older.get(),
            }),
            "a chain must refuse an activation minted by a different chain"
        );
        // And symmetrically: a chain must not accept a FUTURE chain's
        // activation either, which a `>=`-style ordering check would.
        assert!(check_same_chain(older, newer).is_err());
    }

    #[test]
    fn the_refusal_names_both_chains() {
        let a = ChainId::mint();
        let b = ChainId::mint();
        let err = check_same_chain(a, b).expect_err("distinct chains must be refused");
        let rendered = err.to_string();
        assert!(
            rendered.contains(&format!("#{}", a.get())),
            "refusal must name the consuming chain: {rendered}"
        );
        assert!(
            rendered.contains(&format!("#{}", b.get())),
            "refusal must name the producing chain: {rendered}"
        );
    }

    /// The counter is shared process-wide, so the uniqueness claim is a
    /// claim about concurrent minting, not just sequential minting.
    #[test]
    fn ids_are_unique_across_threads() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 256;
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..PER_THREAD)
                        .map(|_| ChainId::mint())
                        .collect::<Vec<ChainId>>()
                })
            })
            .collect();
        let mut all: Vec<ChainId> = handles
            .into_iter()
            .flat_map(|h| h.join().expect("mint thread must not panic"))
            .collect();
        let minted = all.len();
        assert_eq!(minted, THREADS * PER_THREAD);
        all.sort_unstable();
        all.dedup();
        assert_eq!(
            all.len(),
            minted,
            "concurrent mints produced a duplicate id: two activations could alias"
        );
    }
}
