//! Process-wide caches of interleaved ("repacked") weight bytes, keyed
//! by the identity of the mapping the bytes came from.
//!
//! A repack rewrites a whole matrix into the row-interleaved layout the
//! `x4` / `x8` GEMV kernels read, so it costs a pass over every weight
//! byte. Done once per matrix that is the load-time cost llama.cpp pays
//! for its `repack` backend. Done once per CALL it is a full copy of the
//! matrix on every token, and that is what shipped: `apply_cpu_q8`, the
//! int-dot matvec the dense FFN gate/up and the MoE experts take, passed
//! a hand-written `/* uncacheable */ None` where every other matvec
//! passed `data.map_id()`. Profiled at one thread on TinyLlama Q8_0,
//! roughly 85% of a decode token was `pack_q8_0_matrix_x4` and the
//! `Arc` copy behind it, against under 10% in the GEMV it fed (#128).
//!
//! The `None` was a second, hand-restated copy of a decision
//! [`WeightBytes::map_id`] already makes: `Owned` and `Shared` bytes
//! answer `None` there, because their address is not an identity, and
//! `Mapped` bytes answer `Some`. Two places deciding one thing, with
//! nothing making them agree, is this repo's dominant bug shape, so the
//! typed lookups below take the [`WeightBytes`] and ask it themselves.
//! A call site can no longer say "uncacheable" on its own authority.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::WeightBytes;

/// Identity of the memory mapping a repacked buffer was built from.
///
/// The repack caches key on a weight's **address**, and an address is
/// only a stable identity for as long as the mapping that published it
/// is alive. Unmap one file and map another and the kernel will hand
/// the same address straight back -- a textbook ABA. The cache then
/// serves one matrix another matrix's interleaved bytes, which panicked
/// with an out-of-range slice when the two shapes differed and was
/// SILENT, i.e. wrong output, when they matched.
///
/// Holding a [`std::sync::Weak`] is what closes it, and it closes both
/// halves at once:
///
/// * while the `Weak` lives, the `Arc`'s control block cannot be
///   recycled, so [`Self::id`] is a unique name for exactly one mapping
///   for as long as the cache entry exists; and
/// * `upgrade()` succeeding proves the mapping itself is still alive,
///   which is what makes the address it published still mean what it
///   meant when the entry was written.
///
/// A dead `Weak` is therefore a *stale entry*, not a hit, and is
/// repacked and replaced. The `Weak` holds no mapping open, so nothing
/// here keeps a file resident.
#[derive(Clone)]
pub struct MapId {
    map: std::sync::Weak<memmap2::Mmap>,
    id: usize,
    offset: usize,
}

impl MapId {
    /// The identity of `range.start` inside `mmap`. Only
    /// [`WeightBytes::map_id`] builds one, and only for `Mapped` bytes.
    pub(super) fn of(mmap: &Arc<memmap2::Mmap>, offset: usize) -> Self {
        MapId {
            map: Arc::downgrade(mmap),
            id: Arc::as_ptr(mmap) as usize,
            offset,
        }
    }

    /// True when `other` names the same, still-live mapping.
    fn matches(&self, other: &MapId) -> bool {
        self.id == other.id
            && self.offset == other.offset
            && self
                .map
                .upgrade()
                .is_some_and(|m| Arc::as_ptr(&m) as usize == other.id)
    }

    /// The cache key this identity contributes to, for `rows x cols`.
    fn key(&self, rows: usize, cols: usize) -> RepackKey {
        (self.id, self.offset, rows, cols)
    }
}

/// `(mapping id, byte offset, rows, cols)`.
///
/// `cols` is in the key because two tensors of equal row count and
/// unequal width are different matrices with different repacked lengths,
/// and the old `(address, rows)` key called them the same one.
type RepackKey = (usize, usize, usize, usize);

/// Interleaved bytes, beside the mapping identity that makes the key
/// meaningful. See [`MapId`].
type Entry = (MapId, Arc<[u8]>);

/// One format's process-wide cache.
type RepackCache = Mutex<HashMap<RepackKey, Entry>>;

/// The one way any of this module takes a cache lock.
///
/// A poisoned lock is recovered from rather than propagated. The map
/// holds no invariant a panic can leave half-built: entries are
/// `(identity, immutable bytes)` pairs inserted whole, and every read
/// re-checks the identity before trusting the bytes. Propagating the
/// poison instead would let one panic anywhere in the process turn
/// EVERY later matvec on this format into a panic, which is a much
/// worse failure than serving a correct cached packing.
fn lock(cache: &'static RepackCache) -> std::sync::MutexGuard<'static, HashMap<RepackKey, Entry>> {
    cache.lock().unwrap_or_else(|e| e.into_inner())
}

/// The one lookup every format's repack shares.
///
/// `id` is `None` for bytes whose address may be recycled under us
/// (owned buffers, and an expert store's leases -- see
/// [`WeightBytes::map_id`]), and those always repack. Private: the only
/// way for a matvec to reach this is through a typed lookup below, which
/// derives `id` from the [`WeightBytes`] rather than accepting one.
fn get_or_repack(
    cache: &'static RepackCache,
    id: Option<MapId>,
    rows: usize,
    cols: usize,
    repack: impl FnOnce() -> Vec<u8>,
) -> Arc<[u8]> {
    let Some(id) = id else {
        return Arc::from(repack().into_boxed_slice());
    };
    let key = id.key(rows, cols);
    {
        let mut cache = lock(cache);
        match cache.get(&key) {
            Some((entry, hit)) if entry.matches(&id) => return Arc::clone(hit),
            // The mapping that published this address is gone, so the
            // address has been handed to somebody else. Drop the entry
            // rather than leaving a `Weak` pinning a dead control block.
            Some(_) => {
                cache.remove(&key);
            }
            None => {}
        }
    }
    let arc: Arc<[u8]> = Arc::from(repack().into_boxed_slice());
    let mut cache = lock(cache);
    // Another thread may have won the race; prefer the existing entry,
    // but only if it is one this caller would have accepted above.
    match cache.get(&key) {
        Some((entry, hit)) if entry.matches(&id) => Arc::clone(hit),
        _ => {
            cache.insert(key, (id, Arc::clone(&arc)));
            arc
        }
    }
}

/// Whether the packing of `data` (as `rows x cols`) is currently held in
/// `cache` under a live identity. Tests only.
#[cfg(test)]
fn is_cached(cache: &'static RepackCache, data: &WeightBytes, rows: usize, cols: usize) -> bool {
    let Some(id) = data.map_id() else {
        return false;
    };
    lock(cache)
        .get(&id.key(rows, cols))
        .is_some_and(|(entry, _)| entry.matches(&id))
}

/// One `static` cache per interleaved format, each behind a function so
/// the `OnceLock` is spelled once.
macro_rules! format_cache {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        fn $name() -> &'static RepackCache {
            static CACHE: OnceLock<RepackCache> = OnceLock::new();
            CACHE.get_or_init(|| Mutex::new(HashMap::new()))
        }
    };
}

format_cache!(
    /// Process-wide cache of interleaved Q4_K (`block_q4_Kx8`) bytes.
    q4k_repack_cache
);
format_cache!(
    /// Process-wide cache of interleaved Q5_K (`block_q5_Kx8`) bytes.
    q5k_repack_cache
);
format_cache!(
    /// Process-wide cache of interleaved Q6_K (`block_q6_Kx8`) bytes.
    q6k_repack_cache
);
format_cache!(
    /// Process-wide cache of interleaved Q8_0 (`block_q8_0x4`) bytes.
    q8x4_repack_cache
);
format_cache!(
    /// Process-wide cache of interleaved Q4_0 (`block_q4_0x4`) bytes.
    q4x4_repack_cache
);

pub(super) fn get_or_repack_q4k(data: &WeightBytes, rows: usize, cols: usize) -> Arc<[u8]> {
    get_or_repack(q4k_repack_cache(), data.map_id(), rows, cols, || {
        ferrox_quant::pack_q4_k_matrix_x8(
            data.as_slice(),
            rows,
            cols,
            ferrox_quant::q4_kx8_interleave(),
        )
    })
}

pub(super) fn get_or_repack_q5k(data: &WeightBytes, rows: usize, cols: usize) -> Arc<[u8]> {
    get_or_repack(q5k_repack_cache(), data.map_id(), rows, cols, || {
        ferrox_quant::pack_q5_k_matrix_x8(
            data.as_slice(),
            rows,
            cols,
            ferrox_quant::q5_kx8_interleave(),
        )
    })
}

pub(super) fn get_or_repack_q6k(data: &WeightBytes, rows: usize, cols: usize) -> Arc<[u8]> {
    get_or_repack(q6k_repack_cache(), data.map_id(), rows, cols, || {
        ferrox_quant::pack_q6_k_matrix_x8(
            data.as_slice(),
            rows,
            cols,
            ferrox_quant::q6_kx8_interleave(),
        )
    })
}

pub(super) fn get_or_repack_q8x4(data: &WeightBytes, rows: usize, cols: usize) -> Arc<[u8]> {
    get_or_repack(q8x4_repack_cache(), data.map_id(), rows, cols, || {
        ferrox_quant::pack_q8_0_matrix_x4(
            data.as_slice(),
            rows,
            cols,
            ferrox_quant::q8_0x4_interleave(),
        )
    })
}

pub(super) fn get_or_repack_q4_0x4(data: &WeightBytes, rows: usize, cols: usize) -> Arc<[u8]> {
    get_or_repack(q4x4_repack_cache(), data.map_id(), rows, cols, || {
        ferrox_quant::pack_q4_0_matrix_x4(
            data.as_slice(),
            rows,
            cols,
            ferrox_quant::q4_0x4_interleave(),
        )
    })
}

/// Whether the `Q8_0x4` packing of `data` is held in the cache. What
/// the `apply_cpu_q8` test below asks after one call.
#[cfg(test)]
pub(super) fn q8x4_is_cached(data: &WeightBytes, rows: usize, cols: usize) -> bool {
    is_cached(q8x4_repack_cache(), data, rows, cols)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{f16_le, ForceIntDot};
    use super::super::{QuantKind, WeightMatrix};
    use super::*;

    // -----------------------------------------------------------------
    // Repack cache identity (see `MapId`)
    //
    // The bug these cover: the caches used to key on `(address, rows)`
    // and gate on an `address_is_stable() -> bool`. Drop one mmap, make
    // another, and the kernel hands the same address back, so the cache
    // served the previous matrix's interleaved bytes -- an out-of-range
    // panic when the shapes differed, silent wrong output when they
    // matched.
    //
    // Address reuse is the OS's decision and cannot be demanded from a
    // test, so these do not wait for it. They fabricate exactly what the
    // cache would SEE in that moment -- a key that collides while the
    // mapping behind it is gone, or while the width differs -- and
    // assert the cache refuses to serve it.
    // -----------------------------------------------------------------

    /// Writes `bytes` to a temp file and maps it. The caller holds the
    /// `Arc`, so when the mapping dies is explicit, which is the whole
    /// subject of these tests.
    fn mapped(tag: &str, bytes: &[u8]) -> (Arc<memmap2::Mmap>, WeightBytes) {
        let path = std::env::temp_dir().join(format!(
            "ferrox_repack_{tag}_{}_{:?}.bin",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, bytes).expect("write fixture");
        let file = std::fs::File::open(&path).expect("open fixture");
        // SAFETY: the file was written and closed above, is named for
        // this process and thread, and nothing mutates it while mapped.
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).expect("map fixture") });
        let _ = std::fs::remove_file(&path);
        let view = WeightBytes::Mapped {
            mmap: Arc::clone(&mmap),
            range: 0..bytes.len(),
        };
        (mmap, view)
    }

    /// Q8_0 bytes with finite scales, `rows * cols/32` blocks.
    fn q8_0_matrix_bytes(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let mut data = Vec::with_capacity(rows * (cols / 32) * 34);
        for _ in 0..rows * (cols / 32) {
            data.extend_from_slice(&f16_le(0.02 + f32::from(next()) * 0.0004));
            for _ in 0..32 {
                data.push(next());
            }
        }
        data
    }

    /// A `MapId` is only an identity while its mapping is alive. This is
    /// the check the old boolean could not express, and it is the one
    /// thing standing between the cache and an ABA.
    #[test]
    fn map_id_stops_matching_once_its_mapping_is_dropped() {
        let (mmap, view) = mapped("live", &q8_0_matrix_bytes(4, 32, 7));
        let id = view.map_id().expect("Mapped bytes must have an identity");
        let held = id.clone();
        assert!(
            held.matches(&id),
            "a live mapping must match its own identity"
        );

        // Everything that could witness the mapping is gone: this is
        // precisely the moment the address becomes reusable.
        drop(view);
        drop(mmap);
        assert!(
            !held.matches(&id),
            "an identity whose mapping is dead must not match, or the \
             cache will trust an address the kernel has already reissued"
        );
    }

    /// A cache entry left behind by a dead mapping must be replaced, not
    /// served. Fabricates the entry rather than waiting on the OS to
    /// reissue an address; the entry is byte-for-byte what the old code
    /// would have left there.
    #[test]
    fn stale_repack_entry_is_replaced_not_served() {
        let (rows, cols) = (8usize, 64usize);
        let bytes = q8_0_matrix_bytes(rows, cols, 11);
        let (_mmap, view) = mapped("stale", &bytes);
        let id = view.map_id().expect("Mapped bytes must have an identity");

        // Some other matrix's packing, parked at the key this live
        // matrix will look up, under an identity that can never upgrade.
        let poison = vec![0xABu8; 16];
        {
            let mut cache = lock(q8x4_repack_cache());
            cache.insert(
                id.key(rows, cols),
                (
                    MapId {
                        map: std::sync::Weak::new(),
                        id: id.id,
                        offset: id.offset,
                    },
                    Arc::from(poison.clone().into_boxed_slice()),
                ),
            );
        }

        let got = get_or_repack_q8x4(&view, rows, cols);
        let want = ferrox_quant::pack_q8_0_matrix_x4(
            view.as_slice(),
            rows,
            cols,
            ferrox_quant::q8_0x4_interleave(),
        );
        assert_ne!(&got[..], &poison[..], "served a dead mapping's bytes");
        assert_eq!(&got[..], &want[..], "stale entry was not repacked");

        // And the dead entry is gone rather than pinning a control block.
        let cache = lock(q8x4_repack_cache());
        let (entry, _) = cache
            .get(&id.key(rows, cols))
            .expect("the live packing should now be cached");
        assert!(
            entry.matches(&id),
            "the replacement entry must carry the LIVE identity"
        );
    }

    /// Two widths at one address are two matrices. The old key was
    /// `(address, rows)`, so a 576x576 and a 576x1536 collided and the
    /// second was served the first's shorter buffer.
    ///
    /// Drives the primitive directly: the collision needs two byte
    /// buffers under ONE identity, which is exactly what the typed
    /// lookups exist to make impossible for production code.
    #[test]
    fn repack_key_separates_two_widths_at_one_address() {
        let rows = 8usize;
        let narrow = q8_0_matrix_bytes(rows, 32, 3);
        let wide = q8_0_matrix_bytes(rows, 64, 5);
        let (_mmap, view) = mapped("widths", &narrow);
        let id = view.map_id().expect("Mapped bytes must have an identity");

        let il = ferrox_quant::q8_0x4_interleave();
        let a = get_or_repack(q8x4_repack_cache(), Some(id.clone()), rows, 32, || {
            ferrox_quant::pack_q8_0_matrix_x4(&narrow, rows, 32, il)
        });
        let b = get_or_repack(q8x4_repack_cache(), Some(id.clone()), rows, 64, || {
            ferrox_quant::pack_q8_0_matrix_x4(&wide, rows, 64, il)
        });
        assert_eq!(
            &a[..],
            &ferrox_quant::pack_q8_0_matrix_x4(&narrow, rows, 32, il)[..]
        );
        assert_eq!(
            &b[..],
            &ferrox_quant::pack_q8_0_matrix_x4(&wide, rows, 64, il)[..],
            "the wider matrix was served the narrower one's packing"
        );
        assert!(b.len() > a.len(), "widths must not share a cache entry");
    }

    /// Owned buffers and expert-store leases are never cacheable. The
    /// lease is the interesting one: its allocation stays alive and keeps
    /// its address while its CONTENTS are replaced by another expert's,
    /// so no liveness check could rescue it.
    #[test]
    fn map_id_is_none_for_owned_and_shared_bytes() {
        let owned = WeightBytes::Owned(q8_0_matrix_bytes(4, 32, 9));
        assert!(owned.map_id().is_none(), "an owned Vec's address is reused");

        let buf = Arc::new(q8_0_matrix_bytes(4, 32, 13));
        let leased = WeightBytes::Shared {
            buf,
            range: 0..34 * 4,
        };
        assert!(
            leased.map_id().is_none(),
            "an expert lease keeps its address across a content swap"
        );
    }

    /// A mapped matrix is packed once, and a second lookup is a hit that
    /// hands back the SAME allocation. An owned matrix is never cached.
    /// Both halves are the contract the matvecs rely on.
    #[test]
    fn a_mapped_matrix_is_packed_once_and_an_owned_one_never_cached() {
        let (rows, cols) = (8usize, 64usize);
        let bytes = q8_0_matrix_bytes(rows, cols, 17);
        let (_mmap, view) = mapped("once", &bytes);
        assert!(!q8x4_is_cached(&view, rows, cols));
        let first = get_or_repack_q8x4(&view, rows, cols);
        assert!(q8x4_is_cached(&view, rows, cols));
        let second = get_or_repack_q8x4(&view, rows, cols);
        assert!(
            Arc::ptr_eq(&first, &second),
            "a second lookup of a live mapping must be a cache hit"
        );

        let owned = WeightBytes::Owned(bytes);
        let a = get_or_repack_q8x4(&owned, rows, cols);
        let b = get_or_repack_q8x4(&owned, rows, cols);
        assert!(!q8x4_is_cached(&owned, rows, cols));
        assert!(
            !Arc::ptr_eq(&a, &b),
            "owned bytes have no identity and must repack every time"
        );
    }

    /// The #128 regression, at the call site that carried it.
    ///
    /// `apply_cpu_q8` is the int-dot matvec the dense FFN gate/up and
    /// every MoE expert take, and it repacked its matrix on EVERY call:
    /// it passed a hand-written `None` identity where the other matvecs
    /// passed `data.map_id()`. Measured at one thread on TinyLlama Q8_0,
    /// that repack was ~85% of a decode token. The typed lookup now
    /// derives the identity itself, and this asserts the packing of a
    /// mapped matrix is in the cache after one call through that path.
    ///
    /// Sabotage: make `get_or_repack_q8x4` pass `None` instead of
    /// `data.map_id()` and this goes red.
    #[test]
    fn apply_cpu_q8_caches_the_packing_of_a_mapped_matrix() {
        let _force = ForceIntDot::new(true);
        let (rows, cols) = (8usize, 64usize);
        let bytes = q8_0_matrix_bytes(rows, cols, 23);
        let (_mmap, view) = mapped("apply_q8", &bytes);
        let m = WeightMatrix::Quantized {
            data: view,
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        let WeightMatrix::Quantized { data, .. } = &m else {
            unreachable!()
        };
        assert!(!q8x4_is_cached(data, rows, cols), "fresh mapping");

        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let act = ferrox_quant::quantize_activations_q8(&x);
        let out = m
            .apply_cpu_q8(&act)
            .expect("Q8_0 with int-dot on takes the interleaved path");
        assert_eq!(out.len(), rows);
        assert!(
            q8x4_is_cached(data, rows, cols),
            "apply_cpu_q8 repacked a mapped matrix without caching it: \
             that is a full copy of the matrix per token (#128)"
        );
    }
}
