//! A bounded, lease-protected byte cache for routed-expert weights --
//! the storage foundation for running MoE checkpoints whose experts do
//! not all fit in RAM at once (stream cold experts from SSD, keep hot
//! ones resident under one global byte budget).
//!
//! Status: wired into both decode paths as an opt-in
//! (`Decoder::from_gguf_with_expert_cache` for GGUF,
//! `load_kimi_checkpoint_with_expert_cache` for Kimi safetensors, both
//! behind the server's `FERROX_EXPERT_CACHE_BYTES` or
//! `FERROX_SSD_STREAMING=1` which defaults the cache to 2 GiB), each
//! proven
//! bit-identical to its resident path on the committed fixtures at
//! both generous and smaller-than-one-expert budgets. Without the
//! opt-in, experts still load resident (mmap) exactly as before.
//! [`ExpertStore::prefetch`] warms keys for ds4-style SSD streaming
//! overlap (caller supplies the hotlist). Never yet exercised against
//! a real large checkpoint. Reference clone: `.scratch/ds4` (gitignored).
//!
//! Design:
//!
//! - [`ExpertSource`] abstracts where bytes come from (a file with
//!   positional reads, a shard set, an in-memory test source). The
//!   store never caches a partial expert: one `(layer, expert)` key
//!   maps to one complete byte buffer (gate+up+down and any auxiliary
//!   tensors, concatenated by the source in a layout the consumer
//!   defines).
//! - [`ExpertStore::acquire`] returns an [`ExpertLease`] -- a cheap
//!   `Arc` handle that *pins* the entry: eviction skips any entry with
//!   an outstanding lease, so a slot can never be reused while
//!   CPU (or, later, GPU) work still reads it. This is the
//!   slot-reuse-corruption guard, enforced structurally (an `Arc` with
//!   `strong_count > 1` is simply not freeable), not by convention.
//! - When an expert doesn't fit even after evicting every unleased
//!   entry (e.g. a cache configured smaller than one decode step's
//!   expert union), `acquire` still succeeds: the bytes are read and
//!   returned as an *uncached* pass-through lease. Capacity pressure
//!   degrades to more I/O, never to a wrong answer or a deadlock.
//! - Reads happen outside the store lock, so concurrent misses on
//!   different experts overlap their I/O. Two concurrent misses on the
//!   *same* expert may both read it; the first to insert wins and the
//!   loser's buffer becomes that caller's private pass-through copy --
//!   duplicated work under a rare race, never wrong bytes.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Stable identity of one routed expert within a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

/// Where expert bytes come from. Implementations must be cheap to call
/// concurrently (positional reads, not a shared seek cursor).
pub trait ExpertSource: Send + Sync {
    /// The exact byte length of this expert, or `None` if the key does
    /// not exist. Used for budget accounting *before* reading.
    fn expert_len(&self, key: ExpertKey) -> Option<usize>;

    /// Reads this expert's complete bytes.
    fn read_expert(&self, key: ExpertKey) -> io::Result<Vec<u8>>;
}

/// A pinned handle to one expert's bytes. While any lease for an entry
/// is alive, the store cannot evict or reuse that entry's memory.
pub struct ExpertLease {
    data: Arc<Vec<u8>>,
}

impl ExpertLease {
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// The shared buffer behind this lease, for building
    /// `WeightBytes::Shared` sub-range views (one per matrix packed
    /// into the expert's combined bytes). Every clone extends the
    /// entry's pin: the store cannot evict while any of these views is
    /// alive.
    pub fn shared_buf(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.data)
    }
}

/// Monotonic counters, readable at any time without taking the store
/// lock. `resident_bytes` is a gauge (current cache footprint).
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpertStoreStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Acquires that could not be cached (entry larger than what the
    /// budget could free) and were served as uncached pass-throughs.
    pub pass_throughs: u64,
    pub bytes_read: u64,
    pub resident_bytes: u64,
}

struct Entry {
    data: Arc<Vec<u8>>,
    /// Monotonic recency stamp; smallest = least recently used. A
    /// stamp per touch is simpler and cheaper than reshuffling a
    /// dedicated LRU list under the same lock, at the cost of an O(n)
    /// scan on eviction -- fine for the hundreds-to-low-thousands of
    /// entries a real expert cache holds.
    last_used: u64,
}

struct Inner {
    entries: HashMap<ExpertKey, Entry>,
    resident_bytes: usize,
    clock: u64,
}

/// Bytes every live [`ExpertStore`] in this process has committed to
/// holding expert weights.
///
/// This module is the SINGLE holder of the expert byte budget, because
/// on unified memory two budgets are the same RAM counted twice. That
/// rule only bites when something *else* also wants to retain weight
/// bytes, and something else now does: `weight_matrix::repack_cache`
/// keeps interleaved copies of matrices it has packed. This gauge is
/// how that cache subtracts rather than restates -- see
/// [`crate::host_memory::derived_copy_budget`].
static COMMITTED_EXPERT_BYTES: AtomicU64 = AtomicU64::new(0);

/// Bytes currently committed to expert caches across the process. See
/// [`COMMITTED_EXPERT_BYTES`].
pub fn committed_expert_bytes() -> u64 {
    COMMITTED_EXPERT_BYTES.load(Ordering::Relaxed)
}

/// The bounded cache. See the module docs for the design contract.
pub struct ExpertStore<S: ExpertSource> {
    source: S,
    budget_bytes: usize,
    inner: Mutex<Inner>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    pass_throughs: AtomicU64,
    bytes_read: AtomicU64,
}

impl<S: ExpertSource> Drop for ExpertStore<S> {
    fn drop(&mut self) {
        COMMITTED_EXPERT_BYTES.fetch_sub(self.budget_bytes as u64, Ordering::Relaxed);
    }
}

impl<S: ExpertSource> ExpertStore<S> {
    pub fn new(source: S, budget_bytes: usize) -> Self {
        COMMITTED_EXPERT_BYTES.fetch_add(budget_bytes as u64, Ordering::Relaxed);
        ExpertStore {
            source,
            budget_bytes,
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                resident_bytes: 0,
                clock: 0,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            pass_throughs: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub fn stats(&self) -> ExpertStoreStats {
        let resident = self
            .inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .resident_bytes as u64;
        ExpertStoreStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            pass_throughs: self.pass_throughs.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            resident_bytes: resident,
        }
    }

    /// Best-effort warm of `keys` into the cache (ds4-style SSD
    /// streaming prefetch): acquires each key and drops the lease so
    /// the entry stays resident under the budget until eviction.
    /// Concurrent acquires on the same key may duplicate I/O; never
    /// returns wrong bytes. Failures on individual keys are skipped.
    pub fn prefetch(&self, keys: &[ExpertKey]) {
        for &key in keys {
            let _ = self.acquire(key);
        }
    }

    /// Returns a pinned lease on `key`'s bytes, reading them from the
    /// source on a miss. Never blocks waiting for other leases to be
    /// released: if the entry cannot fit in the budget right now, the
    /// bytes are returned uncached (see module docs).
    pub fn acquire(&self, key: ExpertKey) -> io::Result<ExpertLease> {
        // Fast path: cache hit under the lock.
        {
            let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
            inner.clock += 1;
            let clock = inner.clock;
            if let Some(entry) = inner.entries.get_mut(&key) {
                entry.last_used = clock;
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(ExpertLease {
                    data: Arc::clone(&entry.data),
                });
            }
        }

        // Miss: read outside the lock so concurrent misses overlap I/O.
        self.misses.fetch_add(1, Ordering::Relaxed);
        let data = self.source.read_expert(key)?;
        self.bytes_read
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        let size = data.len();
        let data = Arc::new(data);

        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        // Someone else may have inserted this key while we read -- use
        // theirs (keeps exactly one cached copy), our read becomes the
        // rare-race duplicate described in the module docs. Already
        // counted as a miss above (each acquire increments exactly one
        // of hits/misses), so no hit is recorded here.
        if let Some(entry) = inner.entries.get_mut(&key) {
            return Ok(ExpertLease {
                data: Arc::clone(&entry.data),
            });
        }

        if size > self.budget_bytes {
            self.pass_throughs.fetch_add(1, Ordering::Relaxed);
            return Ok(ExpertLease { data });
        }

        // Evict least-recently-used *unleased* entries until it fits.
        while inner.resident_bytes + size > self.budget_bytes {
            let victim = inner
                .entries
                .iter()
                .filter(|(_, e)| Arc::strong_count(&e.data) == 1)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| *k);
            match victim {
                Some(k) => {
                    let e = inner.entries.remove(&k).expect("victim key just found");
                    inner.resident_bytes -= e.data.len();
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                }
                None => {
                    // Every resident entry is pinned by a live lease --
                    // serve uncached rather than waiting (a wait here
                    // could deadlock against our own caller's leases).
                    self.pass_throughs.fetch_add(1, Ordering::Relaxed);
                    return Ok(ExpertLease { data });
                }
            }
        }

        inner.clock += 1;
        let clock = inner.clock;
        inner.resident_bytes += size;
        inner.entries.insert(
            key,
            Entry {
                data: Arc::clone(&data),
                last_used: clock,
            },
        );
        Ok(ExpertLease { data })
    }
}

/// A file-backed [`ExpertSource`]: each expert is one contiguous
/// `(offset, len)` byte range in a single file, read with positional
/// reads (`pread`-style on unix, so concurrent misses never contend on
/// a shared seek cursor). This is the portable buffered-I/O default
/// the storage design starts from; `O_DIRECT`/`F_NOCACHE`/`io_uring`
/// variants are future work behind the same trait and must produce
/// identical bytes.
pub struct FileRangeSource {
    file: std::fs::File,
    ranges: HashMap<ExpertKey, (u64, usize)>,
    /// Non-unix fallback only: serializes seek+read pairs.
    #[cfg(not(unix))]
    seek_lock: Mutex<()>,
}

impl FileRangeSource {
    pub fn new(file: std::fs::File, ranges: HashMap<ExpertKey, (u64, usize)>) -> Self {
        FileRangeSource {
            file,
            ranges,
            #[cfg(not(unix))]
            seek_lock: Mutex::new(()),
        }
    }
}

impl ExpertSource for FileRangeSource {
    fn expert_len(&self, key: ExpertKey) -> Option<usize> {
        self.ranges.get(&key).map(|&(_, len)| len)
    }

    fn read_expert(&self, key: ExpertKey) -> io::Result<Vec<u8>> {
        let &(offset, len) = self
            .ranges
            .get(&key)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("{key:?}")))?;
        let mut buf = vec![0u8; len];
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(&mut buf, offset)?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let _guard = self.seek_lock.lock().unwrap_or_else(|p| p.into_inner());
            let mut f = &self.file;
            f.seek(SeekFrom::Start(offset))?;
            f.read_exact(&mut buf)?;
        }
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic per-key content so any slot-reuse/wrong-bytes bug
    /// fails loudly: byte `i` of expert `(l, e)` is a function of all
    /// three.
    struct PatternSource {
        len: usize,
        n_layers: u32,
        n_experts: u32,
    }

    fn expected_bytes(key: ExpertKey, len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (key.layer as usize * 31 + key.expert as usize * 7 + i * 13) as u8)
            .collect()
    }

    impl ExpertSource for PatternSource {
        fn expert_len(&self, key: ExpertKey) -> Option<usize> {
            (key.layer < self.n_layers && key.expert < self.n_experts).then_some(self.len)
        }
        fn read_expert(&self, key: ExpertKey) -> io::Result<Vec<u8>> {
            if self.expert_len(key).is_none() {
                return Err(io::Error::new(io::ErrorKind::NotFound, "no such expert"));
            }
            Ok(expected_bytes(key, self.len))
        }
    }

    fn key(layer: u32, expert: u32) -> ExpertKey {
        ExpertKey { layer, expert }
    }

    fn store(len: usize, budget: usize) -> ExpertStore<PatternSource> {
        ExpertStore::new(
            PatternSource {
                len,
                n_layers: 8,
                n_experts: 8,
            },
            budget,
        )
    }

    /// The process-wide gauge tracks every live store's budget and
    /// releases it on drop.
    ///
    /// It exists so `weight_matrix::repack_cache` can SUBTRACT the
    /// expert budget from the same pool rather than declare a second
    /// one: on unified memory an expert byte and a repacked byte are
    /// the same RAM, and this module is the single holder of the expert
    /// half. A gauge that did not fall on drop would starve the repack
    /// cache for the rest of the process.
    ///
    /// The gauge is process-wide and the harness runs these tests in
    /// parallel, so this asserts against a SENTINEL budget larger than
    /// every other budget in this module put together rather than
    /// against an exact total another test could move under it.
    ///
    /// Sabotage: delete the `fetch_add` in `new` or the `Drop` impl and
    /// this goes red.
    #[test]
    fn the_committed_gauge_rises_with_a_store_and_falls_when_it_drops() {
        const SENTINEL: usize = 1 << 40;
        {
            let _held = store(64, SENTINEL);
            assert!(
                committed_expert_bytes() >= SENTINEL as u64,
                "a live store must have committed its budget"
            );
        }
        assert!(
            committed_expert_bytes() < SENTINEL as u64,
            "a dropped store must return its budget to the pool, or the \
             repack cache is starved of it for the rest of the process"
        );
    }

    #[test]
    fn hits_misses_and_lru_eviction_order() {
        let s = store(100, 250); // fits 2 experts
        assert_eq!(
            s.acquire(key(0, 0)).unwrap().bytes(),
            expected_bytes(key(0, 0), 100)
        );
        assert_eq!(
            s.acquire(key(0, 1)).unwrap().bytes(),
            expected_bytes(key(0, 1), 100)
        );
        // Touch (0,0) so (0,1) is now least recently used.
        s.acquire(key(0, 0)).unwrap();
        // Third expert evicts (0,1), not (0,0).
        s.acquire(key(0, 2)).unwrap();
        let before = s.stats();
        s.acquire(key(0, 0)).unwrap(); // must still be a hit
        let after = s.stats();
        assert_eq!(after.hits, before.hits + 1);
        assert_eq!(after.misses, before.misses);
        assert_eq!(after.evictions, 1);
        assert!(after.resident_bytes <= 250);
        // (0,1) was the eviction victim: acquiring it again is a miss.
        s.acquire(key(0, 1)).unwrap();
        assert_eq!(s.stats().misses, after.misses + 1);
    }

    #[test]
    fn a_live_lease_pins_its_entry_against_eviction() {
        let s = store(100, 250); // fits 2 experts
        let pinned = s.acquire(key(1, 0)).unwrap();
        // Fill and churn the cache well past the budget: the pinned
        // entry must never be evicted or corrupted while the lease
        // lives...
        for e in 1..6 {
            s.acquire(key(1, e)).unwrap();
        }
        assert_eq!(pinned.bytes(), expected_bytes(key(1, 0), 100));
        // ...and it is still resident (a re-acquire is a hit).
        let hits_before = s.stats().hits;
        let again = s.acquire(key(1, 0)).unwrap();
        assert_eq!(s.stats().hits, hits_before + 1);
        assert_eq!(again.bytes(), expected_bytes(key(1, 0), 100));

        // Once every lease is dropped, pressure may evict it like any
        // other entry.
        drop(pinned);
        drop(again);
        for e in 1..6 {
            s.acquire(key(2, e)).unwrap();
        }
        let misses_before = s.stats().misses;
        s.acquire(key(1, 0)).unwrap();
        assert_eq!(
            s.stats().misses,
            misses_before + 1,
            "unpinned entry was evictable"
        );
    }

    /// The plan's "cache smaller than one token's expert union" case:
    /// a budget that can't hold even one expert still serves correct
    /// bytes on every acquire, as uncached pass-throughs.
    #[test]
    fn budget_smaller_than_one_expert_degrades_to_pass_through_not_failure() {
        let s = store(100, 50);
        for e in 0..4 {
            let lease = s.acquire(key(0, e)).unwrap();
            assert_eq!(lease.bytes(), expected_bytes(key(0, e), 100));
        }
        let st = s.stats();
        assert_eq!(st.pass_throughs, 4);
        assert_eq!(st.resident_bytes, 0);
        assert_eq!(st.evictions, 0);
    }

    /// If every resident entry is pinned, a new acquire must not
    /// deadlock or evict a pinned entry -- it passes through.
    #[test]
    fn fully_pinned_cache_serves_new_experts_uncached() {
        let s = store(100, 200);
        let _a = s.acquire(key(0, 0)).unwrap();
        let _b = s.acquire(key(0, 1)).unwrap();
        let c = s.acquire(key(0, 2)).unwrap();
        assert_eq!(c.bytes(), expected_bytes(key(0, 2), 100));
        assert_eq!(s.stats().pass_throughs, 1);
        assert_eq!(s.stats().evictions, 0);
        // The two pinned entries are still hits.
        let hits_before = s.stats().hits;
        s.acquire(key(0, 0)).unwrap();
        s.acquire(key(0, 1)).unwrap();
        assert_eq!(s.stats().hits, hits_before + 2);
    }

    #[test]
    fn missing_expert_is_a_clean_error() {
        let s = store(100, 1000);
        assert!(s.acquire(key(99, 0)).is_err());
    }

    /// Concurrent stress under a deliberately tiny budget: many
    /// threads acquiring random keys while evictions churn constantly.
    /// Every lease's bytes must match that key's expected pattern --
    /// the direct test for slot-reuse corruption.
    #[test]
    fn concurrent_acquires_under_eviction_pressure_never_yield_wrong_bytes() {
        let s = Arc::new(store(64, 200)); // fits 3 of 16 hot keys
        let mut handles = Vec::new();
        for t in 0..8u32 {
            let s = Arc::clone(&s);
            handles.push(std::thread::spawn(move || {
                let mut state = t.wrapping_mul(2654435761).wrapping_add(12345);
                for _ in 0..500 {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let k = key((state >> 8) % 4, (state >> 16) % 4);
                    let lease = s.acquire(k).expect("in-range key must read");
                    assert_eq!(
                        lease.bytes(),
                        expected_bytes(k, 64),
                        "wrong bytes for {k:?} -- slot reuse corruption"
                    );
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let st = s.stats();
        assert_eq!(
            st.hits + st.misses,
            8 * 500,
            "every acquire is a hit or a miss"
        );
        assert!(st.resident_bytes <= 200, "budget held under concurrency");
    }

    /// The bridge to the weight types: a quantized `WeightMatrix` built
    /// over a lease's shared buffer must (a) compute exactly what the
    /// same bytes compute as an owned buffer, and (b) keep the cache
    /// entry pinned for as long as the matrix lives, even after the
    /// original lease is dropped.
    #[test]
    fn weight_matrix_over_a_lease_computes_identically_and_extends_the_pin() {
        use crate::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};

        // A source whose "expert" payload is real Q8_0 rows.
        struct QuantSource {
            rows: Vec<u8>,
        }
        impl ExpertSource for QuantSource {
            fn expert_len(&self, _k: ExpertKey) -> Option<usize> {
                Some(self.rows.len())
            }
            fn read_expert(&self, _k: ExpertKey) -> io::Result<Vec<u8>> {
                Ok(self.rows.clone())
            }
        }

        let cols = 64;
        let rows = 2;
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.11).sin())
            .collect();
        let mut packed = Vec::new();
        for r in 0..rows {
            packed.extend(ferrox_quant::quantize_q8_0(
                &values[r * cols..(r + 1) * cols],
            ));
        }
        let store = ExpertStore::new(
            QuantSource {
                rows: packed.clone(),
            },
            packed.len(),
        );

        let lease = store.acquire(key(0, 0)).unwrap();
        let matrix = WeightMatrix::Quantized {
            data: WeightBytes::Shared {
                buf: lease.shared_buf(),
                range: 0..packed.len(),
            },
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        drop(lease); // the matrix alone must keep the entry pinned

        let owned = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.07).cos()).collect();
        assert_eq!(
            matrix.apply(&x),
            owned.apply(&x),
            "lease-backed == owned, bit for bit"
        );

        // Entry is still resident (pinned by the matrix): re-acquire is
        // a hit even though the budget is exactly one expert.
        let hits_before = store.stats().hits;
        store.acquire(key(0, 0)).unwrap();
        assert_eq!(store.stats().hits, hits_before + 1);
    }

    /// End-to-end through a real on-disk file: positional-range reads
    /// through the bounded store return exactly the bytes written at
    /// each expert's offset, including under concurrent access.
    #[test]
    fn file_range_source_reads_correct_ranges_through_the_store() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "ferrox_expert_store_test_{}.bin",
            std::process::id()
        ));
        let mut file_bytes = Vec::new();
        let mut ranges = HashMap::new();
        for l in 0..3u32 {
            for e in 0..4u32 {
                let content = expected_bytes(key(l, e), 96);
                ranges.insert(key(l, e), (file_bytes.len() as u64, content.len()));
                file_bytes.extend_from_slice(&content);
            }
        }
        std::fs::write(&path, &file_bytes).unwrap();
        let source = FileRangeSource::new(std::fs::File::open(&path).unwrap(), ranges);
        let store = Arc::new(ExpertStore::new(source, 300)); // fits 3 of 12

        let mut handles = Vec::new();
        for t in 0..4u32 {
            let store = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                for i in 0..200u32 {
                    let k = key((t + i) % 3, i % 4);
                    let lease = store.acquire(k).unwrap();
                    assert_eq!(lease.bytes(), expected_bytes(k, 96));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        std::fs::remove_file(&path).ok();
    }
}
