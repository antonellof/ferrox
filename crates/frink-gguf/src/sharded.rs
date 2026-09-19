//! Split-GGUF support: one logical tensor/metadata namespace over a set
//! of shard files, the layout produced by llama.cpp's `gguf-split` tool
//! and used by every published multi-file GGUF checkpoint (including the
//! "Dynamic" quantizations of the very large MoE models frink targets).
//!
//! The shard convention (verified against `gguf-split`'s public
//! behavior, implemented independently — no source copied):
//!
//! - Files are named `<prefix>-NNNNN-of-MMMMM.gguf`, `NNNNN` starting at
//!   `00001`, zero-padded to 5 digits.
//! - Every shard is itself a complete, valid GGUF file with its own
//!   header, metadata, tensor table, and data section.
//! - Each shard carries `split.no` (this shard's index), `split.count`
//!   (total shards), and `split.tensors.count` (total tensors across all
//!   shards).
//! - **`split.no` is 0-based**, even though the *filename* numbering is
//!   1-based (`-00001-of-...`) -- confirmed against llama.cpp's real
//!   `gguf-split.cpp` source (`i_split` starts at -1 and is
//!   pre-incremented before the first shard is built, so the first
//!   shard's `split.no` is 0). An earlier version of this file assumed
//!   `split.no` was 1-based, matching the filename directly --
//!   self-consistent across every synthetic test fixture (which all
//!   built the same wrong assumption), but wrong against a real
//!   published shard set: `Kimi-K3-UD-IQ1_S-00001-of-00014.gguf`'s real
//!   `split.no` is 0, not 1, and the file failed to open with
//!   `ShardNumberMismatch` until this was fixed.
//! - The first shard carries the model's full metadata (architecture
//!   hparams, tokenizer, ...). It may be **metadata-only** (zero
//!   tensors) — real published checkpoints do this.
//!
//! `ShardedGguf` opens all sibling shards from any one shard's path,
//! validates the set (missing/duplicate shards, per-shard `split.*`
//! consistency, duplicate tensor names, total tensor count), and then
//! answers tensor lookups over the union namespace while keeping exactly
//! one `Mmap` per shard file — no merging or copying of tensor bytes.
//! A plain single-file GGUF (no `split.count` metadata, or
//! `split.count` = 1) loads as a one-shard set, so callers can use this
//! type unconditionally.
//!
//! Not yet covered (disclosed, not hidden): shard files are not
//! re-validated after indexing (a file swapped out from under the mmap
//! is not detected), and there is no content-hash identity check tying
//! the opened set to a specific published checkpoint.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use thiserror::Error;

use crate::{GgufError, GgufFile, GgufValue, TensorInfo};

/// Metadata keys written by `gguf-split` into every shard.
pub const SPLIT_NO_KEY: &str = "split.no";
pub const SPLIT_COUNT_KEY: &str = "split.count";
pub const SPLIT_TENSORS_COUNT_KEY: &str = "split.tensors.count";

#[derive(Debug, Error)]
pub enum ShardError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(
        "'{0}' declares split.count={1} but its filename does not match the \
         canonical '<prefix>-NNNNN-of-MMMMM.gguf' shard pattern, so sibling \
         shards cannot be located"
    )]
    NonCanonicalName(String, u64),
    #[error("missing shard file '{0}' (of {1} expected)")]
    MissingShard(String, u64),
    #[error(
        "shard '{path}': filename implies 0-based split.no={expected} but metadata says {found}"
    )]
    ShardNumberMismatch {
        path: String,
        expected: u64,
        found: u64,
    },
    #[error("shard '{path}': filename says {expected} total shards but split.count metadata says {found}")]
    ShardCountMismatch {
        path: String,
        expected: u64,
        found: u64,
    },
    #[error("shard '{path}': missing required metadata key '{key}'")]
    MissingSplitKey { path: String, key: String },
    #[error("duplicate tensor name '{0}' (appears in shard {1} and shard {2})")]
    DuplicateTensorName(String, u64, u64),
    #[error("split.tensors.count says {expected} tensors but the shard set contains {found}")]
    TensorCountMismatch { expected: u64, found: u64 },
    #[error(
        "shard '{path}': metadata key '{key}' disagrees with the first shard's value \
         (the shard set is inconsistent or mixes files from different checkpoints)"
    )]
    InconsistentMetadata { path: String, key: String },
}

/// The parsed canonical shard filename: `<prefix>-NNNNN-of-MMMMM.gguf`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardName {
    /// Everything before `-NNNNN-of-MMMMM.gguf`, including any directory
    /// components — joining `prefix` back with a shard number reproduces
    /// a sibling's full path.
    pub prefix: String,
    /// 1-based shard number from the filename.
    pub no: u64,
    /// Total shard count from the filename.
    pub count: u64,
}

impl ShardName {
    /// Parses `<prefix>-NNNNN-of-MMMMM.gguf`. Returns `None` for any
    /// path not matching the canonical pattern exactly (5-digit fields,
    /// `.gguf` extension).
    pub fn parse(path: &Path) -> Option<ShardName> {
        let s = path.to_str()?;
        let stem = s.strip_suffix(".gguf")?;
        // stem must end with -NNNNN-of-MMMMM (5+4+5 chars plus the leading '-')
        if stem.len() < 15 {
            return None;
        }
        let (rest, count_str) = stem.split_at(stem.len() - 5);
        let count: u64 = count_str.parse().ok()?;
        if !count_str.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let rest = rest.strip_suffix("-of-")?;
        if rest.len() < 6 {
            return None;
        }
        let (prefix_dash, no_str) = rest.split_at(rest.len() - 5);
        if !no_str.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let no: u64 = no_str.parse().ok()?;
        let prefix = prefix_dash.strip_suffix('-')?;
        if prefix.is_empty() {
            return None;
        }
        Some(ShardName {
            prefix: prefix.to_string(),
            no,
            count,
        })
    }

    /// The canonical path of shard `no` in this set.
    pub fn sibling(&self, no: u64) -> PathBuf {
        PathBuf::from(format!(
            "{}-{:05}-of-{:05}.gguf",
            self.prefix, no, self.count
        ))
    }
}

/// One logical GGUF namespace over one or more shard files. See the
/// module docs for the shard convention and validation rules.
pub struct ShardedGguf {
    shards: Vec<GgufFile>,
    paths: Vec<PathBuf>,
    /// tensor name -> (shard index, index into that shard's tensor table)
    index: HashMap<String, (usize, usize)>,
    /// Every tensor name a loader has looked up on this set.
    ///
    /// A tensor a loader never asks for is a tensor whose contribution
    /// to the graph is silently missing: `blk.N.attn_sinks.weight` on
    /// gpt-oss, `blk.N.ffn_exp_probs_b` on the newer MoE recipes. The
    /// file still loads, runs at full speed, and emits the wrong
    /// distribution. Recording lookups here is what lets the loader
    /// fail closed on that instead (see
    /// `frink_models::loader::assert_every_tensor_consumed`).
    ///
    /// A lookup counts as consumption even when the caller only probes
    /// for presence: that is deliberately conservative, since the point
    /// is to catch tensors nothing in the code path knows about at all.
    consumed: std::sync::Mutex<std::collections::HashSet<String>>,
}

impl ShardedGguf {
    /// The token *text* a metadata key names by id, e.g.
    /// `tokenizer.ggml.bos_token_id` -> `"<s>"`.
    ///
    /// Two lookups, and both can legitimately miss: the key may be
    /// absent, and the id may be outside the token list a checkpoint
    /// shipped. `None` for either, because a checkpoint without a BOS
    /// token is a real configuration and not an error -- substituting a
    /// placeholder would put a token in the prompt that the model was
    /// never trained to see.
    pub fn token_text(&self, key: &str) -> Option<String> {
        let id = self.metadata_u64(key)? as usize;
        let crate::GgufValue::Array(items) = self.metadata("tokenizer.ggml.tokens")? else {
            return None;
        };
        match items.get(id)? {
            crate::GgufValue::String(s) => Some(s.clone()),
            _ => None,
        }
    }
    /// Opens the shard set containing `path`. `path` may be any shard of
    /// a split checkpoint (siblings are discovered from the canonical
    /// filename) or a plain single-file GGUF (loaded as a one-shard set).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ShardError> {
        let path = path.as_ref();
        let first = GgufFile::open(path)?;

        let declared_count = first.metadata_u64(SPLIT_COUNT_KEY);
        match declared_count {
            None | Some(0) | Some(1) => Self::from_single(first, path.to_path_buf()),
            Some(count) => {
                let name = ShardName::parse(path).ok_or_else(|| {
                    ShardError::NonCanonicalName(path.display().to_string(), count)
                })?;
                if name.count != count {
                    return Err(ShardError::ShardCountMismatch {
                        path: path.display().to_string(),
                        expected: name.count,
                        found: count,
                    });
                }
                Self::from_shard_set(name)
            }
        }
    }

    fn from_single(file: GgufFile, path: PathBuf) -> Result<Self, ShardError> {
        let mut index = HashMap::with_capacity(file.tensors.len());
        for (ti, t) in file.tensors.iter().enumerate() {
            if index.insert(t.name.clone(), (0, ti)).is_some() {
                return Err(ShardError::DuplicateTensorName(t.name.clone(), 1, 1));
            }
        }
        Ok(ShardedGguf {
            shards: vec![file],
            paths: vec![path],
            index,
            consumed: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }

    fn from_shard_set(name: ShardName) -> Result<Self, ShardError> {
        let count = name.count;
        let mut shards = Vec::with_capacity(count as usize);
        let mut paths = Vec::with_capacity(count as usize);

        for no in 1..=count {
            let shard_path = name.sibling(no);
            if !shard_path.exists() {
                return Err(ShardError::MissingShard(
                    shard_path.display().to_string(),
                    count,
                ));
            }
            let shard = GgufFile::open(&shard_path)?;
            let display = shard_path.display().to_string();

            let meta_no =
                shard
                    .metadata_u64(SPLIT_NO_KEY)
                    .ok_or_else(|| ShardError::MissingSplitKey {
                        path: display.clone(),
                        key: SPLIT_NO_KEY.to_string(),
                    })?;
            // `no` (the loop variable) is the filename's 1-based shard
            // number; real `split.no` metadata is 0-based (see the module
            // doc comment), so the expected value is `no - 1`.
            let expected_meta_no = no - 1;
            if meta_no != expected_meta_no {
                return Err(ShardError::ShardNumberMismatch {
                    path: display,
                    expected: expected_meta_no,
                    found: meta_no,
                });
            }
            let meta_count =
                shard
                    .metadata_u64(SPLIT_COUNT_KEY)
                    .ok_or_else(|| ShardError::MissingSplitKey {
                        path: display.clone(),
                        key: SPLIT_COUNT_KEY.to_string(),
                    })?;
            if meta_count != count {
                return Err(ShardError::ShardCountMismatch {
                    path: display,
                    expected: count,
                    found: meta_count,
                });
            }

            shards.push(shard);
            paths.push(shard_path);
        }

        // Cross-shard metadata consistency: any key a later shard shares
        // with the first shard must carry the same value (except
        // split.no, which necessarily differs per shard). Catches a
        // shard set accidentally mixing files from different
        // checkpoints/conversions.
        for (si, shard) in shards.iter().enumerate().skip(1) {
            for (key, value) in &shard.metadata {
                if key == SPLIT_NO_KEY {
                    continue;
                }
                if let Some(first_value) = shards[0].metadata.get(key) {
                    if !gguf_value_eq(first_value, value) {
                        return Err(ShardError::InconsistentMetadata {
                            path: paths[si].display().to_string(),
                            key: key.clone(),
                        });
                    }
                }
            }
        }

        let mut index = HashMap::new();
        for (si, shard) in shards.iter().enumerate() {
            for (ti, t) in shard.tensors.iter().enumerate() {
                if let Some((prev_si, _)) = index.insert(t.name.clone(), (si, ti)) {
                    return Err(ShardError::DuplicateTensorName(
                        t.name.clone(),
                        prev_si as u64 + 1,
                        si as u64 + 1,
                    ));
                }
            }
        }

        if let Some(expected) = shards[0].metadata_u64(SPLIT_TENSORS_COUNT_KEY) {
            let found = index.len() as u64;
            if expected != found {
                return Err(ShardError::TensorCountMismatch { expected, found });
            }
        }

        Ok(ShardedGguf {
            shards,
            paths,
            index,
            consumed: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn tensor_count(&self) -> usize {
        self.index.len()
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Iterates every tensor in the set as `(shard_index, &TensorInfo)`,
    /// shard by shard in shard order.
    pub fn tensors(&self) -> impl Iterator<Item = (usize, &TensorInfo)> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(si, s)| s.tensors.iter().map(move |t| (si, t)))
    }

    /// The index (into `shard_paths()`) of the shard file holding
    /// `name`'s bytes -- for consumers that need to do their own
    /// positional file I/O against the owning shard (e.g. the expert
    /// store's positional-read source).
    pub fn tensor_shard_index(&self, name: &str) -> Option<usize> {
        self.note_consumed(name);
        self.index.get(name).map(|&(si, _)| si)
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.note_consumed(name);
        let &(si, ti) = self.index.get(name)?;
        Some(&self.shards[si].tensors[ti])
    }

    /// Records `name` as consumed without reading it. For loaders that
    /// resolve a tensor through their own index (or deliberately ignore
    /// one) and still want it accounted for.
    pub fn note_consumed(&self, name: &str) {
        self.consumed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.to_string());
    }

    /// Tensor names present in the file that no loader ever looked up.
    /// Sorted, so an error message built from it is deterministic.
    pub fn unconsumed_tensors(&self) -> Vec<String> {
        let consumed = self.consumed.lock().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<String> = self
            .index
            .keys()
            .filter(|n| !consumed.contains(*n))
            .cloned()
            .collect();
        out.sort();
        out
    }

    /// Copy-free tensor byte access, delegating to the owning shard's
    /// mmap. Same contract as `GgufFile::tensor_bytes`.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        self.note_consumed(name);
        let &(si, _) = self
            .index
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        self.shards[si].tensor_bytes(name)
    }

    /// Zero-copy accessor: the owning shard's `Arc<Mmap>` plus the byte
    /// range for `name`. Same contract as `GgufFile::tensor_mapped_range`.
    pub fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<Mmap>, std::ops::Range<usize>), GgufError> {
        self.note_consumed(name);
        let &(si, _) = self
            .index
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        self.shards[si].tensor_mapped_range(name)
    }

    /// Metadata lookup over the merged namespace: the first shard (which
    /// carries the model's full metadata by convention) wins; later
    /// shards are consulted only for keys the first doesn't have.
    pub fn metadata(&self, key: &str) -> Option<&GgufValue> {
        self.shards.iter().find_map(|s| s.metadata.get(key))
    }

    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata(key).and_then(|v| v.as_u64())
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata(key).and_then(|v| v.as_str())
    }
}

impl crate::TensorSource for ShardedGguf {
    fn metadata(&self, key: &str) -> Option<&GgufValue> {
        ShardedGguf::metadata(self, key)
    }
    fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        ShardedGguf::find_tensor(self, name)
    }
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        ShardedGguf::tensor_bytes(self, name)
    }
    fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<Mmap>, std::ops::Range<usize>), GgufError> {
        ShardedGguf::tensor_mapped_range(self, name)
    }
}

/// Structural equality for metadata values. `GgufValue` deliberately
/// doesn't implement `PartialEq` crate-wide (float metadata comparing
/// bitwise-equal is a validation concern, not a general-purpose one), so
/// the comparison lives here next to its single use.
fn gguf_value_eq(a: &GgufValue, b: &GgufValue) -> bool {
    use GgufValue::*;
    match (a, b) {
        (U8(x), U8(y)) => x == y,
        (I8(x), I8(y)) => x == y,
        (U16(x), U16(y)) => x == y,
        (I16(x), I16(y)) => x == y,
        (U32(x), U32(y)) => x == y,
        (I32(x), I32(y)) => x == y,
        (F32(x), F32(y)) => x.to_bits() == y.to_bits(),
        (Bool(x), Bool(y)) => x == y,
        (String(x), String(y)) => x == y,
        (U64(x), U64(y)) => x == y,
        (I64(x), I64(y)) => x == y,
        (F64(x), F64(y)) => x.to_bits() == y.to_bits(),
        (Array(x), Array(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| gguf_value_eq(p, q))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    // --- synthetic shard writer -------------------------------------
    // Same in-memory GGUF construction approach as lib.rs's tests, but
    // parameterized over metadata and tensors so multi-shard sets and
    // corrupt variants can be built.

    enum Kv {
        U16(u16),
        U32(u32),
        Str(&'static str),
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        buf.write_all(s.as_bytes()).unwrap();
    }

    fn build_gguf(kvs: &[(&str, Kv)], tensors: &[(&str, Vec<f32>)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(crate::GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
        buf.write_u64::<LittleEndian>(kvs.len() as u64).unwrap();

        for (key, val) in kvs {
            write_string(&mut buf, key);
            match val {
                Kv::U16(v) => {
                    buf.write_u32::<LittleEndian>(2).unwrap();
                    buf.write_u16::<LittleEndian>(*v).unwrap();
                }
                Kv::U32(v) => {
                    buf.write_u32::<LittleEndian>(4).unwrap();
                    buf.write_u32::<LittleEndian>(*v).unwrap();
                }
                Kv::Str(v) => {
                    buf.write_u32::<LittleEndian>(8).unwrap();
                    write_string(&mut buf, v);
                }
            }
        }

        let mut offset = 0u64;
        for (name, values) in tensors {
            write_string(&mut buf, name);
            buf.write_u32::<LittleEndian>(1).unwrap(); // n_dims
            buf.write_u64::<LittleEndian>(values.len() as u64).unwrap();
            buf.write_u32::<LittleEndian>(0).unwrap(); // dtype F32
            buf.write_u64::<LittleEndian>(offset).unwrap();
            // F32 rows are 4 bytes/element; keep each tensor's data
            // 32-byte aligned the way real writers do.
            let byte_len = (values.len() as u64) * 4;
            offset += byte_len.div_ceil(32) * 32;
        }

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for (_, values) in tensors {
            let start = buf.len();
            for v in values {
                buf.write_f32::<LittleEndian>(*v).unwrap();
            }
            while buf.len() - start < ((values.len() * 4).div_ceil(32) * 32) {
                buf.push(0);
            }
        }
        buf
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let d =
                std::env::temp_dir().join(format!("frink_shard_test_{tag}_{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            TempDir(d)
        }
        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    /// A valid 3-shard set with a metadata-only first shard: the real
    /// published-checkpoint layout this module exists to support.
    fn write_valid_set(dir: &TempDir) -> PathBuf {
        let split = |no: u16| -> Vec<(&'static str, Kv)> {
            vec![
                ("split.no", Kv::U16(no)),
                ("split.count", Kv::U16(3)),
                ("split.tensors.count", Kv::U32(3)),
            ]
        };
        let mut kv1 = split(0);
        kv1.push(("general.architecture", Kv::Str("llama")));
        std::fs::write(
            dir.path("model-00001-of-00003.gguf"),
            build_gguf(&kv1, &[]), // metadata-only first shard
        )
        .unwrap();
        std::fs::write(
            dir.path("model-00002-of-00003.gguf"),
            build_gguf(
                &split(1),
                &[("blk.0.a", vec![1.0, 2.0]), ("blk.0.b", vec![3.0])],
            ),
        )
        .unwrap();
        std::fs::write(
            dir.path("model-00003-of-00003.gguf"),
            build_gguf(&split(2), &[("blk.1.a", vec![4.0, 5.0, 6.0])]),
        )
        .unwrap();
        dir.path("model-00001-of-00003.gguf")
    }

    #[test]
    fn shard_name_parses_canonical_and_rejects_others() {
        let n = ShardName::parse(Path::new("/x/m-00002-of-00017.gguf")).unwrap();
        assert_eq!(n.prefix, "/x/m");
        assert_eq!(n.no, 2);
        assert_eq!(n.count, 17);
        assert_eq!(n.sibling(3), PathBuf::from("/x/m-00003-of-00017.gguf"));

        for bad in [
            "model.gguf",
            "m-2-of-17.gguf",
            "m-00002-of-00017.bin",
            "m-00002_of_00017.gguf",
            "-00002-of-00017.gguf", // empty prefix would need a leading '-'
        ] {
            assert!(ShardName::parse(Path::new(bad)).is_none(), "{bad}");
        }
    }

    #[test]
    fn opens_metadata_only_first_shard_plus_payload_shards() {
        let dir = TempDir::new("valid");
        let first = write_valid_set(&dir);
        let s = ShardedGguf::open(&first).unwrap();
        assert_eq!(s.shard_count(), 3);
        assert_eq!(s.tensor_count(), 3);
        // Metadata resolves from the first shard.
        assert_eq!(s.metadata_str("general.architecture"), Some("llama"));
        // Tensors resolve across shards, bytes intact.
        let b = s.tensor_bytes("blk.1.a").unwrap();
        assert_eq!(b.len(), 12);
        assert_eq!(f32::from_le_bytes(b[0..4].try_into().unwrap()), 4.0);
        let (mmap, range) = s.tensor_mapped_range("blk.0.b").unwrap();
        assert_eq!(
            f32::from_le_bytes(mmap[range.clone()][0..4].try_into().unwrap()),
            3.0
        );
        // Shape metadata survives the merge.
        assert_eq!(s.find_tensor("blk.0.a").unwrap().shape, vec![2]);
        assert!(s.find_tensor("nope").is_none());
        assert!(matches!(
            s.tensor_bytes("nope"),
            Err(GgufError::TensorNotFound(_))
        ));
    }

    #[test]
    fn opening_via_a_non_first_shard_finds_the_same_set() {
        let dir = TempDir::new("nonfirst");
        write_valid_set(&dir);
        let s = ShardedGguf::open(dir.path("model-00002-of-00003.gguf")).unwrap();
        assert_eq!(s.shard_count(), 3);
        assert_eq!(s.metadata_str("general.architecture"), Some("llama"));
    }

    #[test]
    fn single_file_without_split_metadata_is_a_one_shard_set() {
        let dir = TempDir::new("single");
        let p = dir.path("plain.gguf");
        std::fs::write(
            &p,
            build_gguf(
                &[("general.architecture", Kv::Str("llama"))],
                &[("tok_embd.weight", vec![1.0, 2.0, 3.0, 4.0])],
            ),
        )
        .unwrap();
        let s = ShardedGguf::open(&p).unwrap();
        assert_eq!(s.shard_count(), 1);
        assert_eq!(s.tensor_count(), 1);
        assert_eq!(s.tensor_bytes("tok_embd.weight").unwrap().len(), 16);
    }

    #[test]
    fn missing_shard_is_rejected() {
        let dir = TempDir::new("missing");
        let first = write_valid_set(&dir);
        std::fs::remove_file(dir.path("model-00003-of-00003.gguf")).unwrap();
        assert!(matches!(
            ShardedGguf::open(&first),
            Err(ShardError::MissingShard(_, 3))
        ));
    }

    #[test]
    fn duplicate_tensor_name_across_shards_is_rejected() {
        let dir = TempDir::new("dup");
        let first = write_valid_set(&dir);
        // Rewrite shard 3 so it repeats shard 2's tensor name.
        std::fs::write(
            dir.path("model-00003-of-00003.gguf"),
            build_gguf(
                &[
                    ("split.no", Kv::U16(2)),
                    ("split.count", Kv::U16(3)),
                    ("split.tensors.count", Kv::U32(3)),
                ],
                &[("blk.0.a", vec![9.0])],
            ),
        )
        .unwrap();
        assert!(matches!(
            ShardedGguf::open(&first),
            Err(ShardError::DuplicateTensorName(name, 2, 3)) if name == "blk.0.a"
        ));
    }

    #[test]
    fn wrong_split_no_metadata_is_rejected() {
        let dir = TempDir::new("wrongno");
        let first = write_valid_set(&dir);
        std::fs::write(
            dir.path("model-00002-of-00003.gguf"),
            build_gguf(
                &[
                    ("split.no", Kv::U16(7)), // wrong; real (0-based) value would be 1
                    ("split.count", Kv::U16(3)),
                    ("split.tensors.count", Kv::U32(3)),
                ],
                &[("blk.0.a", vec![1.0, 2.0]), ("blk.0.b", vec![3.0])],
            ),
        )
        .unwrap();
        assert!(matches!(
            ShardedGguf::open(&first),
            Err(ShardError::ShardNumberMismatch {
                expected: 1,
                found: 7,
                ..
            })
        ));
    }

    #[test]
    fn tensor_count_mismatch_is_rejected() {
        // Every shard consistently declares 5 total tensors, but the
        // set only contains 2 — the consistency check passes and the
        // count check must catch it.
        let dir = TempDir::new("count");
        let split = |no: u16| -> Vec<(&'static str, Kv)> {
            vec![
                ("split.no", Kv::U16(no)),
                ("split.count", Kv::U16(2)),
                ("split.tensors.count", Kv::U32(5)),
            ]
        };
        std::fs::write(
            dir.path("model-00001-of-00002.gguf"),
            build_gguf(&split(0), &[("blk.0.a", vec![1.0])]),
        )
        .unwrap();
        std::fs::write(
            dir.path("model-00002-of-00002.gguf"),
            build_gguf(&split(1), &[("blk.1.a", vec![2.0])]),
        )
        .unwrap();
        assert!(matches!(
            ShardedGguf::open(dir.path("model-00001-of-00002.gguf")),
            Err(ShardError::TensorCountMismatch {
                expected: 5,
                found: 2
            })
        ));
    }

    #[test]
    fn inconsistent_shared_metadata_is_rejected() {
        let dir = TempDir::new("meta");
        let first = write_valid_set(&dir);
        // Shard 2 claims a different architecture than shard 1 — a mixed
        // set, e.g. shards from two different conversions.
        std::fs::write(
            dir.path("model-00002-of-00003.gguf"),
            build_gguf(
                &[
                    ("split.no", Kv::U16(1)),
                    ("split.count", Kv::U16(3)),
                    ("split.tensors.count", Kv::U32(3)),
                    ("general.architecture", Kv::Str("qwen2")),
                ],
                &[("blk.0.a", vec![1.0, 2.0]), ("blk.0.b", vec![3.0])],
            ),
        )
        .unwrap();
        assert!(matches!(
            ShardedGguf::open(&first),
            Err(ShardError::InconsistentMetadata { key, .. }) if key == "general.architecture"
        ));
    }

    #[test]
    fn split_count_with_non_canonical_filename_is_rejected() {
        let dir = TempDir::new("noncanon");
        let p = dir.path("weird-name.gguf");
        std::fs::write(
            &p,
            build_gguf(
                &[
                    ("split.no", Kv::U16(0)),
                    ("split.count", Kv::U16(2)),
                    ("split.tensors.count", Kv::U32(0)),
                ],
                &[],
            ),
        )
        .unwrap();
        assert!(matches!(
            ShardedGguf::open(&p),
            Err(ShardError::NonCanonicalName(_, 2))
        ));
    }

    #[test]
    fn out_of_range_tensor_offset_is_a_clean_error() {
        let dir = TempDir::new("badoffset");
        let p = dir.path("plain.gguf");
        let mut bytes = build_gguf(
            &[("general.architecture", Kv::Str("llama"))],
            &[("tok_embd.weight", vec![1.0; 16])],
        );
        // Truncate the data section so the declared tensor span runs
        // past end-of-file.
        bytes.truncate(bytes.len() - 32);
        std::fs::write(&p, &bytes).unwrap();
        let s = ShardedGguf::open(&p).unwrap();
        assert!(matches!(
            s.tensor_bytes("tok_embd.weight"),
            Err(GgufError::TruncatedTensor(_, _, _))
        ));
    }

    #[test]
    fn tensors_iterator_walks_all_shards_in_order() {
        let dir = TempDir::new("iter");
        let first = write_valid_set(&dir);
        let s = ShardedGguf::open(&first).unwrap();
        let names: Vec<(usize, String)> = s.tensors().map(|(si, t)| (si, t.name.clone())).collect();
        assert_eq!(
            names,
            vec![
                (1, "blk.0.a".to_string()),
                (1, "blk.0.b".to_string()),
                (2, "blk.1.a".to_string()),
            ]
        );
    }
}
