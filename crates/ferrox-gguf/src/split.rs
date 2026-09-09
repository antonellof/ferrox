//! Splitting one GGUF into shards and merging shards back: the writing
//! half of what [`crate::sharded`] reads, and the port of llama.cpp's
//! `tools/gguf-split/gguf-split.cpp`.
//!
//! The shard convention is the one `ShardedGguf` already validates, so
//! the two halves cannot drift about it: shard filenames come from
//! [`ShardName::sibling`], the three `split.*` keys from the constants
//! in `sharded`, and a written set is proven by opening it with
//! `ShardedGguf::open` in the tests below.
//!
//! What llama.cpp's tool does, with the lines this port follows:
//!
//! * The first shard carries the whole source metadata
//!   (`gguf-split.cpp:236`); every later shard carries ONLY `split.no`
//!   (u16, 0-based), `split.count` (u16) and `split.tensors.count`
//!   (i32, the total across all shards) (`:238-240`, `:272`).
//! * Tensor mode starts a new shard every N tensors (`:288`); size mode
//!   starts one when the running sum of alignment-padded tensor bytes
//!   would exceed the limit (`:256-258`, `:285`). A shard is never
//!   left empty, which is only possible when the first tensor alone
//!   exceeds the size limit (`:227`), except the first shard under
//!   `--no-tensor-first-split` (`:247-249`).
//! * Filenames are `<prefix>-NNNNN-of-MMMMM.gguf`, 1-based
//!   (`src/llama.cpp:537`, `llama_split_path`).
//! * Merge takes the FIRST shard by name (`:474`), requires
//!   `split.count` (`:449`), keeps the first shard's metadata with
//!   `split.count` rewritten to 0 so the output is not itself read as
//!   a shard (`:486`), and refuses to overwrite an existing output
//!   (`:410`).
//!
//! Two deliberate differences, both from choices the rest of this crate
//! already made: metadata keys are written in sorted order, because the
//! reader keeps them in a `HashMap` and the writer sorts (see
//! [`GgufWriter::create`]); and shards are padded with the alignment
//! the source declares, where llama.cpp's tool pads with 32 whatever
//! `general.alignment` says (`gguf_init_empty` fixes `ctx->alignment`
//! at `GGUF_DEFAULT_ALIGNMENT`, `ggml/src/gguf.cpp:223`, and
//! `gguf_set_kv` never updates it).
//!
//! Tensor bytes are never buffered: each one is written straight from
//! the source file's mapping.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::sharded::{ShardName, SPLIT_COUNT_KEY, SPLIT_NO_KEY, SPLIT_TENSORS_COUNT_KEY};
use crate::writer::{declared_alignment, encode_header};
use crate::{
    GgufError, GgufFile, GgufValue, GgufWriteError, GgufWriter, ShardError, ShardedGguf,
    TensorInfo, TensorPlan,
};

/// llama.cpp's default for `--split-max-tensors` (`gguf-split.cpp:45`).
pub const DEFAULT_MAX_TENSORS: usize = 128;

/// How shard boundaries are chosen. Mirrors `split_mode` in
/// `gguf-split.cpp:35-39`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitMode {
    /// At most this many tensors per shard.
    MaxTensors(usize),
    /// At most this many alignment-padded tensor bytes per shard. The
    /// header is not counted, exactly as llama.cpp does not count it.
    MaxBytes(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitOptions {
    pub mode: SplitMode,
    /// Leave the first shard metadata-only (`--no-tensor-first-split`).
    pub no_tensor_first_split: bool,
}

#[derive(Debug, Error)]
pub enum SplitError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Write(#[from] GgufWriteError),
    #[error(transparent)]
    Shard(#[from] ShardError),
    #[error("{path}: {source}")]
    Io { path: String, source: io::Error },
    #[error("--split-max-tensors must be at least 1")]
    ZeroMaxTensors,
    #[error("--split-max-size must be at least 1 byte")]
    ZeroMaxBytes,
    #[error(
        "the input is itself one shard of a {0}-file split. Splitting one shard would produce \
         a set whose split.tensors.count covers that shard alone; merge the set first (ferrox \
         gguf-split --merge) and split the result"
    )]
    InputIsSplit(u64),
    #[error(
        "tensor '{name}' is {bytes} bytes after padding, more than the {max}-byte shard limit, \
         so the shard it starts would have to stay empty (llama.cpp refuses this as 'one of \
         splits have 0 tensors'); raise --split-max-size"
    )]
    TensorExceedsShard { name: String, bytes: u64, max: u64 },
    #[error("{0} shards do not fit split.count, which GGUF stores as a u16")]
    TooManyShards(usize),
    #[error("{0} tensors do not fit split.tensors.count, which GGUF stores as an i32")]
    TooManyTensors(usize),
    #[error("output prefix '{0}' is not valid UTF-8, so no shard name can be built from it")]
    NonUtf8Prefix(String),
    #[error(
        "'{0}' is not the first shard of a set: merge takes '<prefix>-00001-of-MMMMM.gguf' and \
         finds the rest from it"
    )]
    NotFirstShard(String),
    #[error("'{0}' carries no split.count metadata, so it is not a split GGUF")]
    NotSplit(String),
    #[error("'{0}' declares split.count = 0, which is not a valid shard count")]
    ZeroSplitCount(String),
    #[error("output '{0}' already exists; merge refuses to overwrite it")]
    OutputExists(String),
}

/// One shard of a [`SplitPlan`]: what it will carry and how big it
/// will be, computed before any file is opened for writing.
#[derive(Debug, Clone)]
pub struct ShardPlan {
    /// The shard's metadata: everything the source had for the first
    /// shard, the three `split.*` keys for the rest.
    pub metadata: BTreeMap<String, GgufValue>,
    /// The shard's tensors, in source order.
    pub tensors: Vec<TensorPlan>,
    /// Header bytes, padding to the data section included: what
    /// llama.cpp reports as `gguf_get_meta_size`.
    pub header_bytes: usize,
    /// Tensor bytes, unpadded, the way llama.cpp's `print_info` sums
    /// them (`gguf-split.cpp:302`).
    pub data_bytes: u64,
    /// The size the shard file will have on disk.
    pub file_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct SplitPlan {
    pub shards: Vec<ShardPlan>,
    /// Tensors across all shards.
    pub n_tensors: usize,
}

/// A merge, planned before anything is written: the shard set held
/// open and the output's metadata and tensor order.
pub struct MergePlan {
    set: ShardedGguf,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<TensorPlan>,
}

impl MergePlan {
    pub fn shard_paths(&self) -> &[PathBuf] {
        self.set.shard_paths()
    }
}

impl std::fmt::Debug for MergePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergePlan")
            .field("shard_paths", &self.set.shard_paths())
            .field("metadata_keys", &self.metadata.len())
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

/// A source tensor's header entry for the output, sized from its
/// dtype. `None` from `byte_len` is a dtype whose block layout this
/// build does not know, and the source's own `tensor_bytes` would
/// refuse it the same way.
fn tensor_plan(t: &TensorInfo) -> Result<TensorPlan, GgufError> {
    let byte_len = t
        .byte_len()
        .ok_or_else(|| GgufError::UnsizedTensor(t.name.clone(), t.dtype))?;
    Ok(TensorPlan {
        name: t.name.clone(),
        shape: t.shape.clone(),
        dtype: t.dtype,
        byte_len,
    })
}

fn split_keys(no: u16, count: u16, n_tensors: i32) -> [(&'static str, GgufValue); 3] {
    // The types are llama.cpp's exactly: `gguf_set_val_u16` for
    // split.no and split.count, `gguf_set_val_i32` for
    // split.tensors.count (`gguf-split.cpp:238-240`, `:272`).
    [
        (SPLIT_NO_KEY, GgufValue::U16(no)),
        (SPLIT_COUNT_KEY, GgufValue::U16(count)),
        (SPLIT_TENSORS_COUNT_KEY, GgufValue::I32(n_tensors)),
    ]
}

fn io_err(path: &Path, source: io::Error) -> SplitError {
    SplitError::Io {
        path: path.display().to_string(),
        source,
    }
}

/// Decides which tensors go in which shard and what each shard will
/// look like. Pure: reads the source's header, opens nothing.
pub fn plan_split(source: &GgufFile, opts: &SplitOptions) -> Result<SplitPlan, SplitError> {
    if let Some(count) = source.metadata_u64(SPLIT_COUNT_KEY).filter(|&c| c > 1) {
        return Err(SplitError::InputIsSplit(count));
    }
    match opts.mode {
        SplitMode::MaxTensors(0) => return Err(SplitError::ZeroMaxTensors),
        SplitMode::MaxBytes(0) => return Err(SplitError::ZeroMaxBytes),
        SplitMode::MaxTensors(_) | SplitMode::MaxBytes(_) => {}
    }
    let alignment = declared_alignment(source.metadata.get("general.alignment"))? as u64;
    let n_tensors = source.tensors.len();
    let tensors_count =
        i32::try_from(n_tensors).map_err(|_| SplitError::TooManyTensors(n_tensors))?;

    let plans: Vec<TensorPlan> = source
        .tensors
        .iter()
        .map(tensor_plan)
        .collect::<Result<_, _>>()?;

    // Group source indices. `gguf-split.cpp:243-268`.
    let mut groups: Vec<Vec<usize>> = vec![Vec::new()];
    if opts.no_tensor_first_split {
        groups.push(Vec::new());
    }
    let mut current: u64 = 0;
    for (i, t) in plans.iter().enumerate() {
        let padded = (t.byte_len as u64).next_multiple_of(alignment);
        let next = current.saturating_add(padded);
        let boundary = match opts.mode {
            SplitMode::MaxBytes(max) => next > max,
            SplitMode::MaxTensors(n) => i > 0 && i % n == 0,
        };
        if boundary {
            let last = groups.last().expect("groups starts non-empty");
            if last.is_empty() {
                // Only reachable in size mode at i == 0: tensor mode
                // never closes a shard before it has N tensors, and a
                // later shard always opens with the tensor that
                // overflowed the previous one.
                let max = match opts.mode {
                    SplitMode::MaxBytes(max) => max,
                    SplitMode::MaxTensors(_) => unreachable!("tensor mode never closes an empty shard"),
                };
                return Err(SplitError::TensorExceedsShard {
                    name: t.name.clone(),
                    bytes: padded,
                    max,
                });
            }
            groups.push(vec![i]);
            current = padded;
        } else {
            groups.last_mut().expect("groups starts non-empty").push(i);
            current = next;
        }
    }

    let count = u16::try_from(groups.len()).map_err(|_| SplitError::TooManyShards(groups.len()))?;

    let mut shards = Vec::with_capacity(groups.len());
    for (no, group) in groups.iter().enumerate() {
        let mut metadata: BTreeMap<String, GgufValue> = if no == 0 {
            source
                .metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        } else {
            BTreeMap::new()
        };
        for (key, value) in split_keys(no as u16, count, tensors_count) {
            metadata.insert(key.to_string(), value);
        }
        let tensors: Vec<TensorPlan> = group.iter().map(|&i| plans[i].clone()).collect();
        let header_bytes = encode_header(&metadata, &tensors)?.len();
        let data_bytes: u64 = tensors.iter().map(|t| t.byte_len as u64).sum();
        let padded_bytes: u64 = tensors
            .iter()
            .map(|t| (t.byte_len as u64).next_multiple_of(alignment))
            .sum();
        shards.push(ShardPlan {
            metadata,
            tensors,
            header_bytes,
            data_bytes,
            file_bytes: header_bytes as u64 + padded_bytes,
        });
    }
    Ok(SplitPlan { shards, n_tensors })
}

/// The shard path `plan` will write for 0-based shard `no`.
pub fn shard_path(prefix: &Path, no: usize, count: usize) -> Result<PathBuf, SplitError> {
    let prefix = prefix
        .to_str()
        .ok_or_else(|| SplitError::NonUtf8Prefix(prefix.display().to_string()))?;
    let name = ShardName {
        prefix: prefix.to_string(),
        no: 0,
        count: count as u64,
    };
    Ok(name.sibling(no as u64 + 1))
}

/// Writes every shard of `plan` as `<prefix>-NNNNN-of-MMMMM.gguf`,
/// calling `on_shard` with each path as it is about to be written.
/// Returns the paths written. Tensor bytes go from `source`'s mapping
/// to the file with no intermediate buffer.
pub fn write_split(
    source: &GgufFile,
    plan: &SplitPlan,
    prefix: &Path,
    mut on_shard: impl FnMut(&Path),
) -> Result<Vec<PathBuf>, SplitError> {
    let count = plan.shards.len();
    let mut paths = Vec::with_capacity(count);
    for (no, shard) in plan.shards.iter().enumerate() {
        let path = shard_path(prefix, no, count)?;
        on_shard(&path);
        let file = File::create(&path).map_err(|e| io_err(&path, e))?;
        let mut w = GgufWriter::create(
            BufWriter::new(file),
            &shard.metadata,
            shard.tensors.clone(),
        )?;
        for t in &shard.tensors {
            w.write_tensor(&t.name, source.tensor_bytes(&t.name)?)?;
        }
        w.finish()?;
        paths.push(path);
    }
    Ok(paths)
}

/// Opens the shard set whose first file is `first` and lays out the
/// merged file. Every shard is validated by `ShardedGguf::open`, so a
/// missing sibling refuses here, by path, before anything is written.
pub fn plan_merge(first: &Path) -> Result<MergePlan, SplitError> {
    let display = first.display().to_string();
    // `llama_split_prefix` with i_split = 0 (`gguf-split.cpp:474`):
    // the name has to end in `-00001-of-MMMMM.gguf`.
    let name = ShardName::parse(first)
        .filter(|n| n.no == 1)
        .ok_or_else(|| SplitError::NotFirstShard(display.clone()))?;
    let first_file = GgufFile::open(first)?;
    let count = first_file
        .metadata_u64(SPLIT_COUNT_KEY)
        .ok_or_else(|| SplitError::NotSplit(display.clone()))?;
    if count == 0 {
        return Err(SplitError::ZeroSplitCount(display));
    }
    if name.count != count {
        return Err(ShardError::ShardCountMismatch {
            path: display,
            expected: name.count,
            found: count,
        }
        .into());
    }
    let set = ShardedGguf::open(first)?;

    let mut metadata: BTreeMap<String, GgufValue> = first_file
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    // `gguf-split.cpp:486`: the merged file keeps split.no and
    // split.tensors.count, and gets split.count = 0 so nothing reads it
    // as a shard again.
    metadata.insert(SPLIT_COUNT_KEY.to_string(), GgufValue::U16(0));

    let tensors = set
        .tensors()
        .map(|(_, t)| tensor_plan(t))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MergePlan {
        set,
        metadata,
        tensors,
    })
}

/// Writes the merged file. Refuses an existing `out` the way llama.cpp
/// does (`gguf-split.cpp:410`).
pub fn write_merge(plan: &MergePlan, out: &Path) -> Result<(), SplitError> {
    if out.exists() {
        return Err(SplitError::OutputExists(out.display().to_string()));
    }
    let file = File::create(out).map_err(|e| io_err(out, e))?;
    let mut w = GgufWriter::create(BufWriter::new(file), &plan.metadata, plan.tensors.clone())?;
    for t in &plan.tensors {
        w.write_tensor(&t.name, plan.set.tensor_bytes(&t.name)?)?;
    }
    w.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgmlType;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let d = std::env::temp_dir().join(format!(
                "ferrox_split_test_{tag}_{}_{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            std::fs::remove_dir_all(&d).ok();
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

    /// Five tensors of three dtypes, sized so no two are equal and so
    /// several are not multiples of the 32-byte alignment (the padding
    /// arithmetic has to be exercised, not skipped).
    fn synthetic_tensors() -> Vec<(TensorPlan, Vec<u8>)> {
        let mk = |name: &str, shape: Vec<u64>, dtype: GgmlType, seed: u8| {
            let n = TensorInfo {
                name: name.into(),
                shape: shape.clone(),
                dtype,
                offset: 0,
            }
            .byte_len()
            .unwrap();
            let bytes: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(seed)).collect();
            (
                TensorPlan {
                    name: name.into(),
                    shape,
                    dtype,
                    byte_len: n,
                },
                bytes,
            )
        };
        vec![
            mk("token_embd.weight", vec![8, 6], GgmlType::F32, 3), // 192
            mk("blk.0.attn_q.weight", vec![32, 3], GgmlType::Q8_0, 5), // 102
            mk("blk.0.attn_norm.weight", vec![5], GgmlType::F32, 7), // 20
            mk("blk.1.ffn_up.weight", vec![7, 3], GgmlType::F16, 11), // 42
            mk("output_norm.weight", vec![9], GgmlType::F32, 13),   // 36
        ]
    }

    fn base_metadata() -> BTreeMap<String, GgufValue> {
        [
            ("general.architecture", GgufValue::String("llama".into())),
            ("general.name", GgufValue::String("synthetic".into())),
            ("llama.block_count", GgufValue::U32(2)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(vec![
                    GgufValue::String("<s>".into()),
                    GgufValue::String("hi".into()),
                ]),
            ),
            ("llama.rope.freq_base", GgufValue::F32(10000.0)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    fn write_source(
        path: &Path,
        metadata: &BTreeMap<String, GgufValue>,
        tensors: &[(TensorPlan, Vec<u8>)],
    ) {
        let file = File::create(path).unwrap();
        let plan: Vec<TensorPlan> = tensors.iter().map(|(p, _)| p.clone()).collect();
        let mut w = GgufWriter::create(BufWriter::new(file), metadata, plan).unwrap();
        for (p, bytes) in tensors {
            w.write_tensor(&p.name, bytes).unwrap();
        }
        w.finish().unwrap();
    }

    fn by_tensors(n: usize) -> SplitOptions {
        SplitOptions {
            mode: SplitMode::MaxTensors(n),
            no_tensor_first_split: false,
        }
    }

    fn split_to(dir: &TempDir, src: &Path, opts: &SplitOptions) -> (SplitPlan, Vec<PathBuf>) {
        let source = GgufFile::open(src).unwrap();
        let plan = plan_split(&source, opts).unwrap();
        let paths = write_split(&source, &plan, &dir.path("out"), |_| {}).unwrap();
        (plan, paths)
    }

    fn assert_same_values(a: &GgufValue, b: &GgufValue, key: &str) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "metadata '{key}' changed");
    }

    /// The property the module exists for: the shards this writes are
    /// what `ShardedGguf` reads back, tensor for tensor and key for
    /// key, and each shard carries llama.cpp's three keys with
    /// llama.cpp's types and values.
    #[test]
    fn a_split_set_reads_back_through_the_shard_loader_as_the_original() {
        let dir = TempDir::new("roundtrip");
        let tensors = synthetic_tensors();
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &tensors);
        let (plan, paths) = split_to(&dir, &src, &by_tensors(2));

        assert_eq!(plan.shards.len(), 3, "5 tensors at 2 per shard is 3 shards");
        assert_eq!(
            paths,
            vec![
                dir.path("out-00001-of-00003.gguf"),
                dir.path("out-00002-of-00003.gguf"),
                dir.path("out-00003-of-00003.gguf"),
            ]
        );

        // Per-shard keys, exactly as gguf-split.cpp:238-240 writes them.
        for (no, path) in paths.iter().enumerate() {
            let shard = GgufFile::open(path).unwrap();
            assert!(
                matches!(shard.metadata.get(SPLIT_NO_KEY), Some(GgufValue::U16(n)) if *n as usize == no),
                "shard {no} split.no: {:?}",
                shard.metadata.get(SPLIT_NO_KEY)
            );
            assert!(
                matches!(shard.metadata.get(SPLIT_COUNT_KEY), Some(GgufValue::U16(3))),
                "shard {no} split.count: {:?}",
                shard.metadata.get(SPLIT_COUNT_KEY)
            );
            assert!(
                matches!(
                    shard.metadata.get(SPLIT_TENSORS_COUNT_KEY),
                    Some(GgufValue::I32(5))
                ),
                "shard {no} split.tensors.count: {:?}",
                shard.metadata.get(SPLIT_TENSORS_COUNT_KEY)
            );
            if no == 0 {
                assert_eq!(shard.metadata.len(), base_metadata().len() + 3);
            } else {
                // gguf-split.cpp:234: "Save all metadata in first split only".
                assert_eq!(shard.metadata.len(), 3, "shard {no} carries only split.*");
            }
            assert_eq!(shard.tensors.len(), plan.shards[no].tensors.len());
        }

        let set = ShardedGguf::open(&paths[0]).unwrap();
        assert_eq!(set.shard_count(), 3);
        assert_eq!(set.tensor_count(), tensors.len());
        let source = GgufFile::open(&src).unwrap();
        for (p, bytes) in &tensors {
            assert_eq!(set.tensor_bytes(&p.name).unwrap(), &bytes[..], "{}", p.name);
            let info = set.find_tensor(&p.name).unwrap();
            assert_eq!(info.shape, p.shape);
            assert_eq!(info.dtype, p.dtype);
        }
        for (key, value) in &source.metadata {
            assert_same_values(set.metadata(key).unwrap(), value, key);
        }
        // Order is preserved across shards: shard order, then table order.
        let order: Vec<String> = set.tensors().map(|(_, t)| t.name.clone()).collect();
        let want: Vec<String> = tensors.iter().map(|(p, _)| p.name.clone()).collect();
        assert_eq!(order, want);
    }

    /// Where llama.cpp's own tool is byte-identical, so is this one: a
    /// source that already carries `split.no = 0`, `split.count = 0`
    /// and `split.tensors.count = N` (what a previous merge leaves
    /// behind, `gguf-split.cpp:486`) survives split then merge without
    /// a single byte changing.
    #[test]
    fn merging_the_shards_reproduces_a_merged_source_byte_for_byte() {
        let dir = TempDir::new("identical");
        let tensors = synthetic_tensors();
        let mut meta = base_metadata();
        for (key, value) in split_keys(0, 0, tensors.len() as i32) {
            meta.insert(key.to_string(), value);
        }
        let src = dir.path("src.gguf");
        write_source(&src, &meta, &tensors);
        let (_, paths) = split_to(&dir, &src, &by_tensors(2));

        let out = dir.path("merged.gguf");
        let plan = plan_merge(&paths[0]).unwrap();
        write_merge(&plan, &out).unwrap();
        assert_eq!(
            std::fs::read(&out).unwrap(),
            std::fs::read(&src).unwrap(),
            "merge output differs from the source"
        );
    }

    /// A source WITHOUT the split keys is not byte-identical after a
    /// merge, in llama.cpp either: the merged file gains exactly
    /// `split.no = 0`, `split.count = 0`, `split.tensors.count = N`
    /// and nothing else changes.
    #[test]
    fn merging_a_plain_source_adds_exactly_the_three_split_keys() {
        let dir = TempDir::new("plain");
        let tensors = synthetic_tensors();
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &tensors);
        let (_, paths) = split_to(&dir, &src, &by_tensors(2));
        let out = dir.path("merged.gguf");
        write_merge(&plan_merge(&paths[0]).unwrap(), &out).unwrap();

        let merged = GgufFile::open(&out).unwrap();
        let source = GgufFile::open(&src).unwrap();
        assert_eq!(merged.metadata.len(), source.metadata.len() + 3);
        for (key, value) in &source.metadata {
            assert_same_values(merged.metadata.get(key).unwrap(), value, key);
        }
        for (key, value) in split_keys(0, 0, tensors.len() as i32) {
            assert_same_values(merged.metadata.get(key).unwrap(), &value, key);
        }
        for (p, bytes) in &tensors {
            assert_eq!(merged.tensor_bytes(&p.name).unwrap(), &bytes[..]);
        }
        let order: Vec<&str> = merged.tensors.iter().map(|t| t.name.as_str()).collect();
        let want: Vec<&str> = tensors.iter().map(|(p, _)| p.name.as_str()).collect();
        assert_eq!(order, want);
    }

    /// The refusal the task asked for by name: a merge whose set is
    /// missing a shard stops before writing and says which file.
    #[test]
    fn a_merge_with_a_missing_shard_refuses_naming_the_shard() {
        let dir = TempDir::new("missing");
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &synthetic_tensors());
        let (_, paths) = split_to(&dir, &src, &by_tensors(2));
        std::fs::remove_file(&paths[1]).unwrap();

        let err = plan_merge(&paths[0]).unwrap_err();
        match err {
            SplitError::Shard(ShardError::MissingShard(path, 3)) => {
                assert_eq!(path, paths[1].display().to_string());
            }
            other => panic!("expected MissingShard naming shard 2, got {other:?}"),
        }
        assert!(!dir.path("merged.gguf").exists());
    }

    /// Size mode groups by alignment-PADDED bytes (`gguf-split.cpp:256`),
    /// so a limit that the unpadded sizes fit and the padded ones do
    /// not has to start a new shard. Tensor sizes: 192, 102 (pads to
    /// 128), 20 (32), 42 (64), 36 (64).
    #[test]
    fn size_mode_groups_by_padded_bytes_like_llama_cpp() {
        let dir = TempDir::new("size");
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &synthetic_tensors());
        let source = GgufFile::open(&src).unwrap();

        // 192 + 128 = 320 fits; + 32 = 352 does not.
        let plan = plan_split(
            &source,
            &SplitOptions {
                mode: SplitMode::MaxBytes(330),
                no_tensor_first_split: false,
            },
        )
        .unwrap();
        let groups: Vec<usize> = plan.shards.iter().map(|s| s.tensors.len()).collect();
        // [192+128] [32+64+64=160]
        assert_eq!(groups, vec![2, 3]);

        // Unpadded 192+102+20 = 314 < 330 would have put three in the
        // first shard; the padded arithmetic is what decides.
        let plan = plan_split(
            &source,
            &SplitMode::MaxBytes(314).with_no_first(false),
        )
        .unwrap();
        let groups: Vec<usize> = plan.shards.iter().map(|s| s.tensors.len()).collect();
        assert_eq!(groups, vec![1, 4], "192 alone; 128+32+64+64 = 288");
    }

    impl SplitMode {
        fn with_no_first(self, no_tensor_first_split: bool) -> SplitOptions {
            SplitOptions {
                mode: self,
                no_tensor_first_split,
            }
        }
    }

    /// The only way a shard can end up empty in size mode is the first
    /// tensor alone exceeding the limit; llama.cpp exits with "one of
    /// splits have 0 tensors" (`gguf-split.cpp:227`). This names the
    /// tensor and the two numbers instead.
    #[test]
    fn a_first_tensor_larger_than_the_size_limit_is_refused_by_name() {
        let dir = TempDir::new("toobig");
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &synthetic_tensors());
        let source = GgufFile::open(&src).unwrap();
        let err = plan_split(&source, &SplitMode::MaxBytes(100).with_no_first(false)).unwrap_err();
        assert!(
            matches!(&err, SplitError::TensorExceedsShard { name, bytes: 192, max: 100 }
                if name == "token_embd.weight"),
            "got {err:?}"
        );
        // Same with a metadata-only first shard: the SECOND shard is the
        // one that would be empty.
        let err = plan_split(&source, &SplitMode::MaxBytes(100).with_no_first(true)).unwrap_err();
        assert!(matches!(err, SplitError::TensorExceedsShard { .. }), "got {err:?}");
    }

    /// `--no-tensor-first-split` produces the metadata-only first shard
    /// real published checkpoints use, and the loader reads the set.
    #[test]
    fn no_tensor_first_split_leaves_the_first_shard_metadata_only() {
        let dir = TempDir::new("nofirst");
        let tensors = synthetic_tensors();
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &tensors);
        let (plan, paths) = split_to(&dir, &src, &SplitMode::MaxTensors(3).with_no_first(true));
        let groups: Vec<usize> = plan.shards.iter().map(|s| s.tensors.len()).collect();
        assert_eq!(groups, vec![0, 3, 2]);
        let first = GgufFile::open(&paths[0]).unwrap();
        assert!(first.tensors.is_empty());
        assert_eq!(first.metadata_str("general.architecture"), Some("llama"));
        let set = ShardedGguf::open(&paths[0]).unwrap();
        assert_eq!(set.tensor_count(), tensors.len());
        for (p, bytes) in &tensors {
            assert_eq!(set.tensor_bytes(&p.name).unwrap(), &bytes[..]);
        }
        // And it merges back.
        let out = dir.path("merged.gguf");
        write_merge(&plan_merge(&paths[0]).unwrap(), &out).unwrap();
        assert_eq!(GgufFile::open(&out).unwrap().tensors.len(), tensors.len());
    }

    /// The dry run's numbers are the numbers the write produces: a plan
    /// that reported one size and wrote another would make `--dry-run`
    /// a guess. Ties `header_bytes` (the shared `encode_header`) and the
    /// padded data sum to the file on disk.
    #[test]
    fn a_dry_run_plan_reports_the_sizes_the_write_produces() {
        let dir = TempDir::new("dryrun");
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &synthetic_tensors());
        let (plan, paths) = split_to(&dir, &src, &by_tensors(2));
        for (shard, path) in plan.shards.iter().zip(&paths) {
            let on_disk = std::fs::metadata(path).unwrap().len();
            assert_eq!(shard.file_bytes, on_disk, "{}", path.display());
            assert!(shard.header_bytes as u64 + shard.data_bytes <= on_disk);
        }
        assert_eq!(plan.n_tensors, 5);
    }

    /// A shard passed to `--split` would be split as if it were a whole
    /// model, with a `split.tensors.count` covering only itself. The
    /// gate is reachable: any file out of `write_split` trips it.
    #[test]
    fn an_input_that_is_already_a_shard_is_refused() {
        let dir = TempDir::new("reshard");
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &synthetic_tensors());
        let (_, paths) = split_to(&dir, &src, &by_tensors(2));
        let shard = GgufFile::open(&paths[1]).unwrap();
        let err = plan_split(&shard, &by_tensors(2)).unwrap_err();
        assert!(
            matches!(err, SplitError::InputIsSplit(3)),
            "got {err:?}"
        );
    }

    #[test]
    fn merge_refuses_a_non_first_shard_a_non_split_file_and_an_existing_output() {
        let dir = TempDir::new("mergerefuse");
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &synthetic_tensors());
        let (_, paths) = split_to(&dir, &src, &by_tensors(2));

        // gguf-split.cpp:474: the input must be shard 1 by name.
        let err = plan_merge(&paths[1]).unwrap_err();
        assert!(matches!(err, SplitError::NotFirstShard(_)), "got {err:?}");

        // gguf-split.cpp:449: no split.count means not a split. The name
        // has to pass the shape check first, so give the plain file one.
        let plain = dir.path("plain-00001-of-00001.gguf");
        std::fs::copy(&src, &plain).unwrap();
        let err = plan_merge(&plain).unwrap_err();
        assert!(matches!(err, SplitError::NotSplit(_)), "got {err:?}");

        // gguf-split.cpp:410: never overwrite.
        let plan = plan_merge(&paths[0]).unwrap();
        let err = write_merge(&plan, &src).unwrap_err();
        assert!(matches!(err, SplitError::OutputExists(_)), "got {err:?}");
    }

    /// `split.count` in the filename and in the metadata are two
    /// statements of one number; a first shard whose two disagree is
    /// refused before any sibling is opened.
    #[test]
    fn merge_refuses_a_first_shard_whose_name_and_metadata_disagree_on_count() {
        let dir = TempDir::new("countmismatch");
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &synthetic_tensors());
        let (_, paths) = split_to(&dir, &src, &by_tensors(2));
        let renamed = dir.path("out-00001-of-00004.gguf");
        std::fs::rename(&paths[0], &renamed).unwrap();
        let err = plan_merge(&renamed).unwrap_err();
        assert!(
            matches!(
                err,
                SplitError::Shard(ShardError::ShardCountMismatch {
                    expected: 4,
                    found: 3,
                    ..
                })
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn zero_limits_are_refused() {
        let dir = TempDir::new("zero");
        let src = dir.path("src.gguf");
        write_source(&src, &base_metadata(), &synthetic_tensors());
        let source = GgufFile::open(&src).unwrap();
        assert!(matches!(
            plan_split(&source, &by_tensors(0)),
            Err(SplitError::ZeroMaxTensors)
        ));
        assert!(matches!(
            plan_split(&source, &SplitMode::MaxBytes(0).with_no_first(false)),
            Err(SplitError::ZeroMaxBytes)
        ));
    }
}
