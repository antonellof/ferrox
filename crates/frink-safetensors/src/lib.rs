//! frink-safetensors: a from-scratch reader for the safetensors file format
//! (spec + reference implementation: https://github.com/huggingface/safetensors).
//! Needed because Kimi K3 ships its real checkpoint as safetensors shards
//! (96 of them, plus a `model.safetensors.index.json`), not GGUF.
//!
//! Implemented independently against the public format: an 8-byte
//! little-endian header length, followed by a UTF-8 JSON header describing
//! each tensor's dtype/shape/byte range, followed by the raw tensor bytes.
//! The validation rules below (offsets must be contiguous starting at 0,
//! non-overlapping, size-consistent with dtype+shape, and must exactly
//! cover the rest of the file) were read from and verified against the
//! real reference implementation's `Metadata::validate` (a shallow clone of
//! `huggingface/safetensors`, scratchpad-only, not vendored) rather than
//! guessed — see docs/THIRD_PARTY_NOTICES.md for the general design-credit
//! policy this project follows. No source code is copied from that
//! reference implementation.
//!
//! Only the dtypes that actually appear in real LLM checkpoints are
//! supported (`BOOL`/`U8`/`I8`/`I16`/`U16`/`I32`/`U32`/`I64`/`U64`/`F16`/
//! `BF16`/`F32`/`F64`). The real spec also defines sub-byte MX micro-scaling
//! dtypes (`F4`, `F6_E2M3`, `F6_E3M2`) and extra FP8 variants, but Kimi K3's
//! real checkpoint (verified directly against its own
//! `model.safetensors.index.json` and a real shard header, not assumed)
//! stores its MXFP4-packed expert weights as plain `U8` byte buffers under
//! `*.weight_packed`/`*.weight_scale` tensor-name pairs, not a dedicated MX
//! dtype tag — so those exotic dtypes aren't needed here.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;
use thiserror::Error;

const HEADER_LEN_BYTES: usize = 8;
/// Matches the real reference implementation's own guard against a
/// corrupt/malicious header length claiming to be huge.
const MAX_HEADER_SIZE: u64 = 100_000_000;

#[derive(Debug, Error)]
pub enum SafetensorsError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("file too small to contain a safetensors header")]
    HeaderTooSmall,
    #[error("header length {0} exceeds maximum {MAX_HEADER_SIZE}")]
    HeaderTooLarge(u64),
    #[error("header end {0} is past the end of the file ({1} bytes)")]
    TruncatedHeader(usize, usize),
    #[error("header is not valid UTF-8: {0}")]
    InvalidHeaderUtf8(#[from] std::str::Utf8Error),
    #[error("header JSON is malformed: {0}")]
    InvalidHeaderJson(#[from] serde_json::Error),
    #[error("unknown dtype string '{0}'")]
    UnknownDtype(String),
    #[error("tensor '{0}' has invalid data_offsets {1:?} (must start where the previous tensor ended, and end >= start)")]
    InvalidOffsets(String, (u64, u64)),
    #[error("tensor '{0}': shape/dtype imply {1} bytes but data_offsets span {2} bytes")]
    SizeMismatch(String, u64, u64),
    #[error(
        "tensor '{0}' declares shape {1:?}, whose size in bytes is not representable, so there \
         is nothing for its data_offsets span to be checked against"
    )]
    ImplausibleShape(String, Vec<usize>),
    #[error("data section is {0} bytes but tensor offsets claim {1} bytes")]
    IncompleteBuffer(usize, usize),
    #[error("tensor '{0}' not found")]
    TensorNotFound(String),
    #[error("shard '{0}' referenced by the index but its file could not be opened: {1}")]
    ShardOpenFailed(String, String),
}

/// The safetensors dtypes frink actually needs. See the module doc comment
/// for why the real spec's sub-byte MX dtypes are intentionally omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetensorsDtype {
    Bool,
    U8,
    I8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

impl SafetensorsDtype {
    fn from_tag(tag: &str) -> Result<Self, SafetensorsError> {
        Ok(match tag {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "I64" => Self::I64,
            "U64" => Self::U64,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            other => return Err(SafetensorsError::UnknownDtype(other.to_string())),
        })
    }

    /// Bytes per element.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: SafetensorsDtype,
    pub shape: Vec<usize>,
    /// Byte range relative to the start of the data section (i.e. relative
    /// to the end of the header), matching the real spec.
    pub data_offsets: (u64, u64),
}

/// The single fold of a file-supplied shape into an element count.
///
/// It used to be a bare `product()` in two places, and that made
/// [`SafetensorsError::SizeMismatch`] -- whose only job is to check a
/// shape against the bytes the header says it occupies -- unable to fire
/// for any overflowing shape, because the product it compared against
/// had already wrapped to whatever would agree. A gate that cannot fire
/// is worse than no gate.
///
/// No separate file-size bound is needed on top: `parse` already
/// requires each span to equal this count times the dtype width, the
/// spans to be contiguous from zero, and the last of them to end exactly
/// at the end of the file. Once the fold cannot lie, that chain is the
/// bound, and it is derived entirely from the input.
fn fold_shape(shape: &[usize]) -> Option<u64> {
    shape
        .iter()
        .try_fold(1u64, |n, &dim| n.checked_mul(dim as u64))
}

impl TensorInfo {
    /// The declared element count, or `None` when folding the
    /// file-supplied dims wraps or does not fit an address on this
    /// machine.
    pub fn element_count(&self) -> Option<usize> {
        fold_shape(&self.shape).and_then(|n| usize::try_from(n).ok())
    }

    /// The element count for REPORTING -- never for sizing a read.
    /// `parse` refuses any shape whose byte count is not representable,
    /// so for a `TensorInfo` that came out of a parsed file this is
    /// exact; the saturating arm exists only for a hand-built one, and
    /// saturates UP, so it can only ever cause a refusal.
    pub fn n_elements(&self) -> usize {
        self.element_count().unwrap_or(usize::MAX)
    }

    pub fn byte_len(&self) -> u64 {
        self.data_offsets.1 - self.data_offsets.0
    }
}

/// A single memory-mapped safetensors file. Tensor bytes are read straight
/// out of the mmap -- never copied into an owned buffer -- the same
/// zero-copy strategy `frink-gguf::GgufFile` uses, since these checkpoints
/// are far too large to read into RAM wholesale.
pub struct SafetensorsFile {
    pub metadata: Option<HashMap<String, String>>,
    pub tensors: Vec<TensorInfo>,
    mmap: Arc<Mmap>,
    data_start: usize,
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self, SafetensorsError> {
        if mmap.len() < HEADER_LEN_BYTES {
            return Err(SafetensorsError::HeaderTooSmall);
        }
        let mut cursor = io::Cursor::new(&mmap[..HEADER_LEN_BYTES]);
        let header_len = cursor.read_u64::<LittleEndian>()?;
        if header_len > MAX_HEADER_SIZE {
            return Err(SafetensorsError::HeaderTooLarge(header_len));
        }
        let header_end = HEADER_LEN_BYTES
            .checked_add(header_len as usize)
            .filter(|&e| e <= mmap.len())
            .ok_or(SafetensorsError::TruncatedHeader(
                HEADER_LEN_BYTES + header_len as usize,
                mmap.len(),
            ))?;
        let header_str = std::str::from_utf8(&mmap[HEADER_LEN_BYTES..header_end])?;
        let mut raw: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header_str)?;

        let metadata = match raw.remove("__metadata__") {
            Some(v) => Some(serde_json::from_value::<HashMap<String, String>>(v)?),
            None => None,
        };

        let mut entries: Vec<(String, RawTensorInfo)> = raw
            .into_iter()
            .map(|(k, v)| Ok((k, serde_json::from_value::<RawTensorInfo>(v)?)))
            .collect::<Result<_, serde_json::Error>>()?;

        // Real safetensors writers don't guarantee JSON key order matches
        // `data_offsets` order -- the reference implementation explicitly
        // re-sorts by offset before validating contiguity. Confirmed
        // directly in its source, not assumed.
        entries.sort_by_key(|(_, info)| info.data_offsets);

        let mut tensors = Vec::with_capacity(entries.len());
        let mut running_end = 0u64;
        for (name, raw_info) in entries {
            let dtype = SafetensorsDtype::from_tag(&raw_info.dtype)?;
            let (start, end) = raw_info.data_offsets;
            if start != running_end || end < start {
                return Err(SafetensorsError::InvalidOffsets(name, (start, end)));
            }
            let expected_bytes = fold_shape(&raw_info.shape)
                .and_then(|n| n.checked_mul(dtype.byte_size() as u64))
                .ok_or_else(|| {
                    SafetensorsError::ImplausibleShape(name.clone(), raw_info.shape.clone())
                })?;
            if end - start != expected_bytes {
                return Err(SafetensorsError::SizeMismatch(
                    name,
                    expected_bytes,
                    end - start,
                ));
            }
            running_end = end;
            tensors.push(TensorInfo {
                name,
                dtype,
                shape: raw_info.shape,
                data_offsets: (start, end),
            });
        }

        let data_len = (mmap.len() - header_end) as u64;
        if running_end != data_len {
            return Err(SafetensorsError::IncompleteBuffer(
                data_len as usize,
                running_end as usize,
            ));
        }

        Ok(SafetensorsFile {
            metadata,
            tensors,
            mmap: Arc::new(mmap),
            data_start: header_end,
        })
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }

    /// The single computation of a tensor's byte range in this file,
    /// rather than the verbatim copy each accessor used to hold -- the
    /// same collapse `frink-gguf::GgufFile::tensor_range` got, and for
    /// the same reason: two copies is two places for the reasoning that
    /// makes them total to rot independently.
    ///
    /// Total by construction, and it is `parse` that makes it so: the
    /// spans are contiguous from zero, each equals its shape's byte
    /// count, and the last ends exactly at the end of the data section,
    /// so `data_start + end` cannot exceed the mmap.
    fn tensor_range(&self, name: &str) -> Result<Range<usize>, SafetensorsError> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;
        let start = self.data_start + info.data_offsets.0 as usize;
        let end = self.data_start + info.data_offsets.1 as usize;
        debug_assert!(start <= end && end <= self.mmap.len());
        Ok(start..end)
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], SafetensorsError> {
        Ok(&self.mmap[self.tensor_range(name)?])
    }

    /// Zero-copy accessor: a cheaply-cloneable handle to the underlying
    /// mmap plus `name`'s byte range, so a caller can hold onto the bytes
    /// without keeping the whole `SafetensorsFile` (or a borrow of it)
    /// alive -- same pattern as `frink-gguf::GgufFile::tensor_mapped_range`.
    pub fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<Mmap>, Range<usize>), SafetensorsError> {
        Ok((Arc::clone(&self.mmap), self.tensor_range(name)?))
    }
}

#[derive(Debug, Deserialize)]
struct IndexFile {
    weight_map: HashMap<String, String>,
}

/// A checkpoint split across multiple safetensors shards, addressed by a
/// `<prefix>.safetensors.index.json` file (real convention: a flat
/// `weight_map` of tensor name -> shard filename, plus a `metadata` block
/// frink doesn't need). Kimi K3 ships this way: 96 shards, tens of
/// thousands of tensor entries. Each referenced shard is opened once (mmap,
/// zero-copy) and shared across every tensor lookup that lands on it.
pub struct ShardedSafetensors {
    shards: HashMap<String, SafetensorsFile>,
    weight_map: HashMap<String, String>,
    /// Shard filename -> full on-disk path, for consumers doing their
    /// own positional file I/O (e.g. the expert store's source).
    shard_paths: HashMap<String, PathBuf>,
}

impl ShardedSafetensors {
    pub fn open_index(index_path: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        let index_path = index_path.as_ref();
        let dir: PathBuf = index_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let text = std::fs::read_to_string(index_path)?;
        let index: IndexFile = serde_json::from_str(&text)?;

        let mut shard_names: Vec<&String> = index.weight_map.values().collect();
        shard_names.sort();
        shard_names.dedup();

        let mut shards = HashMap::with_capacity(shard_names.len());
        let mut shard_paths = HashMap::with_capacity(shard_names.len());
        for shard_name in shard_names {
            let path = dir.join(shard_name);
            let file = SafetensorsFile::open(&path).map_err(|e| {
                SafetensorsError::ShardOpenFailed(shard_name.clone(), e.to_string())
            })?;
            shards.insert(shard_name.clone(), file);
            shard_paths.insert(shard_name.clone(), path);
        }

        Ok(ShardedSafetensors {
            shards,
            weight_map: index.weight_map,
            shard_paths,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let shard_name = self.weight_map.get(name)?;
        self.shards.get(shard_name)?.tensor_info(name)
    }

    fn shard_for<'a>(&'a self, name: &str) -> Result<&'a SafetensorsFile, SafetensorsError> {
        let shard_name = self
            .weight_map
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;
        self.shards
            .get(shard_name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], SafetensorsError> {
        self.shard_for(name)?.tensor_bytes(name)
    }

    pub fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<Mmap>, Range<usize>), SafetensorsError> {
        self.shard_for(name)?.tensor_mapped_range(name)
    }

    /// The on-disk path of the shard holding `name` plus the tensor's
    /// byte range within that file (a `SafetensorsFile` mmap covers
    /// the whole file, so its mapped range IS the file-offset range) --
    /// for positional-read consumers like the expert store's source.
    pub fn tensor_file_location(
        &self,
        name: &str,
    ) -> Result<(&Path, Range<usize>), SafetensorsError> {
        let shard_name = self
            .weight_map
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;
        let (_, range) = self.tensor_mapped_range(name)?;
        let path = self
            .shard_paths
            .get(shard_name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;
        Ok((path.as_path(), range))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Write;

    /// Builds a real on-disk safetensors file (not a synthetic in-memory
    /// buffer) exactly the way the real format lays out: 8-byte LE header
    /// length, UTF-8 JSON header, then raw tensor bytes back to back at the
    /// offsets the header claims.
    fn build_file(header_json: &str, data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u64::<LittleEndian>(header_json.len() as u64)
            .unwrap();
        buf.write_all(header_json.as_bytes()).unwrap();
        buf.write_all(data).unwrap();
        buf
    }

    #[test]
    fn parses_a_real_single_tensor_file() {
        let data = [1.0f32, -2.5, 3.25]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();
        let header = r#"{"weight":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}}"#;
        let bytes = build_file(header, &data);

        let tmp = std::env::temp_dir().join(format!(
            "frink_st_single_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&tmp, &bytes).unwrap();
        let file = SafetensorsFile::open(&tmp).expect("must parse");
        std::fs::remove_file(&tmp).ok();

        assert_eq!(file.tensors.len(), 1);
        let info = file.tensor_info("weight").unwrap();
        assert_eq!(info.dtype, SafetensorsDtype::F32);
        assert_eq!(info.shape, vec![3]);
        assert_eq!(info.n_elements(), 3);

        let got = file.tensor_bytes("weight").unwrap();
        assert_eq!(got, &data[..]);
    }

    #[test]
    fn parses_metadata_block_and_multiple_tensors_out_of_offset_order() {
        // JSON key order deliberately does NOT match data_offsets order,
        // to exercise the real spec's "must re-sort by offset" behavior.
        let header = r#"{"__metadata__":{"format":"pt"},"b":{"dtype":"U8","shape":[2],"data_offsets":[2,4]},"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let data = [10u8, 11, 20, 21];
        let bytes = build_file(header, &data);

        let tmp =
            std::env::temp_dir().join(format!("frink_st_multi_{}.safetensors", std::process::id()));
        std::fs::write(&tmp, &bytes).unwrap();
        let file = SafetensorsFile::open(&tmp).expect("must parse");
        std::fs::remove_file(&tmp).ok();

        assert_eq!(
            file.metadata.as_ref().unwrap().get("format"),
            Some(&"pt".to_string())
        );
        assert_eq!(file.tensor_bytes("a").unwrap(), &[10u8, 11]);
        assert_eq!(file.tensor_bytes("b").unwrap(), &[20u8, 21]);
    }

    /// `SizeMismatch` is the only thing standing between a
    /// file-supplied shape and every consumer that trusts it, and it
    /// compared two values that wrapped identically -- so for an
    /// overflowing shape it could never fire. Executed before this fix,
    /// shape `[2^63 + 2, 2]` with a 16-byte span parsed OK and reported
    /// `n_elements=4`, and the wrapped count is what every downstream
    /// bounds check then agreed with.
    #[test]
    fn a_shape_whose_byte_count_wraps_is_refused_rather_than_matched() {
        // Chosen to fail rather than to be round:
        //   [2^63 + 2, 2]           the element fold wraps to exactly 4
        //   [2^32, 2^32]            it wraps to exactly 0, and a 0-byte
        //                           span then agrees with it perfectly
        //   [2^62]                  the count fits; it is the * 4 for
        //                           F32 that wraps, to 0
        //   [u64::MAX / 64, 65]     the fold wraps to a large nonsense
        for (i, (shape, span)) in [
            ("[9223372036854775810,2]", 16usize),
            ("[4294967296,4294967296]", 0),
            ("[4611686018427387904]", 0),
            ("[288230376151711743,65]", 0),
        ]
        .into_iter()
        .enumerate()
        {
            let header =
                format!(r#"{{"w":{{"dtype":"F32","shape":{shape},"data_offsets":[0,{span}]}}}}"#);
            let bytes = build_file(&header, &vec![0u8; span]);
            let tmp = std::env::temp_dir().join(format!(
                "frink_st_wrapshape{i}_{}.safetensors",
                std::process::id()
            ));
            std::fs::write(&tmp, &bytes).unwrap();
            let result = SafetensorsFile::open(&tmp);
            std::fs::remove_file(&tmp).ok();

            match result {
                Err(SafetensorsError::ImplausibleShape(name, _)) => assert_eq!(name, "w"),
                Err(other) => panic!("shape {shape}: expected ImplausibleShape, got {other}"),
                Ok(f) => panic!(
                    "shape {shape} parsed: n_elements={} byte_len={}",
                    f.tensors[0].n_elements(),
                    f.tensors[0].byte_len()
                ),
            }
        }

        // And the gate it was hiding still fires when it CAN: a shape
        // that does not overflow but does not match its span is a
        // SizeMismatch, not an ImplausibleShape.
        let header = r#"{"w":{"dtype":"F32","shape":[7],"data_offsets":[0,16]}}"#;
        let bytes = build_file(header, &[0u8; 16]);
        let tmp = std::env::temp_dir().join(format!(
            "frink_st_plainmismatch_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&tmp, &bytes).unwrap();
        let result = SafetensorsFile::open(&tmp);
        std::fs::remove_file(&tmp).ok();
        match result {
            Err(SafetensorsError::SizeMismatch(name, expected, got)) => {
                assert_eq!((name.as_str(), expected, got), ("w", 28, 16));
            }
            Err(other) => panic!("expected SizeMismatch, got {other}"),
            Ok(_) => panic!("expected SizeMismatch, got a parsed file"),
        }
    }

    #[test]
    fn rejects_a_gap_in_offsets() {
        // Tensor "a" claims [0,2) and "b" claims [4,6) -- a 2-byte gap that
        // the real reference implementation's `validate()` rejects (offsets
        // must be contiguous, not merely non-overlapping).
        let header = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[4,6]}}"#;
        let data = [0u8; 6];
        let bytes = build_file(header, &data);

        let tmp =
            std::env::temp_dir().join(format!("frink_st_gap_{}.safetensors", std::process::id()));
        std::fs::write(&tmp, &bytes).unwrap();
        let result = SafetensorsFile::open(&tmp);
        std::fs::remove_file(&tmp).ok();

        assert!(matches!(
            result,
            Err(SafetensorsError::InvalidOffsets(_, _))
        ));
    }

    #[test]
    fn rejects_a_size_mismatch_between_shape_and_offsets() {
        // 4 F32 elements should be 16 bytes, but data_offsets only claims 12.
        let header = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,12]}}"#;
        let data = [0u8; 12];
        let bytes = build_file(header, &data);

        let tmp = std::env::temp_dir().join(format!(
            "frink_st_sizemismatch_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&tmp, &bytes).unwrap();
        let result = SafetensorsFile::open(&tmp);
        std::fs::remove_file(&tmp).ok();

        assert!(matches!(
            result,
            Err(SafetensorsError::SizeMismatch(_, _, _))
        ));
    }

    #[test]
    fn rejects_trailing_bytes_the_header_does_not_account_for() {
        let header = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let data = [0u8; 5]; // 3 bytes more than the header's tensor claims
        let bytes = build_file(header, &data);

        let tmp = std::env::temp_dir().join(format!(
            "frink_st_trailing_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&tmp, &bytes).unwrap();
        let result = SafetensorsFile::open(&tmp);
        std::fs::remove_file(&tmp).ok();

        assert!(matches!(
            result,
            Err(SafetensorsError::IncompleteBuffer(_, _))
        ));
    }

    #[test]
    fn rejects_an_unknown_dtype_string() {
        let header = r#"{"a":{"dtype":"MADE_UP","shape":[2],"data_offsets":[0,2]}}"#;
        let data = [0u8; 2];
        let bytes = build_file(header, &data);

        let tmp = std::env::temp_dir().join(format!(
            "frink_st_baddtype_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&tmp, &bytes).unwrap();
        let result = SafetensorsFile::open(&tmp);
        std::fs::remove_file(&tmp).ok();

        assert!(matches!(result, Err(SafetensorsError::UnknownDtype(s)) if s == "MADE_UP"));
    }

    #[test]
    fn rejects_a_file_too_small_for_even_the_header_length_field() {
        let tmp =
            std::env::temp_dir().join(format!("frink_st_tiny_{}.safetensors", std::process::id()));
        std::fs::write(&tmp, [0u8; 4]).unwrap();
        let result = SafetensorsFile::open(&tmp);
        std::fs::remove_file(&tmp).ok();

        assert!(matches!(result, Err(SafetensorsError::HeaderTooSmall)));
    }

    #[test]
    fn sharded_index_resolves_tensors_across_multiple_shard_files() {
        let dir =
            std::env::temp_dir().join(format!("frink_st_sharded_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let shard0 = build_file(
            r#"{"layer0.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[1.0f32, 2.0]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let shard1 = build_file(
            r#"{"layer1.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[3.0f32, 4.0]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        std::fs::write(dir.join("shard0.safetensors"), &shard0).unwrap();
        std::fs::write(dir.join("shard1.safetensors"), &shard1).unwrap();

        let index_json = r#"{"weight_map":{"layer0.weight":"shard0.safetensors","layer1.weight":"shard1.safetensors"}}"#;
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, index_json).unwrap();

        let sharded = ShardedSafetensors::open_index(&index_path).expect("must open index");
        assert_eq!(sharded.tensor_names().count(), 2);

        let a = sharded.tensor_bytes("layer0.weight").unwrap();
        assert_eq!(
            a,
            &1.0f32.to_le_bytes()[..]
                .iter()
                .chain(2.0f32.to_le_bytes().iter())
                .copied()
                .collect::<Vec<u8>>()[..]
        );
        let b = sharded.tensor_info("layer1.weight").unwrap();
        assert_eq!(b.dtype, SafetensorsDtype::F32);
        assert_eq!(b.shape, vec![2]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sharded_lookup_of_a_missing_tensor_is_a_clean_error_not_a_panic() {
        let dir = std::env::temp_dir().join(format!(
            "frink_st_sharded_missing_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let shard0 = build_file(
            r#"{"layer0.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            &[7u8],
        );
        std::fs::write(dir.join("shard0.safetensors"), &shard0).unwrap();
        let index_json = r#"{"weight_map":{"layer0.weight":"shard0.safetensors"}}"#;
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, index_json).unwrap();

        let sharded = ShardedSafetensors::open_index(&index_path).expect("must open index");
        let result = sharded.tensor_bytes("does.not.exist");
        std::fs::remove_dir_all(&dir).ok();

        assert!(matches!(result, Err(SafetensorsError::TensorNotFound(_))));
    }
}
