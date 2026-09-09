//! A streaming GGUF *writer*: the inverse of the reader in
//! [`crate`], and the only one in this workspace.
//!
//! Everything that wrote GGUF bytes before this module was a
//! `#[cfg(test)]` fixture builder -- there were seven of them, each
//! with its own idea of alignment and padding, and none of them
//! produced a file a second tool could read. This one exists because
//! `ferrox quantize` has to emit a checkpoint the reader accepts.
//!
//! Two design points earn their keep:
//!
//! * **The plan is separate from the data.** A caller declares every
//!   tensor's name, shape, dtype and byte length up front; the writer
//!   lays out the header and the tensor-data offsets from that, then
//!   takes the bytes one tensor at a time. A multi-gigabyte checkpoint
//!   is never held in memory, and the offsets in the header cannot
//!   disagree with the bytes that follow, because
//!   [`GgufWriter::write_tensor`] refuses a name or a length the plan
//!   did not declare.
//! * **The type tags come from [`GgmlType::to_tag`]**, which reads the
//!   same `GGML_TYPE_TAGS` table [`GgmlType::from_tag`] reads. A writer
//!   with its own tag list is two tables that must agree about thirty
//!   numbers with nothing enforcing it.

use std::collections::BTreeMap;
use std::io::{self, Write};

use byteorder::{LittleEndian, WriteBytesExt};
use thiserror::Error;

use crate::{GgmlType, GgufValue, GGUF_MAGIC};

/// GGUF version this writer emits. The reader accepts 2 and 3; 3 is
/// what every current converter writes.
pub const GGUF_WRITE_VERSION: u32 = 3;

/// The alignment used when the metadata does not carry
/// `general.alignment`. Matches the spec default and the reader's.
pub const DEFAULT_ALIGNMENT: usize = 32;

#[derive(Debug, Error)]
pub enum GgufWriteError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error(
        "metadata key '{0}' is an empty array, whose element type GGUF stores on the wire but \
         the reader does not keep. Writing it would have to guess a type tag, and a wrong guess \
         is a file that parses into different values"
    )]
    EmptyArray(String),
    #[error(
        "metadata key '{0}' is an array mixing element types ({1} then {2}). GGUF arrays carry \
         one element type tag, so this value cannot be written without changing what it means"
    )]
    HeterogeneousArray(String, &'static str, &'static str),
    #[error("metadata key '{0}' is a nested array; GGUF's array-of-array encoding is not written")]
    NestedArray(String),
    #[error(
        "general.alignment is {0}: the GGUF spec requires a power of two, and it must be one a \
         file can actually be aligned to"
    )]
    BadAlignment(u64),
    #[error("duplicate tensor name '{0}' in the write plan")]
    DuplicateTensor(String),
    #[error(
        "tensor '{0}' declares no dimensions; GGUF requires at least one, and a zero-dimension \
         tensor has no row for a kernel to walk"
    )]
    NoDimensions(String),
    #[error(
        "write_tensor called with '{got}' but the plan's next tensor is '{want}': the header's \
         offsets are laid out in plan order, so writing out of order would put every following \
         tensor at the wrong place"
    )]
    OutOfOrder { want: String, got: String },
    #[error(
        "tensor '{name}' was planned as {want} bytes but {got} were supplied; the header's \
         offsets are already written and cannot absorb the difference"
    )]
    WrongLength {
        name: String,
        want: usize,
        got: usize,
    },
    #[error("write_tensor called with '{0}' but every planned tensor has already been written")]
    TooManyTensors(String),
    #[error("finish() called with {0} planned tensor(s) still unwritten, starting at '{1}'")]
    Unfinished(usize, String),
}

/// One tensor's header entry, declared before any bytes are written.
#[derive(Debug, Clone)]
pub struct TensorPlan {
    pub name: String,
    /// GGUF dimension order (`ne[0]` first), exactly as the reader
    /// hands it back in [`crate::TensorInfo::shape`].
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    /// Bytes this tensor occupies in the data section. The caller owns
    /// this number because only the caller knows whether it is copying
    /// source bytes or encoding new ones; [`GgufWriter::write_tensor`]
    /// holds it to it.
    pub byte_len: usize,
}

/// Streaming GGUF writer. See the module docs for the plan/data split.
#[derive(Debug)]
pub struct GgufWriter<W: Write> {
    out: W,
    alignment: usize,
    plan: Vec<TensorPlan>,
    /// Index of the next tensor whose bytes are expected.
    next: usize,
    /// Bytes written into the data section so far, padding included.
    /// The invariant this maintains is the whole point of the type:
    /// when tensor `i` is about to be written, `written` equals the
    /// offset the header recorded for it.
    written: usize,
}

impl<W: Write> GgufWriter<W> {
    /// Writes the header (metadata + tensor descriptors + the pad up to
    /// the data section) and returns a writer ready to take tensor
    /// bytes in plan order.
    ///
    /// `metadata` is a `BTreeMap` rather than a `HashMap` so the key
    /// order in the output is a function of the keys alone. GGUF does
    /// not care about the order, but a byte-identical rerun over the
    /// same input is worth more than matching some source file's order,
    /// which the reader has already thrown away.
    pub fn create(
        mut out: W,
        metadata: &BTreeMap<String, GgufValue>,
        plan: Vec<TensorPlan>,
    ) -> Result<Self, GgufWriteError> {
        let alignment = declared_alignment(metadata.get("general.alignment"))?;
        // The header goes into a buffer first: the tensor-data offsets
        // are relative to the start of the data section, so they do not
        // depend on the header's own length, but writing to a buffer
        // keeps the whole header one `write_all` and lets `create` fail
        // before touching the file for a metadata value it cannot
        // encode.
        let buf = encode_header(metadata, &plan)?;
        out.write_all(&buf)?;
        Ok(GgufWriter {
            out,
            alignment,
            plan,
            next: 0,
            written: 0,
        })
    }

    /// Appends one tensor's bytes. Must be called once per planned
    /// tensor, in plan order, with exactly the planned number of bytes.
    pub fn write_tensor(&mut self, name: &str, bytes: &[u8]) -> Result<(), GgufWriteError> {
        let Some(planned) = self.plan.get(self.next) else {
            return Err(GgufWriteError::TooManyTensors(name.to_string()));
        };
        if planned.name != name {
            return Err(GgufWriteError::OutOfOrder {
                want: planned.name.clone(),
                got: name.to_string(),
            });
        }
        if planned.byte_len != bytes.len() {
            return Err(GgufWriteError::WrongLength {
                name: name.to_string(),
                want: planned.byte_len,
                got: bytes.len(),
            });
        }
        self.out.write_all(bytes)?;
        let end = self.written + bytes.len();
        let padded = end.next_multiple_of(self.alignment);
        self.out.write_all(&vec![0u8; padded.saturating_sub(end)])?;
        self.written = padded;
        self.next += 1;
        Ok(())
    }

    /// Flushes and returns the underlying writer, refusing if any
    /// planned tensor never received its bytes -- the header already
    /// promises they are there.
    pub fn finish(mut self) -> Result<W, GgufWriteError> {
        if self.next < self.plan.len() {
            return Err(GgufWriteError::Unfinished(
                self.plan.len() - self.next,
                self.plan[self.next].name.clone(),
            ));
        }
        self.out.flush()?;
        Ok(self.out)
    }
}

/// The data-section alignment a `general.alignment` value declares, or
/// [`DEFAULT_ALIGNMENT`] when the key is absent. One function for both
/// the writer and `split`, which has to size shards with the alignment
/// the writer will pad with: two readings of the key would be two
/// structures that must agree about one number.
pub fn declared_alignment(value: Option<&GgufValue>) -> Result<usize, GgufWriteError> {
    let declared = value
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_ALIGNMENT as u64);
    usize::try_from(declared)
        .ok()
        .filter(|a| a.is_power_of_two())
        .ok_or(GgufWriteError::BadAlignment(declared))
}

/// The complete header for `metadata` and `plan`: magic, counts, every
/// key/value, every tensor descriptor, and the pad up to the data
/// section. These are EXACTLY the bytes [`GgufWriter::create`] writes
/// before the first tensor, and its length is what llama.cpp calls
/// `gguf_get_meta_size`, so a dry run that reports a shard's size and
/// the write that produces it cannot disagree.
pub fn encode_header(
    metadata: &BTreeMap<String, GgufValue>,
    plan: &[TensorPlan],
) -> Result<Vec<u8>, GgufWriteError> {
    let alignment = declared_alignment(metadata.get("general.alignment"))?;

    let mut seen = std::collections::HashSet::with_capacity(plan.len());
    for t in plan {
        if t.shape.is_empty() {
            return Err(GgufWriteError::NoDimensions(t.name.clone()));
        }
        if !seen.insert(t.name.as_str()) {
            return Err(GgufWriteError::DuplicateTensor(t.name.clone()));
        }
    }

    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC)?;
    buf.write_u32::<LittleEndian>(GGUF_WRITE_VERSION)?;
    buf.write_u64::<LittleEndian>(plan.len() as u64)?;
    buf.write_u64::<LittleEndian>(metadata.len() as u64)?;

    for (key, value) in metadata {
        write_string(&mut buf, key)?;
        buf.write_u32::<LittleEndian>(value_tag(value))?;
        write_value(&mut buf, key, value)?;
    }

    let mut offset: usize = 0;
    for t in plan {
        write_string(&mut buf, &t.name)?;
        buf.write_u32::<LittleEndian>(t.shape.len() as u32)?;
        for &dim in &t.shape {
            buf.write_u64::<LittleEndian>(dim)?;
        }
        buf.write_u32::<LittleEndian>(t.dtype.to_tag())?;
        buf.write_u64::<LittleEndian>(offset as u64)?;
        // Every tensor starts on an alignment boundary, so the
        // offset advances by the padded length. `checked_*`
        // throughout: `byte_len` may come from a file's own header.
        offset = offset
            .checked_add(t.byte_len)
            .and_then(|o| o.checked_next_multiple_of(alignment))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "tensor '{}' pushes the data section past the address space",
                        t.name
                    ),
                )
            })?;
    }

    // Pad the header up to the data section, exactly as the reader
    // computes `data_start`.
    let pad = buf.len().next_multiple_of(alignment) - buf.len();
    buf.extend(std::iter::repeat_n(0u8, pad));
    Ok(buf)
}

fn write_string(buf: &mut Vec<u8>, s: &str) -> Result<(), GgufWriteError> {
    buf.write_u64::<LittleEndian>(s.len() as u64)?;
    buf.extend_from_slice(s.as_bytes());
    Ok(())
}

/// The GGUF value-type tag for a value.
///
/// The `match` is exhaustive with no `_` arm on purpose: a new
/// [`GgufValue`] variant must be given a tag here, and the reader's
/// `read_gguf_value_typed` must learn the same number, which
/// `every_value_variant_round_trips` checks.
fn value_tag(v: &GgufValue) -> u32 {
    match v {
        GgufValue::U8(_) => 0,
        GgufValue::I8(_) => 1,
        GgufValue::U16(_) => 2,
        GgufValue::I16(_) => 3,
        GgufValue::U32(_) => 4,
        GgufValue::I32(_) => 5,
        GgufValue::F32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::U64(_) => 10,
        GgufValue::I64(_) => 11,
        GgufValue::F64(_) => 12,
    }
}

/// The variant's name, for the heterogeneous-array message only.
fn variant_name(v: &GgufValue) -> &'static str {
    match v {
        GgufValue::U8(_) => "u8",
        GgufValue::I8(_) => "i8",
        GgufValue::U16(_) => "u16",
        GgufValue::I16(_) => "i16",
        GgufValue::U32(_) => "u32",
        GgufValue::I32(_) => "i32",
        GgufValue::F32(_) => "f32",
        GgufValue::Bool(_) => "bool",
        GgufValue::String(_) => "string",
        GgufValue::Array(_) => "array",
        GgufValue::U64(_) => "u64",
        GgufValue::I64(_) => "i64",
        GgufValue::F64(_) => "f64",
    }
}

fn write_value(buf: &mut Vec<u8>, key: &str, v: &GgufValue) -> Result<(), GgufWriteError> {
    match v {
        GgufValue::U8(x) => buf.write_u8(*x)?,
        GgufValue::I8(x) => buf.write_i8(*x)?,
        GgufValue::U16(x) => buf.write_u16::<LittleEndian>(*x)?,
        GgufValue::I16(x) => buf.write_i16::<LittleEndian>(*x)?,
        GgufValue::U32(x) => buf.write_u32::<LittleEndian>(*x)?,
        GgufValue::I32(x) => buf.write_i32::<LittleEndian>(*x)?,
        GgufValue::F32(x) => buf.write_f32::<LittleEndian>(*x)?,
        GgufValue::Bool(x) => buf.write_u8(u8::from(*x))?,
        GgufValue::String(s) => write_string(buf, s)?,
        GgufValue::U64(x) => buf.write_u64::<LittleEndian>(*x)?,
        GgufValue::I64(x) => buf.write_i64::<LittleEndian>(*x)?,
        GgufValue::F64(x) => buf.write_f64::<LittleEndian>(*x)?,
        GgufValue::Array(items) => {
            // The element type tag is on the wire but the reader does
            // not keep it: `GgufValue::Array(Vec<GgufValue>)` throws it
            // away and the elements carry their own variant instead.
            // For a non-empty array the first element restores it. For
            // an empty one there is nothing to restore it from, and a
            // guess is a file whose meaning changed, so this refuses
            // and names the key.
            let first = items
                .first()
                .ok_or_else(|| GgufWriteError::EmptyArray(key.to_string()))?;
            if matches!(first, GgufValue::Array(_)) {
                return Err(GgufWriteError::NestedArray(key.to_string()));
            }
            let elem_tag = value_tag(first);
            for item in items {
                if value_tag(item) != elem_tag {
                    return Err(GgufWriteError::HeterogeneousArray(
                        key.to_string(),
                        variant_name(first),
                        variant_name(item),
                    ));
                }
            }
            buf.write_u32::<LittleEndian>(elem_tag)?;
            buf.write_u64::<LittleEndian>(items.len() as u64)?;
            for item in items {
                write_value(buf, key, item)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GgufFile, TensorSource};

    fn kv(pairs: &[(&str, GgufValue)]) -> BTreeMap<String, GgufValue> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    fn write_to_temp(
        metadata: &BTreeMap<String, GgufValue>,
        tensors: &[(TensorPlan, Vec<u8>)],
    ) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ferrox-gguf-writer-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.gguf");
        let file = std::fs::File::create(&path).unwrap();
        let plan: Vec<TensorPlan> = tensors.iter().map(|(p, _)| p.clone()).collect();
        let mut w = GgufWriter::create(std::io::BufWriter::new(file), metadata, plan).unwrap();
        for (p, bytes) in tensors {
            w.write_tensor(&p.name, bytes).unwrap();
        }
        w.finish().unwrap().into_inner().unwrap();
        path
    }

    /// The property the whole module exists for: bytes this writer
    /// emits are bytes [`GgufFile`] reads back as the same tensors.
    /// Before it existed the only GGUF-producing code in the workspace
    /// was seven test fixtures, none of which was ever read by anything
    /// but its own test.
    #[test]
    fn a_written_file_reads_back_with_the_same_tensors_and_metadata() {
        let meta = kv(&[
            ("general.architecture", GgufValue::String("llama".into())),
            ("llama.block_count", GgufValue::U32(2)),
            ("general.file_type", GgufValue::U32(7)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(vec![
                    GgufValue::String("<s>".into()),
                    GgufValue::String("hi".into()),
                ]),
            ),
            (
                "llama.attention.head_count_kv",
                GgufValue::Array(vec![GgufValue::U32(8), GgufValue::U32(4)]),
            ),
        ]);
        let a: Vec<u8> = (0..64u8).collect();
        let b: Vec<u8> = (0..12u8).map(|x| x.wrapping_mul(7)).collect();
        let path = write_to_temp(
            &meta,
            &[
                (
                    TensorPlan {
                        name: "blk.0.attn_q.weight".into(),
                        shape: vec![4, 4],
                        dtype: GgmlType::F32,
                        byte_len: a.len(),
                    },
                    a.clone(),
                ),
                (
                    TensorPlan {
                        name: "output_norm.weight".into(),
                        shape: vec![3],
                        dtype: GgmlType::F32,
                        byte_len: b.len(),
                    },
                    b.clone(),
                ),
            ],
        );

        let f = GgufFile::open(&path).unwrap();
        assert_eq!(f.version, GGUF_WRITE_VERSION);
        assert_eq!(f.metadata_str("general.architecture"), Some("llama"));
        assert_eq!(f.metadata_u64("llama.block_count"), Some(2));
        assert_eq!(f.tensors.len(), 2);
        assert_eq!(f.tensor_bytes("blk.0.attn_q.weight").unwrap(), &a[..]);
        assert_eq!(f.tensor_bytes("output_norm.weight").unwrap(), &b[..]);
        let q = f.find_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(q.shape, vec![4, 4]);
        assert_eq!(q.dtype, GgmlType::F32);
        match f.metadata("tokenizer.ggml.tokens") {
            Some(GgufValue::Array(items)) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[1].as_str(), Some("hi"));
            }
            other => panic!("tokens came back as {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    /// Every [`GgufValue`] variant survives the write/read round trip.
    /// This is the test that walks BOTH tag tables -- `value_tag` here
    /// and `read_gguf_value_typed` in the reader -- so a disagreement
    /// about any of the thirteen numbers goes red rather than shipping
    /// a file whose f64 reads back as an i64.
    #[test]
    fn every_value_variant_round_trips() {
        let meta = kv(&[
            ("a.u8", GgufValue::U8(200)),
            ("a.i8", GgufValue::I8(-42)),
            ("a.u16", GgufValue::U16(60000)),
            ("a.i16", GgufValue::I16(-3000)),
            ("a.u32", GgufValue::U32(4_000_000_000)),
            ("a.i32", GgufValue::I32(-2_000_000_000)),
            ("a.f32", GgufValue::F32(0.125)),
            ("a.bool", GgufValue::Bool(true)),
            ("a.string", GgufValue::String("hello".into())),
            ("a.u64", GgufValue::U64(u64::MAX)),
            ("a.i64", GgufValue::I64(i64::MIN)),
            ("a.f64", GgufValue::F64(-0.5)),
            (
                "a.array",
                GgufValue::Array(vec![GgufValue::F32(1.5), GgufValue::F32(-2.5)]),
            ),
        ]);
        let path = write_to_temp(
            &meta,
            &[(
                TensorPlan {
                    name: "t".into(),
                    shape: vec![1],
                    dtype: GgmlType::F32,
                    byte_len: 4,
                },
                vec![0, 0, 0, 0],
            )],
        );
        let f = GgufFile::open(&path).unwrap();
        for (key, want) in &meta {
            let got = f.metadata(key).unwrap_or_else(|| panic!("missing {key}"));
            assert_eq!(
                format!("{got:?}"),
                format!("{want:?}"),
                "{key} did not round trip"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    /// An empty array has no element to recover its wire type tag from,
    /// and GGUF stores that tag even when the array is empty. Guessing
    /// writes a file that parses into a different value, so the writer
    /// refuses and names the key. The gate is reachable: the reader
    /// produces `Array(vec![])` for any empty array in a real file.
    #[test]
    fn an_empty_metadata_array_is_refused_by_name() {
        let meta = kv(&[("tokenizer.ggml.merges", GgufValue::Array(vec![]))]);
        let err = GgufWriter::create(Vec::new(), &meta, vec![]).unwrap_err();
        assert!(
            matches!(&err, GgufWriteError::EmptyArray(k) if k == "tokenizer.ggml.merges"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_mixed_type_metadata_array_is_refused_by_name() {
        let meta = kv(&[(
            "x",
            GgufValue::Array(vec![GgufValue::U32(1), GgufValue::String("two".into())]),
        )]);
        let err = GgufWriter::create(Vec::new(), &meta, vec![]).unwrap_err();
        assert!(
            matches!(&err, GgufWriteError::HeterogeneousArray(k, a, b)
                if k == "x" && *a == "u32" && *b == "string"),
            "got {err:?}"
        );
    }

    /// The header's offsets are laid out from the plan before a single
    /// tensor byte exists. If `write_tensor` accepted a different
    /// length the file would parse and every following tensor would be
    /// read from the wrong place -- the silent-corruption shape this
    /// repo keeps finding.
    #[test]
    fn a_tensor_whose_bytes_do_not_match_its_plan_is_refused() {
        let plan = vec![TensorPlan {
            name: "t".into(),
            shape: vec![8],
            dtype: GgmlType::F32,
            byte_len: 32,
        }];
        let mut w = GgufWriter::create(Vec::new(), &BTreeMap::new(), plan).unwrap();
        let err = w.write_tensor("t", &[0u8; 16]).unwrap_err();
        assert!(
            matches!(
                err,
                GgufWriteError::WrongLength {
                    want: 32,
                    got: 16,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn writing_tensors_out_of_plan_order_is_refused() {
        let plan = vec![
            TensorPlan {
                name: "a".into(),
                shape: vec![8],
                dtype: GgmlType::F32,
                byte_len: 32,
            },
            TensorPlan {
                name: "b".into(),
                shape: vec![8],
                dtype: GgmlType::F32,
                byte_len: 32,
            },
        ];
        let mut w = GgufWriter::create(Vec::new(), &BTreeMap::new(), plan).unwrap();
        let err = w.write_tensor("b", &[0u8; 32]).unwrap_err();
        assert!(
            matches!(&err, GgufWriteError::OutOfOrder { want, got } if want == "a" && got == "b"),
            "got {err:?}"
        );
    }

    /// `finish` is the last place a truncated file can be caught. The
    /// header already promises N tensors; returning the writer with
    /// fewer written produces a file whose last tensor reads past the
    /// end (or, worse, into the padding) at load time.
    #[test]
    fn finishing_with_a_planned_tensor_unwritten_is_refused() {
        let plan = vec![TensorPlan {
            name: "a".into(),
            shape: vec![8],
            dtype: GgmlType::F32,
            byte_len: 32,
        }];
        let w = GgufWriter::create(Vec::new(), &BTreeMap::new(), plan).unwrap();
        let err = w.finish().unwrap_err();
        assert!(
            matches!(&err, GgufWriteError::Unfinished(1, n) if n == "a"),
            "got {err:?}"
        );
    }

    /// A non-default `general.alignment` has to reach both the header
    /// pad and every inter-tensor pad, or the reader's `data_start`
    /// lands somewhere the writer never put anything. 64 is a value
    /// real converters emit.
    #[test]
    fn a_non_default_alignment_is_honoured_by_both_pads() {
        let meta = kv(&[("general.alignment", GgufValue::U32(64))]);
        // Deliberately not a multiple of 64, so the inter-tensor pad is
        // exercised rather than accidentally zero.
        let a = vec![1u8; 100];
        let b = vec![2u8; 40];
        let path = write_to_temp(
            &meta,
            &[
                (
                    TensorPlan {
                        name: "a".into(),
                        shape: vec![25],
                        dtype: GgmlType::F32,
                        byte_len: a.len(),
                    },
                    a.clone(),
                ),
                (
                    TensorPlan {
                        name: "b".into(),
                        shape: vec![10],
                        dtype: GgmlType::F32,
                        byte_len: b.len(),
                    },
                    b.clone(),
                ),
            ],
        );
        let f = GgufFile::open(&path).unwrap();
        assert_eq!(f.tensor_bytes("a").unwrap(), &a[..]);
        assert_eq!(f.tensor_bytes("b").unwrap(), &b[..]);
        assert_eq!(f.find_tensor("b").unwrap().offset, 128);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_non_power_of_two_alignment_is_refused_the_way_the_reader_refuses_it() {
        let meta = kv(&[("general.alignment", GgufValue::U32(3))]);
        let err = GgufWriter::create(Vec::new(), &meta, vec![]).unwrap_err();
        assert!(
            matches!(err, GgufWriteError::BadAlignment(3)),
            "got {err:?}"
        );
    }
}
