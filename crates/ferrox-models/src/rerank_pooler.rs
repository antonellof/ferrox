//! `ferrox splice-pooler`: put a `BertForSequenceClassification`
//! reranker's pooler back into the GGUF llama.cpp's converter dropped it
//! from -- issue #82's missing half.
//!
//! # What is wrong with every reranker GGUF in circulation
//!
//! `conversion/bert.py`, `BertModel.filter_tensors`, deletes
//! `pooler.dense.{weight,bias}` by name for every BERT conversion,
//! classification heads included ("we are only using BERT for
//! embeddings so we don't need the pooling layer"; still unconditional
//! on `master` as of 2026-09-11). A cross-encoder trained as
//! `classifier(tanh(pooler(cls)))` is therefore served as
//! `classifier(cls)`: the same ORDER, on `ms-marco-MiniLM-L6-v2`, and a
//! score range about fifty times narrower (about +-0.2 instead of
//! about +-11). [`crate::rank_head`] already runs the pooler whenever a
//! file carries one and refuses to invent it when the file does not.
//! This module is how a file comes to carry one.
//!
//! # Why the pooler goes INTO the GGUF and not beside it
//!
//! Three places the pooler could come from were weighed:
//!
//! 1. **Upstream's converter keeping it.** The right fix, and the only
//!    one that helps files people already downloaded, once they
//!    re-convert. Not landed upstream, and not something ferrox controls.
//! 2. **A sidecar file read at load time.** Two files that must agree
//!    about one checkpoint, tied by a digest, discovered by filename --
//!    exactly the two-structures-with-nothing-enforcing-agreement shape
//!    this repo keeps shipping bugs in, and llama.cpp could not use it.
//! 3. **Writing a GGUF that carries the tensor**, which is this module.
//!    The output is what route 1 would have produced: `cls.weight` /
//!    `cls.bias` under llama.cpp's own names, so `load_rank_head`
//!    needs no change, `/v1/rerank` reports
//!    `classifier(tanh(pooler(cls)))` from the weights it actually
//!    loaded, and llama.cpp's `build_pooling` RANK arm runs the same
//!    graph on the same file. One file, no load-time identity question.
//!
//! # How the pooler is tied to the checkpoint
//!
//! A pooler from the wrong checkpoint scores on a range that LOOKS
//! calibrated and is not -- the silent-wrong-answer class, and worse
//! than the uncalibrated file it replaced. The GGUF cannot vouch for
//! itself by name: the published `ms-marco-MiniLM-L6-v2-Q8_0.gguf`
//! carries `general.name = "Ms Marco MiniLM L 12 v2"` and a
//! `base_model.0.repo_url` pointing at the L12 checkpoint, while its
//! six layers and its scores are L6's. A name-keyed check would pair it
//! with the wrong pooler and pass.
//!
//! So the tie is the one tensor BOTH files carry: the classifier.
//! `cls.output.{weight,bias}` in the GGUF is `classifier.{weight,bias}`
//! in the safetensors, and [`classifier_matches`] requires the two to
//! agree element-wise to within the GGUF's own storage precision.
//! A pooler whose classifier the GGUF does not contain is refused by
//! name, with the measured deviation. The slot-save fingerprint
//! (`ferrox-server::slots::identity`) is not reused here on purpose: it
//! identifies one GGUF to a later reader of the same GGUF, and the
//! question at splice time is whether a *safetensors* and a GGUF are
//! one checkpoint, which no digest of either can answer and the shared
//! classifier can.
//!
//! # What the output is, exactly
//!
//! Every metadata key and every tensor of the input, byte for byte, in
//! the input's order, plus `cls.weight` (`[n_embd, n_embd]`, F32) and
//! `cls.bias` (`[n_embd]`, F32), plus one string key
//! [`POOLER_SOURCE_KEY`] naming the safetensors file so `ferrox inspect`
//! can say the file is not the converter's own output. The written
//! file is then reopened and passed through [`load_rank_head`] before
//! this function returns, so the only GGUF it ever leaves on disk is
//! one the loader that will consume it has already accepted.

use std::collections::BTreeMap;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use ferrox_gguf::{
    GgmlType, GgufFile, GgufValue, GgufWriter, ShardedGguf, TensorPlan, TensorSource,
};
use ferrox_safetensors::SafetensorsFile;

use crate::loader::{load_f32_vec_optional, load_weight_matrix, LoadError};
use crate::rank_head::load_rank_head;
use crate::safetensors_f32::widen_to_f32;

/// The metadata key the spliced file carries, so its provenance is
/// visible in `ferrox inspect` and a second splice refuses by name.
pub const POOLER_SOURCE_KEY: &str = "ferrox.rerank.pooler_source";

/// llama.cpp's names for the head (`llama-arch.cpp`): `cls` is
/// HuggingFace's `bert.pooler.dense`, `cls.output` its `classifier`.
const CLS_W: &str = "cls.weight";
const CLS_B: &str = "cls.bias";
const CLS_OUT_W: &str = "cls.output.weight";
const CLS_OUT_B: &str = "cls.output.bias";

/// HuggingFace's names. The converter strips a leading `bert.` before
/// filtering, so a checkpoint may spell the pooler either way; the
/// classifier sits outside the `bert.` module and has one spelling.
const HF_POOLER_W: [&str; 2] = ["bert.pooler.dense.weight", "pooler.dense.weight"];
const HF_POOLER_B: [&str; 2] = ["bert.pooler.dense.bias", "pooler.dense.bias"];
const HF_CLASSIFIER_W: &str = "classifier.weight";
const HF_CLASSIFIER_B: &str = "classifier.bias";

/// How far the GGUF's classifier may sit from the safetensors' before
/// the two are different checkpoints, as a fraction of the reference
/// tensor's largest magnitude.
///
/// Derived from the storage precisions a one-row head tensor is ever
/// written at, not chosen: `llama-quantize` never quantizes a tensor
/// ggml sees as one-dimensional, and `cls.output.weight` is one (its
/// trailing dimension of 1 is dropped on disk), so the tensor is F32,
/// F16 or BF16 from the converter, or Q8_0 from a hand-built file. The
/// worst half-step among those is BF16's `2^-8` of the value and
/// Q8_0's `1/254` of the block maximum; `1/128` admits both with a
/// factor of two to spare and nothing coarser. A classifier that
/// differs from the file's by less than the file's own rounding cannot
/// change a score by more than that rounding does, which is what makes
/// the bound the identity and not a heuristic. [`SPLICEABLE_HEAD_DTYPES`]
/// is the list this number is true for, and a head stored coarser is
/// refused rather than admitted under a bound it could pass by accident.
pub const IDENTITY_TOLERANCE: f32 = 1.0 / 128.0;

/// The `cls.output.weight` storage types [`IDENTITY_TOLERANCE`] is
/// derived from. Anything else refuses by name.
pub const SPLICEABLE_HEAD_DTYPES: [GgmlType; 4] =
    [GgmlType::F32, GgmlType::F16, GgmlType::BF16, GgmlType::Q8_0];

#[derive(Debug, thiserror::Error)]
pub enum SpliceError {
    #[error(transparent)]
    Gguf(#[from] ferrox_gguf::GgufError),
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error(transparent)]
    Safetensors(#[from] ferrox_safetensors::SafetensorsError),
    #[error("writing {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: ferrox_gguf::GgufWriteError,
    },
    #[error("reopening the written file {path}: {source}")]
    Reopen {
        path: PathBuf,
        #[source]
        source: ferrox_gguf::ShardError,
    },
    #[error(
        "{path} is a '{arch}' checkpoint; only a `bert` classification head is known to run \
         classifier(tanh(pooler(cls))), so only a `bert` GGUF can take a pooler"
    )]
    NotBert { path: PathBuf, arch: String },
    #[error(
        "{path} is a split checkpoint ({shards} shards); merge it first (`ferrox gguf-split \
         --merge`) so the pooler goes into one file"
    )]
    Split { path: PathBuf, shards: u64 },
    #[error("{path} is missing `{key}`, which sizes the pooler")]
    MissingHparam { path: PathBuf, key: String },
    #[error(
        "{path} already carries {CLS_W}{spliced_from}; splicing a second pooler over it would \
         replace the head the file was converted with"
    )]
    AlreadyPooled { path: PathBuf, spliced_from: String },
    #[error(
        "{path} carries no {CLS_OUT_W}: there is no classifier for a pooler to feed, and \
         no classifier to tie the pooler to. A plain embedding model has no rerank head"
    )]
    NoClassifier { path: PathBuf },
    #[error(
        "{path} stores {CLS_OUT_W} as {dtype:?}; the classifier identity check is derived \
         for {allowed:?} and a coarser storage could pass it by accident"
    )]
    HeadDtype {
        path: PathBuf,
        dtype: GgmlType,
        allowed: [GgmlType; 4],
    },
    #[error("{path} carries none of {tried:?}; it is not a BertForSequenceClassification export")]
    MissingSafetensor {
        path: PathBuf,
        tried: Vec<&'static str>,
    },
    #[error("{path}: `{name}` is {dtype:?}, which is not a float type this splice reads")]
    SafetensorDtype {
        path: PathBuf,
        name: String,
        dtype: ferrox_safetensors::SafetensorsDtype,
    },
    #[error("{path}: `{name}` is {shape:?}, but the GGUF's encoder is {n_embd} wide so it must be {want:?}")]
    Shape {
        path: PathBuf,
        name: String,
        shape: Vec<usize>,
        n_embd: usize,
        want: Vec<usize>,
    },
    #[error(
        "the pooler in {safetensors} does not belong to {gguf}: {mismatch}. A pooler from \
         another checkpoint produces scores that look calibrated and are not, so nothing was \
         written. Check that the safetensors is the exact HuggingFace repo this GGUF was \
         converted from -- the GGUF's own `general.name` is not evidence, the published \
         ms-marco-MiniLM-L6-v2 file names the L12 model"
    )]
    Mismatch {
        gguf: PathBuf,
        safetensors: PathBuf,
        mismatch: IdentityMismatch,
    },
    #[error(
        "the written file {path} loads without a pooler, which means the splice wrote the \
         tensors under names the loader does not read; the file was removed"
    )]
    NotPooledAfterWrite { path: PathBuf },
}

/// Where and by how much the GGUF's classifier differs from the
/// safetensors'. Carried in the refusal so an operator sees a number,
/// not just "mismatch".
#[derive(Debug, Clone, PartialEq)]
pub struct IdentityMismatch {
    /// The GGUF tensor that disagreed.
    pub tensor: &'static str,
    /// Element index of the largest deviation.
    pub index: usize,
    pub gguf: f32,
    pub reference: f32,
    /// The bound that was exceeded, in absolute units.
    pub allowed: f32,
}

impl std::fmt::Display for IdentityMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} differs at element {} (GGUF {}, safetensors {}, allowed |diff| <= {:.3e})",
            self.tensor, self.index, self.gguf, self.reference, self.allowed
        )
    }
}

/// What was spliced, for the CLI's report.
#[derive(Debug, Clone, PartialEq)]
pub struct SplicedPooler {
    pub output: PathBuf,
    pub n_embd: usize,
    pub n_out: usize,
    /// How `cls.output.weight` is stored in the source, which is what
    /// [`IDENTITY_TOLERANCE`] was checked against.
    pub head_dtype: GgmlType,
    /// The largest element-wise deviation measured between the two
    /// classifiers, so the report shows the tie was tight and not merely
    /// under the bound.
    pub classifier_max_abs_diff: f32,
    /// The bound that deviation was held to, in absolute units.
    pub classifier_allowed: f32,
}

/// The identity check: `gguf` and `reference` are one tensor stored at
/// two precisions, or they are two tensors.
///
/// Same length, and every element within
/// `IDENTITY_TOLERANCE * max|reference|` -- a bound relative to the
/// tensor's scale rather than absolute, because a classifier's
/// magnitude is whatever training left it at. Returns the largest
/// deviation found, so a pass can be reported as a number.
pub fn classifier_matches(
    tensor: &'static str,
    gguf: &[f32],
    reference: &[f32],
) -> Result<(f32, f32), IdentityMismatch> {
    let absmax = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let allowed = absmax * IDENTITY_TOLERANCE;
    if gguf.len() != reference.len() {
        return Err(IdentityMismatch {
            tensor,
            index: gguf.len().min(reference.len()),
            gguf: f32::NAN,
            reference: f32::NAN,
            allowed,
        });
    }
    let mut worst = (0usize, 0.0f32);
    for (i, (g, r)) in gguf.iter().zip(reference).enumerate() {
        let diff = (g - r).abs();
        if diff > worst.1 || diff.is_nan() {
            worst = (i, diff);
        }
    }
    if worst.1 > allowed || worst.1.is_nan() {
        return Err(IdentityMismatch {
            tensor,
            index: worst.0,
            gguf: gguf[worst.0],
            reference: reference[worst.0],
            allowed,
        });
    }
    Ok((worst.1, allowed))
}

/// The first of `names` the file carries, widened to `f32`, with its
/// declared shape.
fn read_hf(
    file: &SafetensorsFile,
    path: &Path,
    names: &[&'static str],
) -> Result<(Vec<usize>, Vec<f32>), SpliceError> {
    let Some(name) = names.iter().find(|n| file.tensor_info(n).is_some()) else {
        return Err(SpliceError::MissingSafetensor {
            path: path.to_path_buf(),
            tried: names.to_vec(),
        });
    };
    let info = file.tensor_info(name).expect("found above");
    let data = widen_to_f32(info.dtype, file.tensor_bytes(name)?).ok_or_else(|| {
        SpliceError::SafetensorDtype {
            path: path.to_path_buf(),
            name: name.to_string(),
            dtype: info.dtype,
        }
    })?;
    Ok((info.shape.clone(), data))
}

fn want_shape(
    path: &Path,
    name: &str,
    shape: &[usize],
    want: &[usize],
    n_embd: usize,
) -> Result<(), SpliceError> {
    if shape == want {
        return Ok(());
    }
    Err(SpliceError::Shape {
        path: path.to_path_buf(),
        name: name.to_string(),
        shape: shape.to_vec(),
        n_embd,
        want: want.to_vec(),
    })
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Writes `out`: `gguf` plus the pooler from `safetensors`, after the
/// classifier in both has been shown to be one tensor. See the module
/// docs for every decision in here.
pub fn splice_pooler(
    gguf: &Path,
    safetensors: &Path,
    out: &Path,
) -> Result<SplicedPooler, SpliceError> {
    let file = GgufFile::open(gguf)?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("")
        .to_string();
    if arch != crate::bert_gguf_loader::BERT_ARCH {
        return Err(SpliceError::NotBert {
            path: gguf.to_path_buf(),
            arch,
        });
    }
    if let Some(shards @ 2..) = file.metadata_u64("split.count") {
        return Err(SpliceError::Split {
            path: gguf.to_path_buf(),
            shards,
        });
    }
    let n_embd_key = format!("{arch}.embedding_length");
    let n_embd = file
        .metadata_u64(&n_embd_key)
        .ok_or_else(|| SpliceError::MissingHparam {
            path: gguf.to_path_buf(),
            key: n_embd_key,
        })? as usize;
    if file.find_tensor(CLS_W).is_some() {
        let source = file
            .metadata_str(POOLER_SOURCE_KEY)
            .map(|s| format!(" (spliced from {s})"))
            .unwrap_or_default();
        return Err(SpliceError::AlreadyPooled {
            path: gguf.to_path_buf(),
            spliced_from: source,
        });
    }
    let Some(head_info) = file.find_tensor(CLS_OUT_W) else {
        return Err(SpliceError::NoClassifier {
            path: gguf.to_path_buf(),
        });
    };
    let head_dtype = head_info.dtype;
    if !SPLICEABLE_HEAD_DTYPES.contains(&head_dtype) {
        return Err(SpliceError::HeadDtype {
            path: gguf.to_path_buf(),
            dtype: head_dtype,
            allowed: SPLICEABLE_HEAD_DTYPES,
        });
    }

    // The GGUF's classifier, dequantized row by row through the same
    // loader `load_rank_head` uses, so the orientation checked here is
    // the orientation that will score.
    let head = load_weight_matrix(&file, CLS_OUT_W)?;
    let n_out = head.rows();
    let gguf_w: Vec<f32> = (0..n_out).flat_map(|r| head.dequant_row(r)).collect();
    let gguf_b = load_f32_vec_optional(&file, CLS_OUT_B)?;

    let hf = SafetensorsFile::open(safetensors)?;
    let (cw_shape, hf_w) = read_hf(&hf, safetensors, &[HF_CLASSIFIER_W])?;
    want_shape(
        safetensors,
        HF_CLASSIFIER_W,
        &cw_shape,
        &[n_out, n_embd],
        n_embd,
    )?;
    let (pw_shape, pooler_w) = read_hf(&hf, safetensors, &HF_POOLER_W)?;
    want_shape(
        safetensors,
        HF_POOLER_W[0],
        &pw_shape,
        &[n_embd, n_embd],
        n_embd,
    )?;
    let (pb_shape, pooler_b) = read_hf(&hf, safetensors, &HF_POOLER_B)?;
    want_shape(safetensors, HF_POOLER_B[0], &pb_shape, &[n_embd], n_embd)?;

    let mismatch = |mismatch| SpliceError::Mismatch {
        gguf: gguf.to_path_buf(),
        safetensors: safetensors.to_path_buf(),
        mismatch,
    };
    let (mut worst, allowed) = classifier_matches(CLS_OUT_W, &gguf_w, &hf_w).map_err(mismatch)?;
    // The bias is compared when both files carry one. A bias in one
    // file and not the other is a shape the loader already treats as
    // two different heads, so it is refused as a mismatch too.
    match (gguf_b, hf.tensor_info(HF_CLASSIFIER_B).is_some()) {
        (Some(gguf_b), true) => {
            let (_, hf_b) = read_hf(&hf, safetensors, &[HF_CLASSIFIER_B])?;
            let (worst_b, _) = classifier_matches(CLS_OUT_B, &gguf_b, &hf_b).map_err(mismatch)?;
            worst = worst.max(worst_b);
        }
        (None, false) => {}
        (gguf_b, _) => {
            return Err(mismatch(IdentityMismatch {
                tensor: CLS_OUT_B,
                index: 0,
                gguf: gguf_b.map(|b| b[0]).unwrap_or(f32::NAN),
                reference: f32::NAN,
                allowed,
            }));
        }
    }

    // Everything the input has, in the input's order, then the pooler.
    let mut metadata: BTreeMap<String, GgufValue> = file
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    metadata.insert(
        POOLER_SOURCE_KEY.to_string(),
        GgufValue::String(
            safetensors
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| safetensors.display().to_string()),
        ),
    );
    let mut plan: Vec<TensorPlan> = Vec::with_capacity(file.tensors.len() + 2);
    for t in &file.tensors {
        plan.push(TensorPlan {
            name: t.name.clone(),
            shape: t.shape.clone(),
            dtype: t.dtype,
            byte_len: file.tensor_bytes(&t.name)?.len(),
        });
    }
    let pooler_w_bytes = f32_bytes(&pooler_w);
    let pooler_b_bytes = f32_bytes(&pooler_b);
    // GGUF `ne[]` is fastest-dimension-first; a square pooler makes the
    // order invisible here, so it is the loader's `cols() != n_embd`
    // check on the reopen below, and `rerank_pooler_present.rs`'s
    // non-square fixture, that pin it.
    plan.push(TensorPlan {
        name: CLS_W.to_string(),
        shape: vec![n_embd as u64, n_embd as u64],
        dtype: GgmlType::F32,
        byte_len: pooler_w_bytes.len(),
    });
    plan.push(TensorPlan {
        name: CLS_B.to_string(),
        shape: vec![n_embd as u64],
        dtype: GgmlType::F32,
        byte_len: pooler_b_bytes.len(),
    });

    let write_err = |source| SpliceError::Write {
        path: out.to_path_buf(),
        source,
    };
    let sink = std::fs::File::create(out).map_err(|e| write_err(e.into()))?;
    let mut w = GgufWriter::create(BufWriter::new(sink), &metadata, plan).map_err(write_err)?;
    for t in &file.tensors {
        w.write_tensor(&t.name, file.tensor_bytes(&t.name)?)
            .map_err(write_err)?;
    }
    w.write_tensor(CLS_W, &pooler_w_bytes).map_err(write_err)?;
    w.write_tensor(CLS_B, &pooler_b_bytes).map_err(write_err)?;
    w.finish().map_err(write_err)?;

    // The file is not done until the loader that will consume it has
    // accepted it with the pooler in place.
    let eps = file
        .metadata_f32(&format!("{arch}.attention.layer_norm_epsilon"))
        .unwrap_or(1e-12);
    let reopened = ShardedGguf::open(out).map_err(|source| SpliceError::Reopen {
        path: out.to_path_buf(),
        source,
    })?;
    let pooled = load_rank_head(&reopened, &arch, n_embd, eps)
        .map(|h| h.is_some_and(|h| h.has_pooler()))
        .unwrap_or(false);
    if !pooled {
        std::fs::remove_file(out).ok();
        return Err(SpliceError::NotPooledAfterWrite {
            path: out.to_path_buf(),
        });
    }

    Ok(SplicedPooler {
        output: out.to_path_buf(),
        n_embd,
        n_out,
        head_dtype,
        classifier_max_abs_diff: worst,
        classifier_allowed: allowed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic classifier-shaped vector: signed, spanning three
    /// orders of magnitude, so a bound relative to the maximum is
    /// exercised on small elements too.
    fn reference(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = i as f32;
                ((x * 0.37).sin() * 0.8 + (x * 0.011).cos() * 0.05)
                    * if i % 7 == 0 { 3.0 } else { 1.0 }
            })
            .collect()
    }

    /// `IDENTITY_TOLERANCE` is a claim about the storage types in
    /// `SPLICEABLE_HEAD_DTYPES`, so it is checked against the actual
    /// round trips rather than asserted: the reference quantized to
    /// Q8_0 and read back, rounded to BF16 (nearest-even, which is what
    /// torch writes) and to F16, must all pass. A tolerance tightened
    /// below any of these would refuse the checkpoint's OWN pooler.
    #[test]
    fn every_spliceable_storage_precision_passes_the_identity_bound() {
        let r = reference(384);

        let q8 = ferrox_quant::dequant_q8_0(&ferrox_quant::quantize_q8_0(&r)).unwrap();
        let (worst, allowed) = classifier_matches("q8_0", &q8, &r).expect("Q8_0 round trip");
        assert!(
            worst > 0.0,
            "the Q8_0 round trip must actually perturb something"
        );
        assert!(worst <= allowed);

        let bf16: Vec<f32> = r
            .iter()
            .map(|x| {
                let bits = x.to_bits();
                let rounded = (bits.wrapping_add(0x7FFF + ((bits >> 16) & 1))) >> 16;
                f32::from_bits(rounded << 16)
            })
            .collect();
        let (worst, allowed) = classifier_matches("bf16", &bf16, &r).expect("BF16 round trip");
        assert!(worst > 0.0);
        assert!(worst <= allowed);

        let f16: Vec<f32> = r.iter().map(|x| half::f16::from_f32(*x).to_f32()).collect();
        classifier_matches("f16", &f16, &r).expect("F16 round trip");
        classifier_matches("f32", &r, &r).expect("F32 is exact");
    }

    /// The case the check exists for: a classifier that is NOT the
    /// GGUF's. Not a random vector -- a copy with one element moved by
    /// twice the bound, which is the tightest mismatch worth refusing
    /// and far tighter than two fine-tunes ever are. The refusal names
    /// the element and both values.
    #[test]
    fn a_classifier_off_by_more_than_the_files_own_rounding_is_refused_by_element() {
        let r = reference(384);
        let absmax = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let mut other = r.clone();
        other[200] += 2.0 * absmax * IDENTITY_TOLERANCE;
        let err = classifier_matches(CLS_OUT_W, &other, &r).unwrap_err();
        assert_eq!(err.tensor, CLS_OUT_W);
        assert_eq!(err.index, 200);
        assert_eq!(err.gguf, other[200]);
        assert_eq!(err.reference, r[200]);
        assert!(err.to_string().contains("element 200"), "{err}");
    }

    /// A different width is two heads, whatever the values.
    #[test]
    fn a_classifier_of_another_width_is_refused_before_any_element_is_compared() {
        let r = reference(384);
        assert!(classifier_matches(CLS_OUT_W, &r[..383], &r).is_err());
        assert!(classifier_matches(CLS_OUT_W, &r, &r[..383]).is_err());
    }

    /// A NaN in the GGUF's head is not "within tolerance" of anything.
    #[test]
    fn a_nan_never_matches() {
        let r = reference(8);
        let mut g = r.clone();
        g[3] = f32::NAN;
        assert!(classifier_matches(CLS_OUT_W, &g, &r).is_err());
    }
}
