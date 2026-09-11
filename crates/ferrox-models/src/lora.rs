//! Reading a LoRA adapter GGUF: the file format llama.cpp's
//! `convert_lora_to_gguf.py` writes and `src/llama-adapter.cpp` reads.
//!
//! The contract, line by line from `llama_adapter_lora_init_impl`
//! (`llama-adapter.cpp:169-497`):
//!
//! * `general.type` must be `"adapter"` (`:202-205`), `adapter.type`
//!   must be `"lora"` (`:213-216`), and `general.architecture` must be
//!   the base model's (`:207-211`, "model arch and LoRA arch mismatch").
//! * `adapter.lora.alpha` is read as f32, absent meaning `0`, which
//!   `get_scale` treats as "no alpha scaling" (`llama-adapter.h:53-57`).
//! * Every tensor is `<base name>.lora_a` or `<base name>.lora_b`,
//!   bundled into pairs by base name (`:273-292`). A `_norm.weight`
//!   tensor is SKIPPED ("we don't really care because most adapters
//!   still work fine without it", `:287-290`); any other suffix is
//!   refused (`:291-293`). A pair missing one half is refused
//!   (`:342-344`).
//! * `adapter.alora.invocation_tokens` marks an *activated* LoRA, whose
//!   delta is applied only from an invocation sequence onward
//!   (`:219-238`, `server-context.cpp:1752-1800`). ferrox does not
//!   implement that gating, so the key is refused by name here rather
//!   than the adapter being applied to every position.
//!
//! What this module does NOT decide is whether a pair fits the base
//! model: that needs the base's shapes and lives in
//! [`crate::lora_attach`], beside the walk over the decoder's
//! projections. Shapes here are `[rows, cols]` in ferrox's row-major
//! sense, i.e. ggml's `ne` reversed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ferrox_gguf::{GgufFile, TensorSource};

use crate::loader::{load_f32_vec, LoadError};

/// A tensor of the adapter file, widened to f32.
#[derive(Debug, Clone)]
pub struct LoraTensor {
    /// `[rows, cols]`: ggml's `ne` reversed.
    pub shape: [usize; 2],
    pub data: Vec<f32>,
}

impl LoraTensor {
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.shape[1]
    }
}

/// One base tensor's `(lora_a, lora_b)` pair.
#[derive(Debug, Clone)]
pub struct LoraPair {
    pub a: LoraTensor,
    pub b: LoraTensor,
}

/// A parsed adapter file, not yet matched against a base model.
#[derive(Debug)]
pub struct LoraAdapter {
    pub path: PathBuf,
    /// `general.architecture`, compared against the base's.
    pub arch: String,
    /// `adapter.lora.alpha`, `0.0` when the file carries none.
    pub alpha: f32,
    /// `adapter.lora.task_name` / `adapter.lora.prompt_prefix`, the two
    /// metadata strings `GET /lora-adapters` reports upstream
    /// (`common.cpp:1274-1277`, `server-task.cpp:1616-1617`); empty
    /// when absent, as there.
    pub task_name: String,
    pub prompt_prefix: String,
    /// Base tensor name -> its pair, in name order so an attach walks
    /// deterministically.
    pub pairs: BTreeMap<String, LoraPair>,
    /// `_norm.weight` tensors the file carries and llama.cpp ignores.
    pub skipped_norms: Vec<String>,
}

/// Why an adapter file, or its match against a base, was refused.
#[derive(Debug, thiserror::Error)]
pub enum LoraError {
    #[error("{0}")]
    Load(#[from] LoadError),
    #[error("{path}: expect general.type to be 'adapter', but got: {got:?}")]
    NotAnAdapter { path: PathBuf, got: String },
    #[error("{path}: expect adapter.type to be 'lora', but got: {got:?}")]
    NotLora { path: PathBuf, got: String },
    #[error(
        "{path}: model arch and LoRA arch mismatch (adapter declares {adapter:?}, base is \
         {base:?})"
    )]
    ArchMismatch {
        path: PathBuf,
        adapter: String,
        base: String,
    },
    #[error("{path}: LoRA tensor '{name}' has unexpected suffix (want .lora_a or .lora_b)")]
    UnexpectedSuffix { path: PathBuf, name: String },
    #[error("{path}: LoRA tensor pair for '{name}' is missing one component")]
    MissingComponent { path: PathBuf, name: String },
    #[error(
        "{path}: this is an activated LoRA (`adapter.alora.invocation_tokens`, {n} tokens), \
         which llama.cpp applies only from the invocation sequence onward \
         (server-context.cpp:1752-1800); ferrox applies an adapter to every position and \
         refuses rather than activate it early"
    )]
    Alora { path: PathBuf, n: usize },
    #[error("{path}: LoRA tensor '{name}' is not 2-D (shape {shape:?})")]
    NotTwoD {
        path: PathBuf,
        name: String,
        shape: Vec<u64>,
    },
    #[error(
        "{path}: LoRA tensor '{name}' does not exist in base model (hint: maybe wrong base \
         model?)"
    )]
    NotInBase { path: PathBuf, name: String },
    #[error(
        "{path}: tensor '{name}' has incorrect shape (hint: maybe wrong base model?): base is \
         [{rows} x {cols}], lora_a is {a:?}, lora_b is {b:?}"
    )]
    Shape {
        path: PathBuf,
        name: String,
        rows: usize,
        cols: usize,
        a: [usize; 2],
        b: [usize; 2],
    },
    #[error(
        "{path}: lora_a tensor for '{name}' is not transposed (hint: adapter from \"finetune\" \
         example is no longer supported): lora_a is {a:?}, lora_b is {b:?}"
    )]
    NotTransposed {
        path: PathBuf,
        name: String,
        a: [usize; 2],
        b: [usize; 2],
    },
    #[error(
        "{path}: '{name}' adapts a routed-expert tensor; llama.cpp applies that through \
         `build_lora_mm_id` (llama-graph.cpp:1517-1550) and ferrox has no per-expert delta, \
         so the adapter is refused rather than applied to the dense projections only"
    )]
    RoutedExperts { path: PathBuf, name: String },
    #[error(
        "{path}: '{name}' adapts a tensor ferrox holds under no projection (norms, biases and \
         side tables are not `WeightMatrix`), so the delta would be dropped; refused rather \
         than applied partially"
    )]
    NoProjection { path: PathBuf, name: String },
    #[error(
        "{path}: 'token_embd.weight' is adapted but the base model ties its output head to the \
         embedding (no `output.weight`); llama.cpp builds `build_lora_mm(model.output, ..)` \
         (llama-graph.cpp:1490) over the FLIPPED embedding pair there and aborts with \
         `ggml.c:3282: GGML_ASSERT(ggml_can_mul_mat(a, b))` (measured), so no engine serves \
         this combination"
    )]
    TiedHead { path: PathBuf },
    #[error(
        "{path}: '{name}' targets a store-backed expert layer, whose weights are leased \
             per use; run with resident experts to attach an adapter"
    )]
    StoredExperts { path: PathBuf, name: String },
}

impl LoraAdapter {
    /// Parses the file and its tensors. Every check llama.cpp makes on
    /// the file ALONE is made here; the shape checks need the base and
    /// are made at attach.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoraError> {
        let path = path.as_ref().to_path_buf();
        let file = GgufFile::open(&path).map_err(LoadError::from)?;
        Self::from_file(path, &file)
    }

    fn from_file(path: PathBuf, file: &GgufFile) -> Result<Self, LoraError> {
        let general_type = file.metadata_str("general.type").unwrap_or("");
        if general_type != "adapter" {
            return Err(LoraError::NotAnAdapter {
                path,
                got: general_type.to_string(),
            });
        }
        let adapter_type = file.metadata_str("adapter.type").unwrap_or("");
        if adapter_type != "lora" {
            return Err(LoraError::NotLora {
                path,
                got: adapter_type.to_string(),
            });
        }
        if let Some(v) = file.metadata("adapter.alora.invocation_tokens") {
            let n = match v {
                ferrox_gguf::GgufValue::Array(items) => items.len(),
                _ => 1,
            };
            return Err(LoraError::Alora { path, n });
        }
        let arch = file
            .metadata_str("general.architecture")
            .unwrap_or("")
            .to_string();
        let alpha = file.metadata_f32("adapter.lora.alpha").unwrap_or(0.0);
        let task_name = file
            .metadata_str("adapter.lora.task_name")
            .unwrap_or("")
            .to_string();
        let prompt_prefix = file
            .metadata_str("adapter.lora.prompt_prefix")
            .unwrap_or("")
            .to_string();

        // Bundle `lora_a` / `lora_b` into pairs by base name, exactly
        // as `llama-adapter.cpp:273-293` does, including which suffixes
        // are skipped and which are refused.
        let mut halves: BTreeMap<String, (Option<LoraTensor>, Option<LoraTensor>)> =
            BTreeMap::new();
        let mut skipped_norms = Vec::new();
        for info in &file.tensors {
            let name = info.name.as_str();
            let (base, is_a) = if let Some(base) = name.strip_suffix(".lora_a") {
                (base, true)
            } else if let Some(base) = name.strip_suffix(".lora_b") {
                (base, false)
            } else if name.ends_with("_norm.weight") {
                skipped_norms.push(name.to_string());
                continue;
            } else {
                return Err(LoraError::UnexpectedSuffix {
                    path,
                    name: name.to_string(),
                });
            };
            if info.shape.len() != 2 {
                return Err(LoraError::NotTwoD {
                    path,
                    name: name.to_string(),
                    shape: info.shape.clone(),
                });
            }
            let tensor = LoraTensor {
                // ggml `ne = [n_cols, n_rows]`.
                shape: [info.shape[1] as usize, info.shape[0] as usize],
                data: load_f32_vec(file, name)?,
            };
            let entry = halves.entry(base.to_string()).or_default();
            if is_a {
                entry.0 = Some(tensor);
            } else {
                entry.1 = Some(tensor);
            }
        }
        let mut pairs = BTreeMap::new();
        for (name, (a, b)) in halves {
            match (a, b) {
                (Some(a), Some(b)) => {
                    pairs.insert(name, LoraPair { a, b });
                }
                _ => return Err(LoraError::MissingComponent { path, name }),
            }
        }
        Ok(Self {
            path,
            arch,
            alpha,
            task_name,
            prompt_prefix,
            pairs,
            skipped_norms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_gguf::writer::{GgufWriter, TensorPlan};
    use ferrox_gguf::GgmlType;

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    /// A hand-written adapter file: `kv` as metadata, each tensor F32
    /// of the given `ne` (GGUF dimension order), filled with 0.5.
    fn write_adapter(
        name: &str,
        kv: &[(&str, ferrox_gguf::GgufValue)],
        tensors: &[(&str, Vec<u64>)],
    ) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ferrox-lora-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("adapter.gguf");
        let metadata: BTreeMap<String, ferrox_gguf::GgufValue> =
            kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        let plan: Vec<TensorPlan> = tensors
            .iter()
            .map(|(t, ne)| TensorPlan {
                name: t.to_string(),
                shape: ne.clone(),
                dtype: GgmlType::F32,
                byte_len: ne.iter().product::<u64>() as usize * 4,
            })
            .collect();
        let file = std::fs::File::create(&path).unwrap();
        let mut w = GgufWriter::create(std::io::BufWriter::new(file), &metadata, plan).unwrap();
        for (t, ne) in tensors {
            let n = ne.iter().product::<u64>() as usize;
            let bytes: Vec<u8> = std::iter::repeat_n(0.5f32.to_le_bytes(), n)
                .flatten()
                .collect();
            w.write_tensor(t, &bytes).unwrap();
        }
        w.finish().unwrap();
        path
    }

    fn base_kv() -> Vec<(&'static str, ferrox_gguf::GgufValue)> {
        use ferrox_gguf::GgufValue as V;
        vec![
            ("general.type", V::String("adapter".into())),
            ("adapter.type", V::String("lora".into())),
            ("general.architecture", V::String("llama".into())),
            ("adapter.lora.alpha", V::F32(8.0)),
        ]
    }

    /// The real converter's output parses, with every pair and the
    /// flipped embedding shape read as `[rows, cols]`.
    #[test]
    fn the_converter_s_file_parses_into_pairs() {
        let a = LoraAdapter::open(fixture_dir().join("lora_a_tiny.gguf")).unwrap();
        assert_eq!(a.arch, "llama");
        assert_eq!(a.alpha, 8.0);
        assert_eq!(a.pairs.len(), 16, "7 per layer x 2 + embedding + head");
        let q = &a.pairs["blk.0.attn_q.weight"];
        assert_eq!(q.a.shape, [4, 24], "lora_a is [rank, n_in]");
        assert_eq!(q.b.shape, [24, 4], "lora_b is [n_out, rank]");
        let e = &a.pairs["token_embd.weight"];
        assert_eq!(
            e.a.shape,
            [48, 4],
            "the embedding's lora_a is flipped: [n_vocab, rank]"
        );
        assert_eq!(e.b.shape, [24, 4], "[n_embd, rank]");
        assert!(a.skipped_norms.is_empty());
    }

    #[test]
    fn a_plain_model_file_is_not_an_adapter() {
        let err = LoraAdapter::open(fixture_dir().join("lora_base_tiny.gguf")).unwrap_err();
        assert!(matches!(err, LoraError::NotAnAdapter { .. }), "{err}");
        assert!(err.to_string().contains("general.type"), "{err}");
    }

    #[test]
    fn a_control_vector_adapter_is_refused_by_type() {
        let mut kv = base_kv();
        kv[1] = (
            "adapter.type",
            ferrox_gguf::GgufValue::String("control_vector".into()),
        );
        let p = write_adapter("cvec", &kv, &[]);
        let err = LoraAdapter::open(&p).unwrap_err();
        assert!(matches!(err, LoraError::NotLora { .. }), "{err}");
    }

    #[test]
    fn a_tensor_with_another_suffix_is_refused_and_a_norm_is_skipped() {
        let p = write_adapter(
            "suffix",
            &base_kv(),
            &[("blk.0.attn_q.weight.lora_c", vec![4, 4])],
        );
        let err = LoraAdapter::open(&p).unwrap_err();
        assert!(matches!(err, LoraError::UnexpectedSuffix { .. }), "{err}");

        let p = write_adapter(
            "norm",
            &base_kv(),
            &[
                ("blk.0.attn_norm.weight", vec![24]),
                ("blk.0.attn_q.weight.lora_a", vec![24, 4]),
                ("blk.0.attn_q.weight.lora_b", vec![4, 24]),
            ],
        );
        let a = LoraAdapter::open(&p).unwrap();
        assert_eq!(a.skipped_norms, vec!["blk.0.attn_norm.weight".to_string()]);
        assert_eq!(a.pairs.len(), 1);
    }

    #[test]
    fn a_pair_missing_one_half_is_refused() {
        let p = write_adapter(
            "half",
            &base_kv(),
            &[("blk.0.attn_q.weight.lora_a", vec![24, 4])],
        );
        let err = LoraAdapter::open(&p).unwrap_err();
        assert!(matches!(err, LoraError::MissingComponent { .. }), "{err}");
        assert!(err.to_string().contains("blk.0.attn_q.weight"), "{err}");
    }

    #[test]
    fn an_activated_lora_is_refused_by_name() {
        let mut kv = base_kv();
        kv.push((
            "adapter.alora.invocation_tokens",
            ferrox_gguf::GgufValue::Array(vec![
                ferrox_gguf::GgufValue::U32(5),
                ferrox_gguf::GgufValue::U32(9),
            ]),
        ));
        let p = write_adapter("alora", &kv, &[]);
        let err = LoraAdapter::open(&p).unwrap_err();
        assert!(matches!(err, LoraError::Alora { n: 2, .. }), "{err}");
        assert!(err.to_string().contains("invocation"), "{err}");
    }

    #[test]
    fn a_missing_alpha_reads_as_zero() {
        let kv: Vec<_> = base_kv().into_iter().take(3).collect();
        let p = write_adapter("noalpha", &kv, &[]);
        let a = LoraAdapter::open(&p).unwrap();
        assert_eq!(a.alpha, 0.0);
    }
}
