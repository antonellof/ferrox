//! `prism.hadamard.*`: which weights of a PrismML checkpoint carry a
//! folded Hadamard rotation, and the per-weight transform that undoes
//! it (`frink_core::weight_matrix::hadamard`).
//!
//! Read as `llama-model.cpp:1196-1330` (PrismML-Eng/llama.cpp, branch
//! `prism`) reads it, refusal for refusal: `version` must be 1,
//! `block_size` a power of two, `transform` the normalized Sylvester
//! Walsh-Hadamard, `axis` the input's last dimension, `sign_mode`
//! `identity` or `explicit` (explicit: `sign_widths` partitions
//! `sign_values`, each width a multiple of the block, every value
//! `+/-1`), every `weight_names` entry a foldable kind
//! (`:1276-1309`), every `inverse_weight_names` entry
//! `token_embd.weight` (`:1323-1330`: the graph applies the inverse
//! only to the token-embedding lookup). `gdn_v_grouped` makes
//! `ssm_out`'s input arrive tiled and the fold expect it grouped
//! (`:2083-2092`). The reference also refuses architectures it has not
//! verified route every matmul through the folding helper
//! (`:1262-1275`: llama, qwen3, qwen3moe, qwen35, qwen35moe,
//! qwen3next); frink's route is `WeightMatrix::apply`, which every
//! matmul of every architecture goes through, so that list is not
//! copied.
//!
//! `load_weight_matrix` asks [`fold_for`] for every 2-D tensor it
//! loads and wraps the matrix (`WeightMatrix::fold_hadamard`) when the
//! name is listed. The metadata is parsed once per file (memoised on
//! the file's identity and checked on every hit) so the 400 lookups a
//! Bonsai load makes do not each re-read 28k sign values.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use frink_core::weight_matrix::hadamard::{FoldSite, HadamardFold, HeadPerm};
use frink_gguf::{GgufValue, TensorSource};

use crate::loader::LoadError;

const TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";
const AXIS: &str = "input-last-dimension";

/// The weight kinds the reference folds (`llama-model.cpp:1278-1284`).
const FOLDABLE_KINDS: &[&str] = &[
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_gate",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_up_exps",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
    "ssm_out",
];

/// One checkpoint's fold table.
pub struct HadamardMeta {
    block: usize,
    /// Sign vector per input width; empty for `identity`.
    signs: HashMap<usize, Arc<[f32]>>,
    weights: HashMap<String, ()>,
    inverses: HashMap<String, ()>,
    /// `ssm_out`'s tiled-to-grouped permutation: `(n_v_heads, n_k_heads)`.
    gdn_v_grouped: Option<(usize, usize)>,
    /// One `Arc` per distinct fold, so q/k/v split from one fused
    /// projection share theirs and `apply_gpu_multi` can see that.
    folds: Mutex<HashMap<(usize, FoldSite, bool), Arc<HadamardFold>>>,
}

fn refuse(what: impl Into<String>) -> LoadError {
    LoadError::UnsupportedFeature("prism.hadamard".to_string(), what.into())
}

fn str_array(file: &impl TensorSource, key: &str) -> Result<Vec<String>, LoadError> {
    match file.metadata(key) {
        None => Ok(Vec::new()),
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|v| match v {
                GgufValue::String(s) => Ok(s.clone()),
                other => Err(refuse(format!("{key}: expected strings, found {other:?}"))),
            })
            .collect(),
        Some(other) => Err(refuse(format!("{key}: expected an array, found {other:?}"))),
    }
}

fn int_array(file: &impl TensorSource, key: &str) -> Result<Vec<i64>, LoadError> {
    match file.metadata(key) {
        None => Ok(Vec::new()),
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|v| {
                let signed = match v {
                    GgufValue::I32(i) => Some(*i as i64),
                    GgufValue::I16(i) => Some(*i as i64),
                    GgufValue::I8(i) => Some(*i as i64),
                    GgufValue::I64(i) => Some(*i),
                    _ => None,
                };
                v.as_u64()
                    .map(|u| u as i64)
                    .or(signed)
                    .ok_or_else(|| refuse(format!("{key}: expected integers, found {v:?}")))
            })
            .collect(),
        Some(other) => Err(refuse(format!("{key}: expected an array, found {other:?}"))),
    }
}

/// `blk.N.<kind>.weight` for a foldable kind, or `output.weight`.
fn is_foldable_weight(name: &str) -> bool {
    if name == "output.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return false;
    }
    let Some(kind) = rest[digits..].strip_prefix('.') else {
        return false;
    };
    FOLDABLE_KINDS.iter().any(|k| kind == format!("{k}.weight"))
}

impl HadamardMeta {
    /// `None` for a file with no `prism.hadamard.version`; an error for
    /// one that declares a version and gets any of the rest wrong.
    pub fn from_gguf(file: &impl TensorSource) -> Result<Option<Self>, LoadError> {
        let Some(version) = file.metadata_u64("prism.hadamard.version") else {
            return Ok(None);
        };
        if version != 1 {
            return Err(refuse(format!(
                "version {version}; this build reads version 1"
            )));
        }
        let block = file
            .metadata_u64("prism.hadamard.block_size")
            .ok_or_else(|| refuse("block_size missing"))? as usize;
        if block == 0 || !block.is_power_of_two() {
            return Err(refuse(format!("block_size {block} is not a power of two")));
        }
        let transform = file.metadata_str("prism.hadamard.transform").unwrap_or("");
        if transform != TRANSFORM {
            return Err(refuse(format!(
                "transform {transform:?}; this build applies {TRANSFORM:?}"
            )));
        }
        let axis = file.metadata_str("prism.hadamard.axis").unwrap_or("");
        if axis != AXIS {
            return Err(refuse(format!(
                "axis {axis:?}; this build applies {AXIS:?}"
            )));
        }
        let sign_mode = file.metadata_str("prism.hadamard.sign_mode").unwrap_or("");
        let weight_names = str_array(file, "prism.hadamard.weight_names")?;
        if weight_names.is_empty() {
            return Err(refuse("weight_names is empty"));
        }
        let mut signs = HashMap::new();
        match sign_mode {
            "identity" => {}
            "explicit" => {
                let widths = int_array(file, "prism.hadamard.sign_widths")?;
                let values = int_array(file, "prism.hadamard.sign_values")?;
                if widths.is_empty() {
                    return Err(refuse("sign_mode is explicit but sign_widths is empty"));
                }
                let mut off = 0usize;
                for w in widths {
                    let w = usize::try_from(w).unwrap_or(0);
                    if w == 0 || !w.is_multiple_of(block) || off + w > values.len() {
                        return Err(refuse(format!("invalid sign width {w}")));
                    }
                    let mut v = Vec::with_capacity(w);
                    for &x in &values[off..off + w] {
                        if x != 1 && x != -1 {
                            return Err(refuse(format!(
                                "sign value {x}; every sign must be +1 or -1"
                            )));
                        }
                        v.push(x as f32);
                    }
                    signs.insert(w, Arc::from(v));
                    off += w;
                }
                if off != values.len() {
                    return Err(refuse("sign_values length does not match sign_widths"));
                }
            }
            other => return Err(refuse(format!("sign_mode {other:?}; identity or explicit"))),
        }
        let mut weights = HashMap::new();
        for name in weight_names {
            if !is_foldable_weight(&name) {
                return Err(refuse(format!(
                    "weight {name:?} is not on a verified Hadamard-aware matmul path"
                )));
            }
            if weights.insert(name.clone(), ()).is_some() {
                return Err(refuse(format!("duplicate weight {name:?}")));
            }
        }
        let mut inverses = HashMap::new();
        for name in str_array(file, "prism.hadamard.inverse_weight_names")? {
            if name != "token_embd.weight" {
                return Err(refuse(format!(
                    "weight {name:?} is not a verified inverse-after-lookup table"
                )));
            }
            inverses.insert(name, ());
        }
        let gdn_v_grouped = if file
            .metadata_bool("prism.hadamard.gdn_v_grouped")
            .unwrap_or(false)
        {
            let arch = file
                .metadata_str("general.architecture")
                .ok_or_else(|| refuse("general.architecture missing"))?;
            let n_v = file.metadata_u64(&format!("{arch}.ssm.time_step_rank"));
            let n_k = file.metadata_u64(&format!("{arch}.ssm.group_count"));
            match (n_v, n_k) {
                (Some(v), Some(k)) if v > 0 && k > 0 && v.is_multiple_of(k) => {
                    Some((v as usize, k as usize))
                }
                other => {
                    return Err(refuse(format!(
                        "gdn_v_grouped with bad GDN head geometry {other:?} \
                         (ssm.time_step_rank, ssm.group_count)"
                    )))
                }
            }
        } else {
            None
        };
        Ok(Some(Self {
            block,
            signs,
            weights,
            inverses,
            gdn_v_grouped,
            folds: Mutex::new(HashMap::new()),
        }))
    }

    /// The fold for `name`, a matrix of `cols` input columns, or `None`
    /// when the name is not listed. Errors: a listed weight whose width
    /// the block does not divide, or that has no sign vector.
    pub fn fold_for(
        &self,
        name: &str,
        cols: usize,
    ) -> Result<Option<Arc<HadamardFold>>, LoadError> {
        let site = if self.weights.contains_key(name) {
            FoldSite::Input
        } else if self.inverses.contains_key(name) {
            FoldSite::RowLookup
        } else {
            return Ok(None);
        };
        if !cols.is_multiple_of(self.block) {
            return Err(refuse(format!(
                "block size {} does not divide input dimension {cols} for {name}",
                self.block
            )));
        }
        let signs = if self.signs.is_empty() {
            None
        } else {
            Some(
                self.signs
                    .get(&cols)
                    .cloned()
                    .ok_or_else(|| refuse(format!("no sign vector for width {cols} ({name})")))?,
            )
        };
        let perm = match (self.gdn_v_grouped, name.contains(".ssm_out.")) {
            (Some((n_v, n_k)), true) => {
                if !cols.is_multiple_of(n_v) {
                    return Err(refuse(format!("bad GDN head geometry for {name}")));
                }
                Some(HeadPerm {
                    hd: cols / n_v,
                    nk: n_k,
                    rep: n_v / n_k,
                })
            }
            _ => None,
        };
        let key = (cols, site, perm.is_some());
        let mut folds = self.folds.lock().unwrap_or_else(|p| p.into_inner());
        Ok(Some(
            folds
                .entry(key)
                .or_insert_with(|| {
                    Arc::new(HadamardFold {
                        block: self.block,
                        signs,
                        perm,
                        site,
                    })
                })
                .clone(),
        ))
    }
}

/// Memo per file: keyed on the source's address, validated by a
/// fingerprint of the metadata so a different file at a reused address
/// cannot be served the previous one's table. Process-wide, not
/// thread-local: the fold `Arc`s hang off the meta, and `apply_gpu_multi`
/// fuses two matrices' launches only when they share ONE `Arc`, so a
/// per-thread memo (the first version) handed gate and up different
/// folds whenever the loader read them on different threads, and the
/// pair ran as two launches with nothing saying why.
static META: Mutex<Option<MetaMemo>> = Mutex::new(None);
type MetaMemo = HashMap<usize, (u64, Option<Arc<HadamardMeta>>)>;

fn fingerprint(file: &impl TensorSource) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for k in [
        "general.name",
        "general.architecture",
        "prism.hadamard.version",
        "prism.hadamard.weight_names",
    ] {
        let s = format!("{:?}", file.metadata(k).map(|v| format!("{v:?}").len()));
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The checkpoint's fold table, parsed once per file.
pub fn meta_for<F: TensorSource>(file: &F) -> Result<Option<Arc<HadamardMeta>>, LoadError> {
    let key = file as *const F as usize;
    let fp = fingerprint(file);
    let hit = META
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&key))
        .filter(|(f, _)| *f == fp)
        .map(|(_, meta)| meta.clone());
    if let Some(hit) = hit {
        return Ok(hit);
    }
    let meta = HadamardMeta::from_gguf(file)?.map(Arc::new);
    META.lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key, (fp, meta.clone()));
    Ok(meta)
}

/// [`HadamardMeta::fold_for`] through the memo: the one call
/// `load_weight_matrix` makes.
pub fn fold_for<F: TensorSource>(
    file: &F,
    name: &str,
    cols: usize,
) -> Result<Option<Arc<HadamardFold>>, LoadError> {
    match meta_for(file)? {
        Some(meta) => meta.fold_for(name, cols),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foldable_names_are_the_reference_s_list() {
        assert!(is_foldable_weight("output.weight"));
        assert!(is_foldable_weight("blk.12.attn_qkv.weight"));
        assert!(is_foldable_weight("blk.0.ssm_out.weight"));
        assert!(!is_foldable_weight("blk.0.ssm_in.weight"));
        assert!(!is_foldable_weight("token_embd.weight"));
        assert!(!is_foldable_weight("blk..attn_q.weight"));
        assert!(!is_foldable_weight("blk.3.attn_q.bias"));
    }

    fn bonsai_like() -> crate::test_source::StubSource {
        use frink_gguf::GgufValue as V;
        crate::test_source::StubSource::with_tensors(&[])
            .with_key("prism.hadamard.version", V::U32(1))
            .with_key("prism.hadamard.block_size", V::U32(4))
            .with_key("prism.hadamard.transform", V::String(TRANSFORM.into()))
            .with_key("prism.hadamard.axis", V::String(AXIS.into()))
            .with_key("prism.hadamard.sign_mode", V::String("identity".into()))
            .with_key(
                "prism.hadamard.weight_names",
                V::Array(vec![
                    V::String("blk.0.ffn_gate.weight".into()),
                    V::String("blk.0.ffn_up.weight".into()),
                ]),
            )
    }

    /// `apply_gpu_multi` fuses gate and up into one launch only when the
    /// two carry ONE `Arc`; the loader reads tensors on several threads,
    /// so the memo has to be process-wide. A thread-local memo passed
    /// every single-threaded test and cost Bonsai a GPU round trip per
    /// layer.
    #[test]
    fn folds_for_one_width_are_one_arc_across_threads() {
        let file = bonsai_like();
        let a = fold_for(&file, "blk.0.ffn_gate.weight", 8)
            .unwrap()
            .unwrap();
        let b = std::thread::scope(|s| {
            s.spawn(|| fold_for(&file, "blk.0.ffn_up.weight", 8).unwrap().unwrap())
                .join()
                .unwrap()
        });
        assert!(Arc::ptr_eq(&a, &b), "gate and up got different folds");
    }
}
