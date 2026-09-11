//! The consumer side of an importance matrix in `ferrox quantize`:
//! reading it, recording it in the output's metadata, and picking each
//! tensor's slice, all as `llama-quantize` does.
//!
//! `src/llama-quant.cpp:913-934` (b7650) is the per-tensor rule and
//! `tools/quantize/quantize.cpp:532-566` the metadata; both are cited
//! at the function that transcribes them. The file formats themselves
//! are `crate::imatrix::file`, shared with `ferrox imatrix`, so the
//! reader the quantizer uses is the writer's own inverse rather than a
//! second reading of the spec.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Result};
use ferrox_gguf::GgufValue;

use super::Planned;
use crate::imatrix::file::ImatrixWeights;

/// Reads the imatrix and refuses a non-finite value anywhere in it,
/// as `llama-quant.cpp:619-625` does: an `inf` weight would fit a
/// super-block to nothing and the file would carry no sign of it.
pub(crate) fn load_imatrix(path: &Path) -> Result<ImatrixWeights> {
    let im = crate::imatrix::file::read(path)?;
    for (name, vals) in &im.entries {
        if let Some(bad) = vals.iter().find(|v| !v.is_finite()) {
            bail!("imatrix contains non-finite value {bad} in entry {name}");
        }
    }
    println!(
        "quantize: have weights data with {} entries from {} (computed on {} chunks)",
        im.entries.len(),
        path.display(),
        im.chunk_count
    );
    Ok(im)
}

/// `LLAMA_KV_OVERRIDE_TYPE_STR` values are copied with `strncpy(...,
/// 127)` into a fixed buffer (`quantize.cpp:538-548`), so a longer
/// path is cut at 127 bytes in llama.cpp's output and here.
fn truncate_127(s: &str) -> String {
    let mut end = s.len().min(127);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// The four `quantize.imatrix.*` keys `llama-quantize` records
/// (`quantize.cpp:532-566`), with the types `llama-quant.cpp:651-660`
/// writes them as: strings for the file and dataset, and `u32` for the
/// two counts (`LLAMA_KV_OVERRIDE_TYPE_INT` lands as `gguf_set_val_u32`
/// there). `chunks_count` is written only when it is positive, as
/// upstream does.
pub(crate) fn imatrix_metadata(
    metadata: &mut BTreeMap<String, GgufValue>,
    im: &ImatrixWeights,
    path: &Path,
) {
    metadata.insert(
        "quantize.imatrix.file".to_string(),
        GgufValue::String(truncate_127(&path.to_string_lossy())),
    );
    if let Some(d) = im.datasets.first() {
        metadata.insert(
            "quantize.imatrix.dataset".to_string(),
            GgufValue::String(truncate_127(d)),
        );
    }
    metadata.insert(
        "quantize.imatrix.entries_count".to_string(),
        GgufValue::U32(im.entries.len() as u32),
    );
    if im.chunk_count > 0 {
        metadata.insert(
            "quantize.imatrix.chunks_count".to_string(),
            GgufValue::U32(im.chunk_count),
        );
    }
}

/// The imatrix slice for one tensor, or `None` with a printed notice
/// when the file has no entry for it (`llama-quant.cpp:916-919`). An
/// entry of the wrong size is refused (`:920-934`), except for
/// `token_embd.weight`, which upstream lets through unweighted because
/// old imatrix files carry a wrong-shaped entry for it.
pub(crate) fn imatrix_for_tensor<'a>(
    im: &'a ImatrixWeights,
    p: &Planned,
) -> Result<Option<&'a [f32]>> {
    let Some(vals) = im.entries.get(&p.name) else {
        println!("quantize: did not find weights for {}", p.name);
        return Ok(None);
    };
    let ne0 = p.shape[0] as usize;
    let ne2 = p.shape.get(2).copied().unwrap_or(1) as usize;
    if vals.len() == ne0 * ne2 {
        return Ok(Some(vals));
    }
    if p.name == "token_embd.weight" {
        println!(
            "quantize: imatrix size {} is different from tensor size {} for {}; quantizing it \
             unweighted, as llama.cpp does",
            vals.len(),
            ne0 * ne2,
            p.name
        );
        return Ok(None);
    }
    bail!(
        "imatrix size {} is different from tensor size {} for {}",
        vals.len(),
        ne0 * ne2,
        p.name
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_gguf::GgmlType;

    fn planned(name: &str, shape: Vec<u64>) -> Planned {
        Planned {
            name: name.into(),
            shape,
            source_dtype: GgmlType::F16,
            out_dtype: GgmlType::Q4K,
            source_bytes: 0,
            out_bytes: 0,
            copy_reason: None,
        }
    }

    fn weights(entries: &[(&str, usize)]) -> ImatrixWeights {
        ImatrixWeights {
            entries: entries
                .iter()
                .map(|(n, len)| (n.to_string(), vec![1.0; *len]))
                .collect(),
            datasets: vec!["calib.txt".into()],
            chunk_count: 3,
        }
    }

    /// A 2-D tensor gets its `n_cols` slice; a 3-D expert stack gets
    /// `n_cols * n_expert`, one set per expert, which is what
    /// `llama-quant.cpp:920` checks against `ne[0]*ne[2]`.
    #[test]
    fn a_tensor_gets_its_slice_when_the_width_matches() {
        let im = weights(&[
            ("blk.0.attn_q.weight", 256),
            ("blk.0.ffn_gate_exps.weight", 256 * 4),
        ]);
        let p = planned("blk.0.attn_q.weight", vec![256, 8]);
        assert_eq!(imatrix_for_tensor(&im, &p).unwrap().unwrap().len(), 256);
        let p = planned("blk.0.ffn_gate_exps.weight", vec![256, 8, 4]);
        assert_eq!(imatrix_for_tensor(&im, &p).unwrap().unwrap().len(), 1024);
    }

    /// No entry: unweighted, not an error (`llama-quant.cpp:916-918`).
    /// Wrong width: an error, except for `token_embd.weight`, which
    /// upstream exempts (`:929-933`). Both halves are what makes a
    /// ferrox file match a llama.cpp file quantized from the same
    /// imatrix, since `output.weight` normally has no entry.
    #[test]
    fn a_missing_entry_is_unweighted_and_a_wrong_width_is_refused_except_for_the_embedding() {
        let im = weights(&[("blk.0.attn_q.weight", 128), ("token_embd.weight", 128)]);
        let p = planned("output.weight", vec![256, 8]);
        assert!(imatrix_for_tensor(&im, &p).unwrap().is_none());
        let p = planned("blk.0.attn_q.weight", vec![256, 8]);
        let err = imatrix_for_tensor(&im, &p).unwrap_err().to_string();
        assert!(
            err.contains("imatrix size 128 is different from tensor size 256"),
            "{err}"
        );
        let p = planned("token_embd.weight", vec![256, 8]);
        assert!(imatrix_for_tensor(&im, &p).unwrap().is_none());
        // Too WIDE is wrong too: an entry for a 4-expert stack handed to
        // a 2-D tensor would silently weight it by expert 0's columns.
        let im = weights(&[("blk.0.attn_q.weight", 1024)]);
        let p = planned("blk.0.attn_q.weight", vec![256, 8]);
        assert!(imatrix_for_tensor(&im, &p).is_err());
    }

    /// The four keys, their types, and the 127-byte truncation that
    /// `strncpy(kvo.val_str, ..., 127)` imposes on the file path.
    #[test]
    fn the_metadata_keys_match_llama_quantizes_types_and_truncation() {
        let im = weights(&[("blk.0.attn_q.weight", 4)]);
        let long = format!("/{}/imatrix.gguf", "x".repeat(200));
        let mut md = BTreeMap::new();
        imatrix_metadata(&mut md, &im, Path::new(&long));
        assert_eq!(md["quantize.imatrix.file"].as_str(), Some(&long[..127]));
        assert_eq!(md["quantize.imatrix.dataset"].as_str(), Some("calib.txt"));
        assert!(matches!(
            md["quantize.imatrix.entries_count"],
            GgufValue::U32(1)
        ));
        assert!(matches!(
            md["quantize.imatrix.chunks_count"],
            GgufValue::U32(3)
        ));
        // A legacy file without a trailer has no chunk count, and
        // upstream then writes no key rather than a zero.
        let mut md = BTreeMap::new();
        let mut im0 = im.clone();
        im0.chunk_count = 0;
        imatrix_metadata(&mut md, &im0, Path::new("im.dat"));
        assert!(!md.contains_key("quantize.imatrix.chunks_count"));
    }

    /// A 127-byte cut must not split a multi-byte character; a path
    /// with one straddling the boundary is cut before it.
    #[test]
    fn the_truncation_respects_char_boundaries() {
        let s = format!("{}\u{e9}", "a".repeat(126));
        assert_eq!(truncate_127(&s).len(), 126);
    }
}
