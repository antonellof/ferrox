//! Loading Q/K/V from a checkpoint that fuses them into one tensor.
//!
//! Some architectures store `blk.N.attn_q/k/v.weight` separately and
//! some store one `blk.N.attn_qkv.weight` with Q's rows, then K's, then
//! V's. llama.cpp resolves that in `create_tensor_qkv`
//! (`src/llama-model.cpp:2886-2900`): a fused `wqkv` wins when present,
//! and **the bias follows the weight** -- a fused weight gets
//! `attn_qkv.bias`, split weights get `attn_{q,k,v}.bias`. `build_qkv`
//! (`src/llama-graph.cpp:1591-1664`) then adds the fused bias to the
//! fused projection *before* slicing it into the three views, which is
//! arithmetically the same as adding each slice of the bias to the
//! matching projection.
//!
//! ferrox split the fused WEIGHT and then looked for the bias only
//! under the split names. On any checkpoint that fuses both -- ChatGLM2
//! and ChatGLM3 set `add_qkv_bias: true` and gguf-py maps
//! `self_attention.query_key_value` onto `blk.N.attn_qkv`, so the file
//! really does hold `attn_qkv.weight` AND `attn_qkv.bias` -- the bias
//! was silently dropped and all three projections ran unbiased. It
//! loaded, and it answered fluently.
//!
//! **That is this repo's dominant bug shape**: two structures that had
//! to agree about one thing (which spelling this layer uses) with
//! nothing enforcing it, because the weight decided independently of
//! the bias. The shape of the fix matters more than the fix: the choice
//! is made ONCE, in [`load_fused_or_split_qkv`], and both halves are
//! sliced by the same [`FusedQkvRows`] spans, so a future change to the
//! row arithmetic cannot move one and leave the other behind.

use crate::config::ModelConfig;
use crate::loader::{
    load_f32_vec, load_f32_vec_optional, load_weight_matrix, slice_quantized_rows,
};
use crate::LoadError;
use ferrox_core::WeightMatrix;
use ferrox_gguf::{GgufError, TensorSource};

/// Where Q, K and V live inside a fused `attn_qkv` tensor.
///
/// llama.cpp computes the same three offsets twice -- once to size the
/// tensor (`llama-model.cpp:2889`) and once to view it
/// (`llama-graph.cpp:1615-1622`). Here they exist once so the weight
/// split, the bias split and the length check cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FusedQkvRows {
    q: usize,
    k: usize,
    v: usize,
}

impl FusedQkvRows {
    /// Layer `layer`'s spans. Per layer because `openelm.cpp:27,34`
    /// sizes each layer's `wqkv` from that layer's own head counts; a
    /// uniform model gives the same answer for every layer.
    ///
    /// Only a [`crate::layer_shapes::AttnShape::Gqa`] layer has a fused
    /// tensor to split; the other two shapes never reach this.
    pub(crate) fn of(config: &ModelConfig, layer: usize) -> Self {
        let shape = config.layer_shape(layer).attention;
        // K at the K width, V at the V width: `mimo2.cpp:132-140` views
        // the fused tensor exactly so, and for every other architecture
        // the two widths are one (`crate::kv_head_dims`).
        Self {
            q: shape.n_heads() * config.head_dim,
            k: shape.n_kv_heads() * config.head_dim,
            v: shape.n_kv_heads() * config.v_head_dim(),
        }
    }

    /// Total rows a fused `attn_qkv` tensor must have, and the length a
    /// fused `attn_qkv.bias` must have.
    pub(crate) fn total(self) -> usize {
        self.q + self.k + self.v
    }

    /// `(start, len)` for Q, K and V, in file order.
    pub(crate) fn spans(self) -> [(usize, usize); 3] {
        [(0, self.q), (self.q, self.k), (self.q + self.k, self.v)]
    }
}

/// One layer's Q/K/V projections and their optional biases, resolved
/// from whichever spelling the file uses.
pub(crate) struct QkvProjections {
    pub(crate) q: WeightMatrix,
    pub(crate) k: WeightMatrix,
    pub(crate) v: WeightMatrix,
    pub(crate) q_bias: Option<Vec<f32>>,
    pub(crate) k_bias: Option<Vec<f32>>,
    pub(crate) v_bias: Option<Vec<f32>>,
}

/// Loads Q/K/V (and their biases) for one layer: prefers split
/// `attn_{q,k,v}.weight`, falls back to a fused `attn_qkv.weight`
/// (Phi-3, Qwen-1, ChatGLM, BailingMoE-2) by slicing quantized rows --
/// zero-copy for mmapped GGUFs, dequantizing only when the storage is
/// not block-quantized.
///
/// Mirrors llama.cpp's `create_tensor_qkv`, including the part that was
/// missing: the bias comes from the same spelling as the weight.
pub(crate) fn load_fused_or_split_qkv(
    file: &impl TensorSource,
    layer: usize,
    config: &ModelConfig,
) -> Result<QkvProjections, LoadError> {
    let q_name = format!("blk.{layer}.attn_q.weight");
    let k_name = format!("blk.{layer}.attn_k.weight");
    let v_name = format!("blk.{layer}.attn_v.weight");
    let fused_name = format!("blk.{layer}.attn_qkv.weight");

    if file.find_tensor(&q_name).is_some() {
        // Split spelling: each bias is independently optional, exactly
        // as `create_tensor_qkv`'s else-branch creates them
        // (`TENSOR_NOT_REQUIRED`, llama-model.cpp:2897-2899).
        return Ok(QkvProjections {
            q: load_weight_matrix(file, &q_name)?,
            k: load_weight_matrix(file, &k_name)?,
            v: load_weight_matrix(file, &v_name)?,
            q_bias: load_f32_vec_optional(file, &format!("blk.{layer}.attn_q.bias"))?,
            k_bias: load_f32_vec_optional(file, &format!("blk.{layer}.attn_k.bias"))?,
            v_bias: load_f32_vec_optional(file, &format!("blk.{layer}.attn_v.bias"))?,
        });
    }
    if file.find_tensor(&fused_name).is_none() {
        return Err(LoadError::Gguf(GgufError::TensorNotFound(q_name)));
    }

    let rows = FusedQkvRows::of(config, layer);
    let fused = load_weight_matrix(file, &fused_name)?;
    if fused.rows() != rows.total() {
        // Phi-3 sometimes stores Q as full n_embd (== q rows when MHA).
        return Err(LoadError::UnsupportedFeature(
            config.name.to_string(),
            format!(
                "{fused_name} has {} rows; expected q+k+v = {} \
                 (n_heads*head_dim + n_kv_heads*head_dim + n_kv_heads*v_head_dim)",
                fused.rows(),
                rows.total()
            ),
        ));
    }
    let [q_span, k_span, v_span] = rows.spans();
    let (q, k, v) = split_fused_weight(&fused, rows)?;
    let (q_bias, k_bias, v_bias) = match split_fused_bias(file, layer, config, rows)? {
        None => (None, None, None),
        Some(b) => (
            Some(b[q_span.0..q_span.0 + q_span.1].to_vec()),
            Some(b[k_span.0..k_span.0 + k_span.1].to_vec()),
            Some(b[v_span.0..v_span.0 + v_span.1].to_vec()),
        ),
    };
    Ok(QkvProjections {
        q,
        k,
        v,
        q_bias,
        k_bias,
        v_bias,
    })
}

/// Slices the fused weight into three matrices by [`FusedQkvRows`].
///
/// Quantized storage is sliced by row range with no dequantization, so
/// Q/K/V stay on the quantized (Metal-capable) matvec path; any other
/// storage is widened once and split.
fn split_fused_weight(
    fused: &WeightMatrix,
    rows: FusedQkvRows,
) -> Result<(WeightMatrix, WeightMatrix, WeightMatrix), LoadError> {
    let [q_span, k_span, v_span] = rows.spans();
    if let (Some(q), Some(k), Some(v)) = (
        slice_quantized_rows(fused, q_span.0, q_span.1),
        slice_quantized_rows(fused, k_span.0, k_span.1),
        slice_quantized_rows(fused, v_span.0, v_span.1),
    ) {
        return Ok((q, k, v));
    }
    let cols = fused.cols();
    let mut full = Vec::with_capacity(fused.rows() * cols);
    for r in 0..fused.rows() {
        full.extend_from_slice(&fused.dequant_row(r));
    }
    let take = |span: (usize, usize)| {
        WeightMatrix::F32(ferrox_core::Tensor::new(
            full[span.0 * cols..(span.0 + span.1) * cols].to_vec(),
            vec![span.1, cols],
        ))
    };
    Ok((take(q_span), take(k_span), take(v_span)))
}

/// The fused `attn_qkv.bias`, checked against the same row arithmetic
/// the weight was checked against.
///
/// A length mismatch is a hard refusal rather than a dropped bias: this
/// whole module exists because a dropped bias loads and answers
/// fluently.
fn split_fused_bias(
    file: &impl TensorSource,
    layer: usize,
    config: &ModelConfig,
    rows: FusedQkvRows,
) -> Result<Option<Vec<f32>>, LoadError> {
    let name = format!("blk.{layer}.attn_qkv.bias");
    if file.find_tensor(&name).is_none() {
        return Ok(None);
    }
    let bias = load_f32_vec(file, &name)?;
    if bias.len() != rows.total() {
        return Err(LoadError::UnsupportedFeature(
            config.name.to_string(),
            format!(
                "{name} has {} elements; expected q+k+v = {} \
                 (n_heads*head_dim + n_kv_heads*head_dim + n_kv_heads*v_head_dim)",
                bias.len(),
                rows.total()
            ),
        ));
    }
    Ok(Some(bias))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spans must tile the whole tensor with no gap and no overlap,
    /// and must end exactly at `total()`.
    ///
    /// The bug this file exists for was two independent computations of
    /// these offsets; this is the assertion that makes a third one
    /// impossible to add quietly.
    #[test]
    fn the_three_spans_tile_the_fused_tensor_exactly() {
        for (n_heads, n_kv_heads, head_dim, v_head_dim) in [
            (4, 2, 8, 8),
            (32, 32, 128, 128),
            (7, 1, 64, 64),
            (4, 2, 12, 8),
        ] {
            let rows = FusedQkvRows {
                q: n_heads * head_dim,
                k: n_kv_heads * head_dim,
                v: n_kv_heads * v_head_dim,
            };
            let spans = rows.spans();
            assert_eq!(spans[0].0, 0);
            for w in spans.windows(2) {
                assert_eq!(
                    w[0].0 + w[0].1,
                    w[1].0,
                    "spans must be contiguous for {n_heads}/{n_kv_heads}/{head_dim}"
                );
            }
            let last = spans[2];
            assert_eq!(
                last.0 + last.1,
                rows.total(),
                "spans must end at total() for {n_heads}/{n_kv_heads}/{head_dim}"
            );
        }
    }

    /// K and V are the same width; Q is the wide one. Getting that
    /// backwards is the classic GQA off-by-a-factor.
    #[test]
    fn k_and_v_have_the_same_width_and_q_is_the_gqa_multiple() {
        let rows = FusedQkvRows {
            q: 32,
            k: 16,
            v: 16,
        };
        let spans = rows.spans();
        assert_eq!(spans[1].1, spans[2].1);
        assert_eq!(spans[0].1, 2 * spans[1].1);
        assert_eq!(rows.total(), 64);
    }

    /// A tensor-only GGUF holding `blk.0.attn_qkv.weight` and
    /// `blk.0.attn_qkv.bias` as F32, so the refusals below run against
    /// a real `TensorSource` rather than a mock of one.
    ///
    /// No metadata at all: these tests hand [`load_fused_or_split_qkv`]
    /// the `ModelConfig` directly, which is the point -- the mismatch
    /// under test is between what the config says and what the FILE
    /// holds.
    fn fused_qkv_gguf(
        weight_rows: usize,
        cols: usize,
        bias_len: Option<usize>,
    ) -> ferrox_gguf::GgufFile {
        // `ferrox_gguf::GgufWriter`, not an eighth hand-rolled byte
        // builder: its module docs record that there used to be seven
        // and that none of them produced a file a second tool could
        // read.
        let weight_bytes = weight_rows * cols * 4;
        let mut plan = vec![ferrox_gguf::TensorPlan {
            name: "blk.0.attn_qkv.weight".into(),
            // GGUF `ne` is column-major: [cols, rows].
            shape: vec![cols as u64, weight_rows as u64],
            dtype: ferrox_gguf::GgmlType::F32,
            byte_len: weight_bytes,
        }];
        if let Some(n) = bias_len {
            plan.push(ferrox_gguf::TensorPlan {
                name: "blk.0.attn_qkv.bias".into(),
                shape: vec![n as u64],
                dtype: ferrox_gguf::GgmlType::F32,
                byte_len: n * 4,
            });
        }
        let mut w = ferrox_gguf::GgufWriter::create(Vec::new(), &Default::default(), plan).unwrap();
        w.write_tensor("blk.0.attn_qkv.weight", &vec![0u8; weight_bytes])
            .unwrap();
        if let Some(n) = bias_len {
            w.write_tensor("blk.0.attn_qkv.bias", &vec![0u8; n * 4])
                .unwrap();
        }
        let bytes = w.finish().unwrap();

        let tag = bias_len.map_or("none".to_string(), |n| n.to_string());
        let tmp =
            std::env::temp_dir().join(format!("ferrox_qkv_fused_{weight_rows}_{cols}_{tag}.gguf"));
        std::fs::write(&tmp, &bytes).unwrap();
        let file = ferrox_gguf::GgufFile::open(&tmp).expect("the written file must parse");
        std::fs::remove_file(&tmp).ok();
        file
    }

    fn config_4x2_head8() -> ModelConfig {
        let mut cfg = crate::config::glm_5_2();
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 8;
        cfg
    }

    /// A fused bias whose length disagrees with the head arithmetic is
    /// REFUSED, not dropped and not truncated.
    ///
    /// This is the reachability half. A refusal nothing can reach reads
    /// as coverage while checking nothing, and this one guards the exact
    /// failure the module exists for: the file's bias and the config's
    /// idea of Q/K/V widths disagreeing, with three projections about to
    /// be built from it.
    #[test]
    fn a_fused_bias_of_the_wrong_length_is_refused_by_name() {
        let cfg = config_4x2_head8();
        let rows = FusedQkvRows::of(&cfg, 0);
        assert_eq!(rows.total(), 64);
        // The WEIGHT is the right shape, so the refusal below can only
        // come from the bias.
        let file = fused_qkv_gguf(rows.total(), 8, Some(rows.total() - 1));
        let Err(err) = load_fused_or_split_qkv(&file, 0, &cfg) else {
            panic!("a fused bias one element short must refuse");
        };
        let msg = err.to_string();
        assert!(msg.contains("attn_qkv.bias"), "{msg}");
        assert!(
            msg.contains("63"),
            "the message must name what it found: {msg}"
        );
        assert!(msg.contains("64"), "and what it expected: {msg}");
    }

    /// The same file with the RIGHT bias length loads, so the test above
    /// is not passing on some unrelated error.
    #[test]
    fn a_fused_bias_of_the_right_length_is_split_into_three() {
        let cfg = config_4x2_head8();
        let rows = FusedQkvRows::of(&cfg, 0);
        let file = fused_qkv_gguf(rows.total(), 8, Some(rows.total()));
        let Ok(p) = load_fused_or_split_qkv(&file, 0, &cfg) else {
            panic!("a fused bias of the declared length must load");
        };
        assert_eq!(p.q_bias.expect("q").len(), 32);
        assert_eq!(p.k_bias.expect("k").len(), 16);
        assert_eq!(p.v_bias.expect("v").len(), 16);
    }

    /// A fused file with NO bias tensor keeps all three as `None`.
    ///
    /// Phi-3 is exactly this and is audited, so the arm must not have
    /// turned an absent bias into a zero one -- adding zeros would be
    /// numerically inert here and wrong the moment anything downstream
    /// branches on `is_some()`.
    #[test]
    fn a_fused_weight_with_no_bias_leaves_all_three_unset() {
        let cfg = config_4x2_head8();
        let rows = FusedQkvRows::of(&cfg, 0);
        let file = fused_qkv_gguf(rows.total(), 8, None);
        assert!(file.find_tensor("blk.0.attn_qkv.bias").is_none());
        let Ok(p) = load_fused_or_split_qkv(&file, 0, &cfg) else {
            panic!("a fused weight with no bias must still load");
        };
        assert!(p.q_bias.is_none());
        assert!(p.k_bias.is_none());
        assert!(p.v_bias.is_none());
    }
}
