//! The gated delta-net block (Qwen3.5's linear attention), at the site
//! attention occupies on a recurrent layer
//! ([`crate::layer_shapes::AttnShape::Gdn`]).
//!
//! `qwen35.cpp:236-317` (`build_layer_attn_linear`), per token:
//!
//! ```text
//! qkv   = attn_qkv(normed)                       {2 key_dim + value_dim}    :248-250
//! z     = attn_gate(normed)                      {value_dim}                :250
//! beta  = sigmoid(ssm_beta(normed))              {n_v_heads}                :251-253
//! g     = softplus(ssm_alpha(normed) + ssm_dt) * ssm_a   {n_v_heads}        :254-262
//! qkv   = silu(conv(qkv))                        causal, width d_conv, NO bias  :266-273
//! q, k, v = qkv split [key_dim, key_dim, value_dim]                         :276-295
//! q, k  = l2_norm(q), l2_norm(k) per head        eps = rms_eps              :296-297
//! o     = delta_step(state, q / sqrt(S), k, v, g, beta)   ferrox_core::gdn   :302-308
//! o     = rms_norm(o per head, ssm_norm) * silu(z per head)                 :311-313, :171-178
//! out   = ssm_out(o)                                                        :315
//! ```
//!
//! with `head_k_dim = head_v_dim = ssm.state_size`, `n_k_heads =
//! ssm.group_count`, `n_v_heads = ssm.time_step_rank` and `ssm.inner_size
//! = n_v_heads * head_v_dim` (`:52-58`), V head `h` reading K head
//! `h % n_k_heads` (`HeadMap::Tiled`, `llama-model.cpp:524-526`; the
//! converter reorders V heads into that order). The state is a
//! [`RecurrentState`] on the sequence's layer cache: the conv window
//! `[d_conv - 1][2 key_dim + value_dim]` and the delta state
//! `[n_v_heads][S][S]`.
//!
//! # Which layers
//!
//! `qwen35.cpp:17-24`: `{arch}.attention.recurrent_layers` (a bool per
//! layer over `n_layer_all`, the MTP block included) when the file
//! carries it, else `(i + 1) % full_attention_interval != 0` with the
//! interval from `{arch}.full_attention_interval` (default 4) and the
//! MTP block never recurrent. [`recurrent_layers`] is that rule.
//!
//! # Reach
//!
//! Three graphs of 140 build the block (`grep -l build_layer_attn_linear
//! src/models/*.cpp`: `qwen35`, `qwen35moe`, `qwen3next`) over the ONE
//! `delta-net-base.cpp`. `qwen3next` differs in two places, both
//! tables here: its V heads read K heads GROUPED (`h / ratio`,
//! `qwen3next.cpp:521-539`, `llama-model.cpp:525`;
//! [`GROUPED_HEAD_ARCHITECTURES`]), and beta and alpha come from ONE
//! `ssm_ba` projection laid out `[k_group][beta * ratio, alpha * ratio]`
//! (`:96,422-436`; [`BetaAlpha::Fused`]). Its legacy fused `ssm_in`
//! (q/k/v/z in one, `:88-90,336-360`) is refused by name: every
//! current export splits it (`conversion/qwen.py:389-416`).

use ferrox_core::gdn::{delta_step, l2_normalize, DeltaDims, HeadMap};
use ferrox_core::mamba2::{conv_step, softplus};
use ferrox_core::matmul::rms_norm;
use ferrox_core::recurrent_state::RecurrentState;
use ferrox_core::weight_matrix::WeightMatrix;
use ferrox_gguf::TensorSource;

use crate::loader::{load_f32_vec, load_weight_matrix, LoadError};

/// Architectures whose V heads read K heads grouped (`HeadMap::Grouped`,
/// `qwen3next.cpp:521-539`: `ggml_repeat_4d` over a `[head_dim, 1,
/// n_k]` view repeats each K head `ratio` times consecutively). Every
/// other reader of the block tiles (`llama-model.cpp:524-526`).
pub const GROUPED_HEAD_ARCHITECTURES: &[&str] = &["qwen3next"];

/// How `arch`'s V heads find their K heads.
pub fn head_map(arch: &str) -> HeadMap {
    if GROUPED_HEAD_ARCHITECTURES.contains(&arch) {
        HeadMap::Grouped
    } else {
        HeadMap::Tiled
    }
}

/// The five `ssm.*` hparams as one value (`qwen35.cpp:7-11`), and the
/// architecture's head map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnHparams {
    pub d_conv: usize,
    /// `ssm.state_size`: the key AND value head width.
    pub head_dim: usize,
    /// `ssm.group_count`.
    pub n_k_heads: usize,
    /// `ssm.time_step_rank`.
    pub n_v_heads: usize,
    pub map: HeadMap,
}

impl GdnHparams {
    pub fn read(file: &impl TensorSource, arch: &str) -> Result<Self, LoadError> {
        let key = |k: &str| format!("{arch}.ssm.{k}");
        let read = |k: &str| {
            file.metadata_u64(&key(k))
                .map(|v| v as usize)
                .ok_or_else(|| LoadError::MissingHparam(key(k)))
        };
        let h = Self {
            d_conv: read("conv_kernel")?,
            head_dim: read("state_size")?,
            n_k_heads: read("group_count")?,
            n_v_heads: read("time_step_rank")?,
            map: head_map(arch),
        };
        let d_inner = read("inner_size")?;
        if h.d_conv < 2
            || h.head_dim == 0
            || h.n_k_heads == 0
            || !h.n_v_heads.is_multiple_of(h.n_k_heads)
            || d_inner != h.n_v_heads * h.head_dim
        {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!(
                    "ssm.* hparams {h:?} with inner_size {d_inner}: qwen35.cpp:52-58 sizes the \
                     block as inner_size = time_step_rank * state_size with time_step_rank a \
                     multiple of group_count (delta-net-base.cpp:308)"
                ),
            ));
        }
        Ok(h)
    }

    pub fn key_dim(self) -> usize {
        self.n_k_heads * self.head_dim
    }

    pub fn value_dim(self) -> usize {
        self.n_v_heads * self.head_dim
    }

    /// The conv's channel count, `2 key_dim + value_dim`.
    pub fn conv_dim(self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    pub fn delta_dims(self) -> DeltaDims {
        DeltaDims {
            n_k_heads: self.n_k_heads,
            n_v_heads: self.n_v_heads,
            head_dim: self.head_dim,
            map: self.map,
        }
    }

    /// Floats of state per sequence per layer: `n_embd_r + n_embd_s`.
    pub fn state_floats(self) -> (usize, usize) {
        (
            (self.d_conv - 1) * self.conv_dim(),
            self.delta_dims().state_len(),
        )
    }
}

/// Where beta and alpha come from.
pub enum BetaAlpha {
    /// `blk.N.ssm_beta.weight` and `blk.N.ssm_alpha.weight`, each
    /// `[n_v_heads, n_embd]` (`qwen35.cpp:71-72`).
    Split {
        beta: WeightMatrix,
        alpha: WeightMatrix,
    },
    /// `blk.N.ssm_ba.weight`, `[2 n_v_heads, n_embd]`, its rows laid out
    /// per K group as `ratio` betas then `ratio` alphas
    /// (`qwen3next.cpp:96,422-436`).
    Fused { ba: WeightMatrix },
}

/// One GDN layer's weights (`qwen35.cpp:66-74`).
pub struct Gdn {
    pub h: GdnHparams,
    /// `blk.N.attn_qkv.weight`, `[conv_dim, n_embd]`.
    pub qkv: WeightMatrix,
    /// `blk.N.attn_gate.weight`, `[value_dim, n_embd]`: the `z` gate.
    pub z_proj: WeightMatrix,
    /// `blk.N.ssm_conv1d.weight`, `[conv_dim][d_conv]`; no bias.
    pub conv1d: Vec<f32>,
    /// `blk.N.ssm_dt.bias`, `[n_v_heads]`.
    pub dt_bias: Vec<f32>,
    /// `blk.N.ssm_a`, `[n_v_heads]`, stored negative.
    pub a: Vec<f32>,
    pub beta_alpha: BetaAlpha,
    /// `blk.N.ssm_norm.weight`, `[head_dim]`, one weight for every head.
    pub norm: Vec<f32>,
    /// `blk.N.ssm_out.weight`, `[n_embd, value_dim]`.
    pub out_proj: WeightMatrix,
}

impl Gdn {
    /// Loads layer `layer`'s tensors and checks them against
    /// `qwen35.cpp:66-74`'s shapes.
    pub fn load(
        file: &impl TensorSource,
        arch: &str,
        layer: usize,
        hidden_dim: usize,
    ) -> Result<Self, LoadError> {
        let h = GdnHparams::read(file, arch)?;
        let name = |t: &str| format!("blk.{layer}.{t}");
        let matrix = |t: &str, rows: usize, cols: usize| -> Result<WeightMatrix, LoadError> {
            let m = load_weight_matrix(file, &name(t))?;
            if m.rows() != rows || m.cols() != cols {
                return Err(LoadError::UnsupportedFeature(
                    name(t),
                    format!(
                        "{}x{}; qwen35.cpp:66-74 sizes it {rows}x{cols}",
                        m.rows(),
                        m.cols()
                    ),
                ));
            }
            Ok(m)
        };
        let vector = |t: &str, len: usize| -> Result<Vec<f32>, LoadError> {
            let v = load_f32_vec(file, &name(t))?;
            if v.len() != len {
                return Err(LoadError::UnsupportedFeature(
                    name(t),
                    format!("{} elements; qwen35.cpp:66-74 sizes it {len}", v.len()),
                ));
            }
            Ok(v)
        };
        // `:66` creates `attn_qkv` TENSOR_NOT_REQUIRED because qwen3next
        // may store it fused as `ssm_in`; qwen35 has no other spelling.
        if file.find_tensor(&name("attn_qkv.weight")).is_none() {
            return Err(LoadError::UnsupportedFeature(
                name("attn_qkv.weight"),
                "absent: the gated delta net's fused q/k/v projection (qwen35.cpp:66); the \
                 legacy `ssm_in` / `ssm_ba` spelling (qwen3next.cpp:88-95) is not served"
                    .to_string(),
            ));
        }
        let beta_alpha = if file.find_tensor(&name("ssm_ba.weight")).is_some() {
            BetaAlpha::Fused {
                ba: matrix("ssm_ba.weight", 2 * h.n_v_heads, hidden_dim)?,
            }
        } else {
            BetaAlpha::Split {
                beta: matrix("ssm_beta.weight", h.n_v_heads, hidden_dim)?,
                alpha: matrix("ssm_alpha.weight", h.n_v_heads, hidden_dim)?,
            }
        };
        Ok(Self {
            h,
            qkv: matrix("attn_qkv.weight", h.conv_dim(), hidden_dim)?,
            z_proj: matrix("attn_gate.weight", h.value_dim(), hidden_dim)?,
            conv1d: vector("ssm_conv1d.weight", h.d_conv * h.conv_dim())?,
            dt_bias: vector("ssm_dt.bias", h.n_v_heads)?,
            a: vector("ssm_a", h.n_v_heads)?,
            beta_alpha,
            norm: vector("ssm_norm.weight", h.head_dim)?,
            out_proj: matrix("ssm_out.weight", hidden_dim, h.value_dim())?,
        })
    }

    /// A fresh sequence's state for this layer.
    pub fn zero_state(&self) -> RecurrentState {
        let (conv, ssm) = self.h.state_floats();
        RecurrentState::zeros(conv, ssm)
    }

    /// `rows` consecutive tokens of ONE sequence (`normed` is
    /// `[rows][n_embd]`) through the block, advancing `state` in place.
    pub fn forward_rows(
        &self,
        normed: &[f32],
        rows: usize,
        state: &mut RecurrentState,
        rms_eps: f32,
    ) -> Vec<f32> {
        let h = self.h;
        let n_embd = self.out_proj.rows();
        assert_eq!(normed.len(), rows * n_embd);
        let (conv_len, ssm_len) = h.state_floats();
        assert_eq!(
            state.conv.len(),
            conv_len,
            "conv state sized by these weights"
        );
        assert_eq!(
            state.ssm.len(),
            ssm_len,
            "delta state sized by these weights"
        );
        let (s, n_k, n_v) = (h.head_dim, h.n_k_heads, h.n_v_heads);
        let (key_dim, value_dim, conv_dim) = (h.key_dim(), h.value_dim(), h.conv_dim());
        let dims = h.delta_dims();
        let project = |m: &WeightMatrix| {
            if rows == 1 {
                m.apply(normed)
            } else {
                m.apply_batch(normed, rows)
            }
        };
        // `qkv` and `z` read the same input: one launch on a GPU
        // backend (`apply_pair`), two overlapped regions on the CPU.
        //
        // The gate projections are NOT in that launch, and that is a
        // measurement: sending all of them together as one list was
        // neutral on Bonsai (tg32 7.2 either way) because those
        // matrices are tiny and unquantized, and it made the fused
        // launch all-or-nothing across two different folds.
        let (qkv_all, z_all) = if rows == 1 {
            WeightMatrix::apply_pair(&self.qkv, &self.z_proj, normed)
        } else {
            // A batch rotates the shared input ONCE for the pair.
            WeightMatrix::apply_batch_pair_with_acts(&self.qkv, &self.z_proj, normed, rows, None)
        };
        // The two per-head gate logits, whichever projection spells
        // them: `[rows][n_v]` each.
        let (beta_all, alpha_all) = match &self.beta_alpha {
            BetaAlpha::Split { beta, alpha } => (project(beta), project(alpha)),
            BetaAlpha::Fused { ba } => {
                let mixed = project(ba);
                let ratio = n_v / n_k;
                let mut beta_all = vec![0.0f32; rows * n_v];
                let mut alpha_all = vec![0.0f32; rows * n_v];
                for r in 0..rows {
                    let row = &mixed[r * 2 * n_v..(r + 1) * 2 * n_v];
                    for hd in 0..n_v {
                        // qwen3next.cpp:422-436: group `hd / ratio`, its
                        // `ratio` betas then its `ratio` alphas.
                        let base = (hd / ratio) * 2 * ratio + hd % ratio;
                        beta_all[r * n_v + hd] = row[base];
                        alpha_all[r * n_v + hd] = row[base + ratio];
                    }
                }
                (beta_all, alpha_all)
            }
        };
        let mut ys = vec![0.0f32; rows * value_dim];
        let mut conv_out = vec![0.0f32; conv_dim];
        let mut o = vec![0.0f32; value_dim];
        let mut g = vec![0.0f32; n_v];
        let mut beta = vec![0.0f32; n_v];
        // A PREFILL batch takes the CHUNKED delta rule
        // (`ferrox_core::gdn_chunk`): the same recurrence with the
        // state read once per chunk of rows instead of once per row.
        // The sequential step is bandwidth-bound (36 GB/s of state,
        // measured) and a 128-token Bonsai prefill moves 38 GB through
        // it, so trading 1.5x the multiply-adds for a 32nd of the
        // traffic is 2.1x on the step. A decode token still steps one
        // row at a time, where there is no traffic to amortise, and
        // running the row step on the GPU is a loss three ways
        // (`docs/plans/gdn-resident-state.md`).
        if rows > 1 {
            let (q_all, k_all, v_all, g_all, beta_gate) =
                self.conv_and_gates_for_rows(rows, &qkv_all, &beta_all, &alpha_all, state, rms_eps);
            let mut o_all = vec![0.0f32; rows * value_dim];
            ferrox_core::gdn_chunk::delta_chunk(
                dims,
                rows,
                &mut state.ssm,
                &q_all,
                &k_all,
                &v_all,
                &g_all,
                &beta_gate,
                &mut o_all,
            );
            for r in 0..rows {
                let o = &o_all[r * value_dim..(r + 1) * value_dim];
                let y = &mut ys[r * value_dim..(r + 1) * value_dim];
                let z = &z_all[r * value_dim..(r + 1) * value_dim];
                for hd in 0..n_v {
                    let normed_head = rms_norm(&o[hd * s..(hd + 1) * s], &self.norm, rms_eps);
                    for i in 0..s {
                        y[hd * s + i] = normed_head[i] * silu(z[hd * s + i]);
                    }
                }
            }
            return self.out_proj.apply_batch(&ys, rows);
        }
        for r in 0..rows {
            // :251-262: the two per-head gates from the layer input.
            for hd in 0..n_v {
                beta[hd] = sigmoid(beta_all[r * n_v + hd]);
                g[hd] = softplus(alpha_all[r * n_v + hd] + self.dt_bias[hd]) * self.a[hd];
            }
            // :266-273: the conv over this token and the state, SiLU.
            conv_step(
                &mut state.conv,
                &self.conv1d,
                h.d_conv,
                &qkv_all[r * conv_dim..(r + 1) * conv_dim],
                &mut conv_out,
            );
            for x in conv_out.iter_mut() {
                *x = silu(*x);
            }
            let (q, rest) = conv_out.split_at_mut(key_dim);
            let (k, v) = rest.split_at_mut(key_dim);
            // :296-297: l2 per head, with the RMS epsilon.
            for hd in 0..n_k {
                l2_normalize(&mut q[hd * s..(hd + 1) * s], rms_eps);
                l2_normalize(&mut k[hd * s..(hd + 1) * s], rms_eps);
            }
            // The recurrence stays on the HOST, and that is THREE
            // measurements rather than an omission. Running it, the
            // gated norm and `ssm_out` as one Metal submission works
            // and is slower every way it has been tried on
            // Bonsai-2-27B: 6.0 tok/s against 7.1 with the state copied
            // both ways, 6.6 with it wrapped in place (the state buffer
            // is page-aligned for exactly that, `AlignedF32`), and 6.9
            // with the wrapper cached so the pages are mapped once.
            // The kernels are real and pinned against this code
            // (`ferrox_metal::gdn`); what beats a host recurrence is a
            // CHUNKED delta rule, which is a different algorithm.
            // `docs/plans/gdn-resident-state.md` carries all of it.
            delta_step(dims, &mut state.ssm, q, k, v, &g, &beta, &mut o);
            // :311-313 (`build_norm_gated`, :171-178): per head,
            // rms_norm(o, ssm_norm) * silu(z).
            let y = &mut ys[r * value_dim..(r + 1) * value_dim];
            let z = &z_all[r * value_dim..(r + 1) * value_dim];
            for hd in 0..n_v {
                let normed_head = rms_norm(&o[hd * s..(hd + 1) * s], &self.norm, rms_eps);
                for i in 0..s {
                    y[hd * s + i] = normed_head[i] * silu(z[hd * s + i]);
                }
            }
        }
        if rows == 1 {
            self.out_proj.apply(&ys)
        } else {
            self.out_proj.apply_batch(&ys, rows)
        }
    }
}

impl Gdn {
    /// Every row's conv step, gates and l2 norms, which the chunked
    /// recurrence needs up front: none of them reads the delta state,
    /// so they do not have to interleave with it the way the
    /// row-at-a-time loop does.
    ///
    /// Returns `(q, k, v, g, beta)`, each `[rows][...]`.
    #[allow(clippy::type_complexity)]
    fn conv_and_gates_for_rows(
        &self,
        rows: usize,
        qkv_all: &[f32],
        beta_all: &[f32],
        alpha_all: &[f32],
        state: &mut RecurrentState,
        rms_eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = self.h;
        let (s, n_k, n_v) = (h.head_dim, h.n_k_heads, h.n_v_heads);
        let (key_dim, value_dim, conv_dim) = (h.key_dim(), h.value_dim(), h.conv_dim());
        let mut q_all = vec![0.0f32; rows * key_dim];
        let mut k_all = vec![0.0f32; rows * key_dim];
        let mut v_all = vec![0.0f32; rows * value_dim];
        let mut g_all = vec![0.0f32; rows * n_v];
        let mut beta_gate = vec![0.0f32; rows * n_v];
        let mut conv_out = vec![0.0f32; conv_dim];
        for r in 0..rows {
            for hd in 0..n_v {
                beta_gate[r * n_v + hd] = sigmoid(beta_all[r * n_v + hd]);
                g_all[r * n_v + hd] =
                    softplus(alpha_all[r * n_v + hd] + self.dt_bias[hd]) * self.a[hd];
            }
            conv_step(
                &mut state.conv,
                &self.conv1d,
                h.d_conv,
                &qkv_all[r * conv_dim..(r + 1) * conv_dim],
                &mut conv_out,
            );
            for x in conv_out.iter_mut() {
                *x = silu(*x);
            }
            let (q, rest) = conv_out.split_at_mut(key_dim);
            let (k, v) = rest.split_at_mut(key_dim);
            for hd in 0..n_k {
                l2_normalize(&mut q[hd * s..(hd + 1) * s], rms_eps);
                l2_normalize(&mut k[hd * s..(hd + 1) * s], rms_eps);
            }
            q_all[r * key_dim..(r + 1) * key_dim].copy_from_slice(q);
            k_all[r * key_dim..(r + 1) * key_dim].copy_from_slice(k);
            v_all[r * value_dim..(r + 1) * value_dim].copy_from_slice(v);
        }
        (q_all, k_all, v_all, g_all, beta_gate)
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Architectures whose recurrent layers are decided by
/// `{arch}.attention.recurrent_layers` / `{arch}.full_attention_interval`
/// rather than by a zero KV count (`qwen35.cpp:17-24`; `qwen35moe.cpp`
/// and `qwen3next.cpp` read the same two keys).
pub const INTERVAL_RECURRENT_ARCHITECTURES: &[&str] = &["qwen35", "qwen35moe", "qwen3next"];

/// Which trunk layers of `arch` are recurrent, or `None` for an
/// architecture that decides by its head counts.
///
/// `qwen35.cpp:17-24`: the array wins when present (read at
/// `block_count` length, the MTP block included, and cut to the trunk
/// here); else layer `i` is recurrent iff `(i + 1) % interval != 0`,
/// with `interval` from `{arch}.full_attention_interval` (default 4).
pub fn recurrent_layers(
    file: &impl TensorSource,
    arch: &str,
    block_count: usize,
    n_layers: usize,
) -> Result<Option<Vec<bool>>, LoadError> {
    if !INTERVAL_RECURRENT_ARCHITECTURES.contains(&arch) {
        return Ok(None);
    }
    let key = format!("{arch}.attention.recurrent_layers");
    if let Some(ferrox_gguf::GgufValue::Array(items)) = file.metadata(&key) {
        if items.len() != block_count {
            return Err(LoadError::UnsupportedFeature(
                key,
                format!(
                    "{} entries for block_count {block_count}; llama.cpp reads it at n_layer_all \
                     length (qwen35.cpp:17)",
                    items.len()
                ),
            ));
        }
        let mut out = Vec::with_capacity(n_layers);
        for (il, item) in items.iter().enumerate().take(n_layers) {
            out.push(item.as_bool().ok_or_else(|| {
                LoadError::UnsupportedFeature(key.clone(), format!("entry {il} is not a bool"))
            })?);
        }
        return Ok(Some(out));
    }
    let interval = file
        .metadata_u64(&format!("{arch}.full_attention_interval"))
        .unwrap_or(4) as usize;
    if interval == 0 {
        return Err(LoadError::UnsupportedFeature(
            format!("{arch}.full_attention_interval"),
            "0: qwen35.cpp:22 takes `(i + 1) % interval`".to_string(),
        ));
    }
    Ok(Some(
        (0..n_layers)
            .map(|i| !(i + 1).is_multiple_of(interval))
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_core::Tensor;

    fn hp() -> GdnHparams {
        GdnHparams {
            d_conv: 3,
            head_dim: 2,
            n_k_heads: 1,
            n_v_heads: 2,
            map: HeadMap::Tiled,
        }
    }

    #[test]
    fn the_widths_are_the_graph_s() {
        let h = hp();
        assert_eq!((h.key_dim(), h.value_dim(), h.conv_dim()), (2, 4, 8));
        assert_eq!(h.state_floats(), (2 * 8, 2 * 2 * 2));
    }

    /// Batched rows and one-at-a-time rows agree and leave the same
    /// state.
    #[test]
    fn rows_and_one_at_a_time_agree_and_leave_the_same_state() {
        let h = hp();
        let n_embd = 3;
        let mut seed = 5u32;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
                })
                .collect()
        };
        let mat = |rows: usize, cols: usize, v: Vec<f32>| {
            WeightMatrix::F32(Tensor::new(v, vec![rows, cols]))
        };
        let m = Gdn {
            h,
            qkv: mat(h.conv_dim(), n_embd, rnd(h.conv_dim() * n_embd)),
            z_proj: mat(h.value_dim(), n_embd, rnd(h.value_dim() * n_embd)),
            conv1d: rnd(h.d_conv * h.conv_dim()),
            dt_bias: rnd(h.n_v_heads),
            a: vec![-0.7, -1.2],
            beta_alpha: BetaAlpha::Split {
                beta: mat(h.n_v_heads, n_embd, rnd(h.n_v_heads * n_embd)),
                alpha: mat(h.n_v_heads, n_embd, rnd(h.n_v_heads * n_embd)),
            },
            norm: vec![1.1, 0.9],
            out_proj: mat(n_embd, h.value_dim(), rnd(n_embd * h.value_dim())),
        };
        let tokens: Vec<Vec<f32>> = (0..4).map(|_| rnd(n_embd)).collect();
        let flat: Vec<f32> = tokens.concat();
        let mut s_batch = m.zero_state();
        let batched = m.forward_rows(&flat, 4, &mut s_batch, 1e-5);
        let mut s_seq = m.zero_state();
        let mut seq = Vec::new();
        for t in &tokens {
            seq.extend(m.forward_rows(t, 1, &mut s_seq, 1e-5));
        }
        for (a, b) in batched.iter().zip(&seq) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        // Close, not identical: a batch takes the CHUNKED delta rule
        // (`ferrox_core::gdn_chunk`), which is the same recurrence
        // with the rank-one updates unrolled across the chunk, so the
        // float association differs from stepping row by row. The
        // chunked module pins the two against each other directly; what
        // this test is for is that the batched path has not lost a
        // FEATURE, which a tolerance still catches.
        assert_eq!(s_batch.conv, s_seq.conv, "the conv window is exact");
        for (a, b) in s_batch.ssm.iter().zip(s_seq.ssm.iter()) {
            assert!((a - b).abs() < 1e-6, "state: {a} vs {b}");
        }
        let again = m.forward_rows(&flat, 4, &mut s_batch, 1e-5);
        assert!(again
            .iter()
            .zip(&batched)
            .any(|(a, b)| (a - b).abs() > 1e-6));
    }
}
