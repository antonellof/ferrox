//! Kimi K3's KDA (Kimi Delta Attention): a gated delta-rule linear
//! attention mechanism used on the majority of Kimi K3's layers
//! (69 of 93, per `AttentionKind::KimiHybrid`), interleaved with Gated
//! MLA (`frink_models::mla`) on the remainder.
//!
//! Transcribed directly from real reference source fetched live (not
//! guessed, not derived by analogy to other gated-linear-attention
//! designs):
//! - `moonshotai/Kimi-K3`'s `modeling_kimi_linear.py`,
//!   `KimiDeltaAttention` (q/k/v projections, short causal convolutions,
//!   decay-gate and beta projections, output gate, `RMSNormGated`).
//! - `fla-org/flash-linear-attention`'s `fla/ops/kda/naive.py`
//!   (`naive_recurrent_kda`) for the core state recurrence: per position,
//!   decay the state by `exp(g)`, add a rank-1 correction
//!   `beta * k ⊗ (v - kᵀS)`, read the output as `qᵀS`.
//! - That same project's `fla/ops/kda/fused_recurrent.py` Triton kernel
//!   source (the kernel actually invoked in decode, read directly since
//!   `naive_recurrent_kda` takes its inputs pre-transformed) for the
//!   exact preprocessing: L2-normalize q/k (`eps=1e-6`) before the
//!   recurrence, then scale q by `head_dim^-0.5`.
//! - `fla`'s `fla/modules/conv/short_conv.py` (`ShortConvolution`: a
//!   depthwise causal `Conv1d`, `padding=kernel_size-1`, no bias) and
//!   `fla/modules/fused_norm_gate.py` (`FusedRMSNormGated`) for the short
//!   causal conv and output-gate formulas.
//!
//! Two real, non-obvious facts confirmed by reading source rather than
//! assuming standard conventions:
//!
//! 1. The decay gate `g` is per-(head, key-dim), not one scalar per
//!    head: `g = gate_lower_bound * sigmoid(exp(A_log) * (raw_g +
//!    dt_bias))`, with `A_log`/`gate_lower_bound` per-head but
//!    `raw_g`/`dt_bias` per-(head, dim). State decay `S *= exp(g)` is
//!    applied per key dimension, broadcast across the value dimension
//!    (`S` is `[head_dim, head_dim]` per head here, since KDA's K and V
//!    head dims are both `head_dim` — unlike Gated MLA, where they
//!    differ).
//! 2. `FusedRMSNormGated`'s output-gate activation is **sigmoid**, not
//!    silu/swish (the more common choice in gated-linear-attention
//!    literature) — confirmed directly by `KimiDeltaAttention.__init__`
//!    passing `activation='sigmoid'` explicitly.
//!
//! `KdaConfig::use_full_rank_gate` is `true` for Kimi K3's real
//! configuration, so only that output-gate path (`g_proj` projecting
//! `hidden_size -> num_heads*head_dim` directly) is implemented; the
//! real reference's low-rank alternative (`g_a_proj`/`g_b_proj`) is
//! unused by Kimi K3 and intentionally not implemented here.
//!
//! Not yet wired into `Decoder`'s forward pass. Tested here against
//! synthetic weights, cross-validated against an independent Python
//! transcription of the same real algorithm run
//! one position at a time (matching this module's incremental decode
//! API) over a 5-position sequence — long enough to exercise the short
//! causal conv's full window (`short_conv_kernel_size` = 4) past its
//! initial zero-padded steps.

use frink_core::matmul::{rms_norm, silu};
use frink_core::weight_matrix::WeightMatrix;

use crate::config::KdaConfig;

pub struct KdaAttnWeights {
    pub q_proj: WeightMatrix, // [n_heads*head_dim, hidden_dim]
    pub k_proj: WeightMatrix, // [n_heads*head_dim, hidden_dim]
    pub v_proj: WeightMatrix, // [n_heads*head_dim, hidden_dim]
    /// Depthwise causal conv taps, row-major `[n_heads*head_dim,
    /// short_conv_kernel_size]` — one independent kernel per channel.
    pub q_conv_weight: Vec<f32>,
    pub k_conv_weight: Vec<f32>,
    pub v_conv_weight: Vec<f32>,
    pub a_log: Vec<f32>,         // [n_heads]
    pub f_a_proj: WeightMatrix,  // [head_dim, hidden_dim]
    pub f_b_proj: WeightMatrix,  // [n_heads*head_dim, head_dim]
    pub dt_bias: Vec<f32>,       // [n_heads*head_dim]
    pub b_proj: WeightMatrix,    // [n_heads, hidden_dim]
    pub g_proj: WeightMatrix,    // [n_heads*head_dim, hidden_dim] (full-rank output gate)
    pub o_norm_weight: Vec<f32>, // [head_dim]
    pub o_proj: WeightMatrix,    // [hidden_dim, n_heads*head_dim]
}

/// Per-layer decode-time state: the short causal convs' recent-input
/// history (up to `short_conv_kernel_size - 1` raw projected vectors
/// each) and the recurrent state `S` (`[n_heads, head_dim, head_dim]`,
/// flattened, zero-initialized) -- fundamentally different from
/// `frink_core::cache::KvCache`'s growing K/V history, since KDA's
/// per-layer state is fixed-size regardless of sequence length.
pub struct KdaState {
    conv_hist_q: Vec<f32>,
    conv_hist_k: Vec<f32>,
    conv_hist_v: Vec<f32>,
    recurrent: Vec<f32>,
}

impl KdaState {
    pub fn new(cfg: &KdaConfig) -> Self {
        KdaState {
            conv_hist_q: Vec::new(),
            conv_hist_k: Vec::new(),
            conv_hist_v: Vec::new(),
            recurrent: vec![0f32; cfg.num_heads * cfg.head_dim * cfg.head_dim],
        }
    }
}

/// One depthwise causal-conv step for all `dim` channels at once, plus
/// SiLU activation. `history` holds up to `kernel_size - 1` previous raw
/// (pre-activation) projected vectors, oldest first, flattened; updated
/// in place to hold the trailing window after this call. `weight` is
/// row-major `[dim, kernel_size]`.
///
/// Matches `ShortConvolution`'s real math exactly: `y[d] = silu(sum_{j=0
/// ..kernel_size} weight[d,j] * x[t-(kernel_size-1)+j, d])`, treating
/// any position before the start of the sequence as zero (the same
/// effect as `nn.Conv1d`'s `padding=kernel_size-1` followed by dropping
/// the trailing, non-causal output positions).
fn causal_conv_step(
    weight: &[f32],
    history: &mut Vec<f32>,
    current: &[f32],
    kernel_size: usize,
    dim: usize,
) -> Vec<f32> {
    let hist_len = history.len() / dim;
    let missing = (kernel_size - 1).saturating_sub(hist_len);

    let mut y = vec![0f32; dim];
    for j in 0..kernel_size {
        if j < missing {
            continue; // implicit zero (before the start of the sequence)
        }
        let src: &[f32] = if j == kernel_size - 1 {
            current
        } else {
            let hist_idx = j - missing;
            &history[hist_idx * dim..(hist_idx + 1) * dim]
        };
        for d in 0..dim {
            y[d] += weight[d * kernel_size + j] * src[d];
        }
    }
    for v in y.iter_mut() {
        *v = silu(*v);
    }

    history.extend_from_slice(current);
    let max_hist_len = (kernel_size - 1) * dim;
    if history.len() > max_hist_len {
        let excess = history.len() - max_hist_len;
        history.drain(0..excess);
    }
    y
}

fn l2_normalize(v: &mut [f32], eps: f32) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let scale = 1.0 / (norm_sq + eps).sqrt();
    for x in v.iter_mut() {
        *x *= scale;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One decode step.
pub fn kda_forward_token(
    weights: &KdaAttnWeights,
    cfg: &KdaConfig,
    hidden: &[f32],
    rms_norm_eps: f32,
    state: &mut KdaState,
) -> Vec<f32> {
    let projection_size = cfg.num_heads * cfg.head_dim;
    let k_size = cfg.short_conv_kernel_size;

    let q_lin = weights.q_proj.apply(hidden);
    let k_lin = weights.k_proj.apply(hidden);
    let v_lin = weights.v_proj.apply(hidden);

    let q = causal_conv_step(
        &weights.q_conv_weight,
        &mut state.conv_hist_q,
        &q_lin,
        k_size,
        projection_size,
    );
    let k = causal_conv_step(
        &weights.k_conv_weight,
        &mut state.conv_hist_k,
        &k_lin,
        k_size,
        projection_size,
    );
    let v = causal_conv_step(
        &weights.v_conv_weight,
        &mut state.conv_hist_v,
        &v_lin,
        k_size,
        projection_size,
    );

    let f_a = weights.f_a_proj.apply(hidden); // [head_dim]
    let g_raw_full = weights.f_b_proj.apply(&f_a); // [projection_size]
    let beta_raw = weights.b_proj.apply(hidden); // [n_heads]

    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let mut attn_out = vec![0f32; projection_size];

    // `h` indexes several independent slices (q/k/v, weights.a_log,
    // beta_raw, state.recurrent, attn_out) at once, which doesn't map
    // cleanly onto a single `.iter().enumerate()`.
    #[allow(clippy::needless_range_loop)]
    for h in 0..cfg.num_heads {
        let base = h * cfg.head_dim;
        let mut q_h = q[base..base + cfg.head_dim].to_vec();
        let mut k_h = k[base..base + cfg.head_dim].to_vec();
        let v_h = &v[base..base + cfg.head_dim];

        l2_normalize(&mut q_h, 1e-6);
        l2_normalize(&mut k_h, 1e-6);
        for x in q_h.iter_mut() {
            *x *= scale;
        }

        let a_log_h_exp = weights.a_log[h].exp();
        let beta_h = sigmoid(beta_raw[h]);

        let s_base = h * cfg.head_dim * cfg.head_dim;
        let s = &mut state.recurrent[s_base..s_base + cfg.head_dim * cfg.head_dim];

        let mut retrieval = vec![0f32; cfg.head_dim];
        for k_idx in 0..cfg.head_dim {
            let raw_gate = g_raw_full[base + k_idx] + weights.dt_bias[base + k_idx];
            let gate = cfg.gate_lower_bound * sigmoid(a_log_h_exp * raw_gate);
            let decay = gate.exp();
            for v_idx in 0..cfg.head_dim {
                let cell = &mut s[k_idx * cfg.head_dim + v_idx];
                *cell *= decay;
                retrieval[v_idx] += k_h[k_idx] * *cell;
            }
        }

        let mut v_scaled = vec![0f32; cfg.head_dim];
        for v_idx in 0..cfg.head_dim {
            v_scaled[v_idx] = (v_h[v_idx] - retrieval[v_idx]) * beta_h;
        }

        for k_idx in 0..cfg.head_dim {
            for v_idx in 0..cfg.head_dim {
                s[k_idx * cfg.head_dim + v_idx] += k_h[k_idx] * v_scaled[v_idx];
            }
        }

        for v_idx in 0..cfg.head_dim {
            let mut acc = 0f32;
            for k_idx in 0..cfg.head_dim {
                acc += q_h[k_idx] * s[k_idx * cfg.head_dim + v_idx];
            }
            attn_out[base + v_idx] = acc;
        }
    }

    let g_out_raw = weights.g_proj.apply(hidden); // [projection_size]
    let mut gated = vec![0f32; projection_size];
    for h in 0..cfg.num_heads {
        let base = h * cfg.head_dim;
        let o_h = &attn_out[base..base + cfg.head_dim];
        let o_normed = rms_norm(o_h, &weights.o_norm_weight, rms_norm_eps);
        for d in 0..cfg.head_dim {
            gated[base + d] = o_normed[d] * sigmoid(g_out_raw[base + d]);
        }
    }

    weights.o_proj.apply(&gated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::tensor::Tensor;

    const HIDDEN_SIZE: usize = 8;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 3;
    const PROJECTION_SIZE: usize = NUM_HEADS * HEAD_DIM;
    const CONV_SIZE: usize = 4;
    const GATE_LOWER_BOUND: f32 = -5.0;
    const EPS: f32 = 1e-5;

    fn wm(data: &[f32], rows: usize, cols: usize) -> WeightMatrix {
        assert_eq!(data.len(), rows * cols);
        WeightMatrix::F32(Tensor::new(data.to_vec(), vec![rows, cols]))
    }

    fn cfg() -> KdaConfig {
        KdaConfig {
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
            short_conv_kernel_size: CONV_SIZE,
            gate_lower_bound: GATE_LOWER_BOUND,
            use_full_rank_gate: true,
        }
    }

    // Generated by an independent Python reference -- do not hand-edit.
    const KDA_Q_PROJ: [f32; 48] = [
        -0.237937, 0.0721714, -0.568898, 0.418732, 0.191488, -0.0876142, -0.0935848, 0.0911506,
        -0.0802981, -0.0677727, 0.21602, 0.154412, -0.0192384, -0.025643, 0.0482749, -0.184206,
        -0.121125, 0.164478, -0.0391448, -0.412328, -0.143184, 0.196986, -0.0696848, -0.0446198,
        0.192551, 0.547383, -0.213957, 0.404462, -0.369004, 0.0524933, -0.350859, 0.405437,
        0.250177, 0.341315, -0.26566, 0.205367, -0.155704, -0.137216, 0.151961, 0.263015,
        0.0613261, -0.188396, -0.247745, 0.433295, 0.178184, 0.215918, 0.655046, -0.244759,
    ];
    const KDA_K_PROJ: [f32; 48] = [
        0.767837,
        0.945273,
        0.485524,
        0.248132,
        -0.199147,
        0.298346,
        -0.132808,
        -0.00649524,
        -0.08713,
        0.085149,
        0.386423,
        -0.166675,
        -0.295621,
        -0.300887,
        -0.290483,
        -0.429331,
        -0.27388,
        0.38798,
        -0.177994,
        0.0771323,
        -0.365068,
        0.0508952,
        -0.522234,
        -0.209627,
        0.676362,
        -0.174891,
        0.335993,
        0.136516,
        -0.0458957,
        -0.195632,
        0.386073,
        -0.053215,
        0.458227,
        -0.215724,
        0.0172017,
        0.13965,
        0.111948,
        -0.37014,
        -0.199219,
        -0.0587941,
        -0.25611,
        0.203198,
        0.176406,
        -0.587125,
        -0.541576,
        -0.384469,
        0.0351779,
        0.609952,
    ];
    const KDA_V_PROJ: [f32; 48] = [
        -0.114707,
        0.0751952,
        -0.318934,
        -0.314051,
        -0.587168,
        -0.00850327,
        0.284165,
        -0.107014,
        0.418936,
        0.0593565,
        -0.0109237,
        0.155176,
        0.146213,
        0.344342,
        -0.240586,
        -0.686415,
        0.034424,
        -0.183503,
        -0.00815316,
        0.49929,
        -0.330853,
        0.229439,
        0.28376,
        0.138221,
        0.335552,
        -0.137509,
        -0.204453,
        0.311695,
        0.21609,
        0.417113,
        0.0636869,
        0.487069,
        -0.0846241,
        -0.318099,
        -0.611127,
        -0.33042,
        0.245551,
        -0.439479,
        -0.135709,
        0.631975,
        0.254039,
        0.537295,
        -0.297469,
        -0.737993,
        0.454914,
        -0.425083,
        0.0272771,
        0.065128,
    ];
    const KDA_Q_CONV_W: [f32; 24] = [
        -0.379559, 0.621053, 0.619735, 0.220578, -0.0117963, 0.0765747, -0.438363, -0.0956255,
        -0.0419733, -0.340893, 0.339686, -0.570593, -0.181368, -0.901151, 0.198727, 0.297042,
        0.248517, 1.16723, 0.375916, -0.376796, 1.06542, -0.368127, 0.315495, 0.297251,
    ];
    const KDA_K_CONV_W: [f32; 24] = [
        -0.351889, 0.55814, 0.141837, 0.228382, -0.270293, 0.521053, -0.344375, -0.711664,
        0.570051, -0.170692, 0.017348, 0.769103, 0.454597, 0.253084, 0.427966, -0.291003, 0.301235,
        0.443975, 0.302169, -0.0816539, -1.49897, 0.222921, 0.0459506, -0.147821,
    ];
    const KDA_V_CONV_W: [f32; 24] = [
        0.311239, 0.106281, 0.0815823, -0.593436, -0.583729, 0.148214, 0.0304228, 0.0554964,
        0.0665052, 0.164583, 0.0144263, 0.0631083, 0.102446, 0.371764, -0.205876, 0.314558,
        0.254553, 0.0421608, -0.190045, 0.147656, 0.345589, -0.429991, 0.0248583, 0.0258976,
    ];
    const KDA_A_LOG: [f32; 2] = [1.39784, 2.63702];
    const KDA_F_A_PROJ: [f32; 24] = [
        -0.465344, -0.330938, -0.461243, -0.184972, -0.167996, -0.162386, 0.111553, -0.05627,
        -0.218674, -0.216565, -0.339453, -0.0588504, -0.0642164, 0.486887, 0.460246, 0.443371,
        0.623773, -0.36512, 0.201895, 0.228316, 0.142865, 0.638001, 0.761363, 0.29381,
    ];
    const KDA_F_B_PROJ: [f32; 18] = [
        0.195944, -0.336719, -0.336108, 0.210067, -0.0226898, -0.399168, -0.0372073, 0.108002,
        0.280778, -0.0323038, -0.0807652, 0.00186235, 0.051895, -0.34894, 0.249993, 0.657098,
        0.24426, -0.434459,
    ];
    const KDA_DT_BIAS: [f32; 6] = [
        0.0189017, -0.353126, -0.365842, 0.13185, 0.0155298, -0.137422,
    ];
    const KDA_B_PROJ: [f32; 16] = [
        -0.626234, -0.38068, 0.0486588, 0.00956812, 0.20691, 0.457018, -0.101777, 0.14069,
        -0.0413395, -0.148962, -0.0734372, 0.288226, -0.308526, 0.342312, 0.443186, 0.0107199,
    ];
    const KDA_G_PROJ: [f32; 48] = [
        0.0764755, -0.42354, 0.70816, 0.293666, 0.0516828, 0.0172313, 0.135292, -0.195371,
        0.0849785, -0.277073, 0.149196, -0.203464, 0.268656, -0.0361107, 0.0806238, 0.888267,
        0.24127, 0.0803401, 0.0133166, -0.311749, 0.325266, 0.480702, 0.0193065, 0.503025,
        -0.0336833, -0.531916, -0.195972, -0.317098, -0.082362, 0.188094, -0.278692, -0.0240735,
        0.0598056, -0.219908, -0.272796, 0.0261548, 0.261711, -0.180968, -0.225418, -0.042376,
        0.0869846, -0.162905, -0.100244, -0.0693806, -0.325737, -0.0532611, -0.168014, 0.475017,
    ];
    const KDA_O_NORM_W: [f32; 3] = [0.948268, 1.04571, 1.03795];
    const KDA_O_PROJ: [f32; 48] = [
        -0.33541, -0.374685, -0.133646, 0.198772, -0.171972, -0.196846, -0.870066, -0.119052,
        0.0937437, 0.133406, 0.128758, -0.0983533, -0.0681474, 0.249683, -0.0407133, -0.690913,
        0.306003, -0.323304, 0.412587, 0.508244, 0.228732, 0.271235, 0.55734, -0.330637, 0.0761556,
        0.0717198, -0.290133, 0.603544, -0.174208, -0.219374, 0.0803143, 0.309335, -0.180314,
        -0.17987, 0.151233, -0.427684, -0.231776, -0.251471, -0.1303, -0.0426799, 0.553263,
        0.272208, 0.00671946, -0.0674106, 0.262305, 0.12852, 0.267363, 0.414293,
    ];

    const KDA_HIDDEN_0: [f32; 8] = [
        -0.504092, -1.12643, -0.0109012, 0.0397307, 0.143389, -0.299412, -0.11631, 0.279368,
    ];
    const KDA_HIDDEN_1: [f32; 8] = [
        0.306279, 0.665489, -1.08314, -0.824329, 0.730971, 0.304099, 0.396717, -0.142417,
    ];
    const KDA_HIDDEN_2: [f32; 8] = [
        -0.237078, 0.0999019, -0.469894, -0.765081, -0.555285, 0.0582631, 0.318596, 0.82563,
    ];
    const KDA_HIDDEN_3: [f32; 8] = [
        0.185911, 0.236259, 0.84407, -0.388328, -0.479646, -0.268966, -0.048112, 1.01162,
    ];
    const KDA_HIDDEN_4: [f32; 8] = [
        -0.17442, 0.653557, -0.23633, 0.245708, 0.340658, 0.266476, 0.375523, -0.34452,
    ];

    const KDA_GOLDEN_OUT_0: [f32; 8] = [
        0.334593, 0.599535, -0.141337, -0.849879, 0.0520445, -0.161366, -0.356024, -0.259803,
    ];
    const KDA_GOLDEN_OUT_1: [f32; 8] = [
        -0.178719, -0.104539, 0.510691, 0.283403, -0.331164, 0.229022, 0.24393, 0.0242566,
    ];
    const KDA_GOLDEN_OUT_2: [f32; 8] = [
        -0.0640815, -0.296412, 0.488677, 0.684492, 0.289487, 0.664769, -0.493374, -0.528477,
    ];
    const KDA_GOLDEN_OUT_3: [f32; 8] = [
        0.286145, 0.513164, -0.532459, -1.06872, -0.442514, -0.844267, 0.659196, 0.568707,
    ];
    const KDA_GOLDEN_OUT_4: [f32; 8] = [
        -0.0696118, -0.371894, -0.388843, -0.283232, -0.109181, -0.399766, 0.265563, 0.319472,
    ];

    fn make_weights() -> KdaAttnWeights {
        KdaAttnWeights {
            q_proj: wm(&KDA_Q_PROJ, PROJECTION_SIZE, HIDDEN_SIZE),
            k_proj: wm(&KDA_K_PROJ, PROJECTION_SIZE, HIDDEN_SIZE),
            v_proj: wm(&KDA_V_PROJ, PROJECTION_SIZE, HIDDEN_SIZE),
            q_conv_weight: KDA_Q_CONV_W.to_vec(),
            k_conv_weight: KDA_K_CONV_W.to_vec(),
            v_conv_weight: KDA_V_CONV_W.to_vec(),
            a_log: KDA_A_LOG.to_vec(),
            f_a_proj: wm(&KDA_F_A_PROJ, HEAD_DIM, HIDDEN_SIZE),
            f_b_proj: wm(&KDA_F_B_PROJ, PROJECTION_SIZE, HEAD_DIM),
            dt_bias: KDA_DT_BIAS.to_vec(),
            b_proj: wm(&KDA_B_PROJ, NUM_HEADS, HIDDEN_SIZE),
            g_proj: wm(&KDA_G_PROJ, PROJECTION_SIZE, HIDDEN_SIZE),
            o_norm_weight: KDA_O_NORM_W.to_vec(),
            o_proj: wm(&KDA_O_PROJ, HIDDEN_SIZE, PROJECTION_SIZE),
        }
    }

    #[test]
    fn matches_independent_python_reference_across_five_decode_steps() {
        let weights = make_weights();
        let cfg = cfg();
        let mut state = KdaState::new(&cfg);

        let hiddens = [
            &KDA_HIDDEN_0[..],
            &KDA_HIDDEN_1[..],
            &KDA_HIDDEN_2[..],
            &KDA_HIDDEN_3[..],
            &KDA_HIDDEN_4[..],
        ];
        let goldens = [
            &KDA_GOLDEN_OUT_0[..],
            &KDA_GOLDEN_OUT_1[..],
            &KDA_GOLDEN_OUT_2[..],
            &KDA_GOLDEN_OUT_3[..],
            &KDA_GOLDEN_OUT_4[..],
        ];

        for (pos, (hidden, golden)) in hiddens.iter().zip(goldens.iter()).enumerate() {
            let out = kda_forward_token(&weights, &cfg, hidden, EPS, &mut state);
            assert_eq!(out.len(), golden.len());
            for (i, (a, b)) in out.iter().zip(golden.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "position {pos} element {i}: rust={a} python={b}"
                );
            }
        }
    }

    #[test]
    fn causal_conv_step_zero_pads_before_the_start_of_the_sequence() {
        // With no history yet and kernel_size=3, only the tap aligned
        // with the current position (the last one) should contribute.
        let weight = vec![100.0, 100.0, 2.0]; // dim=1, kernel_size=3
        let mut history = Vec::new();
        let y = causal_conv_step(&weight, &mut history, &[3.0], 3, 1);
        // silu(2.0 * 3.0) = silu(6.0)
        let expected = silu(6.0);
        assert!((y[0] - expected).abs() < 1e-5);
        assert_eq!(history, vec![3.0]);
    }

    #[test]
    fn causal_conv_step_history_caps_at_kernel_size_minus_one() {
        let weight = vec![1.0, 1.0, 1.0];
        let mut history = Vec::new();
        for x in [1.0, 2.0, 3.0, 4.0] {
            causal_conv_step(&weight, &mut history, &[x], 3, 1);
        }
        // Only the last (kernel_size - 1) = 2 raw values are retained.
        assert_eq!(history, vec![3.0, 4.0]);
    }
}
