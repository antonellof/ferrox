//! A recurrent layer's whole BRANCH in one command buffer: the two
//! gates, the causal convolution with its SiLU, the per-head l2 norms,
//! the delta rule, the gated output norm, the folded rotation and the
//! output projection.
//!
//! # Why this shape and not another
//!
//! `docs/plans/gdn-resident-state.md` prices a Bonsai decode token: 192
//! command buffers, 71.6 ms of GPU, 29.9 ms of latency beyond it, and
//! about 35 ms of host of which two thirds is the recurrence. The GPU
//! work is already faster than the reference's whole token, the 0.15 ms
//! a submission costs is the OS wake-up, and every way around that has
//! been measured and lost. So the only thing left to remove is host
//! time inside a layer, and this is where nearly all of it is.
//!
//! The projections that FEED this stay where they are. `attn_qkv` and
//! `attn_gate` are one launch already (`WeightMatrix::apply_pair`), and
//! the two gate projections are tiny unquantized matrices whose own
//! measurement said sending them with the pair was neutral. So a
//! recurrent layer keeps the two submissions it has and loses the host
//! step between them, rather than gaining a third.
//!
//! # Why it can win where three earlier attempts lost
//!
//! Those three moved the state once per row through a kernel that read
//! it UNCOALESCED, and measured the kernel against six CPU cores while
//! still paying a submission for it. Two of those three facts have
//! changed: `crate::gdn`'s step reads the state coalesced now, and the
//! submission this rides in already existed. The third has not, and is
//! why the host body stays as the fallback and the caller measures.

use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

use crate::gdn::{buffer_no_copy, encode_delta_step_at, encode_gated_norm, DeltaShape};
use crate::gdn_head::{encode_conv_silu, encode_gates, encode_l2_norm_heads, HeadShape};
use crate::gpu::{encode_matvec, resident_weight_buffer, shared_metal, MatvecLaunch, MetalError};
use crate::hadamard::FoldPlan;

/// Everything about one recurrent layer that does not change between
/// tokens, so the launch takes ONE argument and cannot be handed the
/// convolution of one layer with the output projection of another.
pub struct BranchWeights<'a> {
    pub shape: DeltaShape,
    pub head: HeadShape,
    /// `blk.N.ssm_conv1d.weight`, `[conv_dim][d_conv]`.
    pub conv1d: &'a [f32],
    /// `blk.N.ssm_dt.bias` and `blk.N.ssm_a`, `[n_v_heads]` each.
    pub dt_bias: &'a [f32],
    pub a: &'a [f32],
    /// `blk.N.ssm_norm.weight`, `[head_dim]`.
    pub ssm_norm: &'a [f32],
    /// The model's RMS epsilon, which the l2 norms clamp by and the
    /// gated norm uses.
    pub eps: f32,
    /// `blk.N.ssm_out.weight`, and the rotation its input needs.
    pub out_proj: &'a MatvecLaunch<'a>,
    pub fold_y: Option<&'a FoldPlan<'a>>,
}

impl BranchWeights<'_> {
    /// Whether the shapes agree with each other and with what the
    /// kernels serve. A caller checks this once per layer rather than
    /// discovering it per token.
    pub fn is_supported(&self) -> bool {
        let h = self.head;
        h.n_k_heads > 0
            && h.n_v_heads > 0
            && h.head_dim.is_power_of_two()
            && h.head_dim <= crate::gdn::MAX_HEAD_DIM
            && h.d_conv >= 2
            && h.d_conv <= 8
            && h.n_v_heads.is_multiple_of(h.n_k_heads)
            && self.shape.n_k_heads == h.n_k_heads
            && self.shape.n_v_heads == h.n_v_heads
            && self.shape.head_dim == h.head_dim
            && self.conv1d.len() == h.conv_dim() * h.d_conv
            && self.dt_bias.len() == h.n_v_heads
            && self.a.len() == h.n_v_heads
            && self.ssm_norm.len() == h.head_dim
            && self.out_proj.row_bytes / self.out_proj.block_bytes * self.out_proj.block_elems
                == h.value_dim()
    }
}

/// One token through the whole branch, in one submission.
///
/// BOTH states are page-aligned host allocations wrapped in place and
/// never copied: the delta state (`ssm_ptr`) and the convolution window
/// (`conv_ptr`). `qkv` is `attn_qkv`'s output for this token and `z` is
/// `attn_gate`'s.
///
/// # Safety
///
/// Each pointer must address that many readable-writable, page-aligned
/// bytes which outlive the call and which the caller holds exclusively
/// across it. The GPU writes those bytes rather than a copy of them,
/// and this call waits for it.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gdn_branch(
    w: &BranchWeights<'_>,
    ssm_ptr: *mut f32,
    ssm_bytes: usize,
    conv_ptr: *mut f32,
    conv_bytes: usize,
    conv_len: usize,
    qkv: &[f32],
    z: &[f32],
    beta_in: &[f32],
    alpha_in: &[f32],
) -> Result<Vec<f32>, MetalError> {
    let h = w.head;
    let (key_dim, value_dim, conv_dim) = (h.key_dim(), h.value_dim(), h.conv_dim());
    if !w.is_supported()
        || conv_len != h.conv_state_len()
        || qkv.len() != conv_dim
        || z.len() != value_dim
        || beta_in.len() != h.n_v_heads
        || alpha_in.len() != h.n_v_heads
    {
        return Err(MetalError::CommandFailed);
    }

    let shared = shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;

    // SAFETY: the caller's contract, forwarded.
    let ssm_buf = unsafe { buffer_no_copy(device, ssm_ptr, ssm_bytes) }
        .ok_or(MetalError::BufferAllocFailed)?;
    // The layer's CONSTANTS are resident, cached by address: `conv1d`
    // alone is `d_conv * conv_dim` floats, 164 KB on Bonsai, and
    // uploading it per layer per token is 7.9 MB a token of pure copy
    // for bytes that never change. The first version of this launch
    // did exactly that for five operands and measured SLOWER than the
    // host body it replaced, which is what the ledger's GPU time going
    // UP by 3.1 ms was.
    let taps_buf = crate::gpu::resident_f32_buffer(device, w.conv1d)?;
    let dt_buf = crate::gpu::resident_f32_buffer(device, w.dt_bias)?;
    let a_buf = crate::gpu::resident_f32_buffer(device, w.a)?;
    let norm_buf = crate::gpu::resident_f32_buffer(device, w.ssm_norm)?;
    // SAFETY: the caller's contract, forwarded.
    let conv_buf = unsafe { buffer_no_copy(device, conv_ptr, conv_bytes) }
        .ok_or(MetalError::BufferAllocFailed)?;
    // Scratch from the pool rather than six fresh allocations: `[q | k
    // | v]` in one buffer, which is the layout the convolution writes
    // and the layout the delta step reads by offset, so nothing is
    // copied between those two either. Returned when `scratch` drops,
    // which is after the wait below.
    let scratch = crate::scratch_pool::Scratch::take(
        device,
        &[
            conv_dim,
            h.n_v_heads,
            h.n_v_heads,
            value_dim,
            value_dim,
            w.out_proj.rows,
            // The per-token inputs, which really do have to travel: 66
            // KB on Bonsai, against the 123 KB of convolution window
            // that does not. They travel as a memcpy into a pooled
            // buffer rather than as four allocations.
            conv_dim,
            value_dim,
            h.n_v_heads,
            h.n_v_heads,
        ],
    )
    .ok_or(MetalError::BufferAllocFailed)?;
    let conv_out = scratch.buf(0);
    let (beta_buf, g_buf) = (scratch.buf(1), scratch.buf(2));
    let o_buf = scratch.buf(3);
    let y_buf = scratch.buf(4);
    let out_buf = scratch.buf(5);
    let fill = |i: usize, xs: &[f32]| scratch.write(i, xs).ok_or(MetalError::CommandFailed);
    let qkv_buf = fill(6, qkv)?;
    let z_buf = fill(7, z)?;
    let beta_in_buf = fill(8, beta_in)?;
    let alpha_in_buf = fill(9, alpha_in)?;
    let weights_buf = resident_weight_buffer(device, w.out_proj.weights)?;
    let signs_buf = match w.fold_y.and_then(|p| p.signs) {
        None => None,
        Some(signs) => {
            // SAFETY: a `&[f32]` viewed as its own bytes, for a read.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    signs.as_ptr() as *const u8,
                    std::mem::size_of_val(signs),
                )
            };
            Some(resident_weight_buffer(device, bytes)?)
        }
    };

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_gates(
        &encoder,
        device,
        h.n_v_heads,
        beta_in_buf,
        alpha_in_buf,
        &dt_buf.buffer,
        &a_buf.buffer,
        beta_buf,
        g_buf,
    )?;
    encode_conv_silu(
        &encoder,
        device,
        h,
        &conv_buf,
        &taps_buf.buffer,
        qkv_buf,
        conv_out,
    )?;
    // Q and K only: V is the convolution's output as it stands.
    encode_l2_norm_heads(
        &encoder,
        device,
        conv_out,
        0,
        h.head_dim,
        h.n_k_heads,
        w.eps,
    )?;
    encode_l2_norm_heads(
        &encoder,
        device,
        conv_out,
        key_dim,
        h.head_dim,
        h.n_k_heads,
        w.eps,
    )?;
    encode_delta_step_at(
        &encoder,
        device,
        w.shape,
        &ssm_buf,
        (conv_out, 0),
        (conv_out, key_dim),
        (conv_out, 2 * key_dim),
        (g_buf, 0),
        (beta_buf, 0),
        (o_buf, 0),
    )?;
    encode_gated_norm(
        &encoder,
        device,
        h.n_v_heads,
        h.head_dim,
        w.eps,
        o_buf,
        z_buf,
        &norm_buf.buffer,
        y_buf,
    )?;
    if let Some(plan) = w.fold_y {
        plan.check(value_dim)?;
        crate::hadamard::encode_fold(
            &encoder,
            device,
            y_buf,
            value_dim,
            plan,
            signs_buf.as_ref().map(|b| &*b.buffer),
        )?;
    }
    encode_matvec(&encoder, device, w.out_proj, &weights_buf, y_buf, out_buf)?;
    encoder.endEncoding();
    let clock = crate::timing::SubmitClock::start();
    crate::timing::commit_wait_note(&cmd_buf, "gdn-branch", 32, clock);

    // SAFETY: shared storage of exactly `out_proj.rows` floats, written
    // by kernels this call has waited for. The convolution window needs
    // no read-back: the kernel wrote the caller's own bytes.
    unsafe {
        Ok(
            std::slice::from_raw_parts(out_buf.contents().as_ptr() as *const f32, w.out_proj.rows)
                .to_vec(),
        )
    }
}
