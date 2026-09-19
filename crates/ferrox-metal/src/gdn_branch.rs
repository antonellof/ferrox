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
    /// The layer's HEAD, when the caller owns it too: the input norm
    /// and the four projections that feed the branch.
    ///
    /// `Some` removes the second submission a recurrent layer costs,
    /// and with `ffn` set as well the layer is ONE.
    pub head_in: Option<LayerHeadIn<'a>>,
    /// The rest of the layer, when the caller owns it too.
    ///
    /// `Some` turns three submissions a layer into ONE: the branch's
    /// output is added to the residual, normed and run through the FFN
    /// without the hidden state coming back to the host in between,
    /// which is the only thing left between this engine and the
    /// reference's decode rate
    /// (`docs/plans/gdn-resident-state.md`). `None` stops at
    /// `ssm_out` and the caller finishes the layer itself.
    pub ffn: Option<LayerFfn<'a>>,
}

/// The head of a layer: `attn_norm(x)`, then the four projections the
/// branch reads.
///
/// `beta` and `alpha` are the SPLIT spelling only; the fused one's
/// per-group interleave is host arithmetic with no kernel here, and a
/// caller carrying it passes `None` for the whole head.
pub struct LayerHeadIn<'a> {
    pub norm: &'a [f32],
    pub norm_eps: f32,
    pub qkv: &'a MatvecLaunch<'a>,
    pub z: &'a MatvecLaunch<'a>,
    pub beta: &'a MatvecLaunch<'a>,
    pub alpha: &'a MatvecLaunch<'a>,
    /// The rotation the projections' shared input needs. One plan, not
    /// four: they all read `attn_norm(x)`, and a checkpoint whose four
    /// disagree about their input basis is not this shape.
    pub fold_x: Option<&'a FoldPlan<'a>>,
}

/// The dense half of a layer: `x + ffn(norm(x + branch))`.
///
/// One shape only, and a caller whose layer is not that shape passes
/// `None` rather than having this approximate it. What that excludes is
/// named at the call site, field by field.
pub struct LayerFfn<'a> {
    /// `blk.N.ffn_norm.weight` and the model's RMS epsilon.
    pub norm: &'a [f32],
    pub norm_eps: f32,
    pub gate: &'a MatvecLaunch<'a>,
    pub up: &'a MatvecLaunch<'a>,
    pub down: &'a MatvecLaunch<'a>,
    /// The rotations the FFN's input and its activation need.
    pub fold_x: Option<&'a FoldPlan<'a>>,
    pub fold_act: Option<&'a FoldPlan<'a>>,
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
            && match &self.head_in {
                None => true,
                Some(hd) => {
                    let hidden = self.out_proj.rows;
                    hd.norm.len() == hidden
                        && hd.qkv.rows == h.conv_dim()
                        && hd.z.rows == h.value_dim()
                        && hd.beta.rows == h.n_v_heads
                        && hd.alpha.rows == h.n_v_heads
                        && [hd.qkv, hd.z, hd.beta, hd.alpha]
                            .iter()
                            .all(|m| m.row_bytes / m.block_bytes * m.block_elems == hidden)
                }
            }
            && match &self.ffn {
                None => true,
                Some(f) => {
                    let hidden = self.out_proj.rows;
                    f.norm.len() == hidden
                        && f.gate.rows == f.up.rows
                        && f.down.rows == hidden
                        && f.gate.row_bytes / f.gate.block_bytes * f.gate.block_elems == hidden
                        && f.up.row_bytes / f.up.block_bytes * f.up.block_elems == hidden
                        && f.down.row_bytes / f.down.block_bytes * f.down.block_elems == f.gate.rows
                }
            }
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
    qkv: Option<&[f32]>,
    z: Option<&[f32]>,
    beta_in: Option<&[f32]>,
    alpha_in: Option<&[f32]>,
    residual: Option<&[f32]>,
) -> Result<Vec<f32>, MetalError> {
    let h = w.head;
    let (key_dim, value_dim, conv_dim) = (h.key_dim(), h.value_dim(), h.conv_dim());
    if !w.is_supported()
        || conv_len != h.conv_state_len()
        // The four projections arrive from the host exactly when the
        // head is NOT on the device, so a caller cannot half-fuse.
        || w.head_in.is_some() == qkv.is_some()
        || qkv.is_some_and(|x| x.len() != conv_dim)
        || z.is_some_and(|x| x.len() != value_dim)
        || beta_in.is_some_and(|x| x.len() != h.n_v_heads)
        || alpha_in.is_some_and(|x| x.len() != h.n_v_heads)
        || qkv.is_some() != z.is_some()
        || qkv.is_some() != beta_in.is_some()
        || qkv.is_some() != alpha_in.is_some()
        // The residual is needed by the FFN, and by the head as the
        // vector it norms.
        || (w.ffn.is_some() || w.head_in.is_some()) != residual.is_some()
        // The residual travels exactly when the FFN does, so a caller
        // cannot ask for the fused layer and forget what to add the
        // branch to.
        || residual.is_some_and(|r| r.len() != w.out_proj.rows)
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
    let hidden = w.out_proj.rows;
    // The head's own scratch: one buffer for `attn_norm(x)`, which the
    // two gate projections read BEFORE it is rotated in place and the
    // other two read after.
    let head_scratch = match &w.head_in {
        None => None,
        // Two: the residual as it arrives, and the norm of it. Not
        // one in place -- an RMS norm reduces the whole vector before
        // it writes any of it, and a kernel reading a buffer it is
        // writing is a hazard nothing here would report.
        Some(_) => Some(
            crate::scratch_pool::Scratch::take(device, &[hidden, hidden])
                .ok_or(MetalError::BufferAllocFailed)?,
        ),
    };
    // The FFN's own scratch, taken only when there is an FFN: the
    // residual (which becomes the layer's output), the normed copy, the
    // two projections, the activation and the FFN output.
    let ffn_scratch = match &w.ffn {
        None => None,
        Some(f) => Some(
            crate::scratch_pool::Scratch::take(
                device,
                &[hidden, hidden, f.gate.rows, f.up.rows, f.gate.rows, hidden],
            )
            .ok_or(MetalError::BufferAllocFailed)?,
        ),
    };
    let conv_out = scratch.buf(0);
    let (beta_buf, g_buf) = (scratch.buf(1), scratch.buf(2));
    let o_buf = scratch.buf(3);
    let y_buf = scratch.buf(4);
    let out_buf = scratch.buf(5);
    // The four projection OUTPUTS: filled from the host when the head
    // is not fused, written by matvecs below when it is.
    let (qkv_buf, z_buf, beta_in_buf, alpha_in_buf) = match (qkv, z, beta_in, alpha_in) {
        (Some(a), Some(b), Some(c), Some(d)) => {
            let fill = |i: usize, xs: &[f32]| scratch.write(i, xs).ok_or(MetalError::CommandFailed);
            (fill(6, a)?, fill(7, b)?, fill(8, c)?, fill(9, d)?)
        }
        _ => (
            scratch.buf(6),
            scratch.buf(7),
            scratch.buf(8),
            scratch.buf(9),
        ),
    };
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
    // The layer's HEAD, when the caller owns it: the norm and the four
    // projections, which is the second of a recurrent layer's three
    // submissions.
    if let (Some(hd), Some(sc), Some(res)) = (&w.head_in, &head_scratch, residual) {
        let x_buf = sc.write(0, res).ok_or(MetalError::CommandFailed)?;
        let normed = sc.buf(1);
        let norm_w = crate::gpu::resident_f32_buffer(device, hd.norm)?;
        crate::norm::encode_rms_norm(
            &encoder,
            device,
            x_buf,
            &norm_w.buffer,
            normed,
            hidden as u32,
            hd.norm_eps,
        )?;
        // BEFORE the rotation: the two gate projections are stored
        // unfolded, so they read `attn_norm(x)` in the primal basis,
        // and the fold below rewrites that buffer in place.
        let beta_w = resident_weight_buffer(device, hd.beta.weights)?;
        let alpha_w = resident_weight_buffer(device, hd.alpha.weights)?;
        encode_matvec(&encoder, device, hd.beta, &beta_w, normed, beta_in_buf)?;
        encode_matvec(&encoder, device, hd.alpha, &alpha_w, normed, alpha_in_buf)?;
        if let Some(plan) = hd.fold_x {
            plan.check(hidden)?;
            let signs = match plan.signs {
                None => None,
                Some(sg) => {
                    // SAFETY: a `&[f32]` viewed as its own bytes, read only.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            sg.as_ptr() as *const u8,
                            std::mem::size_of_val(sg),
                        )
                    };
                    Some(resident_weight_buffer(device, bytes)?)
                }
            };
            crate::hadamard::encode_fold(
                &encoder,
                device,
                normed,
                hidden,
                plan,
                signs.as_ref().map(|b| &*b.buffer),
            )?;
        }
        let qkv_w = resident_weight_buffer(device, hd.qkv.weights)?;
        let z_w = resident_weight_buffer(device, hd.z.weights)?;
        encode_matvec(&encoder, device, hd.qkv, &qkv_w, normed, qkv_buf)?;
        encode_matvec(&encoder, device, hd.z, &z_w, normed, z_buf)?;
    }
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

    // The rest of the layer, when the caller owns it: the whole point
    // of the file, because none of this needs the host and each piece
    // of it used to cost a submission.
    let result_buf = match (&w.ffn, &ffn_scratch, residual) {
        (Some(f), Some(sc), Some(res)) => {
            let h_buf = sc.write(0, res).ok_or(MetalError::CommandFailed)?;
            let (normed, gate_buf) = (sc.buf(1), sc.buf(2));
            let (up_buf, act_buf, ffn_out) = (sc.buf(3), sc.buf(4), sc.buf(5));
            let ffn_signs = |plan: Option<&FoldPlan<'_>>| match plan.and_then(|p| p.signs) {
                None => Ok(None),
                Some(signs) => {
                    // SAFETY: a `&[f32]` viewed as its own bytes, read only.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            signs.as_ptr() as *const u8,
                            std::mem::size_of_val(signs),
                        )
                    };
                    resident_weight_buffer(device, bytes).map(Some)
                }
            };
            let x_signs = ffn_signs(f.fold_x)?;
            let act_signs = ffn_signs(f.fold_act)?;
            let norm_w = crate::gpu::resident_f32_buffer(device, f.norm)?;
            let gate_w = resident_weight_buffer(device, f.gate.weights)?;
            let up_w = resident_weight_buffer(device, f.up.weights)?;
            let down_w = resident_weight_buffer(device, f.down.weights)?;

            crate::elem::encode_vec_add(&encoder, device, h_buf, out_buf, hidden as u32)?;
            crate::norm::encode_rms_norm(
                &encoder,
                device,
                h_buf,
                &norm_w.buffer,
                normed,
                hidden as u32,
                f.norm_eps,
            )?;
            if let Some(plan) = f.fold_x {
                plan.check(hidden)?;
                crate::hadamard::encode_fold(
                    &encoder,
                    device,
                    normed,
                    hidden,
                    plan,
                    x_signs.as_ref().map(|b| &*b.buffer),
                )?;
            }
            encode_matvec(&encoder, device, f.gate, &gate_w, normed, gate_buf)?;
            encode_matvec(&encoder, device, f.up, &up_w, normed, up_buf)?;
            crate::elem::encode_silu_mul(
                &encoder,
                device,
                gate_buf,
                up_buf,
                act_buf,
                f.gate.rows as u32,
            )?;
            if let Some(plan) = f.fold_act {
                plan.check(f.gate.rows)?;
                crate::hadamard::encode_fold(
                    &encoder,
                    device,
                    act_buf,
                    f.gate.rows,
                    plan,
                    act_signs.as_ref().map(|b| &*b.buffer),
                )?;
            }
            encode_matvec(&encoder, device, f.down, &down_w, act_buf, ffn_out)?;
            crate::elem::encode_vec_add(&encoder, device, h_buf, ffn_out, hidden as u32)?;
            h_buf
        }
        _ => out_buf,
    };
    encoder.endEncoding();
    let clock = crate::timing::SubmitClock::start();
    let label = match (w.head_in.is_some(), w.ffn.is_some()) {
        (true, true) => "gdn-layer-full",
        (false, true) => "gdn-layer",
        _ => "gdn-branch",
    };
    crate::timing::commit_wait_note(&cmd_buf, label, 32, clock);

    // SAFETY: shared storage of exactly `out_proj.rows` floats, written
    // by kernels this call has waited for. The convolution window needs
    // no read-back: the kernel wrote the caller's own bytes.
    unsafe {
        Ok(
            std::slice::from_raw_parts(result_buf.contents().as_ptr() as *const f32, hidden)
                .to_vec(),
        )
    }
}
