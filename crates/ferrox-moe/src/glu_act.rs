//! The activation an expert FFN runs between its `up` (and `gate`)
//! projections and its `down` projection, and the one place its
//! arithmetic is spelled.
//!
//! Split out of `lib.rs` when the fourth variant arrived. Three of the
//! four are a unit: SwiGLU, GeGLU, ReGLU. The fourth, xIELU, CARRIES
//! PARAMETERS -- four scalars that llama.cpp reads from the GGUF as
//! per-layer arrays (`apertus.cpp:6-9`) and hands to `ggml_xielu` per
//! layer (`:132-138`). So a `GluAct` is a value a LAYER has, not a
//! model, and `ferrox_models::ModelConfig::layer_ffn_act(il)` is the
//! accessor that answers it; the uniform case is the special case where
//! every layer answers the same thing.
//!
//! Adding a variant here is a non-exhaustive `match` at every consumer
//! in this crate and in `ferrox-models`, which is the point: the last
//! time an activation was added, six Metal launch sites had derived a
//! `gelu: bool` as `!is_swiglu()` and would have run it as GELU.

use ferrox_core::matmul::{geglu, swiglu};

/// Which gated activation an expert's `down(act(gate(x)) * up(x))` FFN
/// uses.
///
/// A named type rather than a `bool` or an implicit default, and a
/// REQUIRED argument of [`run_expert`](crate::run_expert) /
/// [`run_expert_placed`](crate::run_expert_placed), because the
/// alternative already failed once: every routed-expert path in
/// `ferrox-models` hardcoded SwiGLU while only the dense arm consulted
/// `ModelConfig::ffn_activation`, so a GeGLU MoE would have computed the
/// wrong activation with nothing to notice. A caller cannot forget an
/// argument the compiler demands.
///
/// [`Geglu`](GluAct::Geglu) is `gelu(gate) * up` with llama.cpp's tanh
/// GELU approximation (`ferrox_core::matmul::gelu`), which is what
/// `build_moe_ffn` does under `LLM_FFN_GELU` -- the real shape of
/// llama.cpp's `grok` (`src/models/grok.cpp`, `LLM_FFN_GELU` passed to
/// `build_moe_ffn`).
///
/// [`ReluSqr`](GluAct::ReluSqr) is `relu(up)^2`, for an FFN that has
/// NO gate at all: llama.cpp's `LLM_FFN_RELU_SQR` under `LLM_FFN_SEQ`
/// with a null gate (`arcee.cpp:123-128`, also `plm`, `nemotron`,
/// `jais2`, `nemotron-h`) is `down(relu(up(x))^2)`. The loader aliases
/// the expert's `gate` to its `up` matrix so the gated struct serves
/// it; [`GluAct::combine`] reads the `up` operand and ignores the
/// aliased gate, and [`GluAct::ungated`] lets the dense hot paths skip
/// the aliased matmul.
///
/// [`Reglu`](GluAct::Reglu) is `relu(gate) * up` on a REAL gate --
/// llama.cpp's `LLM_FFN_RELU` in `build_moe_ffn` with `gate_exps`
/// present, `ggml_reglu_split` (`llama-graph.cpp:2195-2197`,
/// `smallthinker.cpp:158`). It USED to be the spelling of `ReluSqr` as
/// well, "with the gate aliased to up", and that overload is the
/// defect the SmallThinker fixture found: `ungated()` answered
/// `relu(up)^2` for it, so `run_expert` skipped the gate matmul on a
/// model whose gate is a real tensor and computed `relu(up)^2` where
/// libllama computed `relu(gate) * up` -- silently, at full speed. The
/// two are two variants now; a variant whose meaning depends on what
/// the loader did to the weights is two structures that must agree
/// with nothing enforcing it. See `ferrox_core::matmul::reglu`.
///
/// [`Xielu`](GluAct::Xielu) is the second ungated one, and the first
/// that carries parameters: `down(xielu(up(x)))` with the four scalars
/// of THIS layer (`apertus.cpp:132-138`). The loader aliases gate to up
/// exactly as for `ReluSqr`, and [`GluAct::combine`] reads the `up`
/// operand and ignores the aliased gate -- there is no `f(gate) * up`
/// spelling of it, so every site that used to assume that shape
/// (`gate_fn`) is a `combine` now.
///
/// [`SwigluClamped`](GluAct::SwigluClamped) is SwiGLU with ONE scalar
/// that llama.cpp reads per layer (`step35.cpp:28-29`, and
/// `deepseek4` / `dflash` on their own engines) and applies in its
/// generic `build_ffn` / `build_moe_ffn` (`llama-graph.cpp:1751-1768`,
/// `:2146-2164`) whenever the layer's entry is above `1e-6`:
///
/// ```text
/// up  = clamp(up, -limit, limit)
/// act = min(silu(gate), limit)
/// out = act * up
/// ```
///
/// (the `else` branch there; the `swiglu_split` branch is DeepSeek-4's
/// own engine). Not the gpt-oss clamp, which is a different formula
/// with an `alpha` and a `+ 1`. A layer whose entry is zero runs plain
/// [`Swiglu`](GluAct::Swiglu), and `ferrox_models::act_layers` is what
/// decides that per layer.
///
/// Not `Eq`: two variants hold `f32`s. Every comparison in the tree is
/// `assert_eq!` on values, which wants only `PartialEq`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GluAct {
    /// `silu(gate) * up`.
    Swiglu,
    /// `min(silu(gate), limit) * clamp(up, -limit, limit)`.
    SwigluClamped {
        /// The layer's `swiglu_clamp_exp` / `_shexp` entry, above `1e-6`.
        limit: f32,
    },
    /// `gelu(gate) * up`.
    Geglu,
    /// `relu(gate) * up`, on a real gate.
    Reglu,
    /// `relu(up)^2`; `gate` is an alias of `up` and is not read.
    ReluSqr,
    /// `gelu(up)`; `gate` is an alias of `up` and is not read --
    /// llama.cpp's `LLM_FFN_GELU` under `LLM_FFN_SEQ` with a null gate
    /// (`starcoder2.cpp:125-131`, `codeshell.cpp:120-126`), `ggml_gelu`'s
    /// tanh form, the same function [`Self::Geglu`] applies to its gate.
    GeluUngated,
    /// `xielu(up)` with this layer's parameters; `gate` is an alias of
    /// `up` and is not read.
    Xielu(XieluParams),
}

/// One layer's xIELU parameters, AFTER the fold `ggml_xielu` applies at
/// graph build (ggml.c:2837-2856).
///
/// llama.cpp stores the raw checkpoint scalars in `hparams` and folds
/// them when it builds the op:
///
/// ```text
/// alpha_n' = beta + softplus(alpha_n)      (op param 1)
/// alpha_p' = softplus(alpha_p)             (op param 2)
/// ```
///
/// with `softplus(x) = x > 20 ? x : ln(1 + e^x)`
/// (`ggml_compute_softplus_f32`, ggml-impl.h:107-109). The CPU op
/// (ggml-cpu/unary-ops.cpp:55-62) is then
///
/// ```text
/// x > 0:   alpha_p' * x * x + beta * x
/// x <= 0:  (expm1(min(x, eps)) - x) * alpha_n' + beta * x
/// ```
///
/// [`XieluParams::from_gguf`] takes the raw four and folds them once,
/// so the per-element function never recomputes a `ln(1 + e^x)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XieluParams {
    /// `beta + softplus(alpha_n)`, the negative branch's slope.
    pub alpha_n: f32,
    /// `softplus(alpha_p)`, the positive branch's quadratic coefficient.
    pub alpha_p: f32,
    /// The linear term on both branches.
    pub beta: f32,
    /// The clamp on the negative branch's exponent, `min(x, eps)`.
    pub eps: f32,
}

impl XieluParams {
    /// Folds the four scalars as the GGUF carries them (and as the
    /// checkpoint's `act_fn.{alpha_n,alpha_p,beta,eps}` hold them) the
    /// way `ggml_xielu` does.
    pub fn from_gguf(alpha_n: f32, alpha_p: f32, beta: f32, eps: f32) -> Self {
        Self {
            alpha_n: beta + softplus(alpha_n),
            alpha_p: softplus(alpha_p),
            beta,
            eps,
        }
    }

    /// `op_xielu`, ggml-cpu/unary-ops.cpp:55-62, one element.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        if x > 0.0 {
            self.alpha_p * x * x + self.beta * x
        } else {
            let min_x_eps = x.min(self.eps);
            (min_x_eps.exp_m1() - x) * self.alpha_n + self.beta * x
        }
    }
}

/// `ggml_compute_softplus_f32`, ggml-impl.h:107-109.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

impl GluAct {
    /// The gated combine itself. One place, so a new variant is a
    /// compile error at every site instead of a silent SwiGLU.
    pub fn apply(self, gate: &[f32], up: &[f32]) -> Vec<f32> {
        match self {
            GluAct::Swiglu => swiglu(gate, up),
            GluAct::Geglu => geglu(gate, up),
            GluAct::Reglu => ferrox_core::matmul::reglu(gate, up),
            GluAct::ReluSqr => relu_sqr(up),
            GluAct::GeluUngated => up.iter().map(|&x| ferrox_core::matmul::gelu(x)).collect(),
            GluAct::SwigluClamped { .. } | GluAct::Xielu(_) => gate
                .iter()
                .zip(up)
                .map(|(&g, &u)| self.combine(g, u))
                .collect(),
        }
    }

    /// One element of [`Self::apply`], for callers that fuse the
    /// combine into a loop of their own
    /// (`cpu_moe_topk_parallel_slots`).
    ///
    /// This used to be `gate_fn() -> fn(f32) -> f32`, a function of the
    /// GATE alone that the caller multiplied by `up`. That shape cannot
    /// spell xIELU: with gate aliased to up, `xielu(gate) * up` is
    /// `xielu(up) * up`, which is the wrong function by a factor of the
    /// input. Taking both operands is the only signature all four fit.
    #[inline]
    pub fn combine(self, gate: f32, up: f32) -> f32 {
        match self {
            GluAct::Swiglu => ferrox_core::matmul::silu(gate) * up,
            GluAct::SwigluClamped { limit } => {
                ferrox_core::matmul::silu(gate).min(limit) * up.clamp(-limit, limit)
            }
            GluAct::Geglu => ferrox_core::matmul::gelu(gate) * up,
            GluAct::Reglu => ferrox_core::matmul::relu(gate) * up,
            GluAct::ReluSqr => {
                let r = ferrox_core::matmul::relu(up);
                r * r
            }
            GluAct::GeluUngated => ferrox_core::matmul::gelu(up),
            GluAct::Xielu(p) => p.apply(up),
        }
    }

    /// Whether the fused device kernels, which only implement SwiGLU,
    /// may serve this activation.
    pub fn is_swiglu(self) -> bool {
        matches!(self, GluAct::Swiglu)
    }

    /// Whether the fused dense kernels that take a `gelu: bool` uniform
    /// may serve this activation, and what to pass them.
    ///
    /// `None` is a refusal, and it is the whole reason this exists: six
    /// launch sites in `decoder.rs` used to derive that flag as
    /// `!act.is_swiglu()`, which reads "not SwiGLU, therefore GELU" --
    /// true while the enum had two variants and silently wrong for the
    /// third, which those kernels would have run as GELU. A site that
    /// asks this question cannot get a `bool` for a variant no kernel
    /// implements.
    pub fn fused_kernel_gelu_flag(self) -> Option<bool> {
        match self {
            GluAct::Swiglu => Some(false),
            GluAct::Geglu => Some(true),
            GluAct::Reglu
            | GluAct::ReluSqr
            | GluAct::GeluUngated
            | GluAct::SwigluClamped { .. }
            | GluAct::Xielu(_) => None,
        }
    }

    /// The elementwise function this activation applies to `up` ALONE
    /// when the loader has aliased `gate` to `up`, or `None` for a
    /// genuinely gated activation.
    ///
    /// The dense hot path (`run_expert`) uses it to skip the aliased
    /// gate matmul. Every other path runs `combine` on the aliased
    /// pair, which reads `up` alone for these two and is the same
    /// arithmetic -- `relu_sqr_reads_up_alone_and_reglu_reads_the_gate`
    /// and `xielu_ignores_the_aliased_gate` pin that.
    ///
    /// `Reglu` is `None`: its gate is real, and answering `ReluSqr` for
    /// it is exactly the bug the two variants exist to make
    /// unspellable.
    pub fn ungated(self) -> Option<Ungated> {
        match self {
            GluAct::Swiglu | GluAct::SwigluClamped { .. } | GluAct::Geglu | GluAct::Reglu => None,
            GluAct::ReluSqr => Some(Ungated::ReluSqr),
            GluAct::GeluUngated => Some(Ungated::Gelu),
            GluAct::Xielu(p) => Some(Ungated::Xielu(p)),
        }
    }
}

/// The elementwise function [`GluAct::ungated`] hands back.
///
/// An enum rather than a `fn(&[f32]) -> Vec<f32>` because the xIELU
/// form has parameters and a plain function pointer cannot carry them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ungated {
    /// `relu(x)^2`.
    ReluSqr,
    /// `gelu(x)`, the tanh form.
    Gelu,
    /// `xielu(x)` with the layer's parameters.
    Xielu(XieluParams),
}

impl Ungated {
    /// The function over a whole `up` projection.
    pub fn apply(self, up: &[f32]) -> Vec<f32> {
        match self {
            Ungated::ReluSqr => relu_sqr(up),
            Ungated::Gelu => up.iter().map(|&x| ferrox_core::matmul::gelu(x)).collect(),
            Ungated::Xielu(p) => up.iter().map(|&x| p.apply(x)).collect(),
        }
    }
}

/// `relu(x)^2`, elementwise: [`GluAct::ReluSqr`] over a whole
/// projection.
pub fn relu_sqr(up: &[f32]) -> Vec<f32> {
    up.iter()
        .map(|&x| {
            let r = ferrox_core::matmul::relu(x);
            r * r
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fold and the two branches, against ggml's arithmetic written
    /// out by hand for one point on each side of zero and one inside
    /// the `eps` clamp.
    #[test]
    fn xielu_matches_ggml_s_op_on_both_branches_and_inside_eps() {
        let p = XieluParams::from_gguf(0.8, 0.8, 0.5, -0.3);
        let sp = |x: f32| (1.0 + x.exp()).ln();
        assert!((p.alpha_n - (0.5 + sp(0.8))).abs() < 1e-6);
        assert!((p.alpha_p - sp(0.8)).abs() < 1e-6);
        // Positive branch: alpha_p' x^2 + beta x.
        let x = 1.5f32;
        assert!((p.apply(x) - (p.alpha_p * x * x + 0.5 * x)).abs() < 1e-6);
        // Negative branch above eps: min(x, eps) = eps.
        let x = -0.1f32;
        let want = ((-0.3f32).exp_m1() - x) * p.alpha_n + 0.5 * x;
        assert!((p.apply(x) - want).abs() < 1e-6);
        // Negative branch below eps: min(x, eps) = x.
        let x = -2.0f32;
        let want = (x.exp_m1() - x) * p.alpha_n + 0.5 * x;
        assert!((p.apply(x) - want).abs() < 1e-6);
        // The softplus's large-x shortcut.
        assert_eq!(softplus(25.0), 25.0);
    }

    /// With the gate aliased to up, `apply` and `combine` must read the
    /// `up` operand and only it: a `Xielu` that read the gate would
    /// still pass on an aliased pair, so this feeds DIFFERENT gate and
    /// up vectors and pins that the gate has no effect.
    #[test]
    fn xielu_ignores_the_aliased_gate() {
        let p = XieluParams::from_gguf(0.2, 1.5, 0.75, -1e-6);
        let act = GluAct::Xielu(p);
        let up = [-1.0f32, -0.2, 0.0, 0.3, 2.0];
        let gate = [5.0f32; 5];
        let want: Vec<f32> = up.iter().map(|&x| p.apply(x)).collect();
        assert_eq!(act.apply(&gate, &up), want);
        for (i, &u) in up.iter().enumerate() {
            assert_eq!(act.combine(gate[i], u), want[i]);
        }
        assert_eq!(act.ungated().expect("ungated").apply(&up), want);
        assert_eq!(act.fused_kernel_gelu_flag(), None);
        assert!(!act.is_swiglu());
    }

    /// The two ReLU forms are two ops. `ReluSqr` reads `up` alone --
    /// fed DIFFERENT gate and up vectors, the gate has no effect, on
    /// `apply`, `combine` and the ungated shortcut alike -- and `Reglu`
    /// reads the gate and has NO shortcut. Before the split one variant
    /// was both, and `run_expert` computed `relu(up)^2` on
    /// SmallThinker's real gate.
    #[test]
    fn relu_sqr_reads_up_alone_and_reglu_reads_the_gate() {
        let up = [-1.0f32, -0.2, 0.0, 0.3, 2.0];
        let gate = [5.0f32, -5.0, 5.0, -5.0, 0.5];
        let sqr: Vec<f32> = up
            .iter()
            .map(|&x| if x > 0.0 { x * x } else { 0.0 })
            .collect();
        assert_eq!(GluAct::ReluSqr.apply(&gate, &up), sqr);
        for (i, &u) in up.iter().enumerate() {
            assert_eq!(GluAct::ReluSqr.combine(gate[i], u), sqr[i]);
        }
        assert_eq!(GluAct::ReluSqr.ungated().expect("ungated").apply(&up), sqr);

        let gated: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| if g > 0.0 { g * u } else { 0.0 })
            .collect();
        assert_eq!(GluAct::Reglu.apply(&gate, &up), gated);
        for (i, &u) in up.iter().enumerate() {
            assert_eq!(GluAct::Reglu.combine(gate[i], u), gated[i]);
        }
        assert!(GluAct::Reglu.ungated().is_none(), "the gate is real");
        assert_ne!(gated, sqr, "the vectors must tell the two apart");
        for act in [GluAct::Reglu, GluAct::ReluSqr] {
            assert_eq!(act.fused_kernel_gelu_flag(), None);
            assert!(!act.is_swiglu());
        }
    }

    /// The clamp, against llama-graph.cpp:1751-1768 written out by
    /// hand: `up` is clamped on both sides, `silu(gate)` from above
    /// only, and the two multiply. A gate below the limit and an `up`
    /// inside it reduce to plain SwiGLU.
    #[test]
    fn clamped_swiglu_matches_llama_cpp_s_else_branch() {
        let act = GluAct::SwigluClamped { limit: 2.0 };
        let silu = ferrox_core::matmul::silu;
        // Both clamps bite.
        assert!((act.combine(5.0, -7.0) - silu(5.0).min(2.0) * -2.0).abs() < 1e-6);
        assert!(
            (silu(5.0) - 2.0).abs() > 0.5,
            "the gate clamp must bite here"
        );
        // Neither bites: plain SwiGLU.
        assert!((act.combine(1.0, 1.5) - GluAct::Swiglu.combine(1.0, 1.5)).abs() < 1e-7);
        // A very negative gate is NOT clamped from below (min only).
        assert!((act.combine(-9.0, 1.0) - silu(-9.0)).abs() < 1e-7);
        assert!(!act.is_swiglu(), "no fused kernel spells the clamp");
        assert_eq!(act.fused_kernel_gelu_flag(), None);
        assert!(act.ungated().is_none());
    }

    /// `combine` is `apply` one element at a time for every variant --
    /// the site that fuses the loop (`cpu_moe_topk_parallel_slots`) and
    /// the sites that call `apply` must not disagree.
    #[test]
    fn combine_is_apply_elementwise_for_every_variant() {
        let gate = [-1.5f32, -0.4, 0.0, 0.7, 2.2];
        let up = [0.9f32, -1.1, 0.5, -0.3, 1.7];
        for act in [
            GluAct::Swiglu,
            GluAct::SwigluClamped { limit: 0.5 },
            GluAct::Geglu,
            GluAct::Reglu,
            GluAct::ReluSqr,
            GluAct::Xielu(XieluParams::from_gguf(0.8, 0.8, 0.5, -1e-6)),
        ] {
            let whole = act.apply(&gate, &up);
            for i in 0..gate.len() {
                let one = act.combine(gate[i], up[i]);
                assert!(
                    (whole[i] - one).abs() < 1e-6,
                    "{act:?} element {i}: apply {} vs combine {one}",
                    whole[i]
                );
            }
        }
    }
}
