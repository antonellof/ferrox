//! **BIASES ON A DENSE FFN** -- `build_ffn(up, up_b, gate, gate_b,
//! down, down_b, ...)`'s three optional adds, as one type the dense
//! bodies take and one function that applies them where llama.cpp does.
//!
//! # What it is
//!
//! `llama-graph.cpp`'s `build_ffn` adds `up_b` right after the `up`
//! matmul, `gate_b` right after the `gate` matmul (when there is a
//! gate), applies the activation to the BIASED projections, and adds
//! `down_b` right after `down`. So for a gated FFN:
//!
//! ```text
//! down(act(gate(x) + gate_b, up(x) + up_b)) + down_b
//! ```
//!
//! and for an ungated one (`LLM_FFN_SEQ` with a null gate) the same
//! with the gate half absent. The tensors are `blk.N.ffn_{up,gate,
//! down}.bias`.
//!
//! # Reach -- MEASURED
//!
//! `grep -l 'FFN_UP, *"bias"' src/models/*.cpp` over all 140 graphs is
//! twenty-seven files (2026-09-12); most create the biases as
//! `TENSOR_NOT_REQUIRED` (`llama.cpp` itself, `granite`, `deci`,
//! `mistral3`, `minicpm`, `nemotron`), and `starcoder2`, `codeshell`,
//! `jais2`, `starcoder`, `gpt2`, `bloom`, `mpt`, `refact`, `gptneox`,
//! `phi2` REQUIRE them. Which architectures the generic loader reads
//! them for is `ferrox_models::proj_bias`; this module is the
//! arithmetic, which is the same in every one of the twenty-seven.
//!
//! # Where it applies
//!
//! [`run_expert_biased`] is the row body: the gate/up projections come
//! from the same `gate_up_projections` the unbiased [`crate::run_expert`]
//! uses, the biases are added, then the activation, then `down` and
//! its bias. It does NOT take the fused on-device SwiGLU, whose kernel
//! has no site between the projections and the activation. The
//! batched dense body in `ferrox_models` adds the same three vectors
//! per row and fences its fused launch on the same fact.

use crate::glu_act::GluAct;
use crate::{gate_up_projections, ExpertWeights};

/// The three optional biases of one dense FFN. `gate` is `None` for an
/// ungated FFN and for a gated one whose file carries no gate bias.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DenseBias {
    pub gate: Option<Vec<f32>>,
    pub up: Option<Vec<f32>>,
    pub down: Option<Vec<f32>>,
}

impl DenseBias {
    /// Whether any bias is present; a `DenseBias` with none is the
    /// unbiased FFN and callers may take the unbiased paths.
    pub fn is_empty(&self) -> bool {
        self.gate.is_none() && self.up.is_none() && self.down.is_none()
    }

    /// `up_b` and `gate_b` onto the projections, for `rows` rows laid
    /// out contiguously. Before the activation, as `build_ffn` adds
    /// them.
    pub fn add_pre_activation(&self, gate: &mut [f32], up: &mut [f32], rows: usize) {
        if let Some(b) = &self.up {
            add_rows(up, b, rows);
        }
        if let Some(b) = &self.gate {
            add_rows(gate, b, rows);
        }
    }

    /// `down_b` onto the down projection's output, for `rows` rows.
    pub fn add_post_down(&self, out: &mut [f32], rows: usize) {
        if let Some(b) = &self.down {
            add_rows(out, b, rows);
        }
    }
}

fn add_rows(x: &mut [f32], b: &[f32], rows: usize) {
    debug_assert_eq!(x.len(), rows * b.len());
    for row in x.chunks_mut(b.len()) {
        for (v, b) in row.iter_mut().zip(b.iter()) {
            *v += b;
        }
    }
}

/// [`crate::run_expert`] with the three biases: `down(act(gate(x) +
/// gate_b, up(x) + up_b)) + down_b`, or the ungated form `down(act(up(x)
/// + up_b)) + down_b`.
pub fn run_expert_biased(
    hidden: &[f32],
    expert: &ExpertWeights,
    act: GluAct,
    bias: &DenseBias,
) -> Vec<f32> {
    let activated = match act.ungated() {
        Some(f) => {
            // `gate` is an alias of `up` on an ungated FFN, so there is
            // one projection and one bias to add.
            let mut up = expert.up.apply(hidden);
            if let Some(b) = &bias.up {
                add_rows(&mut up, b, 1);
            }
            f.apply(&up)
        }
        None => {
            let (mut gate, mut up) = gate_up_projections(hidden, expert);
            bias.add_pre_activation(&mut gate, &mut up, 1);
            act.apply(&gate, &up)
        }
    };
    let mut out = expert.down.apply(&activated);
    bias.add_post_down(&mut out, 1);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_core::tensor::Tensor;
    use ferrox_core::weight_matrix::WeightMatrix;

    fn wm(rows: usize, cols: usize, seed: f32) -> WeightMatrix {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect();
        WeightMatrix::F32(Tensor::new(data, vec![rows, cols]))
    }

    /// With no bias the biased body IS `run_expert`; with biases it is
    /// the hand-written formula, bias before the activation and after
    /// `down`.
    #[test]
    fn biased_equals_the_formula_and_empty_equals_run_expert() {
        let (hidden, ff) = (8usize, 12usize);
        let expert = ExpertWeights {
            gate: wm(ff, hidden, 1.0),
            up: wm(ff, hidden, 2.0),
            down: wm(hidden, ff, 3.0),
        };
        let x: Vec<f32> = (0..hidden).map(|i| 0.1 * i as f32 - 0.3).collect();
        let none = DenseBias::default();
        assert!(none.is_empty());
        assert_eq!(
            run_expert_biased(&x, &expert, GluAct::Swiglu, &none),
            crate::run_expert(&x, &expert, GluAct::Swiglu)
        );

        let bias = DenseBias {
            gate: Some((0..ff).map(|i| 0.05 * i as f32).collect()),
            up: Some((0..ff).map(|i| -0.02 * i as f32 + 0.1).collect()),
            down: Some((0..hidden).map(|i| 0.3 - 0.04 * i as f32).collect()),
        };
        let got = run_expert_biased(&x, &expert, GluAct::Swiglu, &bias);
        let mut gate = expert.gate.apply(&x);
        let mut up = expert.up.apply(&x);
        for (g, b) in gate.iter_mut().zip(bias.gate.as_ref().unwrap()) {
            *g += b;
        }
        for (u, b) in up.iter_mut().zip(bias.up.as_ref().unwrap()) {
            *u += b;
        }
        let mut want = expert.down.apply(&GluAct::Swiglu.apply(&gate, &up));
        for (o, b) in want.iter_mut().zip(bias.down.as_ref().unwrap()) {
            *o += b;
        }
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "{got:?} vs {want:?}");
        }
        // And the bias really lands BEFORE the activation: adding it
        // after would be a different number.
        let mut after = expert
            .down
            .apply(&GluAct::Swiglu.apply(&expert.gate.apply(&x), &expert.up.apply(&x)));
        for (o, b) in after.iter_mut().zip(bias.down.as_ref().unwrap()) {
            *o += b;
        }
        assert!(got
            .iter()
            .zip(after.iter())
            .any(|(g, a)| (g - a).abs() > 1e-4));
    }

    /// The ungated form adds `up_b` once, to the one projection.
    #[test]
    fn the_ungated_form_biases_the_one_projection() {
        let (hidden, ff) = (8usize, 12usize);
        let up = wm(ff, hidden, 2.0);
        let expert = ExpertWeights {
            gate: wm(ff, hidden, 2.0),
            up,
            down: wm(hidden, ff, 3.0),
        };
        let x: Vec<f32> = (0..hidden).map(|i| 0.1 * i as f32 - 0.3).collect();
        let bias = DenseBias {
            gate: None,
            up: Some((0..ff).map(|i| 0.5 - 0.1 * i as f32).collect()),
            down: None,
        };
        let got = run_expert_biased(&x, &expert, GluAct::ReluSqr, &bias);
        let mut u = expert.up.apply(&x);
        for (v, b) in u.iter_mut().zip(bias.up.as_ref().unwrap()) {
            *v += b;
        }
        let want = expert.down.apply(&crate::relu_sqr(&u));
        assert_eq!(got, want);
    }
}
