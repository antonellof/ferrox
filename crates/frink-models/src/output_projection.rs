//! DeepSeek V4's grouped attention output projection (`wo_a`/`wo_b`,
//! `o_group_count` groups): a real, structural replacement for the usual
//! single dense `wo` output projection, transcribed directly from the
//! real, merged reference implementation (llama.cpp PR #24162,
//! `src/models/deepseek4.cpp::graph::build_attention`, the tail end
//! after `attn_derope`).
//!
//! Real per-layer tensor shapes (`load_arch_tensors`):
//! `wo_a`: `{n_head * n_embd_head / o_groups, o_lora_rank * o_groups}`,
//! `wo_b`: `{o_groups * o_lora_rank, n_embd}`. The real graph code
//! reshapes `wo_a` to `{o_group_dim, o_lora_rank, o_groups}`
//! (`o_group_dim = n_head/o_groups * n_embd_head`) and runs a **batched**
//! `ggml_mul_mat` over the group axis -- i.e. `wo_a` is not one shared
//! matrix but `o_groups` independent learned `[o_group_dim ->
//! o_lora_rank]` down-projections, one per contiguous slice of
//! `n_head/o_groups` attention heads. Their `o_lora_rank`-wide outputs
//! are concatenated (in group order) into one `o_lora_rank*o_groups`-wide
//! vector, then passed through **one** shared `wo_b`
//! `[o_lora_rank*o_groups -> n_embd]` up-projection
//! (`build_lora_mm(layer.wo_b, oa)`) -- only the down-projection is
//! grouped; the up-projection is not.
//!
//! Real op sequence read line-by-line, confirming the grouping is a
//! contiguous head-block split (not e.g. interleaved or transposed): the
//! pre-projection attention output is `[n_embd_head, n_head, n_tokens]`
//! (`ne[0]` fastest-varying), reshaped directly to `[o_group_dim,
//! o_groups, n_tokens]` with no intervening permute -- reinterpreting the
//! same contiguous per-token buffer means group `g` owns exactly heads
//! `[g * (n_head/o_groups), (g+1) * (n_head/o_groups))`, i.e. attention
//! output element range `[g * o_group_dim, (g+1) * o_group_dim)`.
//!
//! **Not implemented here**: the real code applies an inverse RoPE
//! (`ggml_rope_ext_back`) to the attention output's trailing rope-dim
//! slice immediately *before* this grouped projection, for any layer
//! using CSA/HCA compression (`attn_derope`, `out_pe = ggml_rope_ext_back
//! (..., inp_pos, ...)`) -- a consequence of DeepSeek V4's combined K/V
//! tensor ("kv") being used directly as both K and V (`GGML_ASSERT
//! (n_embd_head == n_embd_head_v)`), so V's own rope-rotated slice leaks
//! into the attention output and must be de-rotated before the output
//! projection. Reproducing that correctly requires the same "V is
//! literally K" convention this module hasn't investigated; this module
//! implements the grouped projection itself, taking the (already
//! appropriately de-roped, if needed) attention output as an opaque
//! input vector.

use frink_core::weight_matrix::WeightMatrix;

/// DeepSeek V4's real grouped output projection: splits `attn_out` (the
/// per-token attention output, length `n_head * n_embd_head`) into
/// `n_groups` contiguous, equal-size slices, applies each group's own
/// independent `group_down[g]` projection (real `wo_a`, per-group
/// down-projection to `o_lora_rank`), concatenates the `n_groups`
/// down-projected outputs in group order, then applies the one shared
/// `wo_b` up-projection to `n_embd`. See the module doc comment for the
/// real tensor shapes and citations.
pub fn grouped_output_projection(
    attn_out: &[f32],
    group_down: &[WeightMatrix],
    wo_b: &WeightMatrix,
) -> Vec<f32> {
    let n_groups = group_down.len();
    assert!(n_groups > 0, "must have at least one output group");
    assert_eq!(
        attn_out.len() % n_groups,
        0,
        "attention output width must split evenly across groups"
    );
    let o_group_dim = attn_out.len() / n_groups;

    let mut combined = Vec::new();
    for (g, down) in group_down.iter().enumerate() {
        assert_eq!(
            down.cols(),
            o_group_dim,
            "group {g}'s down-projection must accept exactly one group's slice width"
        );
        let slice = &attn_out[g * o_group_dim..(g + 1) * o_group_dim];
        combined.extend(down.apply(slice));
    }

    wo_b.apply(&combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::tensor::Tensor;

    fn wm(data: &[f32], rows: usize, cols: usize) -> WeightMatrix {
        assert_eq!(data.len(), rows * cols);
        WeightMatrix::F32(Tensor::new(data.to_vec(), vec![rows, cols]))
    }

    #[test]
    fn single_group_matches_a_plain_two_matrix_low_rank_projection() {
        // n_groups=1 must degenerate to exactly: wo_b.apply(wo_a.apply(x)).
        let attn_out = vec![1.0, 2.0, 3.0, 4.0];
        let down_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let down = wm(&down_data, 2, 4); // [o_lora_rank=2, o_group_dim=4]
        let down_again = wm(&down_data, 2, 4);
        let wo_b = wm(&[1.0, 0.0, 0.0, 1.0], 2, 2); // identity, [n_embd=2, o_lora_rank=2]

        let out = grouped_output_projection(&attn_out, &[down], &wo_b);
        let expected = wo_b.apply(&down_again.apply(&attn_out));
        assert_eq!(out.len(), expected.len());
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn groups_are_independent_changing_one_groups_slice_only_affects_its_own_contribution() {
        // 2 groups, each owning 2 of the 4 attn_out elements. Zeroing out
        // group 1's down-projection weights entirely must leave the
        // output identical to what group 0 alone would produce through
        // wo_b -- proving group 1's slice of attn_out has zero path to
        // the output when its projection is zeroed, i.e. the groups truly
        // don't share weights or mix inputs before their own projection.
        let attn_out = vec![1.0, 2.0, 100.0, -50.0]; // group0=[1,2], group1=[100,-50]
        let down0 = wm(&[0.5, -0.5], 1, 2); // [o_lora_rank=1, o_group_dim=2]
        let down1_zero = wm(&[0.0, 0.0], 1, 2);
        let wo_b = wm(&[2.0, 3.0], 1, 2); // [n_embd=1, o_lora_rank*n_groups=2]

        let out = grouped_output_projection(&attn_out, &[down0, down1_zero], &wo_b);

        // Expected: group0 contributes down0.apply([1,2]) = 0.5*1 + -0.5*2 = -0.5;
        // group1 contributes 0 (zeroed weights) regardless of its huge
        // input values. Combined = [-0.5, 0.0]; wo_b.apply -> 2.0*-0.5 + 3.0*0.0 = -1.0.
        assert!((out[0] - (-1.0)).abs() < 1e-5, "out[0]={}", out[0]);
    }

    #[test]
    fn group_order_is_preserved_in_the_concatenation_fed_to_wo_b() {
        // Swap which group produces which down-projected value and
        // confirm wo_b sees them in group order (0 then 1), not reversed
        // -- catches an accidental concatenation-order bug.
        let attn_out = vec![10.0, 20.0]; // group0=[10], group1=[20]
        let down0 = wm(&[1.0], 1, 1); // identity-ish: passes 10 through as 10
        let down1 = wm(&[1.0], 1, 1); // passes 20 through as 20
                                      // wo_b picks out only the *first* combined element (weight 1,0).
        let wo_b = wm(&[1.0, 0.0], 1, 2);

        let out = grouped_output_projection(&attn_out, &[down0, down1], &wo_b);
        assert!(
            (out[0] - 10.0).abs() < 1e-5,
            "expected group 0's value first, out[0]={}",
            out[0]
        );
    }

    #[test]
    #[should_panic(expected = "split evenly")]
    fn mismatched_group_count_panics_rather_than_silently_truncating() {
        let attn_out = vec![1.0, 2.0, 3.0]; // length 3, not divisible by 2 groups
        let down = wm(&[1.0, 1.0, 1.0], 1, 3);
        let down2 = wm(&[1.0, 1.0, 1.0], 1, 3);
        let wo_b = wm(&[1.0, 1.0], 1, 2);
        let _ = grouped_output_projection(&attn_out, &[down, down2], &wo_b);
    }
}
