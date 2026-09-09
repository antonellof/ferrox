//! What `llama-quantize` actually chose, tensor by tensor, for a real
//! checkpoint -- and the test that [`super::recipe`] chooses the same.
//!
//! Every other test of the recipe checks it against a reading of
//! `llama_tensor_get_type`. This one checks it against the BINARY:
//! llama.cpp b7650's `llama-quantize` was run over an F16 Llama-3.2-1B
//! three times, once per mix, and the types below were read back out of
//! its three output files with ferrox's own GGUF reader. A transcription
//! that is wrong in the same way twice -- in the code and in a test that
//! restates the code -- cannot survive this, because nothing here came
//! from reading the C.
//!
//! # Reproducing it
//!
//! The local checkpoints are all already quantized and both quantizers
//! refuse a quantized source, so the F16 input was made by dequantizing
//! `models/Llama-3.2-1B-Instruct-Q8_0.gguf` through ferrox's reader and
//! re-emitting every tensor as F16. Then, for each of `Q4_K_M`,
//! `Q5_K_M`, `Q6_K`:
//!
//! ```text
//! llama-quantize llama1b-f16.gguf llama1b-<MIX>.gguf <MIX> 8
//! ```
//!
//! and the three outputs' tensor types were dumped and pasted below.
//!
//! # Why this model
//!
//! Llama-3.2-1B has TIED EMBEDDINGS: there is no `output.weight`, so
//! `token_embd.weight` takes the output head's arm and comes back Q6_K
//! under every mix. That is the branch a recipe is most likely to miss,
//! because on an untied model the same code path is invisible -- and it
//! is the norm on every modern small model. 16 layers also makes
//! `use_more_bits` pick a non-trivial pattern (0, 1, 4, 7, 10, 13, 14,
//! 15) rather than everything or nothing.
//!
//! # What it does NOT cover
//!
//! One architecture, dense, not Falcon, not 70B, not 8-expert, no fused
//! QKV. Those arms are covered by the synthetic tests in
//! [`super::recipe`] against a reading of the C, which is weaker
//! evidence, and the module says so where it matters.

use ferrox_gguf::GgmlType;

use super::policy::allows_quantization;
use super::policy::Target;
use super::recipe::{ModelShape, Recipe};

/// `llama-quantize` b7650's own answer for every tensor of an F16
/// Llama-3.2-1B.
///
/// The rows are in the order llama.cpp's OUTPUT file carries them,
/// which is `weight_name_comparer` order (layer number, then name) --
/// not the order the F16 input carried them in. The test below feeds
/// them to `resolve_all` in a deliberately different order and expects
/// the same answers, because sorting into llama.cpp's order is
/// `resolve_all`'s job and a test that handed them over already sorted
/// would not notice if it stopped doing it.
///
/// Columns: tensor name, its SHAPE, then the type chosen under
/// `Q4_K_M`, `Q5_K_M` and `Q6_K`. `-` is a tensor llama.cpp left at
/// source precision, so this table is also the keep-list's evidence --
/// and the shape is the real one, because the 1-D rule is half of that
/// keep-list and a first version carrying only the leading dimension
/// made `rope_freqs.weight` look 2-D and hid it.
///
/// ONE table rather than a name list plus a type list: the two would
/// have had to agree about order, and order is exactly what decides
/// which layers `use_more_bits` promotes.
const LLAMA_QUANTIZE_CHOSE: &str = "\
output_norm.weight        2048         -   -   -  
rope_freqs.weight         32           -   -   -  
token_embd.weight         2048x128256  Q6K Q6K Q6K
blk.0.attn_k.weight       2048x512     Q4K Q5K Q6K
blk.0.attn_norm.weight    2048         -   -   -  
blk.0.attn_output.weight  2048x2048    Q4K Q5K Q6K
blk.0.attn_q.weight       2048x2048    Q4K Q5K Q6K
blk.0.attn_v.weight       2048x512     Q6K Q6K Q6K
blk.0.ffn_down.weight     8192x2048    Q6K Q6K Q6K
blk.0.ffn_gate.weight     2048x8192    Q4K Q5K Q6K
blk.0.ffn_norm.weight     2048         -   -   -  
blk.0.ffn_up.weight       2048x8192    Q4K Q5K Q6K
blk.1.attn_k.weight       2048x512     Q4K Q5K Q6K
blk.1.attn_norm.weight    2048         -   -   -  
blk.1.attn_output.weight  2048x2048    Q4K Q5K Q6K
blk.1.attn_q.weight       2048x2048    Q4K Q5K Q6K
blk.1.attn_v.weight       2048x512     Q6K Q6K Q6K
blk.1.ffn_down.weight     8192x2048    Q6K Q6K Q6K
blk.1.ffn_gate.weight     2048x8192    Q4K Q5K Q6K
blk.1.ffn_norm.weight     2048         -   -   -  
blk.1.ffn_up.weight       2048x8192    Q4K Q5K Q6K
blk.2.attn_k.weight       2048x512     Q4K Q5K Q6K
blk.2.attn_norm.weight    2048         -   -   -  
blk.2.attn_output.weight  2048x2048    Q4K Q5K Q6K
blk.2.attn_q.weight       2048x2048    Q4K Q5K Q6K
blk.2.attn_v.weight       2048x512     Q4K Q5K Q6K
blk.2.ffn_down.weight     8192x2048    Q4K Q5K Q6K
blk.2.ffn_gate.weight     2048x8192    Q4K Q5K Q6K
blk.2.ffn_norm.weight     2048         -   -   -  
blk.2.ffn_up.weight       2048x8192    Q4K Q5K Q6K
blk.3.attn_k.weight       2048x512     Q4K Q5K Q6K
blk.3.attn_norm.weight    2048         -   -   -  
blk.3.attn_output.weight  2048x2048    Q4K Q5K Q6K
blk.3.attn_q.weight       2048x2048    Q4K Q5K Q6K
blk.3.attn_v.weight       2048x512     Q4K Q5K Q6K
blk.3.ffn_down.weight     8192x2048    Q4K Q5K Q6K
blk.3.ffn_gate.weight     2048x8192    Q4K Q5K Q6K
blk.3.ffn_norm.weight     2048         -   -   -  
blk.3.ffn_up.weight       2048x8192    Q4K Q5K Q6K
blk.4.attn_k.weight       2048x512     Q4K Q5K Q6K
blk.4.attn_norm.weight    2048         -   -   -  
blk.4.attn_output.weight  2048x2048    Q4K Q5K Q6K
blk.4.attn_q.weight       2048x2048    Q4K Q5K Q6K
blk.4.attn_v.weight       2048x512     Q6K Q6K Q6K
blk.4.ffn_down.weight     8192x2048    Q6K Q6K Q6K
blk.4.ffn_gate.weight     2048x8192    Q4K Q5K Q6K
blk.4.ffn_norm.weight     2048         -   -   -  
blk.4.ffn_up.weight       2048x8192    Q4K Q5K Q6K
blk.5.attn_k.weight       2048x512     Q4K Q5K Q6K
blk.5.attn_norm.weight    2048         -   -   -  
blk.5.attn_output.weight  2048x2048    Q4K Q5K Q6K
blk.5.attn_q.weight       2048x2048    Q4K Q5K Q6K
blk.5.attn_v.weight       2048x512     Q4K Q5K Q6K
blk.5.ffn_down.weight     8192x2048    Q4K Q5K Q6K
blk.5.ffn_gate.weight     2048x8192    Q4K Q5K Q6K
blk.5.ffn_norm.weight     2048         -   -   -  
blk.5.ffn_up.weight       2048x8192    Q4K Q5K Q6K
blk.6.attn_k.weight       2048x512     Q4K Q5K Q6K
blk.6.attn_norm.weight    2048         -   -   -  
blk.6.attn_output.weight  2048x2048    Q4K Q5K Q6K
blk.6.attn_q.weight       2048x2048    Q4K Q5K Q6K
blk.6.attn_v.weight       2048x512     Q4K Q5K Q6K
blk.6.ffn_down.weight     8192x2048    Q4K Q5K Q6K
blk.6.ffn_gate.weight     2048x8192    Q4K Q5K Q6K
blk.6.ffn_norm.weight     2048         -   -   -  
blk.6.ffn_up.weight       2048x8192    Q4K Q5K Q6K
blk.7.attn_k.weight       2048x512     Q4K Q5K Q6K
blk.7.attn_norm.weight    2048         -   -   -  
blk.7.attn_output.weight  2048x2048    Q4K Q5K Q6K
blk.7.attn_q.weight       2048x2048    Q4K Q5K Q6K
blk.7.attn_v.weight       2048x512     Q6K Q6K Q6K
blk.7.ffn_down.weight     8192x2048    Q6K Q6K Q6K
blk.7.ffn_gate.weight     2048x8192    Q4K Q5K Q6K
blk.7.ffn_norm.weight     2048         -   -   -  
blk.7.ffn_up.weight       2048x8192    Q4K Q5K Q6K
blk.8.attn_k.weight       2048x512     Q4K Q5K Q6K
blk.8.attn_norm.weight    2048         -   -   -  
blk.8.attn_output.weight  2048x2048    Q4K Q5K Q6K
blk.8.attn_q.weight       2048x2048    Q4K Q5K Q6K
blk.8.attn_v.weight       2048x512     Q4K Q5K Q6K
blk.8.ffn_down.weight     8192x2048    Q4K Q5K Q6K
blk.8.ffn_gate.weight     2048x8192    Q4K Q5K Q6K
blk.8.ffn_norm.weight     2048         -   -   -  
blk.8.ffn_up.weight       2048x8192    Q4K Q5K Q6K
blk.9.attn_k.weight       2048x512     Q4K Q5K Q6K
blk.9.attn_norm.weight    2048         -   -   -  
blk.9.attn_output.weight  2048x2048    Q4K Q5K Q6K
blk.9.attn_q.weight       2048x2048    Q4K Q5K Q6K
blk.9.attn_v.weight       2048x512     Q4K Q5K Q6K
blk.9.ffn_down.weight     8192x2048    Q4K Q5K Q6K
blk.9.ffn_gate.weight     2048x8192    Q4K Q5K Q6K
blk.9.ffn_norm.weight     2048         -   -   -  
blk.9.ffn_up.weight       2048x8192    Q4K Q5K Q6K
blk.10.attn_k.weight      2048x512     Q4K Q5K Q6K
blk.10.attn_norm.weight   2048         -   -   -  
blk.10.attn_output.weight 2048x2048    Q4K Q5K Q6K
blk.10.attn_q.weight      2048x2048    Q4K Q5K Q6K
blk.10.attn_v.weight      2048x512     Q6K Q6K Q6K
blk.10.ffn_down.weight    8192x2048    Q6K Q6K Q6K
blk.10.ffn_gate.weight    2048x8192    Q4K Q5K Q6K
blk.10.ffn_norm.weight    2048         -   -   -  
blk.10.ffn_up.weight      2048x8192    Q4K Q5K Q6K
blk.11.attn_k.weight      2048x512     Q4K Q5K Q6K
blk.11.attn_norm.weight   2048         -   -   -  
blk.11.attn_output.weight 2048x2048    Q4K Q5K Q6K
blk.11.attn_q.weight      2048x2048    Q4K Q5K Q6K
blk.11.attn_v.weight      2048x512     Q4K Q5K Q6K
blk.11.ffn_down.weight    8192x2048    Q4K Q5K Q6K
blk.11.ffn_gate.weight    2048x8192    Q4K Q5K Q6K
blk.11.ffn_norm.weight    2048         -   -   -  
blk.11.ffn_up.weight      2048x8192    Q4K Q5K Q6K
blk.12.attn_k.weight      2048x512     Q4K Q5K Q6K
blk.12.attn_norm.weight   2048         -   -   -  
blk.12.attn_output.weight 2048x2048    Q4K Q5K Q6K
blk.12.attn_q.weight      2048x2048    Q4K Q5K Q6K
blk.12.attn_v.weight      2048x512     Q4K Q5K Q6K
blk.12.ffn_down.weight    8192x2048    Q4K Q5K Q6K
blk.12.ffn_gate.weight    2048x8192    Q4K Q5K Q6K
blk.12.ffn_norm.weight    2048         -   -   -  
blk.12.ffn_up.weight      2048x8192    Q4K Q5K Q6K
blk.13.attn_k.weight      2048x512     Q4K Q5K Q6K
blk.13.attn_norm.weight   2048         -   -   -  
blk.13.attn_output.weight 2048x2048    Q4K Q5K Q6K
blk.13.attn_q.weight      2048x2048    Q4K Q5K Q6K
blk.13.attn_v.weight      2048x512     Q6K Q6K Q6K
blk.13.ffn_down.weight    8192x2048    Q6K Q6K Q6K
blk.13.ffn_gate.weight    2048x8192    Q4K Q5K Q6K
blk.13.ffn_norm.weight    2048         -   -   -  
blk.13.ffn_up.weight      2048x8192    Q4K Q5K Q6K
blk.14.attn_k.weight      2048x512     Q4K Q5K Q6K
blk.14.attn_norm.weight   2048         -   -   -  
blk.14.attn_output.weight 2048x2048    Q4K Q5K Q6K
blk.14.attn_q.weight      2048x2048    Q4K Q5K Q6K
blk.14.attn_v.weight      2048x512     Q6K Q6K Q6K
blk.14.ffn_down.weight    8192x2048    Q6K Q6K Q6K
blk.14.ffn_gate.weight    2048x8192    Q4K Q5K Q6K
blk.14.ffn_norm.weight    2048         -   -   -  
blk.14.ffn_up.weight      2048x8192    Q4K Q5K Q6K
blk.15.attn_k.weight      2048x512     Q4K Q5K Q6K
blk.15.attn_norm.weight   2048         -   -   -  
blk.15.attn_output.weight 2048x2048    Q4K Q5K Q6K
blk.15.attn_q.weight      2048x2048    Q4K Q5K Q6K
blk.15.attn_v.weight      2048x512     Q6K Q6K Q6K
blk.15.ffn_down.weight    8192x2048    Q6K Q6K Q6K
blk.15.ffn_gate.weight    2048x8192    Q4K Q5K Q6K
blk.15.ffn_norm.weight    2048         -   -   -  
blk.15.ffn_up.weight      2048x8192    Q4K Q5K Q6K";

/// Llama-3.2-1B's header, as `ModelShape::from_header` would read it.
/// Spelled out because the test does not open the 2.3 GB F16 file.
const LLAMA_3_2_1B: ModelShape = ModelShape {
    n_layer: 16,
    n_expert: 0,
    is_falcon: false,
    // 16 layers, not 80.
    is_70b: false,
};

struct Row {
    name: &'static str,
    shape: Vec<u64>,
    /// `None` where llama.cpp kept the tensor at source precision.
    chosen: Option<[GgmlType; 3]>,
}

fn rows() -> Vec<Row> {
    LLAMA_QUANTIZE_CHOSE
        .lines()
        .map(|line| {
            let mut f = line.split_whitespace();
            let name = f.next().expect("name");
            let shape: Vec<u64> = f
                .next()
                .expect("shape")
                .split('x')
                .map(|d| d.parse().expect("shape dimension"))
                .collect();
            let tys: Vec<&str> = f.collect();
            assert_eq!(tys.len(), 3, "{name}: expected three type columns");
            let chosen = if tys[0] == "-" {
                assert!(tys.iter().all(|t| *t == "-"), "{name}: mixed - and types");
                None
            } else {
                Some(std::array::from_fn(|i| match tys[i] {
                    "Q4K" => GgmlType::Q4K,
                    "Q5K" => GgmlType::Q5K,
                    "Q6K" => GgmlType::Q6K,
                    "Q8_0" => GgmlType::Q8_0,
                    other => panic!("{name}: unknown type {other} in the golden"),
                }))
            };
            Row {
                name,
                shape,
                chosen,
            }
        })
        .collect()
}

/// The mixes, in the order their columns appear in the golden.
const MIXES: [Target; 3] = [Target::Q4_K_M, Target::Q5_K_M, Target::Q6_K];

#[cfg(test)]
mod tests {
    use super::*;

    /// For a real model's tensor list, the type ferrox chooses per
    /// tensor equals the type `llama-quantize` chose.
    ///
    /// This is the whole point of `recipe.rs`. Every disagreement it
    /// can catch is a file that would load, run, report the right
    /// `general.file_type`, and not be the file its name promises.
    #[test]
    fn ferrox_chooses_the_same_type_llama_quantize_chose_for_every_tensor() {
        let rows = rows();
        assert_eq!(rows.len(), 147, "the golden lost rows");

        // Reversed, so the rows do NOT arrive in llama.cpp's order.
        // `resolve_all` has to sort them back; before it did, twelve of
        // these tensors came out wrong.
        let scrambled: Vec<(String, Vec<u64>)> = rows
            .iter()
            .rev()
            .map(|r| (r.name.to_string(), r.shape.clone()))
            .collect();

        for (col, mix) in MIXES.iter().enumerate() {
            let got = Recipe::resolve_all(*mix, LLAMA_3_2_1B, &scrambled, |name, shape| {
                allows_quantization(name, shape).is_none()
            });
            let mut quantized = 0;
            for r in &rows {
                let allowed = allows_quantization(r.name, &r.shape);
                match (&r.chosen, allowed) {
                    (Some(want), None) => {
                        let have = got[r.name];
                        assert_eq!(
                            have, want[col],
                            "{:?}: {} -> ferrox {:?}, llama-quantize {:?}",
                            mix, r.name, have, want[col]
                        );
                        quantized += 1;
                    }
                    (None, Some(_)) => assert!(!got.contains_key(r.name), "{}", r.name),
                    (Some(_), Some(reason)) => panic!(
                        "{}: llama-quantize quantized it, ferrox keeps it at source precision \
                         ({reason})",
                        r.name
                    ),
                    (None, None) => panic!(
                        "{}: llama-quantize kept it at source precision, ferrox would quantize it",
                        r.name
                    ),
                }
            }
            assert_eq!(quantized, 113, "{mix:?}: wrong number of quantized tensors");
            assert_eq!(
                got.len(),
                113,
                "{mix:?}: resolve_all returned extra tensors"
            );
        }
    }

    /// The golden is not three copies of one column. If it were, the
    /// test above would pass for a recipe that ignored the mix.
    #[test]
    fn the_three_mixes_actually_disagree_in_the_golden() {
        let rows = rows();
        let differing = rows
            .iter()
            .filter_map(|r| r.chosen.as_ref())
            .filter(|c| c[0] != c[1] || c[1] != c[2])
            .count();
        assert!(
            differing > 50,
            "only {differing} tensors differ between mixes"
        );
        // And within one mix, more than one type is chosen -- otherwise
        // a uniform file would pass.
        for col in 0..3 {
            let distinct: std::collections::BTreeSet<_> = rows
                .iter()
                .filter_map(|r| r.chosen.as_ref())
                .map(|c| format!("{:?}", c[col]))
                .collect();
            let want = if col == 2 { 1 } else { 2 };
            assert_eq!(
                distinct.len(),
                want,
                "{:?} column has types {distinct:?}",
                MIXES[col]
            );
        }
    }
}
