//! HRM-Text's TWO residual streams, and the schedule that recombines
//! them.
//!
//! # What llama.cpp does
//!
//! `src/models/hrm-text.cpp:183-196` runs the model as
//! `h_cycles` outer cycles of `l_cycles` LOW stacks and one HIGH
//! stack. Every stack reads `zH + zL` and REPLACES one of the two with
//! its own output:
//!
//! ```text
//! zH = embeddings * embedding_scale       // :174
//! zL = hrm_z_l_init                       // :182, one [n_embd] row
//! for h in 0..h_cycles {                  // :183
//!     for l in 0..l_cycles { zL = stack(zH + zL) }
//!     zH = stack(zH + zL)
//! }
//! logits = output * zH                    // :198-207, NO final norm
//! ```
//!
//! `crate::layer_loops::LayerLoops::Hrm` is the schedule -- which
//! physical layer a logical slot runs, where a stack starts, which
//! stream a finished stack writes -- and this module is the state the
//! three host bodies carry while they walk it.
//!
//! # Why the state is here and not in the loop
//!
//! Three bodies iterate layers (the row body, the batched prefill and
//! the multi-sequence worker) and each of them would otherwise grow
//! its own copy of "hold two vectors, add them here, store one there".
//! That is the shape this repo keeps paying for, so the state is one
//! type with two methods and each body calls them: [`HrmStreams::
//! stack_input`] before a layer and [`HrmStreams::store`] after it.
//!
//! The weightless norm that closes a stack is NOT here: it is a pass
//! boundary like `nanbeige`'s, so it lives in the one place that
//! already applied one (`Decoder::apply_loop_norm`, at the end of both
//! FFN bodies), and [`HrmStreams::store`] runs after it.

use crate::layer_loops::{HrmStream, LayerLoops};

/// The two residual streams of an HRM-Text decode, `rows * hidden_dim`
/// each.
#[derive(Debug, Clone)]
pub struct HrmStreams {
    /// The HIGH stream: the embeddings at the start, a HIGH stack's
    /// output after each cycle, and the value the lm_head reads.
    zh: Vec<f32>,
    /// The LOW stream: `hrm_z_l_init` broadcast over the rows at the
    /// start, a LOW stack's output after each pass.
    zl: Vec<f32>,
}

impl HrmStreams {
    /// The state a decode starts in: `zH` is the embedded rows,
    /// `zL` is the learned `hrm_z_l_init` row broadcast over them
    /// (`hrm-text.cpp:182`, "binary ops broadcast it over
    /// [n_embd, n_tokens]").
    pub fn new(embedded: &[f32], z_l_init: &[f32]) -> Self {
        let width = z_l_init.len();
        debug_assert!(width > 0 && embedded.len().is_multiple_of(width));
        let rows = embedded.len() / width;
        Self {
            zh: embedded.to_vec(),
            zl: z_l_init.repeat(rows),
        }
    }

    /// `zH + zL`, the residual the stack beginning at this layer reads.
    pub fn stack_input(&self) -> Vec<f32> {
        self.zh.iter().zip(&self.zl).map(|(h, l)| h + l).collect()
    }

    /// Store a finished stack's output in the stream it writes.
    pub fn store(&mut self, stream: HrmStream, hidden: &[f32]) {
        let target = match stream {
            HrmStream::Low => &mut self.zl,
            HrmStream::High => &mut self.zh,
        };
        debug_assert_eq!(target.len(), hidden.len());
        target.clear();
        target.extend_from_slice(hidden);
    }

    /// The value the lm_head reads: the HIGH stream
    /// (`hrm-text.cpp:198`).
    pub fn into_output(self) -> Vec<f32> {
        self.zh
    }
}

/// The schedule's answer for a model that may not be HRM at all: the
/// three bodies call this rather than matching on the enum themselves,
/// so a body cannot forget one of the two halves.
pub fn hrm_schedule(loops: Option<LayerLoops>) -> Option<LayerLoops> {
    matches!(loops, Some(LayerLoops::Hrm { .. })).then_some(loops?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two rows, a width of three: `zL` is the same learned row under
    /// both, the input is the sum, and a stored stack output replaces
    /// exactly one stream.
    #[test]
    fn the_streams_broadcast_add_and_replace_one_at_a_time() {
        let embedded = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let z_l_init = vec![0.5, 0.5, 0.5];
        let mut s = HrmStreams::new(&embedded, &z_l_init);
        assert_eq!(s.stack_input(), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);

        s.store(HrmStream::Low, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(s.stack_input(), vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0]);

        s.store(HrmStream::High, &[9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
        assert_eq!(s.stack_input(), vec![9.0, 9.0, 9.0, 10.0, 10.0, 10.0]);
        assert_eq!(s.into_output(), vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
    }

    /// The helper answers `None` for a looped model that is not HRM,
    /// so a body that carries the state only builds it for one
    /// architecture.
    #[test]
    fn only_an_hrm_schedule_asks_for_the_streams() {
        assert!(hrm_schedule(None).is_none());
        assert!(hrm_schedule(Some(LayerLoops::Repeat {
            n_phys: 2,
            n_loops: 2,
            skip_loop_final_norm: false,
        }))
        .is_none());
        assert!(hrm_schedule(Some(LayerLoops::Hrm {
            lps: 2,
            h_cycles: 1,
            l_cycles: 1,
        }))
        .is_some());
    }
}
