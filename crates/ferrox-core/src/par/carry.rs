//! What a promoted parallel region carries from the thread that
//! submitted it, and which backends may be promoted at all.
//!
//! # The problem this exists for (GitHub issue #166)
//!
//! [`super::on_workers`] moves a whole forward pass onto a rayon worker
//! so its ~150 cold pool entries collapse into one. Moving work between
//! threads is free only for work that reads nothing off the thread it
//! runs on, and a Metal decode step is not that: `ferrox-cli` and
//! `ferrox-server` each decide per request whether the stack may fold
//! `final_norm + lm_head + argmax` into its own command buffer, and
//! they announce that decision into a THREAD-LOCAL.
//!
//! Measured on an M2 Pro, `Llama-3.2-3B-Instruct-Q4_K_M --ngl 99`,
//! `--temp 0 --no-cnv`: a step driven on a worker read the default
//! instead of the announcement and silently took the other `lm_head`
//! path, giving a different completion from the tenth token. The
//! workaround was to decline promotion under any GPU backend, which
//! cost Metal the whole scheduling win.
//!
//! # The shape of the fix
//!
//! [`Carry`] is the ONE list of thread-local *settings* a moved step
//! reads. It is captured by value on the submitting thread and adopted
//! on the worker for exactly the length of the job.
//!
//! [`Carry::adopt`] destructures `self` exhaustively, with no `..`, so a
//! field added to `Carry` and not adopted does not compile. That is the
//! enforcement, and it is the point: this repo's dominant defect is two
//! structures that must agree with nothing making them.
//!
//! # Settings, not caches
//!
//! Only state a CALLER configured belongs here. `ferrox-metal`'s other
//! thread-locals -- `TL_PIPELINE_CACHE`, `TL_WEIGHT_CACHE`,
//! `TL_F32_CACHE`, `TL_MOE_PACKED`, `TL_MOE_LAYER_RESIDENT` -- are
//! per-thread mirrors of process-wide `Mutex` caches holding `Arc`s of
//! the same GPU buffers, and `TL_MOE_PREFILL` / `MOE_SCRATCH` are
//! per-thread scratch. A fresh thread rebuilds a mirror or allocates
//! scratch; neither changes a value, and the mirrors clone `Arc`s
//! rather than GPU memory, so a worker does not duplicate weights. The
//! one hand-off that used to carry a value across a step, the resident
//! activation, no longer lives on a thread at all: it is a host address
//! recorded inside `DecodeScratch`, under the process mutex that owns
//! the buffer it describes (`ferrox_metal::resident_act`).
//!
//! So `Carry` has exactly one field, and the reason there is only one
//! is written down rather than assumed.
//!
//! # What losing the setting costs TODAY, which is not what it cost then
//!
//! Worth stating, because the before/after evidence for relaxing the
//! gate is "bit-identical" and issue #166's evidence was "different from
//! the tenth token", and those look contradictory.
//!
//! Issue #170 landed in between. It split the permission to fold into
//! two predicates and made the device argmax legal only where the host
//! sampler would have chosen the same id anyway, so the folded and
//! unfolded paths now agree by construction WHENEVER the fold is
//! permitted. A worker that loses the setting therefore falls back to a
//! correct answer computed the slow way -- it downloads a whole
//! vocabulary and argmaxes on the host instead of downloading four
//! bytes.
//!
//! Measured with the carry deleted from [`super::on_workers`] and
//! nothing else changed: on `Llama-3.2-3B-Instruct-Q4_K_M --ngl 99
//! --temp 0 --repeat-penalty 1.0` the step still ran on a worker and the
//! completion was byte-for-byte the same, but the fold went from `On` to
//! `Unset` and stopped firing. That is the whole of what this module
//! buys now, and it is why the enforcement above matters more than the
//! one field it currently guards: the NEXT setting to be added has no
//! guarantee of degrading so kindly.

use crate::kernel_registry::Backend;

/// The thread-local settings a moved forward pass reads.
///
/// Captured with [`Self::capture`] on the thread whose caller made the
/// decisions, adopted with [`Self::adopt`] on the thread that runs the
/// work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Carry {
    /// Whether this request's Metal decode steps fold
    /// `final_norm + lm_head + argmax` into the stack.
    ///
    /// Three-state on purpose (`Unset` / `Off` / `On`): a thread nobody
    /// configured has not said "no fold", it has said nothing, and
    /// resolving one into the other is what would let a sampling
    /// request inherit a greedy request's precomputed token id. See
    /// `ferrox_metal::greedy_fold`.
    #[cfg(feature = "metal")]
    greedy_fold: ferrox_metal::greedy_fold::GreedyFold,
}

impl Carry {
    /// This thread's settings, as a value that can cross to another
    /// thread.
    pub(crate) fn capture() -> Self {
        Self {
            #[cfg(feature = "metal")]
            greedy_fold: ferrox_metal::greedy_fold::greedy_fold_setting(),
        }
    }

    /// Installs these settings on the calling thread until the returned
    /// guard drops.
    ///
    /// A rayon worker is reused, so every setting installed here has to
    /// be restored: leaving one behind would make the NEXT job on this
    /// worker inherit a stranger's request, which is issue #166 again
    /// one level down. Each field's guard restores what it replaced.
    #[must_use = "the settings are restored when the guard drops"]
    pub(crate) fn adopt(self) -> Adopted {
        // Exhaustive, and deliberately without `..`: a new field must be
        // adopted here or this stops compiling. That is the only thing
        // standing between a future thread-local setting and a silent
        // repeat of issue #166.
        let Self {
            #[cfg(feature = "metal")]
            greedy_fold,
        } = self;
        Adopted {
            #[cfg(feature = "metal")]
            _greedy_fold: ferrox_metal::greedy_fold::adopt_greedy_fold(greedy_fold),
        }
    }
}

/// Restores every setting [`Carry::adopt`] installed, on drop.
pub(crate) struct Adopted {
    #[cfg(feature = "metal")]
    _greedy_fold: ferrox_metal::greedy_fold::GreedyFoldGuard,
}

/// Whether [`super::on_workers`] may move a step running on `backend`
/// onto a rayon worker.
///
/// Exhaustive with no `_` arm, so a new backend cannot be added without
/// stating its verdict, and the only honest way to write `true` is to
/// have made [`Carry`] reproduce everything that backend reads off the
/// submitting thread AND to have checked on hardware that its output is
/// unchanged.
///
/// The verdicts, and what each rests on:
///
/// - **`Cpu`** -- nothing thread-affine. Token-identical under the move
///   at 135M, 3B and 8B (issue #167).
/// - **`Metal`** -- the one setting is carried by [`Carry`], and the
///   remaining `ferrox-metal` thread-locals are caches, mirrors and
///   scratch (this module's header lists them). Verified bit-identical
///   before and after with `ferrox verify` and `ferrox parity` on a
///   dense Llama, a sandwich-norm Gemma-2 and an OLMoE MoE. On a build
///   without the `metal` feature `active_backend` cannot return this,
///   so the arm is unreachable rather than wrong there.
/// - **`Cuda`** -- audited clean and NOT promoted. `ferrox-cuda` has no
///   thread-locals at all: the device handle, the loaded-module set and
///   the weight cache are process-wide `Mutex`es, and cudarc binds the
///   primary context to whichever thread calls it. The audit is not the
///   evidence this repo asks for, though; a GPU behaviour change merges
///   after it has been RUN on that GPU, and this development machine
///   has none. Flipping this row is a one-line change for whoever has
///   the hardware, plus the same before/after check Metal got.
/// - **`Vulkan`** -- one kernel, reached through MoltenVK, never
///   measured under promotion. Same rule as CUDA.
pub(crate) fn promotable(backend: Backend) -> bool {
    match backend {
        Backend::Cpu => true,
        Backend::Metal => true,
        Backend::Cuda => false,
        Backend::Vulkan => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A captured setting reproduces itself on the thread that adopts
    /// it, which is the whole of the fix for issue #166.
    ///
    /// The `ferrox-metal` half of this is tested at its source; what is
    /// tested HERE is that `par`'s own capture/adopt pair is wired to
    /// it, because the defect was never in `greedy_fold` -- it was in
    /// nobody calling it.
    ///
    /// Sabotage: make `Carry::capture` return `Self::default()`.
    #[cfg(feature = "metal")]
    #[test]
    fn a_captured_setting_crosses_to_the_thread_that_adopts_it() {
        use ferrox_metal::greedy_fold::{
            greedy_fold_setting, metal_greedy_argmax_active, set_metal_greedy_argmax, GreedyFold,
        };

        for (announced, expected) in [(true, GreedyFold::On), (false, GreedyFold::Off)] {
            set_metal_greedy_argmax(announced);
            let carried = Carry::capture();

            let seen = std::thread::spawn(move || {
                let _adopted = carried.adopt();
                (greedy_fold_setting(), metal_greedy_argmax_active())
            })
            .join()
            .expect("worker thread");

            assert_eq!(
                seen,
                (expected, announced),
                "the worker must run with the setting its submitter chose"
            );
        }
    }

    /// And the adopting thread gets its own setting back, so the next
    /// job on a reused worker does not inherit this one's request.
    ///
    /// Sabotage: `drop` the `GreedyFoldGuard` inside `Carry::adopt`
    /// instead of storing it in `Adopted` (the field then holds `()`),
    /// and the setting is already gone at the first assertion.
    #[cfg(feature = "metal")]
    #[test]
    fn an_adopted_setting_does_not_outlive_its_job() {
        use ferrox_metal::greedy_fold::{greedy_fold_setting, set_metal_greedy_argmax, GreedyFold};

        set_metal_greedy_argmax(false);
        let mut carry = Carry::capture();
        carry.greedy_fold = GreedyFold::On;
        {
            let _adopted = carry.adopt();
            assert_eq!(greedy_fold_setting(), GreedyFold::On);
        }
        assert_eq!(
            greedy_fold_setting(),
            GreedyFold::Off,
            "a worker is reused, so the borrowed setting must be given back"
        );
    }

    /// The verdict table covers every backend this build knows about.
    ///
    /// `Backend::ALL` is generated from the same one table the enum is,
    /// so this walks the real set rather than a restatement of it. The
    /// match in [`promotable`] has no `_` arm, which is what makes a new
    /// row a compile error; this asserts the other half, that the two
    /// backends promotion is claimed for are the two that were checked.
    ///
    /// Sabotage: add `Backend::Cuda` to the expected set.
    #[test]
    fn only_the_backends_whose_thread_affine_state_is_carried_are_promoted() {
        let promoted: Vec<Backend> = Backend::ALL
            .iter()
            .copied()
            .filter(|&b| promotable(b))
            .collect();
        assert_eq!(
            promoted,
            vec![Backend::Cpu, Backend::Metal],
            "promotion is claimed only where it was proven; see this \
             module's verdict table"
        );
    }
}
