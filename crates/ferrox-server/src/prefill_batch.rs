//! How many prompt tokens go through one forward pass: llama.cpp's
//! `-b` / `-ub`, and the two environment variables ferrox already had
//! for the same number.
//!
//! # Why this module exists rather than two `set_var` calls
//!
//! ferrox has two prefill paths and each grew its own spelling of the
//! same knob:
//!
//! - the private `generate` loop reads `FERROX_CHUNKED_PREFILL`
//!   ([`generate::forward_prompt_batch`](crate::generate));
//! - the continuous-batching scheduler reads `FERROX_CB_PREFILL_CHUNK`
//!   ([`BatcherConfig::from_env`](crate::serving::batch::BatcherConfig)).
//!
//! That is this repo's dominant bug shape -- two structures that must
//! agree about one thing, with nothing enforcing it. An operator who
//! set one and served on the other path got the default and no warning.
//! So the names live here, in [`PREFILL_CHUNK_ENV_KEYS`], and both
//! readers index that array instead of writing the string themselves:
//! adding a third path means adding an entry, not remembering one.
//!
//! # The `-b` / `-ub` mapping
//!
//! llama.cpp splits the number in two: `-b`/`--batch-size` is the
//! *logical* maximum submitted to `llama_decode`, `-ub`/`--ubatch-size`
//! the *physical* maximum computed at once
//! (`common/arg.cpp:1616-1628`). ferrox has one stage, so it has one
//! number, and the value it takes is llama.cpp's own resolution of the
//! two -- `cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch
//! == 0 ? params.n_batch : params.n_ubatch)`
//! (`src/llama-context.cpp:265`), which is [`effective_chunk`].
//!
//! Naming neither flag leaves both paths on their own defaults, so a
//! command that does not mention batching is not silently re-tuned.

/// The environment variables that carry the prefill chunk, in the order
/// [`crate::generate`] and [`crate::serving::batch`] read them.
///
/// One array rather than two literals: see the module note. Both
/// readers index it, so a rename is a compile error at both sites
/// instead of a silent default at one of them.
pub(crate) const PREFILL_CHUNK_ENV_KEYS: [&str; 2] =
    ["FERROX_CHUNKED_PREFILL", "FERROX_CB_PREFILL_CHUNK"];

/// Index of the private `generate` path's spelling.
pub(crate) const PRIVATE_PATH_KEY: usize = 0;
/// Index of the continuous-batching scheduler's spelling.
pub(crate) const BATCH_PATH_KEY: usize = 1;

/// llama.cpp's resolution of `-b` and `-ub` into the one number ferrox
/// keeps: the smaller of whichever the operator actually named, and
/// `None` when neither was named.
///
/// Mirrors `src/llama-context.cpp:265`. `-ub` alone is the physical
/// batch; `-b` alone stands in for it (llama.cpp's `params.n_ubatch ==
/// 0` branch); both together clamp to the smaller, because a physical
/// batch larger than the logical one cannot be submitted.
pub(crate) fn effective_chunk(batch: Option<usize>, ubatch: Option<usize>) -> Option<usize> {
    match (batch, ubatch) {
        (Some(b), Some(u)) => Some(b.min(u)),
        (Some(b), None) => Some(b),
        (None, Some(u)) => Some(u),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two spellings are one concept. Before this array existed,
    /// `FERROX_CHUNKED_PREFILL` and `FERROX_CB_PREFILL_CHUNK` were
    /// independent literals at their two readers, so setting the one
    /// that belonged to the other serving path silently did nothing.
    #[test]
    fn the_two_prefill_chunk_spellings_are_distinct_and_indexed_by_name() {
        assert_eq!(
            PREFILL_CHUNK_ENV_KEYS[PRIVATE_PATH_KEY],
            "FERROX_CHUNKED_PREFILL"
        );
        assert_eq!(
            PREFILL_CHUNK_ENV_KEYS[BATCH_PATH_KEY],
            "FERROX_CB_PREFILL_CHUNK"
        );
        assert_ne!(
            PREFILL_CHUNK_ENV_KEYS[PRIVATE_PATH_KEY], PREFILL_CHUNK_ENV_KEYS[BATCH_PATH_KEY],
            "two readers indexing the same entry would leave one path unconfigurable"
        );
    }

    /// `-ub` larger than `-b` is not a physical batch of `-ub`:
    /// llama.cpp clamps it (`src/llama-context.cpp:265`) and so does
    /// ferrox, or a `-b 64 -ub 512` command would run 512-token chunks
    /// here and 64-token chunks there.
    #[test]
    fn a_ubatch_larger_than_the_batch_clamps_to_the_batch() {
        assert_eq!(effective_chunk(Some(64), Some(512)), Some(64));
        assert_eq!(effective_chunk(Some(512), Some(64)), Some(64));
    }

    /// Either flag alone names the chunk; llama.cpp's `-b` stands in
    /// for an unset `-ub` by the same line.
    #[test]
    fn either_flag_alone_names_the_chunk() {
        assert_eq!(effective_chunk(Some(256), None), Some(256));
        assert_eq!(effective_chunk(None, Some(256)), Some(256));
    }

    /// A command that mentions neither must not re-tune either path:
    /// the private loop's default is "no chunking at all" and the
    /// scheduler's is 128, and picking one for the other is a
    /// behaviour change nobody asked for.
    #[test]
    fn naming_neither_flag_leaves_both_paths_on_their_own_defaults() {
        assert_eq!(effective_chunk(None, None), None);
    }
}
