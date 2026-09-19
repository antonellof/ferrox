//! Writing the two logit vectors a parity run compared, so the
//! comparison can be redone without either engine.
//!
//! Why this exists: `parity` prints ONE number per run, KL(reference ||
//! frink), and that number is a distance between two points without
//! saying where either point is. When the same checkpoint scored DRIFT
//! against one libllama and WRONG against another
//! ([#102](https://github.com/antonellof/frink/issues/102)) there was
//! no way, from parity's output alone, to tell "frink moved" from "the
//! reference moved" — the two hypotheses produce the same printed line.
//!
//! Dumping the vectors makes the third comparison possible, and it is
//! the one that settles it: reference-A against reference-B, with frink
//! out of the experiment entirely. On Qwen2.5-1.5B Q4_K_M that
//! comparison read KL 2.73e-2 between two llama.cpp builds, larger than
//! either build's disagreement with frink — which is a fact about
//! llama.cpp and could not have been discovered by running `parity`
//! more times.
//!
//! The files are raw little-endian f32, the same wire
//! `tools/llama_logits.c` writes, so a dumped frink vector and a
//! dumped reference vector are interchangeable inputs to anything that
//! reads one.

use anyhow::Context;
use std::path::{Path, PathBuf};

/// Suffixes appended to the caller's prefix. Kept in one place because
/// the point of the dump is that a later run can find the earlier run's
/// files, and a suffix spelled twice is a suffix that drifts.
const REFERENCE_SUFFIX: &str = ".llama.f32";
const FRINK_SUFFIX: &str = ".frink.f32";
const TOKENS_SUFFIX: &str = ".tokens.txt";

/// File name for reference `i`. The primary keeps the original
/// `.llama.f32` so a dump taken before `parity` could run two
/// references is still the same file a later one produces.
fn reference_path(prefix: &str, i: usize) -> PathBuf {
    if i == 0 {
        PathBuf::from(format!("{prefix}{REFERENCE_SUFFIX}"))
    } else {
        PathBuf::from(format!("{prefix}.llama{}.f32", i + 1))
    }
}

/// Writes every compared logit vector and the token ids they were
/// computed from.
///
/// The token ids travel with the vectors because they are the only part
/// of the experiment that is not in the file names: two dumps of the
/// same checkpoint taken with different prompts are not comparable, and
/// nothing else would say so.
pub fn write(
    prefix: &str,
    tokens: &[u32],
    references: &[&[f32]],
    frink: &[f32],
) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(references.len() + 2);
    for (i, reference) in references.iter().enumerate() {
        let p = reference_path(prefix, i);
        write_f32(&p, reference)?;
        paths.push(p);
    }

    let fx_path = PathBuf::from(format!("{prefix}{FRINK_SUFFIX}"));
    write_f32(&fx_path, frink)?;
    paths.push(fx_path);

    let tok_path = PathBuf::from(format!("{prefix}{TOKENS_SUFFIX}"));
    let ids = tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    std::fs::write(&tok_path, format!("{ids}\n"))
        .with_context(|| format!("writing {}", tok_path.display()))?;
    paths.push(tok_path);

    Ok(paths)
}

fn write_f32(path: &Path, values: &[f32]) -> anyhow::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &bytes).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dumped vector must read back bit-for-bit, because the whole
    /// use of the dump is comparing it against a vector produced by a
    /// DIFFERENT build months later. A dump that rounds is a dump that
    /// invents the divergence it is meant to measure.
    #[test]
    fn a_dumped_vector_reads_back_bit_for_bit() {
        let dir = std::env::temp_dir().join(format!("frink-dump-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("case").to_string_lossy().into_owned();

        // Values chosen to break a naive round-trip: a denormal, a
        // negative zero, and a value whose f32 -> f64 -> f32 trip is
        // only exact if nothing in between is text.
        let reference = vec![
            0.1f32,
            -0.0,
            f32::MIN_POSITIVE,
            3.402_823_5e38,
            -1.234_567_8e-9,
        ];
        let frink = vec![0.2f32, 1.0, -5.5, 0.0, 7.7];
        let paths = write(&prefix, &[1, 2, 3], &[&reference], &frink).unwrap();
        assert_eq!(paths.len(), 3);

        let back = std::fs::read(&paths[0]).unwrap();
        let read: Vec<f32> = back
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(read.len(), reference.len());
        for (a, b) in read.iter().zip(&reference) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "dumped {b} came back as {a}: the dump is not bit-exact"
            );
        }

        let ids = std::fs::read_to_string(&paths[2]).unwrap();
        assert_eq!(ids.trim(), "1 2 3");

        for p in &paths {
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_dir(&dir);
    }

    /// EVERY vector lands in its own file.
    ///
    /// They are the same length and the same wire format, so a shared
    /// name would silently leave one overwriting another and the
    /// resulting "reference vs reference" KL would be exactly zero —
    /// which reads as "the two builds agree" and is the one answer this
    /// tool must never fabricate. With `--dumper` repeatable there are
    /// now N of them, so this walks the naming rule rather than the
    /// three suffixes it used to be.
    #[test]
    fn no_two_dumped_vectors_share_a_file_name() {
        assert_ne!(REFERENCE_SUFFIX, FRINK_SUFFIX);
        assert_ne!(REFERENCE_SUFFIX, TOKENS_SUFFIX);
        assert_ne!(FRINK_SUFFIX, TOKENS_SUFFIX);

        let dir = std::env::temp_dir().join(format!("frink-dump-names-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("case").to_string_lossy().into_owned();

        // Three references with DIFFERENT contents: if two shared a
        // path the survivor's bytes would give it away.
        let refs: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32, 1.0]).collect();
        let borrowed: Vec<&[f32]> = refs.iter().map(Vec::as_slice).collect();
        let paths = write(&prefix, &[7], &borrowed, &[9.0f32, 1.0]).unwrap();
        assert_eq!(paths.len(), 5, "3 references + frink + the token ids");

        let unique: std::collections::HashSet<_> = paths.iter().collect();
        assert_eq!(
            unique.len(),
            paths.len(),
            "two dumps share a name: {paths:?}"
        );
        for (i, r) in refs.iter().enumerate() {
            let back = std::fs::read(&paths[i]).unwrap();
            assert_eq!(
                f32::from_le_bytes(back[..4].try_into().unwrap()),
                r[0],
                "reference [{i}] landed in {} with someone else's bytes",
                paths[i].display()
            );
        }

        // The primary keeps the name it had before `--dumper` could be
        // repeated, so an old dump and a new one are still the same
        // file.
        assert!(paths[0].to_string_lossy().ends_with(REFERENCE_SUFFIX));

        for p in &paths {
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_dir(&dir);
    }
}
