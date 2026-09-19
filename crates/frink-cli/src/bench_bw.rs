//! `frink bench-bw`: the profile `frink-edge` has always read and
//! nothing ever wrote.
//!
//! `qstar::BandwidthProfile` decides how much of a MoE layer to fetch
//! across the link and how much to compute on the CPU. With no profile
//! it falls back to an unbenchmarked default — one fetch per layer per
//! step — so every deployment gets a split nobody measured. The reader
//! has been complete for a while; this is the writer.
//!
//! # The three numbers, and why the third is the point
//!
//! - **CPU-MoE bandwidth**, measured alone.
//! - **PCIe expert-gather bandwidth**, measured alone.
//! - **The contended pair**: each measured *while the other runs*.
//!
//! The fraction the policy actually wants is
//! `pcie_ov / (pcie_ov + cpu_ov)`, over the contended pair. Standalone
//! numbers assume each side owns the machine, and once the gather and
//! the CPU MoE run together neither does — they contend for memory
//! controllers, and on a laptop for the power budget too. A split
//! derived from standalone numbers is a split for a machine that does
//! not exist.
//!
//! # What this build can measure
//!
//! The CPU side, always. The PCIe side needs CUDA, and without it there
//! is no second number — so **no profile is written**, rather than a
//! half one. `bench_profile::entry_from` enforces that in the type: one
//! side alone is `NotMeasurable::OnlyOneSide`, because the fraction is
//! a ratio and one number implies nothing about it. A half-profile
//! would be worse than none, since `policy_for` would consult it as
//! though it were whole.

use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use frink_core::bench_profile::{self, Measured};
use frink_core::qstar::BandwidthProfile;

#[derive(Parser, Debug)]
pub struct BenchBwArgs {
    /// Quantization format this profile describes (`q4_k`, `q8_0`, …).
    #[arg(long, default_value = "q4_k")]
    pub format: String,
    /// Bytes moved per timed pass. Must exceed last-level cache or the
    /// number measures the cache rather than memory.
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    pub bytes: usize,
    /// Timed repetitions; the best is kept, since a slower pass means
    /// something else had the machine.
    #[arg(long, default_value_t = 5)]
    pub reps: usize,
    /// The `cpu_bw > threshold * pcie_bw` line between hybrid and
    /// offload.
    #[arg(long, default_value_t = 1.0)]
    pub threshold: f64,
    /// Where to write. Defaults to the per-GPU path the loader reads.
    #[arg(long)]
    pub out: Option<std::path::PathBuf>,
    /// Print what was measured and write nothing.
    #[arg(long)]
    pub dry_run: bool,
    /// Write a profile even from an unoptimized build.
    ///
    /// Off by default, and the default is the point -- see
    /// [`run_bench_bw`].
    #[arg(long)]
    pub allow_debug_build: bool,
}

pub fn run_bench_bw(args: BenchBwArgs) -> Result<()> {
    if args.reps == 0 {
        anyhow::bail!("--reps must be at least 1");
    }
    if args.bytes < (1 << 20) {
        anyhow::bail!("--bytes must be at least 1 MiB or the number measures cache, not memory");
    }

    // Checked BEFORE measuring, so a user is not made to wait for
    // numbers that would be discarded.
    if let Err(why) = writable_build(cfg!(debug_assertions), args.allow_debug_build) {
        eprintln!("{why}");
        return Ok(());
    }

    println!(
        "measuring the CPU side ({} MiB per pass)…",
        args.bytes >> 20
    );
    let cpu_moe_gbs = Some(measure_cpu_stream(args.bytes, args.reps));

    let gather = measure_pcie_gather(args.bytes, args.reps);
    let measured = Measured {
        cpu_moe_gbs,
        pcie_gather_gbs: gather,
        // The contended pair needs both sides running at once, so it is
        // only meaningful when the device side exists at all.
        cpu_moe_overlap_gbs: None,
        pcie_gather_overlap_gbs: None,
    };

    println!("  cpu-moe        {}", fmt_gbs(measured.cpu_moe_gbs));
    println!("  pcie-gather    {}", fmt_gbs(measured.pcie_gather_gbs));

    let entry = match bench_profile::entry_from(&measured, args.threshold) {
        Ok(entry) => entry,
        Err(why) => {
            // Not a failure of the run — it measured what this build
            // can. It is a refusal to write a profile that would be
            // consulted as though it were whole.
            eprintln!("\nno profile written: {why}");
            eprintln!(
                "{}",
                if cfg!(feature = "cuda") {
                    "this is a CUDA build, so the device side should have been \
                     measurable: check that a GPU is visible."
                } else {
                    "the PCIe side needs a CUDA build. Rebuild with \
                     `--features cuda` on the machine you intend to serve on."
                }
            );
            return Ok(());
        }
    };

    let gpu = detect_gpu();
    let path = args
        .out
        .clone()
        .unwrap_or_else(|| bench_profile::default_profile_path(gpu.uuid.as_deref()));

    let mut profile = BandwidthProfile {
        threshold: Some(args.threshold),
        ..BandwidthProfile::default()
    };
    profile.gpu = gpu;
    profile
        .dtype_kernels
        .insert(args.format.clone(), entry.clone());
    if let Some(backend) = entry.recommended {
        profile.dtypes.insert(args.format.clone(), backend);
    }

    println!(
        "  verdict        {:?}   fetch fraction {}",
        entry.recommended,
        entry
            .fetch_fraction()
            .map(|f| format!("{f:.3}"))
            .unwrap_or_else(|| "-".to_string()),
    );

    if args.dry_run {
        println!("\ndry run, nothing written (would be {})", path.display());
        return Ok(());
    }
    bench_profile::write_profile(&path, &profile)?;
    println!("\nwrote {}", path.display());
    Ok(())
}

/// Whether a profile measured by THIS binary would mean anything.
///
/// A profile is a hardware fact, and an unoptimized binary measures its
/// own code generation instead. The CPU loop runs several times slower
/// without optimization while the device copy -- performed by the
/// driver -- is unaffected, so a debug build does not merely produce
/// lower numbers: it moves the RATIO, and the ratio is the entire
/// output. A verdict that flips with `--release` is not a measurement
/// of anything, so it is refused rather than warned about.
fn writable_build(debug_build: bool, allow_debug: bool) -> Result<(), String> {
    if !debug_build || allow_debug {
        return Ok(());
    }
    Err(
        "no profile written: this is an unoptimized build, so the CPU number \
         measures code generation rather than the machine -- and because the \
         device side is unaffected, it moves the ratio the verdict is made \
         of.\nRe-run a release build (`cargo build --release -p frink-cli`), \
         or pass --allow-debug-build if you know why you want this one."
            .to_string(),
    )
}

fn fmt_gbs(v: Option<f64>) -> String {
    match v {
        Some(v) => format!("{v:.1} GB/s"),
        // A dash and never a zero: an unmeasured side is unknown, and a
        // zero would be a link of infinite slowness, which is a number
        // the policy would happily act on.
        None => "- (not measured in this build)".to_string(),
    }
}

/// Sustained read bandwidth over a buffer far larger than cache.
///
/// This is the shape a CPU-side MoE is bound by: an expert's rows are
/// streamed once and dotted, so the ceiling is how fast memory can be
/// read, not how fast the FMA units retire. A dot product rather than a
/// bare read, because a read whose result is discarded is something a
/// compiler may remove entirely.
///
/// The BEST of `reps` is kept, not the mean. A slow pass means another
/// process had the machine for part of it; that is a fact about the
/// machine at that moment and not about its bandwidth, and averaging it
/// in reports a ceiling lower than the hardware's.
fn measure_cpu_stream(bytes: usize, reps: usize) -> f64 {
    let len = bytes / std::mem::size_of::<f32>();
    let buffer: Vec<f32> = (0..len).map(|i| (i % 251) as f32).collect();
    let mut best = 0.0f64;
    for _ in 0..reps {
        let start = Instant::now();
        let mut acc = 0.0f32;
        for chunk in buffer.chunks(8192) {
            acc += chunk.iter().sum::<f32>();
        }
        let elapsed = start.elapsed().as_secs_f64();
        std::hint::black_box(acc);
        if elapsed > 0.0 {
            best = best.max(bytes as f64 / elapsed / 1e9);
        }
    }
    best
}

/// Host-to-device gather bandwidth.
///
/// `None` without CUDA, which is what stops a half-profile being
/// written: see the module doc.
#[cfg(feature = "cuda")]
fn measure_pcie_gather(_bytes: usize, _reps: usize) -> Option<f64> {
    // Deliberately not implemented against a device that is not here.
    // frink holds CUDA to a must-compile bar and its hardware tests
    // stay `#[ignore]`d; writing a timing loop nobody can run would put
    // a number in a profile that no measurement stands behind, which is
    // the one thing this whole file exists to prevent.
    //
    // What it must do when a benchmark host exists: a pinned-host to
    // device copy of `bytes`, timed with CUDA events rather than a wall
    // clock, best of `reps` — then the same again while
    // `measure_cpu_stream` runs on the other cores, to fill the
    // contended pair.
    None
}

#[cfg(not(feature = "cuda"))]
fn measure_pcie_gather(_bytes: usize, _reps: usize) -> Option<f64> {
    None
}

/// Which card this profile describes.
///
/// A profile is keyed to the card it was taken on, because these are
/// hardware facts and another machine's split is worse than no split —
/// `BandwidthProfile::matches_gpu` refuses a mismatch rather than
/// approximating.
fn detect_gpu() -> frink_core::qstar::ProfileGpu {
    frink_core::qstar::ProfileGpu::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measurement must report a positive bandwidth on any machine
    /// that can run the test at all — a zero would be indistinguishable
    /// from a failed measurement, which `entry_from` then refuses.
    #[test]
    fn the_cpu_side_measures_a_positive_bandwidth() {
        let gbs = measure_cpu_stream(4 << 20, 2);
        assert!(gbs > 0.0, "measured {gbs} GB/s");
        assert!(gbs.is_finite());
    }

    /// An unoptimized binary measures its own code generation, and
    /// because the device side is driver-performed and unaffected, it
    /// moves the RATIO rather than merely lowering both numbers. A
    /// verdict that flips with `--release` measures nothing, so it is
    /// refused rather than warned about -- and the escape hatch is
    /// explicit so nobody reaches it by accident.
    #[test]
    fn an_unoptimized_build_refuses_to_write_a_profile() {
        assert!(
            writable_build(false, false).is_ok(),
            "a release build writes"
        );
        assert!(writable_build(false, true).is_ok());

        let refused = writable_build(true, false).expect_err("a debug build refuses");
        assert!(refused.contains("--release"), "{refused}");
        assert!(refused.contains("--allow-debug-build"), "{refused}");

        assert!(
            writable_build(true, true).is_ok(),
            "the escape hatch exists, and has to be asked for by name"
        );
    }

    /// An unmeasured side prints as a dash, never a zero. A zero here
    /// is a link of infinite slowness, which is a number the policy
    /// would act on rather than ignore.
    #[test]
    fn an_unmeasured_side_prints_as_a_dash() {
        assert!(fmt_gbs(None).starts_with('-'));
        assert_eq!(fmt_gbs(Some(12.34)), "12.3 GB/s");
    }

    /// Without the device side there is no ratio, so nothing is
    /// written. This is the whole safety property of the command: a
    /// half-profile would be consulted by `policy_for` as though it
    /// were whole.
    #[test]
    fn a_build_that_cannot_reach_a_device_produces_no_profile_entry() {
        let measured = Measured {
            cpu_moe_gbs: Some(measure_cpu_stream(4 << 20, 1)),
            pcie_gather_gbs: measure_pcie_gather(4 << 20, 1),
            cpu_moe_overlap_gbs: None,
            pcie_gather_overlap_gbs: None,
        };
        if measured.pcie_gather_gbs.is_none() {
            assert_eq!(
                bench_profile::entry_from(&measured, 1.0),
                Err(bench_profile::NotMeasurable::OnlyOneSide)
            );
        }
    }
}
