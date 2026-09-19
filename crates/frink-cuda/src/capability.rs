//! Runtime hardware capability detection: probe once, report a plain
//! struct, and let every performance-relevant decision (thread pool
//! width, SIMD kernel selection, GPU residency) derive from the
//! *detected* machine rather than being hardcoded. Frink
//! today only has a CPU execution path, so `HardwareProfile::detect()`
//! is honest about that: the CUDA fields are always populated (zero /
//! false / None) unless built with `--features cuda`, and even then
//! they report exactly what `frink-cuda`'s device probe finds, no
//! more.
//!
//! Everything in this module is real and testable in any environment,
//! including one with no GPU: CPU core count and SIMD flags are always
//! detectable, and "zero CUDA devices found" is itself a correct,
//! verifiable answer on a CPU-only host, not a stand-in for an
//! untested code path.

/// CPU SIMD instruction-set availability (runtime-detected via
/// `is_x86_feature_detected!`, not compile-time `#[cfg]`), matching
/// the fields `frink_quant`'s kernel dispatch actually checks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SimdCaps {
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub neon: bool,
}

impl SimdCaps {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            SimdCaps {
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                fma: is_x86_feature_detected!("fma"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            SimdCaps {
                avx2: false,
                avx512f: false,
                fma: false,
                neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdCaps::default()
        }
    }

    /// Short human label of the widest available SIMD path frink
    /// actually has a kernel for today (Q8_0/Q4_0 dispatch in
    /// `frink-quant` currently only implements the AVX2+FMA path, so
    /// `avx512f` is reported here as detected-but-unused -- no
    /// AVX-512 kernel exists yet).
    pub fn label(&self) -> &'static str {
        if self.avx2 && self.fma {
            "AVX2+FMA (frink's fastest implemented CPU kernel)"
        } else if self.neon {
            "NEON (frink Q4_K/Q6_K/Q8_0/Q4_0 fused dots live)"
        } else {
            "scalar (no SIMD kernel available for this host)"
        }
    }
}

/// A snapshot of the host's inference-relevant capabilities. CPU/RAM
/// fields are always real; CUDA fields are always present but only
/// ever non-default when built with `--features cuda` on a host that
/// actually has a CUDA-capable device.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu_logical_cores: usize,
    pub host_ram_total_bytes: u64,
    pub simd: SimdCaps,
    pub cuda_available: bool,
    pub cuda_device_count: usize,
    pub cuda_device_name: Option<String>,
    pub cuda_vram_total_bytes: u64,
    /// Free device memory at probe time (`cuMemGetInfo`'s first half).
    /// Zero without `--features cuda` or without a device, exactly like
    /// the other CUDA fields. A memory budget should be drawn against
    /// this rather than the total -- see
    /// `frink_models::device_budget`.
    pub cuda_vram_free_bytes: u64,
}

impl HardwareProfile {
    /// Probe the machine. Cheap and side-effect-free on the CPU side;
    /// the CUDA probe (when built with `--features cuda`) opens a
    /// driver context to enumerate devices and degrades to "no CUDA"
    /// cleanly on hosts without one, in the exact form the fields
    /// below already represent for a CPU-only build.
    pub fn detect() -> Self {
        let cpu_logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let host_ram_total_bytes = detect_total_ram_bytes();
        let simd = SimdCaps::detect();

        #[cfg(feature = "cuda")]
        let (
            cuda_available,
            cuda_device_count,
            cuda_device_name,
            cuda_vram_total_bytes,
            cuda_vram_free_bytes,
        ) = {
            match crate::gpu::probe() {
                Some(info) => (
                    true,
                    info.device_count,
                    info.first_device_name,
                    info.total_vram_bytes,
                    info.free_vram_bytes,
                ),
                None => (false, 0, None, 0, 0),
            }
        };
        #[cfg(not(feature = "cuda"))]
        let (
            cuda_available,
            cuda_device_count,
            cuda_device_name,
            cuda_vram_total_bytes,
            cuda_vram_free_bytes,
        ) = (false, 0, None, 0, 0);

        HardwareProfile {
            cpu_logical_cores,
            host_ram_total_bytes,
            simd,
            cuda_available,
            cuda_device_count,
            cuda_device_name,
            cuda_vram_total_bytes,
            cuda_vram_free_bytes,
        }
    }
}

/// Reads total physical RAM from `/proc/meminfo` on Linux (no external
/// `sysinfo`-style crate dependency, keeping this pure-Rust and
/// dependency-light). Returns 0 on any parse failure or non-Linux host
/// rather than panicking -- this is diagnostic information, not
/// something correctness depends on.
fn detect_total_ram_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let kb: u64 = rest
                        .trim()
                        .trim_end_matches(" kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                    return kb * 1024;
                }
            }
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        // `sysctl -n hw.memsize` rather than a new libc/sysinfo-style
        // dependency, matching this module's existing dependency-light
        // stance for /proc/meminfo above. Real gap found via actually
        // running `frink inspect-plan` on this dev machine: without
        // this, host_ram_total_bytes silently stayed 0 on every macOS
        // host, making --strict always report "DOES NOT FIT" regardless
        // of real available RAM.
        std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_never_panics_and_reports_at_least_one_core() {
        let profile = HardwareProfile::detect();
        assert!(profile.cpu_logical_cores >= 1);
    }

    #[test]
    fn detect_reports_plausible_ram_on_linux_or_macos() {
        let profile = HardwareProfile::detect();
        // Any real Linux or macOS host will always have at least, say,
        // 128 MB, so treat 0 as "couldn't detect" (unsupported OS) and
        // anything absurdly small as a parsing bug, not assert an
        // exact value that would break on other hosts.
        if profile.host_ram_total_bytes > 0 {
            assert!(profile.host_ram_total_bytes > 128 * 1024 * 1024);
        }
    }

    #[test]
    fn simd_caps_label_is_never_empty() {
        let caps = SimdCaps::detect();
        assert!(!caps.label().is_empty());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn without_cuda_feature_profile_always_reports_no_cuda() {
        let profile = HardwareProfile::detect();
        assert!(!profile.cuda_available);
        assert_eq!(profile.cuda_device_count, 0);
        assert_eq!(profile.cuda_device_name, None);
        assert_eq!(profile.cuda_vram_free_bytes, 0);
    }
}
