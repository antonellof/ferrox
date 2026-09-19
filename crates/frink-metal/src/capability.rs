//! Real (not guessed) Metal device detection, mirroring
//! `frink_cuda::capability::HardwareProfile`'s shape: a plain struct
//! that's always constructible, with GPU fields honestly zeroed/false
//! unless built with `--features metal` on macOS and a real device is
//! present.

/// Snapshot of Metal GPU availability. Always constructible; `available`
/// is only ever `true` when built with `--features metal` on macOS and
/// `MTLCreateSystemDefaultDevice()` actually returns a device.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetalProfile {
    pub available: bool,
    pub device_name: Option<String>,
    /// `MTLDevice.recommendedMaxWorkingSetSize` -- Apple's own answer
    /// for how many GPU-resident bytes this process should hold. `0`
    /// when no device was found (or when built without `metal`), same
    /// convention as the fields above. See
    /// `crate::gpu::probe_recommended_working_set_bytes` for what it
    /// does and does not promise.
    pub recommended_working_set_bytes: u64,
}

impl MetalProfile {
    /// Probe the machine. Cheap: on macOS this just asks the system for
    /// the default Metal device's name, no buffers or kernels involved.
    pub fn detect() -> Self {
        #[cfg(feature = "metal")]
        {
            match crate::gpu::probe() {
                Some(name) => MetalProfile {
                    available: true,
                    device_name: Some(name),
                    recommended_working_set_bytes: crate::gpu::probe_recommended_working_set_bytes(
                    )
                    .unwrap_or(0),
                },
                None => MetalProfile::default(),
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            MetalProfile::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_never_panics() {
        let _ = MetalProfile::detect();
    }

    /// Runs in both worlds: a host with no Metal device must report a
    /// zero working set, and a host with one must report a plausible
    /// non-zero ceiling. Asserting only one of those would be testing
    /// the machine, not the probe.
    #[test]
    fn working_set_is_zero_without_a_device_and_plausible_with_one() {
        let profile = MetalProfile::detect();
        if profile.available {
            assert!(
                profile.recommended_working_set_bytes > 128 * 1024 * 1024,
                "a real Metal device recommends more than 128 MiB, got {}",
                profile.recommended_working_set_bytes
            );
        } else {
            assert_eq!(profile.recommended_working_set_bytes, 0);
        }
    }

    #[test]
    #[cfg(not(feature = "metal"))]
    fn without_metal_feature_profile_always_reports_unavailable() {
        let profile = MetalProfile::detect();
        assert!(!profile.available);
        assert_eq!(profile.device_name, None);
        assert_eq!(profile.recommended_working_set_bytes, 0);
    }
}
