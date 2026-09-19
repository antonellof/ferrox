//! Summary statistics over samples that may be missing.
//!
//! Two functions, both of which exist because the obvious version of
//! them reports a number nobody measured. They live here, and not
//! beside either caller, because both `frink-server`'s serving
//! telemetry and `frink-cli`'s benchmark client summarise latency
//! samples, and two copies of a percentile definition are free to
//! disagree about what a p95 means.
//!
//! Ported from FreeToken (Apache-2.0); see docs/THIRD_PARTY_NOTICES.md.

/// The `p`-th percentile of `values`, nearest-rank.
///
/// `p` is a percentage in `[0, 100]`. The result is always a value that
/// is actually in `values`: over a handful of requests an interpolated
/// percentile reports a latency nothing measured, and a UI showing it
/// beside a request list invites the reader to look for the row it came
/// from. Interpolated percentiles are fine for continuous data and
/// wrong for a window of twelve requests.
///
/// `None` for an empty input -- a percentile of nothing is not zero.
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let p = p.clamp(0.0, 100.0);
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    Some(sorted[index])
}

/// The mean of the values that exist.
///
/// `None` when none do. The distinction is the whole function: a
/// non-streamed request has no time-to-first-token, and averaging those
/// in as zero drags the mean toward zero in exact proportion to how many
/// clients did not stream -- which reads as the server getting faster.
pub fn mean_of_present<I: IntoIterator<Item = Option<f64>>>(values: I) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.into_iter().flatten() {
        sum += value;
        count += 1;
    }
    (count > 0).then(|| sum / count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nearest-rank always names a real observation, which is the whole
    /// reason it is used here over an interpolating definition.
    #[test]
    fn a_percentile_is_always_a_value_that_was_actually_measured() {
        let values = [10.0, 20.0, 30.0, 40.0];
        for p in [0.0, 25.0, 50.0, 75.0, 95.0, 100.0] {
            let got = percentile(&values, p).expect("non-empty");
            assert!(values.contains(&got), "p{p} produced {got}");
        }
        assert_eq!(percentile(&values, 50.0), Some(20.0));
        assert_eq!(percentile(&values, 95.0), Some(40.0));
        assert_eq!(percentile(&values, 0.0), Some(10.0));
    }

    #[test]
    fn a_percentile_of_nothing_is_none_rather_than_zero() {
        assert_eq!(percentile(&[], 95.0), None);
        assert_eq!(percentile(&[f64::NAN], 95.0), None);
    }

    #[test]
    fn a_mean_ignores_absent_values_instead_of_reading_them_as_zero() {
        assert_eq!(
            mean_of_present([Some(100.0), None, Some(300.0), None]),
            Some(200.0)
        );
        assert_eq!(mean_of_present([None, None]), None);
        assert_eq!(mean_of_present(Vec::<Option<f64>>::new()), None);
    }
}
