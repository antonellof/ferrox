//! Which MoE layers decode on the CPU.
//!
//! The `q*` split ([`crate::qstar`]) decides how a *step* divides its
//! misses. This module decides something coarser and more permanent:
//! which whole layers never use the GPU expert path at all, because
//! their weights could not be page-locked for DMA in the first place.
//!
//! That is a host-memory question, not a bandwidth one. Pinning host
//! memory so the GPU can DMA from it is a scarce, OS-wide resource; on
//! some systems it is capped near half of RAM. A model whose expert
//! banks exceed that cap cannot have every layer pinned, so some layers
//! must be served the other way -- read as ordinary pageable memory by
//! CPU threads.
//!
//! # Head and tail, not a contiguous block
//!
//! [`auto_cpu_layers`] picks from **both ends**. Expert-cache miss
//! rates across a transformer's layers are U-shaped: the first and last
//! layers route more diffusely (their residuals carry the least
//! task-specific structure), so they hit least and benefit least from
//! GPU residency. Handing the middle layers to the GPU cache and the
//! ends to the CPU therefore costs the least throughput per byte of
//! pinning saved. A contiguous prefix would give up the middle layers,
//! which are exactly the ones the cache serves well.
//!
//! Ported 1:1 from FreeToken's `engine/engine.py` (`_parse_cpu_layers_spec`,
//! `_auto_cpu_layers`) (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::BTreeSet;

/// A CPU-layer spec that does not name a valid set of layers.
#[derive(Debug, Clone, PartialEq)]
pub enum CpuLayerSpecError {
    /// A layer index outside the model.
    LayerOutOfRange { index: i64, num_layers: usize },
    /// A count larger than the model has layers.
    CountOutOfRange { count: i64, num_layers: usize },
    /// A fraction outside `[0, 1]`.
    FractionOutOfRange(f64),
    /// The text is not a layer list, a count, or a fraction.
    Unparsable(String),
}

impl std::fmt::Display for CpuLayerSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpuLayerSpecError::LayerOutOfRange { index, num_layers } => {
                write!(f, "layer {index} is outside a model of {num_layers} layers")
            }
            CpuLayerSpecError::CountOutOfRange { count, num_layers } => write!(
                f,
                "{count} CPU layers is more than the model's {num_layers}"
            ),
            CpuLayerSpecError::FractionOutOfRange(fraction) => {
                write!(f, "a CPU-layer fraction must be in [0, 1], got {fraction}")
            }
            CpuLayerSpecError::Unparsable(text) => write!(
                f,
                "could not read {text:?} as a layer list (`3,7,11`), a count (`8`), or a fraction (`0.5`)"
            ),
        }
    }
}

impl std::error::Error for CpuLayerSpecError {}

/// Read a `--moe-cpu-layers` spec.
///
/// Three shapes, told apart by punctuation:
///
/// - `3,7,11` -- exactly these layers;
/// - `0.5` -- this fraction of the model, evenly strided;
/// - `8` -- this many layers, evenly strided.
///
/// Striding rather than taking a prefix keeps the CPU layers spread
/// through the stack, so the CPU work interleaves with GPU work instead
/// of arriving in one lump that nothing can overlap with.
pub fn parse_cpu_layers_spec(
    spec: &str,
    num_layers: usize,
) -> Result<BTreeSet<u32>, CpuLayerSpecError> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Ok(BTreeSet::new());
    }
    if spec.contains(',') {
        let mut layers = BTreeSet::new();
        for part in spec.split(',') {
            let part = part.trim();
            let index: i64 = part
                .parse()
                .map_err(|_| CpuLayerSpecError::Unparsable(part.to_string()))?;
            if index < 0 || index >= num_layers as i64 {
                return Err(CpuLayerSpecError::LayerOutOfRange { index, num_layers });
            }
            layers.insert(index as u32);
        }
        return Ok(layers);
    }
    let count = if spec.contains('.') {
        let fraction: f64 = spec
            .parse()
            .map_err(|_| CpuLayerSpecError::Unparsable(spec.to_string()))?;
        if !(0.0..=1.0).contains(&fraction) {
            return Err(CpuLayerSpecError::FractionOutOfRange(fraction));
        }
        round_half_even(fraction * num_layers as f64)
    } else {
        let count: i64 = spec
            .parse()
            .map_err(|_| CpuLayerSpecError::Unparsable(spec.to_string()))?;
        if count < 0 || count > num_layers as i64 {
            return Err(CpuLayerSpecError::CountOutOfRange { count, num_layers });
        }
        count
    };
    Ok(strided_layers(count as usize, num_layers))
}

/// `count` layers spread evenly across `num_layers`.
pub fn strided_layers(count: usize, num_layers: usize) -> BTreeSet<u32> {
    if count == 0 {
        return BTreeSet::new();
    }
    (0..count)
        .map(|i| round_half_even((i * num_layers) as f64 / count as f64) as u32)
        .collect()
}

/// The layers to serve on the CPU when the expert banks do not fit the
/// host's page-locking budget.
///
/// `None` means "no cap applies, or the banks fit": every layer can be
/// pinned and served from the GPU expert cache. Otherwise enough layers
/// are moved to the CPU that the remaining pinned bytes fit, taken from
/// both ends of the stack for the reason in the module docs.
pub fn auto_cpu_layers(
    num_layers: usize,
    bank_bytes: u64,
    pin_budget_bytes: Option<u64>,
) -> BTreeSet<u32> {
    let Some(budget) = pin_budget_bytes else {
        return BTreeSet::new();
    };
    if bank_bytes == 0 || bank_bytes <= budget {
        return BTreeSet::new();
    }
    let unpinnable = 1.0 - (budget as f64 / bank_bytes as f64);
    let n = (unpinnable * num_layers as f64).ceil() as usize;
    let n = n.min(num_layers);
    let head = n.div_ceil(2);
    let mut layers: BTreeSet<u32> = (0..head as u32).collect();
    layers.extend(((num_layers - (n - head)) as u32)..num_layers as u32);
    layers
}

/// Python's `round`: halves go to the nearest even integer.
///
/// Ported rather than replaced by `f64::round` (which rounds halves
/// away from zero) so a stride lands on the same layers here as
/// upstream -- a one-layer difference silently changes which experts a
/// deployment serves on which device.
///
/// Public so `frink-models`' DSV4 window tier sizes by the same rule:
/// two copies would be free to drift, and a half-page disagreement
/// between placement and sizing is a page of window the budget did not
/// buy.
pub fn round_half_even(value: f64) -> i64 {
    let floor = value.floor();
    let diff = value - floor;
    let floor = floor as i64;
    match diff.partial_cmp(&0.5) {
        Some(std::cmp::Ordering::Less) => floor,
        Some(std::cmp::Ordering::Greater) => floor + 1,
        // Exactly .5: pick the even neighbour.
        _ => {
            if floor % 2 == 0 {
                floor
            } else {
                floor + 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[u32]) -> BTreeSet<u32> {
        items.iter().copied().collect()
    }

    #[test]
    fn an_explicit_list_names_exactly_those_layers() {
        assert_eq!(
            parse_cpu_layers_spec("3,7,11", 40).unwrap(),
            set(&[3, 7, 11])
        );
        assert_eq!(parse_cpu_layers_spec(" 3 , 7 ", 40).unwrap(), set(&[3, 7]));
        assert_eq!(parse_cpu_layers_spec("", 40).unwrap(), BTreeSet::new());
        assert_eq!(parse_cpu_layers_spec("   ", 40).unwrap(), BTreeSet::new());
    }

    /// A count strides through the stack rather than taking a prefix,
    /// so CPU work interleaves with GPU work.
    #[test]
    fn a_count_is_spread_evenly_through_the_stack() {
        assert_eq!(
            parse_cpu_layers_spec("8", 40).unwrap(),
            set(&[0, 5, 10, 15, 20, 25, 30, 35])
        );
        assert_eq!(parse_cpu_layers_spec("0", 40).unwrap(), BTreeSet::new());
        assert_eq!(parse_cpu_layers_spec("40", 40).unwrap().len(), 40);
    }

    #[test]
    fn a_fraction_is_a_count_of_the_model() {
        assert_eq!(parse_cpu_layers_spec("0.5", 40).unwrap().len(), 20);
        assert_eq!(parse_cpu_layers_spec("1.0", 40).unwrap().len(), 40);
        assert_eq!(parse_cpu_layers_spec("0.0", 40).unwrap(), BTreeSet::new());
    }

    #[test]
    fn a_spec_that_names_layers_the_model_lacks_is_refused() {
        assert!(matches!(
            parse_cpu_layers_spec("40,1", 40),
            Err(CpuLayerSpecError::LayerOutOfRange { index: 40, .. })
        ));
        assert!(matches!(
            parse_cpu_layers_spec("-1", 40),
            Err(CpuLayerSpecError::CountOutOfRange { count: -1, .. })
        ));
        assert!(matches!(
            parse_cpu_layers_spec("99", 40),
            Err(CpuLayerSpecError::CountOutOfRange { count: 99, .. })
        ));
        assert!(matches!(
            parse_cpu_layers_spec("1.5", 40),
            Err(CpuLayerSpecError::FractionOutOfRange(_))
        ));
        assert!(matches!(
            parse_cpu_layers_spec("half", 40),
            Err(CpuLayerSpecError::Unparsable(_))
        ));
    }

    /// No cap, or banks that fit it, means every layer stays on the GPU
    /// path.
    #[test]
    fn a_model_that_fits_the_pin_budget_keeps_every_layer_on_the_gpu() {
        assert_eq!(auto_cpu_layers(40, 8 << 30, None), BTreeSet::new());
        assert_eq!(
            auto_cpu_layers(40, 8 << 30, Some(16 << 30)),
            BTreeSet::new()
        );
        assert_eq!(auto_cpu_layers(40, 0, Some(1 << 30)), BTreeSet::new());
    }

    /// Half the banks unpinnable moves half the layers, taken from both
    /// ends because that is where the expert cache helps least.
    #[test]
    fn an_over_budget_model_gives_up_layers_from_both_ends() {
        let layers = auto_cpu_layers(40, 16 << 30, Some(8 << 30));
        assert_eq!(layers.len(), 20);
        assert!(layers.contains(&0) && layers.contains(&9));
        assert!(layers.contains(&39) && layers.contains(&30));
        assert!(
            !layers.contains(&15) && !layers.contains(&20),
            "the middle layers keep their GPU residency"
        );
    }

    #[test]
    fn a_model_far_over_budget_moves_every_layer() {
        let layers = auto_cpu_layers(8, 100 << 30, Some(1 << 30));
        assert_eq!(layers.len(), 8);
    }

    /// An odd count keeps the extra layer at the head, so the two ends
    /// never overlap and double-count.
    #[test]
    fn an_odd_count_splits_head_heavy_without_overlapping() {
        // A quarter of the banks unpinnable over ten layers: 2.5 -> 3.
        let layers = auto_cpu_layers(10, 1024, Some(768));
        assert_eq!(layers, set(&[0, 1, 9]));
    }

    #[test]
    fn halves_round_the_way_python_does() {
        assert_eq!(round_half_even(0.5), 0);
        assert_eq!(round_half_even(1.5), 2);
        assert_eq!(round_half_even(2.5), 2);
        assert_eq!(round_half_even(2.4), 2);
        assert_eq!(round_half_even(2.6), 3);
    }
}
