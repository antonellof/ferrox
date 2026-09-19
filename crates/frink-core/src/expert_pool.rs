//! Device-side homes for the expert slot pool
//! [`crate::expert_slots::ExpertSlots`] governs.
//!
//! `crate::expert_cache` decides which experts are resident and validates the
//! plans that make them so, but holds no device memory by design. This
//! is the other side of that line: the allocations, and the
//! [`SlotDevice`] implementations that write into them.
//!
//! # What is verified, and what is not
//!
//! The policy is verified: every rule about plans, occupancy,
//! attribution of failures and the zero-copy warm step is tested in
//! `crate::expert_slots` on any host.
//!
//! [`CudaExpertPool`] is **compile-verified only**. frink holds CUDA
//! to a must-compile bar and its hardware tests stay `#[ignore]`d, and
//! no benchmark host has run this. It is written out rather than
//! stubbed because its correctness is mostly the type system's to
//! check -- an allocation per slot, a bounds-checked index, a driver
//! copy -- unlike a timing loop, where writing one without a machine to
//! run it on would put a number nobody measured into a profile. Where
//! this file makes a *performance* claim it says so is unmeasured.
//!
//! # One allocation per slot, not one per bank
//!
//! The obvious layout is one contiguous `slots * row_bytes` buffer per
//! bank. This allocates each slot separately instead, for a reason
//! that is not aesthetic: a slot-to-slot copy needs a shared borrow of
//! the source and a mutable borrow of the destination at the same
//! time, and two sub-views of one `CudaSlice` cannot provide that.
//! Separate allocations can, via `split_at_mut`, so the gather path is
//! ordinary safe code rather than raw driver pointer arithmetic
//! written against hardware nobody here can run.
//!
//! The cost is `slots * banks` allocations at startup instead of
//! `banks`. It is paid once, and it buys back the thing each slot is
//! for: one expert row, one device pointer, which is exactly what a
//! matvec launch wants.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::expert_slots::{SlotDevice, SlotGeometry};
#[cfg(feature = "cuda")]
use crate::residency::CopyRoute;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

/// Borrows two distinct slots of one bank at once: the source shared,
/// the destination mutable.
///
/// This is the whole reason a slot is its own allocation. A
/// device-to-device copy needs both borrows live simultaneously, which
/// two sub-views of one buffer cannot give; `split_at_mut` over
/// separate allocations can. Splitting at the HIGHER of the two indices
/// is what puts exactly one of the pair in each half.
///
/// Lives outside the `cuda` feature gate so the index arithmetic --
/// the one part of the copy path that a compiler cannot check and a
/// GPU-less host can -- is exercised by the ordinary test run. Returns
/// `None` when the two are equal (a self-copy, which every caller
/// should have short-circuited) or when either is out of range.
pub fn split_pair<T>(slots: &mut [T], dst: usize, src: usize) -> Option<(&T, &mut T)> {
    if dst == src || dst.max(src) >= slots.len() {
        return None;
    }
    let (low, high) = slots.split_at_mut(dst.max(src));
    if dst < src {
        // low = [0, src), high = [src, ..): the destination is in low.
        let target = &mut low[dst];
        Some((&high[0], target))
    } else {
        // low = [0, dst), high = [dst, ..): the source is in low.
        Some((&low[src], &mut high[0]))
    }
}

/// Expert slots in CUDA device memory.
///
/// Built from the same [`SlotGeometry`] the
/// [`ExpertSlots`](crate::expert_slots::ExpertSlots) governing it was built
/// from, so the two cannot disagree about how many slots exist or
/// how wide a row is.
#[cfg(feature = "cuda")]
pub struct CudaExpertPool {
    dev: Arc<CudaDevice>,
    /// `[bank][slot]`, one allocation per slot -- see the module
    /// docs for why this is not one buffer per bank.
    banks: Vec<Vec<CudaSlice<u8>>>,
    row_bytes: Vec<usize>,
}

#[cfg(feature = "cuda")]
impl CudaExpertPool {
    /// Allocates the whole pool up front.
    ///
    /// Up front and never on demand: the point of a bounded pool is
    /// that its footprint is known before serving starts, and a
    /// pool that grew as experts were routed to would reintroduce
    /// exactly the unbounded device-memory growth it exists to
    /// prevent.
    pub fn new(dev: Arc<CudaDevice>, geometry: &SlotGeometry) -> Result<Self, String> {
        let mut banks = Vec::with_capacity(geometry.banks());
        for &row_bytes in &geometry.row_bytes {
            let mut slots = Vec::with_capacity(geometry.slots);
            for slot in 0..geometry.slots {
                slots.push(dev.alloc_zeros::<u8>(row_bytes).map_err(|e| {
                    format!(
                        "allocating slot {slot} of {} x {row_bytes} bytes: {e:?}",
                        geometry.slots
                    )
                })?);
            }
            banks.push(slots);
        }
        Ok(CudaExpertPool {
            dev,
            banks,
            row_bytes: geometry.row_bytes.clone(),
        })
    }

    /// The device buffer holding one expert row, for a launch that
    /// wants to read it.
    pub fn slot(&self, bank: usize, slot: u32) -> Option<&CudaSlice<u8>> {
        self.banks.get(bank)?.get(slot as usize)
    }

    /// Device bytes this pool holds.
    pub fn bytes(&self) -> u64 {
        self.row_bytes
            .iter()
            .zip(self.banks.iter())
            .map(|(w, slots)| *w as u64 * slots.len() as u64)
            .sum()
    }
}

#[cfg(feature = "cuda")]
impl SlotDevice for CudaExpertPool {
    /// Nothing to arrange: every copy below is already synchronous,
    /// so neither route needs a different mode. A future version
    /// that captures decode into a CUDA graph must start refusing
    /// [`CopyRoute::WholeLayerPageable`] here.
    fn begin_plan(&mut self, route: CopyRoute) -> Result<(), String> {
        let _ = route;
        Ok(())
    }

    /// Copies one expert row from host memory into its slot.
    ///
    /// Synchronous, and that is a real cost rather than a
    /// simplification: it stalls the host once per row where a
    /// pinned staging buffer plus an async copy on a dedicated
    /// stream would not. The unmeasured claim this file will not
    /// make is which of the two is faster on any given machine --
    /// that is what `frink bench-bw` and a benchmark host are for.
    /// The synchronous form is the one whose correctness needs no
    /// hardware to reason about, so it is what lands first.
    fn write_slot(&mut self, bank: usize, dst_slot: u32, src: &[u8]) -> Result<(), String> {
        let dev = Arc::clone(&self.dev);
        let dst = self
            .banks
            .get_mut(bank)
            .and_then(|b| b.get_mut(dst_slot as usize))
            .ok_or_else(|| format!("no slot {dst_slot} in bank {bank}"))?;
        // cudarc `assert_eq!`s the two lengths, and a panic here takes
        // the whole server down mid-decode. `ExpertSlots` already
        // refuses a row of the wrong width, but this type is public and
        // reachable without it, so the mismatch becomes an error the
        // caller can act on rather than an abort.
        if src.len() != dst.len() {
            return Err(format!(
                "bank {bank} slot {dst_slot} is {} bytes, but the row offered is {}",
                dst.len(),
                src.len()
            ));
        }
        dev.htod_sync_copy_into(src, dst)
            .map_err(|e| format!("host-to-device copy into bank {bank} slot {dst_slot}: {e:?}"))
    }

    /// Copies one slot onto another without touching the host.
    ///
    /// `dtod_copy` asserts the two lengths match too, but unlike
    /// [`write_slot`](Self::write_slot) nothing external can make them
    /// differ: every slot in a bank is allocated at that bank's single
    /// `row_bytes`, so the equality is a property of construction.
    fn copy_slot(&mut self, bank: usize, dst_slot: u32, src_slot: u32) -> Result<(), String> {
        if dst_slot == src_slot {
            return Ok(());
        }
        let dev = Arc::clone(&self.dev);
        let slots = self
            .banks
            .get_mut(bank)
            .ok_or_else(|| format!("no bank {bank}"))?;
        let len = slots.len();
        let (source, target) =
            split_pair(slots, dst_slot as usize, src_slot as usize).ok_or_else(|| {
                format!("bank {bank} holds {len} slots, not {dst_slot} <- {src_slot}")
            })?;
        dev.dtod_copy(source, target).map_err(|e| {
            format!("device-to-device copy {src_slot} -> {dst_slot} in bank {bank}: {e:?}")
        })
    }

    /// Nothing is deferred, so there is nothing to complete. If
    /// `write_slot` ever becomes asynchronous this must start
    /// synchronizing the stream and reporting its error -- the
    /// caller forgets every slot a plan wrote when this fails, and
    /// silently succeeding would leave it trusting copies that
    /// never landed.
    fn flush(&mut self) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod split_tests {
    use super::split_pair;

    /// The one line of the device-to-device path a compiler cannot
    /// check: which half of the split holds the source and which the
    /// destination. Getting it backwards copies the wrong slot, and a
    /// wrong expert multiplies without complaining -- so it is checked
    /// here, on a host with no GPU, rather than left to a card nobody
    /// has run this on.
    #[test]
    fn the_split_hands_back_the_source_and_destination_the_caller_asked_for() {
        for (dst, src) in [(0usize, 3usize), (3, 0), (1, 2), (2, 1), (0, 1), (3, 2)] {
            let mut slots: Vec<u32> = (0..4).collect();
            let (source, target) = split_pair(&mut slots, dst, src).expect("distinct, in range");
            assert_eq!(*source, src as u32, "source for {dst} <- {src}");
            assert_eq!(*target, dst as u32, "destination for {dst} <- {src}");
            *target = *source;
            assert_eq!(slots[dst], src as u32, "the copy landed in {dst}");
            assert_eq!(slots[src], src as u32, "and left the source alone");
        }
    }

    /// A self-copy and an out-of-range slot both come back `None`
    /// rather than panicking: the caller turns them into an error that
    /// names the bank, and a panic in a decode step takes the server
    /// down instead.
    #[test]
    fn a_self_copy_or_an_out_of_range_slot_is_not_a_pair() {
        let mut slots: Vec<u32> = (0..4).collect();
        assert!(split_pair(&mut slots, 2, 2).is_none());
        assert!(split_pair(&mut slots, 4, 0).is_none());
        assert!(split_pair(&mut slots, 0, 4).is_none());
        assert!(split_pair::<u32>(&mut [], 0, 1).is_none());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::expert_cache::{CopyPlan, ExpertId, GatherPlan};
    use crate::expert_slots::{ExpertRows, ExpertSlots};

    /// Host rows whose bytes name the expert they belong to, so a slot
    /// read back from the device can be checked for *which* expert it
    /// holds rather than only for having been written.
    struct NamedRows {
        rows: Vec<Vec<u8>>,
    }

    impl NamedRows {
        fn new(experts: usize, width: usize) -> Self {
            let rows = (0..experts)
                .map(|e| (0..width).map(|i| (e as u8) << 4 | i as u8).collect())
                .collect();
            NamedRows { rows }
        }
    }

    impl ExpertRows for NamedRows {
        fn row(&self, bank: usize, _layer: u32, row: u32) -> Option<&[u8]> {
            if bank > 0 {
                return None;
            }
            self.rows.get(row as usize).map(|r| r.as_slice())
        }
    }

    /// The whole pool end to end on a real card: a plan lands the bytes
    /// the planner named in the slots it named, a gather moves one slot
    /// onto another without touching the host, and a warm step copies
    /// nothing.
    ///
    /// This is the hardware half of `persistent-gpu-expert-cache`'s
    /// acceptance. The policy half is proven on any host in
    /// `crate::expert_slots`; what needs a GPU is that these
    /// driver calls do what they are read as doing.
    #[test]
    #[ignore = "requires real CUDA hardware -- NOT yet run on a GPU; run with --ignored on a CUDA-capable machine"]
    fn a_cuda_pool_lands_each_expert_in_the_slot_the_plan_named() {
        const EXPERTS: usize = 4;
        const WIDTH: usize = 64;

        let dev = CudaDevice::new(0).expect("a CUDA device");
        let geometry = SlotGeometry {
            num_layers: 1,
            slots: EXPERTS,
            row_bytes: vec![WIDTH],
        };
        let mut pool = CudaExpertPool::new(dev.clone(), &geometry).unwrap();
        assert_eq!(pool.bytes(), (EXPERTS * WIDTH) as u64);

        let mut slots = ExpertSlots::new(geometry).unwrap();
        let rows = NamedRows::new(EXPERTS, WIDTH);

        let plan = CopyPlan {
            dst_slots: vec![0, 2],
            src_rows: vec![3, 1],
        };
        let applied = slots.apply_copy_plan(0, &plan, &rows, &mut pool).unwrap();
        assert_eq!(applied.rows, 2);
        assert_eq!(applied.bytes as usize, 2 * WIDTH);

        for (&slot, &row) in plan.dst_slots.iter().zip(plan.src_rows.iter()) {
            let got = dev.dtoh_sync_copy(pool.slot(0, slot).unwrap()).unwrap();
            assert_eq!(got, rows.rows[row as usize], "slot {slot} holds row {row}");
        }

        // Device to device: slot 0 onto slot 1, no host traffic.
        let gather = GatherPlan {
            dst_slots: vec![1],
            src_slots: vec![0],
        };
        slots.apply_gather_plan(&gather, &mut pool).unwrap();
        assert_eq!(
            dev.dtoh_sync_copy(pool.slot(0, 1).unwrap()).unwrap(),
            rows.rows[3],
            "the gather must carry slot 0's contents, not its neighbour's"
        );
        assert_eq!(
            slots.occupant(1),
            Some(ExpertId {
                layer: 0,
                expert: 3
            })
        );

        let stats = slots.stats();
        assert_eq!(stats.host_bytes as usize, 2 * WIDTH);
        assert_eq!(
            stats.device_bytes as usize, WIDTH,
            "a device-to-device copy must not be billed to the link"
        );

        // The acceptance property, on hardware: an empty plan issues no
        // copies at all and moves no bytes.
        let warm = slots
            .apply_copy_plan(0, &CopyPlan::default(), &rows, &mut pool)
            .unwrap();
        assert!(warm.warm && warm.bytes == 0);
        assert_eq!(slots.stats().host_bytes, stats.host_bytes);
    }
}
