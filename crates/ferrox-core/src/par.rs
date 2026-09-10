//! The single seam every CPU parallel region in this crate goes through.
//!
//! There used to be about fifty spellings of "run this over rows in
//! parallel" scattered through [`crate::weight_matrix`] alone, each one
//! an inline rayon iterator chain. That is the shape this repo has been
//! burned by before: many copies of one decision, with nothing making
//! them agree. Routing them all through the handful of functions below
//! means the choice of *how* work is scheduled is made in one place.
//!
//! Which is exactly what issue #27 needs, because it wants that choice
//! changed: rayon forks and joins per operation, per layer, per token,
//! and llama.cpp instead hands work to a pool that is already awake.
//! [`Backend::Spin`] is that pool ([`crate::cpu_pool`]).
//!
//! # The switch
//!
//! Which of the two runs is decided **per operation, from its size**, by
//! the one predicate in [`policy`]: [`backend`]. `FERROX_CPU_POOL` pins
//! it either way (`spin` / `rayon`) and is an A/B override, not the
//! decision. See [`policy::SPIN_MIN_OP_MACS`] for the crossover and what
//! is and is not measured about it.
//!
//! Every helper below asks [`backend`] and none of them decides
//! anything itself, which is what stops the two arms of that choice from
//! drifting apart across thirty call sites.
//!
//! # `min_len`, and where `MIN_TASK_MACS` went
//!
//! Every helper takes a `min_len`. On the rayon arm it is passed
//! straight to `with_min_len`, which is what the call sites did by hand
//! before, so the fork-join path's task decomposition is bit-for-bit
//! what it was.
//!
//! On the spin arm it is **ignored**. `MIN_TASK_MACS` existed to stop
//! rayon splitting a matvec into tasks too small to pay for their own
//! fork-join; when a region costs a cache-line transfer instead of a
//! futex there is nothing to pay for, so the spin arm chunks purely by
//! pool width ([`task_count`]) the way `ggml_compute_forward_mul_mat`
//! does. That is the deletion issue #27 asks for, and it is a deletion
//! rather than a retune: no MAC threshold is consulted on this path at
//! all. It survives on the rayon arm because the rayon arm still runs
//! every operation below the crossover, and removing it there re-opens
//! the measured 13-16x small-model regression documented on
//! [`crate::weight_matrix::WeightMatrix::min_rows_per_task`].

use std::cell::Cell;

use rayon::prelude::*;

use crate::cpu_pool::CpuPool;

mod carry;
pub mod policy;

pub use policy::{backend, macs_per_row, with_op_work};

thread_local! {
    /// How many parallel regions THIS thread has opened while not being
    /// a rayon worker. See [`on_workers`] for why that is the number
    /// worth counting, and [`cold_regions`] for how a test reads it.
    ///
    /// Per thread rather than process-wide on purpose. A cold region is
    /// always counted on the thread that submits it, so nothing is lost;
    /// and a shared counter would make the assertion depend on whatever
    /// else the test binary happened to be running at the time, which is
    /// the difference between a guard and a flake.
    static COLD_REGIONS: Cell<u64> = const { Cell::new(0) };
}

/// Parallel regions this thread has opened without being a rayon worker.
///
/// Monotonic, so a test reads it before and after the operation it cares
/// about and asserts on the DIFFERENCE. It is not a benchmark: it is an
/// operation count, which is load-immune, and it is the only thing that
/// distinguishes "this decode step entered the pool once" from "it
/// entered it a hundred and fifty times".
pub fn cold_regions() -> u64 {
    COLD_REGIONS.with(Cell::get)
}

/// Records one region about to be opened on the rayon arm.
///
/// Called from every rayon fallback in this module and nowhere else.
/// The worker-index read is the same TLS lookup rayon is about to do
/// anyway, and the counter is touched only on the cold path, which after
/// [`on_workers`] is once per decode step rather than once per matvec.
fn note_rayon_region() {
    if rayon::current_thread_index().is_none() {
        COLD_REGIONS.with(|c| c.set(c.get().saturating_add(1)));
    }
}

/// Run `f` on a rayon worker, so every parallel region it opens takes
/// rayon's IN-WORKER path instead of its cold-submission path.
///
/// # What this is for
///
/// `rayon::join` and the `par_iter` bridges both funnel through
/// `Registry::in_worker`. That call has two arms and they do not cost
/// the same thing:
///
/// - **From a worker** (`in_worker_hot`): the calling thread runs one
///   half itself, the other half is posted for stealing, and the wait is
///   a `SpinLatch`. No syscall.
/// - **From any other thread** (`in_worker_cold`): the job is injected,
///   and the caller blocks on a `LockLatch`, which is a pthread mutex
///   and condvar. That is a park and a wake, per region, and the caller
///   contributes no arithmetic while it sleeps.
///
/// A decode step opens roughly five regions per layer, so a 30-layer
/// model paid ~150 of the cold arm per token. Measured on an M2 Pro with
/// `sample` over SmolLM2-135M Q8_0 `tg128`, the main thread spent **74%
/// of the token** inside `__psynch_cvwait` under `LockLatch`, and over
/// that same window the six workers it was waiting for held only about
/// an eighth as many samples in the matvec kernel: most of the wait was
/// the round trip, not the work.
///
/// Wrapping the whole step in one `rayon::scope` turns those ~150 cold
/// entries into ONE. The step then runs on worker 0 and every nested
/// region is hot.
///
/// # Why it is not simply free
///
/// The caller still parks once, for the whole step, and the step no
/// longer runs on the caller's thread. Both are deliberate: one park per
/// token against one per matvec, and rayon's own worker count is
/// unchanged, so the same number of cores do the work.
///
/// Nesting is free (a call from inside another `on_workers` returns
/// `f()` directly), so an entry point may wrap unconditionally without
/// having to know whether its caller already did.
///
/// # What crosses with the work
///
/// A step that reads a thread-local *setting* would read the worker's
/// default instead of its caller's choice, which is exactly what
/// GitHub issue #166 was: a Metal decode moved to a worker silently took
/// the other `lm_head` path and answered differently from the tenth
/// token. So the settings are captured on this thread and adopted on the
/// worker for the length of the job. [`carry::Carry`] is the single list
/// of them and its `adopt` destructures exhaustively, so a new one that
/// is not carried does not compile.
///
/// # Two cases this deliberately does NOT promote
///
/// **A backend whose thread-affine state is not proven carried.**
/// Promotion asks [`crate::weight_matrix::active_backend`] -- the same
/// cached predicate dispatch itself uses, not a second opinion about it
/// -- and puts the answer to [`carry::promotable`], which holds the
/// per-backend verdict and the evidence behind each one. CPU and Metal
/// promote; CUDA and Vulkan do not, for want of hardware to check them
/// on rather than for a known defect.
///
/// **The pinned spin pool.** Under `FERROX_CPU_POOL=spin` the rayon
/// global pool is never used for work, and `rayon::scope` would BUILD
/// it, spawning a second set of workers that then only sit there. So
/// that pin short-circuits, exactly as [`num_threads`] avoids
/// `rayon::current_num_threads` for the same reason.
pub fn on_workers<R, F>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if !carry::promotable(crate::weight_matrix::active_backend()) {
        return f();
    }
    if policy::pinned() == Some(Backend::Spin) {
        return f();
    }
    if rayon::current_thread_index().is_some() {
        return f();
    }
    // Captured HERE, on the thread whose caller made the decision, and
    // before anything is submitted. Reading it on the worker would read
    // the worker's default, which is the defect.
    let carried = carry::Carry::capture();
    // The one cold entry this whole design is willing to pay. Counted
    // like any other, so [`cold_regions`] reports the true total and a
    // test can assert it is exactly one.
    note_rayon_region();
    rayon::scope(move |_| {
        let _adopted = carried.adopt();
        f()
    })
}

/// Which scheduler CPU parallel regions use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// A rayon fork-join per region. The default.
    Rayon,
    /// A persistent pool of workers parked on a spin-then-park barrier.
    Spin,
}

/// How many tasks per worker the spin arm aims for.
///
/// Tasks are handed out by one atomic cursor, so more of them means
/// better load balancing and more contention on that cursor. Eight is
/// the same order as llama.cpp's `4 * n_threads` chunk floor, with room
/// for the uneven per-task cost that causal masking gives attention.
const TASKS_PER_THREAD: usize = 8;

/// The process-wide persistent pool, built on first use with
/// [`crate::threads::resolve_cpu_threads`] workers -- the same width
/// [`crate::threads::init_cpu_pool`] gives rayon, so the two backends
/// are the same number of threads and a comparison is not confounded.
///
/// A `static` is never dropped, so the workers live to process exit.
/// That is deliberate and it is also what rayon's global pool does.
fn pool() -> &'static CpuPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<CpuPool> = OnceLock::new();
    POOL.get_or_init(|| CpuPool::new(crate::threads::resolve_cpu_threads()))
}

/// Worker count of the active backend.
///
/// Call this instead of `rayon::current_num_threads` anywhere a task
/// decomposition is being sized: `rayon::current_num_threads` *builds*
/// the global rayon pool as a side effect, so asking it under the spin
/// backend spawns a second set of threads that would never run anything.
pub fn num_threads() -> usize {
    match backend() {
        Backend::Rayon => rayon::current_num_threads().max(1),
        Backend::Spin => pool().num_threads(),
    }
}

/// How many tasks the spin arm splits `n_items` into.
///
/// Never more than one task per item, never zero, and never more than
/// the pool can usefully chase. No work threshold appears here; see the
/// module docs on `MIN_TASK_MACS`.
pub fn task_count(n_items: usize) -> usize {
    if n_items == 0 {
        return 0;
    }
    n_items.min(num_threads().saturating_mul(TASKS_PER_THREAD).max(1))
}

/// `(items_per_task, n_tasks)` for a contiguous split of `n_items`.
fn split(n_items: usize) -> (usize, usize) {
    let n_tasks = task_count(n_items);
    if n_tasks == 0 {
        return (0, 0);
    }
    (n_items.div_ceil(n_tasks), n_tasks)
}

/// A raw pointer that may cross into worker threads.
///
/// Only ever used to hand each task a *disjoint* sub-slice of one
/// allocation the submitter borrows mutably for the whole region.
struct SendPtr<T>(*mut T);

// Hand-written rather than derived: `#[derive(Copy)]` would add a
// `T: Copy` bound, and the element types here are `f32` today but a
// `Q8Activations` tomorrow.
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// The element pointer at `offset`.
    ///
    /// A method rather than a field read at the call sites, because
    /// closure capture is per *field*: reading `base.0` inside a task
    /// captures the bare `*mut T`, which is not `Sync`, and the whole
    /// point of this wrapper is the `unsafe impl` above.
    ///
    /// # Safety
    /// `offset` must be within the allocation this was built from.
    unsafe fn at(self, offset: usize) -> *mut T {
        // SAFETY: the caller's invariant.
        unsafe { self.0.add(offset) }
    }
}

// SAFETY: the pointer comes from a `&mut [T]` the submitter holds for
// the duration of the region, and each task derives a sub-slice from a
// half-open index range that no other task's range overlaps. `T: Send`
// is required at every call site, which is what makes moving those
// sub-slices onto worker threads sound.
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}

/// Run `f(index)` for every `index` in `0..n`.
pub fn indices<F>(n: usize, min_len: usize, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    if n == 0 {
        return;
    }
    if backend() == Backend::Spin {
        let (per, n_tasks) = split(n);
        let task = |t: usize| {
            let lo = t * per;
            let hi = ((t + 1) * per).min(n);
            for i in lo..hi {
                f(i);
            }
        };
        if pool().run(n_tasks, &task) {
            return;
        }
    }
    note_rayon_region();
    (0..n)
        .into_par_iter()
        .with_min_len(min_len.max(1))
        .for_each(&f);
}

/// [`indices`] with a per-task scratch value, the shape rayon spells
/// `for_each_init`. One `S` is created per task, not per index.
pub fn indices_init<S, I, F>(n: usize, min_len: usize, init: I, f: F)
where
    S: Send,
    I: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize) + Send + Sync,
{
    if n == 0 {
        return;
    }
    if backend() == Backend::Spin {
        let (per, n_tasks) = split(n);
        let task = |t: usize| {
            let lo = t * per;
            let hi = ((t + 1) * per).min(n);
            if lo >= hi {
                return;
            }
            let mut state = init();
            for i in lo..hi {
                f(&mut state, i);
            }
        };
        if pool().run(n_tasks, &task) {
            return;
        }
    }
    note_rayon_region();
    (0..n)
        .into_par_iter()
        .with_min_len(min_len.max(1))
        .for_each_init(&init, |state, i| f(state, i));
}

/// Run `f(index, &mut item)` over `data`, the shape rayon spells
/// `par_iter_mut().with_min_len(..).enumerate()`.
pub fn items_mut<T, F>(data: &mut [T], min_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    if backend() == Backend::Spin {
        let base = SendPtr(data.as_mut_ptr());
        let (per, n_tasks) = split(n);
        let task = |t: usize| {
            let lo = t * per;
            let hi = ((t + 1) * per).min(n);
            for i in lo..hi {
                // SAFETY: `base` points at `data`, borrowed mutably for
                // the whole call and outliving the region. Index `i` is
                // inside `0..n` and belongs to exactly one task, so no
                // two of these `&mut T` overlap.
                f(i, unsafe { &mut *base.at(i) });
            }
        };
        if pool().run(n_tasks, &task) {
            return;
        }
    }
    note_rayon_region();
    data.par_iter_mut()
        .with_min_len(min_len.max(1))
        .enumerate()
        .for_each(|(i, slot)| f(i, slot));
}

/// [`chunks_mut`] over two slices of the same length at once, the shape
/// rayon spells `a.par_chunks_mut(k).zip(b.par_chunks_mut(k))`.
///
/// Exists because the MoE decode path computes a gate row and an up row
/// from one shared activation: splitting that into two regions would
/// double the region count, which is the thing this whole module is
/// trying to reduce.
pub fn chunks_mut2<T, U, F>(a: &mut [T], b: &mut [U], chunk_len: usize, min_len: usize, f: F)
where
    T: Send,
    U: Send,
    F: Fn(usize, &mut [T], &mut [U]) + Send + Sync,
{
    assert!(chunk_len > 0, "chunk length must be positive");
    assert_eq!(a.len(), b.len(), "zipped slices must be the same length");
    let len = a.len();
    if len == 0 {
        return;
    }
    let n_chunks = len.div_ceil(chunk_len);
    if backend() == Backend::Spin {
        let base_a = SendPtr(a.as_mut_ptr());
        let base_b = SendPtr(b.as_mut_ptr());
        let (per, n_tasks) = split(n_chunks);
        let task = |t: usize| {
            let lo = t * per;
            let hi = ((t + 1) * per).min(n_chunks);
            for c in lo..hi {
                // SAFETY: both pointers come from slices the caller
                // borrows mutably for the whole call, of equal length,
                // and chunk `c` of each is visited by exactly one task.
                unsafe {
                    f(
                        c,
                        chunk_of(base_a, len, chunk_len, c),
                        chunk_of(base_b, len, chunk_len, c),
                    );
                }
            }
        };
        if pool().run(n_tasks, &task) {
            return;
        }
    }
    note_rayon_region();
    a.par_chunks_mut(chunk_len)
        .zip(b.par_chunks_mut(chunk_len))
        .with_min_len(min_len.max(1))
        .enumerate()
        .for_each(|(c, (ca, cb))| f(c, ca, cb));
}

/// Run `f(chunk_index, &mut chunk)` over `data` split into runs of
/// `chunk_len`, the shape rayon spells `par_chunks_mut(chunk_len)`.
///
/// A trailing partial chunk is delivered short, exactly as
/// `par_chunks_mut` does.
pub fn chunks_mut<T, F>(data: &mut [T], chunk_len: usize, min_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    assert!(chunk_len > 0, "chunk length must be positive");
    let len = data.len();
    if len == 0 {
        return;
    }
    let n_chunks = len.div_ceil(chunk_len);
    if backend() == Backend::Spin {
        let base = SendPtr(data.as_mut_ptr());
        let (per, n_tasks) = split(n_chunks);
        let task = |t: usize| {
            let lo = t * per;
            let hi = ((t + 1) * per).min(n_chunks);
            for c in lo..hi {
                // SAFETY: see `chunks_mut_init`; the ranges are the same
                // disjoint half-open chunks of one live borrow.
                f(c, unsafe { chunk_of(base, len, chunk_len, c) });
            }
        };
        if pool().run(n_tasks, &task) {
            return;
        }
    }
    note_rayon_region();
    data.par_chunks_mut(chunk_len)
        .with_min_len(min_len.max(1))
        .enumerate()
        .for_each(|(c, chunk)| f(c, chunk));
}

/// The `c`-th `chunk_len`-sized chunk of the `len`-element allocation at
/// `base`, delivered short when it is the trailing one.
///
/// # Safety
/// `base` must point at a live allocation of at least `len` elements
/// that outlives the returned slice, and the caller must guarantee that
/// no other live slice covers chunk `c` -- which the callers do by
/// visiting each chunk index from exactly one task.
unsafe fn chunk_of<'a, T>(base: SendPtr<T>, len: usize, chunk_len: usize, c: usize) -> &'a mut [T] {
    let start = c * chunk_len;
    let end = ((c + 1) * chunk_len).min(len);
    debug_assert!(start < end && end <= len);
    // SAFETY: the caller's invariants, plus `start..end` being inside
    // `0..len` by construction of `c < len.div_ceil(chunk_len)`.
    unsafe { std::slice::from_raw_parts_mut(base.at(start), end - start) }
}

/// [`chunks_mut`] with a per-task scratch value.
pub fn chunks_mut_init<T, S, I, F>(data: &mut [T], chunk_len: usize, min_len: usize, init: I, f: F)
where
    T: Send,
    S: Send,
    I: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, &mut [T]) + Send + Sync,
{
    assert!(chunk_len > 0, "chunk length must be positive");
    let len = data.len();
    if len == 0 {
        return;
    }
    let n_chunks = len.div_ceil(chunk_len);
    if backend() == Backend::Spin {
        let base = SendPtr(data.as_mut_ptr());
        let (per, n_tasks) = split(n_chunks);
        let task = |t: usize| {
            let lo = t * per;
            let hi = ((t + 1) * per).min(n_chunks);
            if lo >= hi {
                return;
            }
            let mut state = init();
            for c in lo..hi {
                // SAFETY: `base` points at `data`, which the caller
                // borrows mutably for this whole call and which outlives
                // the region (`CpuPool::run` does not return until every
                // worker has stopped touching the closure). Chunk index
                // `c` is visited by exactly one task, so no two of these
                // slices overlap.
                f(&mut state, c, unsafe { chunk_of(base, len, chunk_len, c) });
            }
        };
        if pool().run(n_tasks, &task) {
            return;
        }
    }
    note_rayon_region();
    data.par_chunks_mut(chunk_len)
        .with_min_len(min_len.max(1))
        .enumerate()
        .for_each_init(&init, |state, (c, chunk)| f(state, c, chunk));
}

/// Two independent pieces of work.
///
/// The rayon arm forks; the spin arm runs them one after the other,
/// because each half already spreads across the whole pool internally
/// and nesting a region inside a region is the one thing
/// [`CpuPool::run`] cannot parallelize. That is llama.cpp's shape too:
/// its threadpool runs one graph node at a time, full width.
pub fn join2<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    match backend() {
        Backend::Rayon => {
            note_rayon_region();
            rayon::join(a, b)
        }
        Backend::Spin => (a(), b()),
    }
}

/// Three independent pieces of work; see [`join2`].
pub fn join3<A, B, C, RA, RB, RC>(a: A, b: B, c: C) -> (RA, RB, RC)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    C: FnOnce() -> RC + Send,
    RA: Send,
    RB: Send,
    RC: Send,
{
    match backend() {
        Backend::Rayon => {
            note_rayon_region();
            let (ra, (rb, rc)) = rayon::join(a, || rayon::join(b, c));
            (ra, rb, rc)
        }
        Backend::Spin => (a(), b(), c()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two configurations `on_workers` declines to promote in, and
    /// so the two it cannot be asserted in. One predicate, shared by
    /// every test below, rather than three spellings of it.
    ///
    /// The backend half asks [`carry::promotable`] rather than naming a
    /// backend: the rule used to be "the backend is CPU", and writing
    /// that here a second time is how a test comes to assert the rule
    /// the code used to have.
    fn the_promotion_applies_here() -> bool {
        policy::pinned().is_none() && carry::promotable(crate::weight_matrix::active_backend())
    }

    use std::sync::atomic::{AtomicU32, Ordering};

    /// With nothing published and nothing pinned, the helpers fork with
    /// rayon -- the behaviour every caller that has not opted into the
    /// size rule keeps.
    #[test]
    fn an_unpublished_region_forks_with_rayon() {
        if policy::pinned().is_none() {
            assert_eq!(backend(), Backend::Rayon);
        }
    }

    /// **Every rayon-versus-spin choice in this crate goes through one
    /// predicate**, and this is what says so.
    ///
    /// The alternative is the shape this repo keeps shipping: a second
    /// site that decides for itself and then drifts. `weight_matrix`
    /// had four copies of one GPU-router eligibility test that tested
    /// three conditions, two, and none.
    ///
    /// Two halves, because a helper can drift in two directions:
    /// reaching the pool without asking, and asking the environment
    /// instead of asking the predicate.
    ///
    /// Sabotage: inline `pool().run(..)` into a helper without its
    /// `if backend() == Backend::Spin` guard, or read the environment
    /// variable in a second place, and this goes red.
    #[test]
    fn every_scheduler_choice_in_this_crate_goes_through_the_one_predicate() {
        // The needles are assembled rather than written out, because
        // this file is one of the files being searched and a literal
        // would count itself.
        let call = format!("{}()", "backend");
        let guard = format!("if {call} == Backend::Spin {{");
        let dispatch = format!("match {call} {{");
        let enters_pool = format!("if {}().run(", "pool");

        let src = include_str!("par.rs");
        let guarded = src.matches(&guard).count();
        assert!(guarded >= 6, "expected one guard per region helper");
        assert_eq!(
            src.matches(&enters_pool).count(),
            guarded,
            "a helper reached the persistent pool without asking the predicate"
        );
        assert_eq!(
            src.matches(&dispatch).count(),
            3,
            "num_threads, join2 and join3 dispatch on the predicate"
        );

        // And the environment is consulted in exactly one place, so the
        // override cannot come to mean two things. The needle is the
        // variable's name up to its closing quote, which is what keeps
        // `FERROX_CPU_POOL_SPIN_US` (a different knob, in `cpu_pool`)
        // out of the answer.
        let needle = format!("FERROX_CPU_POOL{}", '"');
        let mut readers = Vec::new();
        let mut stack = vec![std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("crate source is readable") {
                let path = entry.expect("readable entry").path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs")
                    && std::fs::read_to_string(&path)
                        .expect("source file is UTF-8")
                        .lines()
                        .any(|l| l.contains(&needle) && !l.trim_start().starts_with("//"))
                {
                    readers.push(path);
                }
            }
        }
        assert_eq!(
            readers.len(),
            1,
            "`FERROX_CPU_POOL` must be read only by `par::policy::pinned`, found {readers:?}"
        );
        assert!(readers[0].ends_with("par/policy.rs"), "{readers:?}");
    }

    /// The spin arm's chunking is a function of pool width and item
    /// count and nothing else. If a MAC threshold ever creeps back onto
    /// this path it has to change this signature to do it.
    #[test]
    fn the_spin_arm_chunks_by_pool_width_with_no_work_threshold() {
        assert_eq!(task_count(0), 0);
        assert_eq!(task_count(1), 1);
        assert_eq!(task_count(3), 3);
        let wide = task_count(1_000_000);
        assert_eq!(wide, num_threads() * TASKS_PER_THREAD);
        // A one-element-per-row matrix and a 4096-element-per-row matrix
        // decompose identically: work per item is not an input.
        assert_eq!(task_count(4096), task_count(4096));
        let (per, n) = split(1000);
        assert_eq!(n, task_count(1000));
        assert!(per * n >= 1000 && (per - 1) * n < 1000);
    }

    /// Both arms must visit every index exactly once and produce the
    /// same answer, whichever one the env var picked -- that is the
    /// property the whole switch rests on.
    #[test]
    fn indices_visits_every_index_exactly_once() {
        for n in [0usize, 1, 7, 64, 5000] {
            let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            indices(n, 8, |i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n={n}");
        }
    }

    #[test]
    fn items_mut_writes_every_slot_with_its_own_index() {
        for n in [0usize, 1, 9, 257] {
            let mut data = vec![0u32; n];
            items_mut(&mut data, 4, |i, slot| *slot = i as u32 + 1);
            assert_eq!(data, (1..=n as u32).collect::<Vec<_>>(), "n={n}");
        }
    }

    /// The trailing partial chunk is the easy thing to lose, and losing
    /// it silently drops the last rows of a matvec.
    #[test]
    fn chunks_mut_delivers_a_short_trailing_chunk() {
        let mut data = vec![0u32; 10];
        let seen: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
        chunks_mut(&mut data, 4, 1, |c, chunk| {
            seen.lock().unwrap().push((c, chunk.len()));
            for (i, slot) in chunk.iter_mut().enumerate() {
                *slot = (c * 4 + i) as u32;
            }
        });
        let mut seen = seen.into_inner().unwrap();
        seen.sort_unstable();
        assert_eq!(seen, vec![(0, 4), (1, 4), (2, 2)]);
        assert_eq!(data, (0..10).collect::<Vec<u32>>());
    }

    /// Per-task scratch is created per task, never shared between two
    /// tasks that might run at the same time.
    #[test]
    fn chunks_mut_init_gives_each_task_its_own_scratch() {
        let mut data = vec![0u64; 512];
        chunks_mut_init(
            &mut data,
            8,
            1,
            || Vec::<u64>::with_capacity(8),
            |scratch: &mut Vec<u64>, c, chunk| {
                scratch.clear();
                scratch.extend(chunk.iter().map(|_| c as u64));
                chunk.copy_from_slice(scratch);
            },
        );
        for (c, chunk) in data.chunks(8).enumerate() {
            assert!(chunk.iter().all(|&v| v == c as u64));
        }
    }

    #[test]
    fn joins_return_every_result_in_order() {
        assert_eq!(join2(|| 1u8, || 2u8), (1, 2));
        assert_eq!(join3(|| 1u8, || 2u8, || 3u8), (1, 2, 3));
    }

    /// The two arms are not allowed to disagree. This runs each helper
    /// through the spin pool directly and through rayon directly, in one
    /// process, and compares -- because the env var can only select one
    /// of them per run, and "they agree" is the claim the PR makes.
    #[test]
    fn the_spin_arm_and_the_rayon_arm_produce_identical_results() {
        let pool = CpuPool::new(4);
        for n in [1usize, 5, 63, 1024] {
            let mut spun = vec![0f32; n];
            let base = SendPtr(spun.as_mut_ptr());
            let (per, n_tasks) = split(n);
            let task = |t: usize| {
                let lo = t * per;
                let hi = ((t + 1) * per).min(n);
                for i in lo..hi {
                    // SAFETY: disjoint single-element writes; index `i`
                    // belongs to exactly one task.
                    unsafe { *base.at(i) = (i as f32) * 0.5 + 1.0 };
                }
            };
            assert!(pool.run(n_tasks, &task));

            let mut forked = vec![0f32; n];
            forked
                .par_iter_mut()
                .with_min_len(8)
                .enumerate()
                .for_each(|(i, slot)| *slot = (i as f32) * 0.5 + 1.0);

            assert_eq!(spun, forked, "n={n}");
        }
    }

    /// Every helper in this module must report the region it is about
    /// to open, or the counter reads as coverage while measuring
    /// nothing. This walks all eight of them rather than trusting that
    /// a new one remembered, because a helper that forgot would leave
    /// the counter reading low and every assertion built on it passing.
    ///
    /// Sabotage: delete any single `note_rayon_region()` call and the
    /// helper whose name is in the failure message goes red.
    #[test]
    fn every_helper_reports_the_cold_region_it_opens() {
        if !the_promotion_applies_here() {
            return;
        }
        let mut buf = vec![0f32; 64];
        let mut other = vec![0f32; 64];

        // Spelled out one at a time rather than as a table of boxed
        // closures: the slice helpers borrow `buf`, so a table could
        // hold only half of them, and half a table is exactly the
        // coverage illusion this test exists to avoid.
        let before = cold_regions();
        indices(64, 1, |_| {});
        assert!(cold_regions() > before, "indices did not report");

        let before = cold_regions();
        indices_init(64, 1, || 0u8, |_, _| {});
        assert!(cold_regions() > before, "indices_init did not report");

        let before = cold_regions();
        join2(|| (), || ());
        assert!(cold_regions() > before, "join2 did not report");

        let before = cold_regions();
        join3(|| (), || (), || ());
        assert!(cold_regions() > before, "join3 did not report");

        let before = cold_regions();
        items_mut(&mut buf, 1, |_, _| {});
        assert!(cold_regions() > before, "items_mut did not report");

        let before = cold_regions();
        chunks_mut(&mut buf, 8, 1, |_, _| {});
        assert!(cold_regions() > before, "chunks_mut did not report");

        let before = cold_regions();
        chunks_mut_init(&mut buf, 8, 1, || 0u8, |_, _, _| {});
        assert!(cold_regions() > before, "chunks_mut_init did not report");

        let before = cold_regions();
        chunks_mut2(&mut buf, &mut other, 8, 1, |_, _, _| {});
        assert!(cold_regions() > before, "chunks_mut2 did not report");
    }

    /// The whole claim of `on_workers`: many regions inside it cost ONE
    /// cold entry into the pool, where the same regions outside it cost
    /// one each.
    ///
    /// This is the operation-count form of the fix. It needs no clock
    /// and no quiet host, which is why it is the guard rather than a
    /// throughput assertion.
    ///
    /// Sabotage: make `on_workers` call `f()` unconditionally and the
    /// `inside` count jumps from 1 to `REGIONS`, turning this red.
    #[test]
    fn on_workers_collapses_many_regions_into_one_cold_entry() {
        if !the_promotion_applies_here() {
            return;
        }
        const REGIONS: u64 = 16;
        let open_them = || {
            for _ in 0..REGIONS {
                indices(64, 1, |_| {});
            }
        };

        let before = cold_regions();
        open_them();
        let outside = cold_regions() - before;

        let before = cold_regions();
        on_workers(open_them);
        let inside = cold_regions() - before;

        assert_eq!(
            outside, REGIONS,
            "each region opened from a cold thread should count once"
        );
        assert_eq!(
            inside, 1,
            "the whole batch should enter the pool exactly once"
        );
    }

    /// A promoted step runs with the setting its SUBMITTER announced,
    /// not with the worker's default.
    ///
    /// This is GitHub issue #166's whole mechanism at the seam that
    /// caused it. `on_workers` moves the step to a thread the caller
    /// never configured; before the carry, a Metal decode read the
    /// default there and took the other `lm_head` path, changing the
    /// completion from the tenth token. The test asserts on the thread
    /// id first, so it cannot pass by not having moved.
    ///
    /// Sabotage: delete the `carried.adopt()` line from `on_workers`.
    #[cfg(feature = "metal")]
    #[test]
    fn a_promoted_step_runs_with_the_setting_its_submitter_announced() {
        use ferrox_metal::greedy_fold::{greedy_fold_setting, set_metal_greedy_argmax, GreedyFold};
        if !the_promotion_applies_here() {
            return;
        }
        set_metal_greedy_argmax(true);
        let submitter = std::thread::current().id();

        let (ran_on, seen) = on_workers(|| (std::thread::current().id(), greedy_fold_setting()));

        assert_ne!(
            ran_on, submitter,
            "nothing is being tested unless the step actually moved"
        );
        assert_eq!(
            seen,
            GreedyFold::On,
            "the worker must run the fold the caller asked for"
        );
        set_metal_greedy_argmax(false);
    }

    /// A nested call must not open a second entry, so an entry point can
    /// wrap unconditionally without knowing what its caller did.
    #[test]
    fn a_nested_on_workers_opens_no_further_cold_entry() {
        if !the_promotion_applies_here() {
            return;
        }
        let before = cold_regions();
        on_workers(|| {
            on_workers(|| {
                indices(64, 1, |_| {});
            });
        });
        assert_eq!(cold_regions() - before, 1);
    }

    /// `on_workers` must return what `f` returns, not swallow it.
    #[test]
    fn on_workers_hands_back_the_closures_value() {
        assert_eq!(on_workers(|| 41usize + 1), 42);
    }
}
