//! A persistent CPU worker pool parked on a spin-then-park barrier.
//!
//! # Why this exists
//!
//! Decode opens a parallel region per weight matrix, per layer, per
//! token -- roughly seven per layer. Rayon answers each of those with a
//! fork-join: a job is pushed onto the global injector, sleeping workers
//! are woken through a mutex and a condvar, and the caller then waits on
//! a latch. That is a fixed per-region cost, so the smaller the model
//! the larger its share: measured at ~75% of decode wall time at 135M
//! parameters and ~9% at 8B (issue #27).
//!
//! llama.cpp does not fork. `ggml_threadpool` starts N workers once and
//! parks them on a spin barrier; a graph node is published by bumping a
//! counter the workers are already watching, and each worker pulls
//! chunks off one shared atomic until the work is gone. Waking a
//! spinning thread costs a cache-line transfer instead of a futex.
//!
//! This module is that shape, in safe-by-construction Rust:
//!
//! - [`CpuPool::new`] spawns the workers once and keeps them.
//! - [`CpuPool::run`] publishes `n_tasks` and a type-erased closure,
//!   bumps the epoch, then *participates* in draining the task counter
//!   alongside the workers.
//! - Workers spin for [`spin_window`] and then park on a condvar, so an
//!   idle `frink-server` does not burn a core per worker. That bound is
//!   the whole reason this is not a plain spin barrier.
//!
//! # What makes it sound
//!
//! The one dangerous thing here is that workers dereference a pointer to
//! a closure the submitter owns. Two rules keep that from being a
//! use-after-free, and both are enforced in [`CpuPool::run`]:
//!
//! 1. **A region ends only when every worker has checked out.** `active`
//!    is set to the worker count before the epoch bump and decremented
//!    by each worker after its last touch of the job; `run` does not
//!    return until it reads zero. So the closure outlives every use.
//! 2. **Only one region at a time.** `submit` is a mutex, and `run`
//!    *tries* it rather than blocking: a second thread that arrives
//!    while a region is in flight is told `false` and falls back to
//!    rayon rather than queueing behind it.
//!
//! Re-entrancy is the third hazard -- a task closure that itself opens a
//! region would deadlock against rule 2 -- so [`CpuPool::run`] runs
//! nested regions inline on the calling thread.
//!
//! # What this module is NOT
//!
//! It is not a general work-stealing runtime. There is no task graph, no
//! nested parallelism, no `join`. It runs one flat `0..n_tasks` loop at a
//! time, because that is the entire shape of a quantized matvec and the
//! shape llama.cpp's threadpool has.

use std::any::Any;
use std::cell::Cell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long a worker keeps spinning after a region ends before it parks.
///
/// Decode submits regions microseconds apart, so a window of tens of
/// microseconds keeps the workers hot right through a token while still
/// letting them sleep when generation stops. A pool that never parked
/// would burn a core per worker on an idle server, which is why this is
/// a window and not a flag.
///
/// `FRINK_CPU_POOL_SPIN_US` overrides it; `0` parks immediately, which
/// is the configuration that proves the park/wake path is exercised.
fn spin_window() -> Duration {
    use std::sync::OnceLock;
    static US: OnceLock<u64> = OnceLock::new();
    Duration::from_micros(*US.get_or_init(|| {
        std::env::var("FRINK_CPU_POOL_SPIN_US")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(100)
    }))
}

/// A parked worker is woken by a broadcast, but a lost wakeup would hang
/// a region forever, so the wait is also bounded. The protocol in
/// [`Shared::wait_for_job`] is designed not to lose one; this timeout is
/// the belt to that pair of braces.
const PARK_TIMEOUT: Duration = Duration::from_millis(20);

/// The type-erased closure a region runs, plus its task count.
///
/// `ptr` is a `*const F` for the `F` the submitter still owns on its own
/// stack, and `call` is a monomorphized shim that casts it back. This is
/// how a fat `dyn Fn` pointer is avoided: a plain thin pointer and a
/// function pointer both fit in a `Copy` struct.
#[derive(Clone, Copy)]
struct Job {
    ptr: *const (),
    call: unsafe fn(*const (), usize),
    n_tasks: usize,
}

impl Default for Job {
    fn default() -> Self {
        // A job nobody will ever run: zero tasks, and a callee that is
        // never reached because the task loop exits first.
        unsafe fn never(_: *const (), _: usize) {}
        Self {
            ptr: std::ptr::null(),
            call: never,
            n_tasks: 0,
        }
    }
}

/// The shim `Job::call` points at for a concrete closure type.
///
/// # Safety
/// `ptr` must be a live `*const F` that outlives every call, and `F`
/// must be `Sync` because several workers call it at once.
unsafe fn call_shim<F: Fn(usize) + Sync>(ptr: *const (), task: usize) {
    // SAFETY: the caller guarantees `ptr` was produced from a live
    // `&F` (see `CpuPool::run`, which does not return until every
    // worker has finished calling this).
    let f = unsafe { &*(ptr as *const F) };
    f(task)
}

/// State shared by the submitter and every worker.
struct Shared {
    /// Bumped once per region. This is the *only* channel that publishes
    /// a job: the `Release` on this store orders every non-atomic write
    /// to `job` before the `Acquire` load a worker does.
    epoch: AtomicU64,
    /// Written by the submitter while no region is in flight, read by
    /// workers after they observe a new `epoch`.
    job: std::cell::UnsafeCell<Job>,
    /// The shared task cursor. `fetch_add` is the whole scheduler.
    next: AtomicUsize,
    /// Set when a task panicked, so the remaining tasks are abandoned
    /// instead of each thread running into the same panic.
    aborted: AtomicBool,
    /// Workers that have not yet checked out of the current region.
    active: AtomicUsize,
    /// Asks every worker to leave its loop. Only [`CpuPool::drop`] sets it.
    shutdown: AtomicBool,
    /// Number of workers currently blocked in `cv`. Guarded by the mutex
    /// so the submitter can decide whether a broadcast is needed without
    /// racing a worker that is about to sleep.
    parked: Mutex<usize>,
    cv: Condvar,
    /// The payload of the first task panic in the current region.
    panic: Mutex<Option<Box<dyn Any + Send + 'static>>>,
}

// SAFETY: `Shared` is only ever reached through an `Arc`, and the two
// non-`Sync` fields are disciplined by the epoch protocol:
//   - `job` is written by the submitter *before* `epoch` is bumped and
//     read by workers *after* they observe that bump, and the submitter
//     does not write again until `active` has drained to zero. So writer
//     and readers never overlap.
//   - `Job::ptr` is a pointer to a closure that, per `CpuPool::run`,
//     outlives every worker's use of it.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

impl Shared {
    /// Block until the epoch differs from `seen` (a new region) or
    /// shutdown is requested, and return the epoch observed.
    ///
    /// Spin first, park second. The park path takes `parked` before it
    /// re-checks the epoch, and the submitter bumps the epoch while
    /// holding that same mutex, so there is no window in which a worker
    /// decides to sleep on a region that has already been published.
    fn wait_for_job(&self, seen: u64) -> u64 {
        let deadline = Instant::now() + spin_window();
        loop {
            let epoch = self.epoch.load(Ordering::Acquire);
            if epoch != seen {
                return epoch;
            }
            for _ in 0..64 {
                std::hint::spin_loop();
            }
            if Instant::now() >= deadline {
                break;
            }
        }
        let mut parked = self.parked.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let epoch = self.epoch.load(Ordering::Acquire);
            if epoch != seen {
                return epoch;
            }
            *parked += 1;
            let (guard, _) = self
                .cv
                .wait_timeout(parked, PARK_TIMEOUT)
                .unwrap_or_else(|e| e.into_inner());
            parked = guard;
            *parked -= 1;
        }
    }

    /// Drain the task cursor, running each task index exactly once.
    ///
    /// A panic in a task is caught, recorded, and turned into an abort
    /// for the rest of the region: without that, a worker would unwind
    /// out of its loop, never decrement `active`, and hang the submitter
    /// forever. The payload is re-raised on the submitter in
    /// [`CpuPool::run`], which is where rayon would have raised it too.
    fn drain(&self, job: Job) {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            loop {
                if self.aborted.load(Ordering::Relaxed) {
                    break;
                }
                let task = self.next.fetch_add(1, Ordering::Relaxed);
                if task >= job.n_tasks {
                    break;
                }
                // SAFETY: `job` was published by `CpuPool::run`, which
                // does not return -- and so does not drop the closure --
                // until `active` reaches zero, which happens strictly
                // after this call returns.
                unsafe { (job.call)(job.ptr, task) };
            }
        }));
        if let Err(payload) = outcome {
            self.aborted.store(true, Ordering::Relaxed);
            let mut slot = self.panic.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(payload);
            }
        }
    }
}

thread_local! {
    /// Set while this thread is executing tasks for a region. A task
    /// that opens its own region must not try to take the pool: it
    /// would deadlock against the submit mutex it is already inside of.
    static IN_REGION: Cell<bool> = const { Cell::new(false) };
}

/// Whether the calling thread is currently running pool tasks.
pub fn in_region() -> bool {
    IN_REGION.with(|c| c.get())
}

/// A persistent pool of parked workers.
///
/// Dropping the pool asks every worker to leave and joins it, so a pool
/// never outlives its threads. The process-wide pool is a `static` and
/// is therefore never dropped, exactly like rayon's global pool.
pub struct CpuPool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    submit: Mutex<()>,
}

impl CpuPool {
    /// Spawn `threads` workers, minus the submitter: the thread that
    /// calls [`Self::run`] is a worker too, which is why a one-thread
    /// pool spawns nothing at all and still works.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            job: std::cell::UnsafeCell::new(Job::default()),
            next: AtomicUsize::new(0),
            aborted: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            parked: Mutex::new(0),
            cv: Condvar::new(),
            panic: Mutex::new(None),
        });
        let mut workers = Vec::with_capacity(threads - 1);
        for idx in 0..threads - 1 {
            let shared = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("frink-cpu-{idx}"))
                .spawn(move || worker_loop(&shared))
                .expect("frink: cannot spawn CPU pool worker");
            workers.push(handle);
        }
        Self {
            shared,
            workers,
            submit: Mutex::new(()),
        }
    }

    /// Total width including the submitting thread.
    pub fn num_threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Run `f(task)` for every `task` in `0..n_tasks`.
    ///
    /// Returns `false` without running anything if another thread is
    /// already inside a region -- the caller is expected to fall back to
    /// rayon rather than serialize behind it. Returns `true` when the
    /// work is complete, including the nested case, where the tasks run
    /// inline on the calling thread.
    ///
    /// A panic in `f` is re-raised here, on the submitting thread.
    pub fn run<F: Fn(usize) + Sync>(&self, n_tasks: usize, f: &F) -> bool {
        if n_tasks == 0 {
            return true;
        }
        if in_region() {
            // Nested region: the pool is already saturated by the outer
            // one, and taking it again would deadlock.
            for task in 0..n_tasks {
                f(task);
            }
            return true;
        }
        let guard = match self.submit.try_lock() {
            Ok(guard) => guard,
            // A previous submitter unwound while holding the lock. The
            // region it was running had already drained (`drain` catches
            // task panics), so the pool state is intact and poisoning
            // here would silently retire the pool for the process.
            Err(std::sync::TryLockError::Poisoned(guard)) => guard.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return false,
        };
        let job = Job {
            ptr: std::ptr::from_ref(f) as *const (),
            call: call_shim::<F>,
            n_tasks,
        };
        // SAFETY: `submit` is held, and the previous region drained
        // `active` to zero before releasing it, so no worker is reading
        // `job` right now. The write is published by the `Release` in
        // the `fetch_add` on `epoch` below.
        unsafe { *self.shared.job.get() = job };
        self.shared.next.store(0, Ordering::Relaxed);
        self.shared.aborted.store(false, Ordering::Relaxed);
        self.shared
            .active
            .store(self.workers.len(), Ordering::Relaxed);
        let parked = {
            let parked = self.shared.parked.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.epoch.fetch_add(1, Ordering::Release);
            *parked
        };
        if parked > 0 {
            self.shared.cv.notify_all();
        }

        IN_REGION.with(|c| c.set(true));
        self.shared.drain(job);
        IN_REGION.with(|c| c.set(false));

        let mut spins = 0u32;
        while self.shared.active.load(Ordering::Acquire) != 0 {
            spins += 1;
            if spins.is_multiple_of(512) {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        let payload = self
            .shared
            .panic
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        // Released before re-raising: unwinding out of `run` while
        // holding it would poison the submit mutex, and a poisoned
        // submit mutex is a pool that refuses every future region.
        drop(guard);
        if let Some(payload) = payload {
            resume_unwind(payload);
        }
        true
    }
}

impl Drop for CpuPool {
    fn drop(&mut self) {
        {
            let _parked = self.shared.parked.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.shutdown.store(true, Ordering::Release);
            self.shared.epoch.fetch_add(1, Ordering::Release);
        }
        self.shared.cv.notify_all();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

fn worker_loop(shared: &Shared) {
    crate::threads::set_user_interactive_qos();
    let mut seen = 0u64;
    loop {
        seen = shared.wait_for_job(seen);
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
        // SAFETY: the `Acquire` on `epoch` inside `wait_for_job` pairs
        // with the submitter's `Release`, so this read sees the fully
        // written `Job` and nothing else is writing it (see the
        // `unsafe impl Sync for Shared` above).
        let job = unsafe { *shared.job.get() };
        IN_REGION.with(|c| c.set(true));
        shared.drain(job);
        IN_REGION.with(|c| c.set(false));
        shared.active.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    /// Every task index runs exactly once, on any width, and `run`
    /// does not return before they all have. Sabotage: change `drain`'s
    /// `fetch_add` to `load` and this reports duplicates.
    #[test]
    fn every_task_runs_exactly_once() {
        for threads in [1usize, 2, 4, 8] {
            let pool = CpuPool::new(threads);
            for n in [1usize, 3, 17, 1000] {
                let counts: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
                assert!(pool.run(n, &|task: usize| {
                    counts[task].fetch_add(1, Ordering::Relaxed);
                }));
                for (task, count) in counts.iter().enumerate() {
                    assert_eq!(
                        count.load(Ordering::Relaxed),
                        1,
                        "task {task} of {n} on {threads} threads"
                    );
                }
            }
        }
    }

    /// A pool that never parks burns a core per worker, and a pool whose
    /// wakeup can be lost hangs. Forcing the spin window to zero makes
    /// every single region go through the park/notify path, so this test
    /// exercises the branch a hot loop would never reach.
    ///
    /// Sabotage: move the `epoch.fetch_add` in `run` outside the
    /// `parked` mutex and this deadlocks or takes `PARK_TIMEOUT` per
    /// round instead of finishing promptly.
    #[test]
    fn regions_still_complete_when_every_worker_has_to_be_woken_from_a_park() {
        // The spin window is process-wide and cached, so drive the park
        // path by sleeping longer than it instead of changing it.
        let pool = CpuPool::new(4);
        for round in 0..20 {
            std::thread::sleep(spin_window() * 3);
            let seen = AtomicU32::new(0);
            assert!(pool.run(64, &|_| {
                seen.fetch_add(1, Ordering::Relaxed);
            }));
            assert_eq!(seen.load(Ordering::Relaxed), 64, "round {round}");
        }
    }

    /// The submitter must not return while a worker can still touch the
    /// closure. Each task writes through a borrow of a local, and the
    /// local is read immediately after `run` returns; a pool that let a
    /// worker outlive the region would be writing to a dead stack slot,
    /// which miri and ASan see and a plain run usually does not -- so
    /// this also asserts the *values*, which a late write corrupts.
    #[test]
    fn no_worker_touches_the_closure_after_run_returns() {
        let pool = CpuPool::new(4);
        for _ in 0..50 {
            let cells: Vec<AtomicU32> = (0..256).map(|_| AtomicU32::new(0)).collect();
            let local = 7u32;
            assert!(pool.run(256, &|task: usize| {
                cells[task].store(local + task as u32, Ordering::Relaxed);
            }));
            for (task, cell) in cells.iter().enumerate() {
                assert_eq!(cell.load(Ordering::Relaxed), 7 + task as u32);
            }
        }
    }

    /// A task that opens its own region would deadlock against the
    /// submit mutex. It runs inline instead, and still runs every task.
    #[test]
    fn a_nested_region_runs_inline_instead_of_deadlocking() {
        let pool = CpuPool::new(4);
        let inner_total = AtomicU32::new(0);
        assert!(pool.run(8, &|_outer: usize| {
            assert!(in_region());
            assert!(pool.run(5, &|_inner: usize| {
                inner_total.fetch_add(1, Ordering::Relaxed);
            }));
        }));
        assert_eq!(inner_total.load(Ordering::Relaxed), 40);
    }

    /// A second submitter is refused rather than queued, so a caller can
    /// fall back instead of blocking a whole request behind another.
    #[test]
    fn a_concurrent_submitter_is_refused_rather_than_serialized() {
        let pool = Arc::new(CpuPool::new(2));
        let refused = AtomicU32::new(0);
        let started = Arc::new(AtomicBool::new(false));
        std::thread::scope(|scope| {
            let held = Arc::clone(&pool);
            let started_w = Arc::clone(&started);
            scope.spawn(move || {
                held.run(1, &|_| {
                    started_w.store(true, Ordering::Release);
                    std::thread::sleep(Duration::from_millis(150));
                });
            });
            while !started.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            if !pool.run(4, &|_| {}) {
                refused.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(
            refused.load(Ordering::Relaxed),
            1,
            "the pool must report a busy region rather than block"
        );
    }

    /// A panic on a worker must reach the submitter, not hang it. The
    /// bug this guards is specific: an unwinding worker never decrements
    /// `active`, so `run` spins forever.
    ///
    /// Sabotage: delete the `catch_unwind` in `drain` and this test
    /// hangs instead of failing.
    #[test]
    fn a_panicking_task_is_re_raised_on_the_submitter() {
        let pool = CpuPool::new(4);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            pool.run(64, &|task: usize| {
                if task == 33 {
                    panic!("frink test panic");
                }
            });
        }));
        assert!(outcome.is_err(), "the panic must not be swallowed");
        // And the pool is still usable afterwards.
        let ran = AtomicU32::new(0);
        assert!(pool.run(10, &|_| {
            ran.fetch_add(1, Ordering::Relaxed);
        }));
        assert_eq!(ran.load(Ordering::Relaxed), 10);
    }

    /// Dropping the pool joins every worker. If shutdown did not reach a
    /// parked worker this test hangs, which is the failure mode of a
    /// pool that outlives the data it was built from.
    #[test]
    fn dropping_the_pool_joins_every_worker() {
        for threads in [1usize, 2, 6] {
            let pool = CpuPool::new(threads);
            assert!(pool.run(4, &|_| {}));
            std::thread::sleep(spin_window() * 2);
            drop(pool);
        }
    }

    #[test]
    fn a_single_thread_pool_runs_everything_on_the_submitter() {
        let pool = CpuPool::new(1);
        assert_eq!(pool.num_threads(), 1);
        let here = std::thread::current().id();
        let same = AtomicU32::new(0);
        assert!(pool.run(32, &|_| {
            if std::thread::current().id() == here {
                same.fetch_add(1, Ordering::Relaxed);
            }
        }));
        assert_eq!(same.load(Ordering::Relaxed), 32);
    }
}
