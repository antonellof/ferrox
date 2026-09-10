//! CPU worker-pool policy, shared by `ferrox` (CLI) and `ferrox-server`.
//!
//! Three things this module exists to control, all of which were measured
//! to matter far more than any kernel change on Apple Silicon:
//!
//! 1. **Thread count.** `available_parallelism()` on an M2 Pro reports 10
//!    (6 performance + 4 efficiency cores). Splitting a decode GEMV
//!    across all 10 makes every fork-join wait on the slowest E-core
//!    slice. llama.cpp defaults to `hw.perflevel0.physicalcpu` for
//!    exactly this reason (`common_cpu_get_num_math`), and collapses when
//!    forced above it -- 346 -> 176 tok/s on SmolLM2-135M Q8_0 going from
//!    `-t 4` to `-t 10`. So default to the performance-core count, not
//!    the logical-core count.
//!
//! 2. **Thread QoS.** macOS schedules threads onto E-cores based on their
//!    Quality-of-Service class, and QoS is *inherited* from whichever
//!    thread spawned them. Rayon builds its global pool lazily, on first
//!    use -- which inside `ferrox-server` is a Tokio `spawn_blocking`
//!    task, not the main thread. If that blocking thread carries a
//!    demoted QoS, every rayon worker inherits it and the whole matvec
//!    runs on efficiency cores. [`init_cpu_pool`] pins the workers to
//!    `USER_INTERACTIVE` explicitly so the pool's placement does not
//!    depend on who happened to touch rayon first.
//!
//! 3. **SMT siblings.** The same argument as (1), for the other kind of
//!    fake core. On a 16C/32T host `available_parallelism()` reports 32,
//!    so an auto-sized pool puts two workers on every physical core.
//!    MoE decode is memory-bandwidth-bound: a sibling adds no bandwidth,
//!    contends for the same core's load ports, and turns a spin barrier
//!    into a livelock-grade tax once the pool is oversubscribed. So the
//!    non-macOS width comes from [`physical_core_count`], which
//!    deduplicates `thread_siblings_list` across this process's affinity
//!    mask; sysfs being unreadable degrades to the
//!    `available_parallelism` answer rather than failing.

use std::collections::HashSet;
use std::path::Path;

/// macOS QoS classes (`sys/qos.h`). Only the ones we name are listed.
#[cfg(target_os = "macos")]
mod qos {
    pub const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    pub const QOS_CLASS_USER_INITIATED: u32 = 0x19;
    pub const QOS_CLASS_DEFAULT: u32 = 0x15;
    pub const QOS_CLASS_UTILITY: u32 = 0x11;
    pub const QOS_CLASS_BACKGROUND: u32 = 0x09;

    extern "C" {
        pub fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
        pub fn qos_class_self() -> u32;
    }

    pub fn name(class: u32) -> &'static str {
        match class {
            QOS_CLASS_USER_INTERACTIVE => "user-interactive",
            QOS_CLASS_USER_INITIATED => "user-initiated",
            QOS_CLASS_DEFAULT => "default",
            QOS_CLASS_UTILITY => "utility",
            QOS_CLASS_BACKGROUND => "background",
            _ => "unspecified",
        }
    }
}

/// Put the calling thread in the QoS class a decode worker needs.
///
/// The whole of point (2) in this module's header applies to *any* pool
/// this crate builds, not just rayon's, so both the rayon `start_handler`
/// in [`init_cpu_pool`] and [`crate::cpu_pool::CpuPool`]'s workers call
/// this one function rather than each spelling the class out. Two pools
/// that disagreed about their QoS would put one of them on E-cores and
/// nothing would say so.
///
/// A no-op off macOS, where QoS does not exist.
pub fn set_user_interactive_qos() {
    #[cfg(target_os = "macos")]
    {
        let before = unsafe { qos::qos_class_self() };
        // SAFETY: sets only the calling thread's QoS class.
        let rc = unsafe { qos::pthread_set_qos_class_self_np(qos::QOS_CLASS_USER_INTERACTIVE, 0) };
        if std::env::var_os("FERROX_QOS_LOG").is_some() {
            eprintln!(
                "ferrox: worker {:?} qos {} -> {} (rc={rc})",
                std::thread::current().id(),
                qos::name(before),
                qos::name(unsafe { qos::qos_class_self() }),
            );
        }
    }
}

/// The calling thread's macOS QoS class, as a human-readable name.
/// `None` off macOS, where the concept does not exist.
pub fn current_qos_name() -> Option<&'static str> {
    #[cfg(target_os = "macos")]
    {
        Some(qos::name(unsafe { qos::qos_class_self() }))
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Number of *performance* cores, which is the useful width for a decode
/// GEMV. On macOS this is `hw.perflevel0.physicalcpu` (llama.cpp reads
/// the same key). Elsewhere it is [`physical_core_count`] — one logical
/// CPU per physical core inside this process's affinity mask — and only
/// a host whose sysfs topology is unreadable falls all the way back to
/// `available_parallelism`.
pub fn perf_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(n) = sysctl_usize("hw.perflevel0.physicalcpu") {
            if n > 0 {
                return n;
            }
        }
    }
    physical_core_count()
}

#[cfg(target_os = "macos")]
fn sysctl_usize(name: &str) -> Option<usize> {
    use std::ffi::CString;
    extern "C" {
        fn sysctlbyname(
            name: *const std::os::raw::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> std::os::raw::c_int;
    }
    let key = CString::new(name).ok()?;
    let mut out: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    // SAFETY: `key` is NUL-terminated, and `out`/`len` describe a live
    // i32 of exactly `len` bytes, which is what these keys return.
    let rc = unsafe {
        sysctlbyname(
            key.as_ptr(),
            &mut out as *mut i32 as *mut std::ffi::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && out > 0 {
        Some(out as usize)
    } else {
        None
    }
}

// ---------------------------------------------------------------------
// SMT topology: one worker per physical core
//
// Ported from FreeToken's `freetoken/moe/cpu_executor.py`
// (`physical_core_cpus`, `resolve_threads_and_affinity`, and the pool
// sizing in `CpuMoeExecutor.__init__`). Apache-2.0; see
// `docs/THIRD_PARTY_NOTICES.md`.
//
// Every rule below is a pure function over an injected [`CpuTopology`],
// with a thin wrapper that reads the real `/sys`. That split is not
// cosmetic: the rules only *matter* on an SMT host with a restricted
// affinity mask, and no CI machine can be assumed to be one, so a
// sysfs-reading implementation would be a set of policies that are never
// actually exercised until a production box gets them wrong.
// ---------------------------------------------------------------------

/// Where Linux publishes per-CPU topology. [`CpuTopology::detect`] reads
/// `{root}/cpu{n}/topology/thread_siblings_list` under this directory.
pub const SYSFS_CPU_ROOT: &str = "/sys/devices/system/cpu";

/// Parse one `thread_siblings_list` line into the logical CPU ids it
/// names.
///
/// The kernel prints a cpulist, and both spellings occur on real
/// hardware: `0-1` on a box that numbers a core's siblings adjacently,
/// `0,64` on one that numbers every first-sibling before any second.
/// Handling only one of the two silently collapses or explodes the
/// deduplicated core count on the other, which is exactly the sizing
/// mistake this module exists to prevent — so both are parsed, and mixed
/// `0-1,64-65` forms with them. Tokens that are neither are dropped
/// rather than poisoning the list with a bogus CPU id.
pub fn parse_thread_siblings_list(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for token in text.trim().split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        match token.split_once('-') {
            Some((lo, hi)) => {
                // A cpulist range may carry a `:used/group` stride
                // suffix; sibling lists never do, so take the plain
                // prefix and ignore anything past it.
                let hi = hi.split(':').next().unwrap_or(hi);
                if let (Ok(lo), Ok(hi)) = (lo.trim().parse::<usize>(), hi.trim().parse::<usize>()) {
                    if lo <= hi {
                        out.extend(lo..=hi);
                    }
                }
            }
            None => {
                if let Ok(cpu) = token.parse::<usize>() {
                    out.push(cpu);
                }
            }
        }
    }
    out
}

/// The logical CPUs this process may actually run on, ascending.
///
/// This is the affinity mask, not the machine: a pool sized to the
/// machine inside a `taskset`- or cpuset-confined container
/// oversubscribes the slice it was given by exactly the factor it was
/// confined by, and every worker then time-slices against every other.
/// Linux reads `sched_getaffinity`; everywhere else (macOS included,
/// which has no equivalent) this is `0..available_parallelism()`, i.e.
/// today's answer.
pub fn process_affinity_cpus() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        if let Some(cpus) = sched_affinity_cpus() {
            return cpus;
        }
    }
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    (0..n).collect()
}

#[cfg(target_os = "linux")]
fn sched_affinity_cpus() -> Option<Vec<usize>> {
    // SAFETY: a zeroed `cpu_set_t` is a valid (empty) mask; the kernel
    // writes at most `cpusetsize` bytes into it, and we pass exactly the
    // size of the live local. `CPU_ISSET` only reads that same local.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let rc = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
        if rc != 0 {
            return None;
        }
        let cpus: Vec<usize> = (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
            .collect();
        if cpus.is_empty() {
            None
        } else {
            Some(cpus)
        }
    }
}

/// The SMT layout of the logical CPUs this process may run on: for each
/// allowed CPU, in ascending id order, the set of CPUs sharing its
/// physical core.
///
/// An **empty** sibling list means "sysfs did not answer for this CPU".
/// Such a CPU is treated as a physical core of its own and is never
/// merged with another unknown one — guessing that two unreadable CPUs
/// are siblings would silently halve the pool on any host that simply
/// has no `topology/` directory (a container with a masked `/sys`, some
/// hypervisors, anything non-Linux), and that regression arrives with no
/// error attached to it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CpuTopology {
    /// `(cpu id, its thread siblings)`, ascending by cpu id, unique.
    entries: Vec<(usize, Vec<usize>)>,
}

impl CpuTopology {
    /// Build a topology from explicit sibling lists — the injection
    /// point that makes every rule in this module testable on a host
    /// with no SMT, no root, and no particular CPU count.
    ///
    /// Entries are sorted by CPU id and deduplicated, because the rules
    /// pick the *first* allowed CPU of each physical core as that core's
    /// representative and that choice must not depend on the order the
    /// caller happened to enumerate in.
    pub fn from_sibling_lists<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = (usize, Vec<usize>)>,
    {
        let mut entries: Vec<(usize, Vec<usize>)> = entries.into_iter().collect();
        entries.sort_by_key(|(cpu, _)| *cpu);
        entries.dedup_by_key(|(cpu, _)| *cpu);
        Self { entries }
    }

    /// Read `{root}/cpu{n}/topology/thread_siblings_list` for each CPU in
    /// `allowed`. A CPU whose file is missing or unreadable gets an empty
    /// sibling list (see the type docs: it becomes its own core), so a
    /// host without sysfs topology degrades to one worker per allowed CPU
    /// instead of erroring out mid-boot.
    pub fn read_from(root: &Path, allowed: &[usize]) -> Self {
        Self::from_sibling_lists(allowed.iter().map(|&cpu| {
            let path = root
                .join(format!("cpu{cpu}"))
                .join("topology")
                .join("thread_siblings_list");
            let siblings = std::fs::read_to_string(&path)
                .map(|text| parse_thread_siblings_list(&text))
                .unwrap_or_default();
            (cpu, siblings)
        }))
    }

    /// This host's topology: [`SYSFS_CPU_ROOT`] restricted to
    /// [`process_affinity_cpus`].
    pub fn detect() -> Self {
        Self::read_from(Path::new(SYSFS_CPU_ROOT), &process_affinity_cpus())
    }

    /// The allowed logical CPUs, ascending.
    pub fn allowed_cpus(&self) -> Vec<usize> {
        self.entries.iter().map(|(cpu, _)| *cpu).collect()
    }

    /// Number of allowed logical CPUs, SMT siblings included.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no CPU at all is described.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One logical CPU per physical core, restricted to `topology`.
///
/// MoE decode is memory-bandwidth-bound, so SMT siblings only contend
/// for the same core's load ports without adding bandwidth; one logical
/// CPU per physical core is the fastest and, more importantly, the most
/// *stable* width. Siblings are deduplicated by the set their
/// `thread_siblings_list` names, and the lowest-numbered allowed CPU of
/// each core wins.
///
/// Never returns an empty list: with no topology it degrades to the
/// allowed CPUs, and with nothing allowed at all to `[0]`, because every
/// caller downstream divides work by this length.
pub fn physical_core_cpus_in(topology: &CpuTopology) -> Vec<usize> {
    let mut reps: Vec<usize> = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    for (cpu, siblings) in &topology.entries {
        if siblings.is_empty() {
            // Unknown topology for this CPU: a core of its own, never
            // merged with another unknown CPU.
            reps.push(*cpu);
            continue;
        }
        let mut key = siblings.clone();
        key.sort_unstable();
        key.dedup();
        if seen.insert(key) {
            reps.push(*cpu);
        }
    }
    if !reps.is_empty() {
        return reps;
    }
    let allowed = topology.allowed_cpus();
    if allowed.is_empty() {
        vec![0]
    } else {
        allowed
    }
}

/// [`physical_core_cpus_in`] against this host's real topology.
pub fn physical_core_cpus() -> Vec<usize> {
    physical_core_cpus_in(&CpuTopology::detect())
}

/// How many physical cores this process may run on.
///
/// Clamped by `available_parallelism`, which is the only figure that
/// accounts for a cgroup CPU *quota* — a container pinned to 8 CPUs but
/// throttled to two cores' worth of runtime reports 8 in its affinity
/// mask and 2 here, and a pool built for 8 spends the difference being
/// throttled mid-GEMV. A host with no readable topology lands on
/// `available_parallelism` outright, which is the pre-topology answer.
pub fn physical_core_count() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    physical_core_cpus().len().clamp(1, logical.max(1))
}

/// `(num_threads, core_ids)` for a pinned worker pool.
///
/// `requested == 0` means one worker per physical core, pinned to it.
/// An explicit count is honoured exactly, spreading across physical-core
/// **representatives first** and only then across the remaining logical
/// CPUs, so distinct hardware threads are used before any core is
/// doubled up — filling CPUs in numeric order instead would put workers
/// 0 and 1 on one core's two siblings on every host that numbers
/// siblings adjacently, i.e. half the pool contending before a second
/// core has been touched at all. A count larger than the allowed CPU set
/// wraps, deliberately: the caller asked for that width.
pub fn resolve_threads_and_affinity_in(
    requested: usize,
    topology: &CpuTopology,
) -> (usize, Vec<usize>) {
    let reps = physical_core_cpus_in(topology);
    if requested == 0 {
        let n = reps.len();
        return (n, reps);
    }
    let rep_set: HashSet<usize> = reps.iter().copied().collect();
    let mut order = reps;
    order.extend(
        topology
            .allowed_cpus()
            .into_iter()
            .filter(|cpu| !rep_set.contains(cpu)),
    );
    if order.is_empty() {
        order.push(0);
    }
    let core_ids = (0..requested).map(|i| order[i % order.len()]).collect();
    (requested, core_ids)
}

/// [`resolve_threads_and_affinity_in`] against this host's real topology.
pub fn resolve_threads_and_affinity(requested: usize) -> (usize, Vec<usize>) {
    resolve_threads_and_affinity_in(requested, &CpuTopology::detect())
}

/// A sized worker pool: how many workers, which logical CPU each pins
/// to, and the CPU set aside for a coordinator, if any.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CpuPoolPlan {
    /// Worker count. Always equal to `core_ids.len()`.
    pub num_threads: usize,
    /// The logical CPU each worker pins to, in worker order.
    pub core_ids: Vec<usize>,
    /// The logical CPU donated to the coordinator, or `None` when no
    /// coordinator was asked for or the pool was too small to donate.
    pub coordinator_cpu: Option<usize>,
}

/// Size a pinned pool, optionally reserving a core for a coordinator
/// thread (the one that polls the device doorbell and drives the pool).
///
/// The donation rule is the point: under *auto* sizing only, and only
/// while more than two workers survive it, the coordinator takes the
/// last physical core and the pool drops from N to N-1 workers. A
/// coordinator that instead time-slices against a full-width pool
/// measurably destabilizes throughput on a fully-subscribed box — it is
/// a spinner, so the worker sharing its core is the one the fork-join
/// barrier waits for, every single step. An explicit thread count is
/// never silently reduced: the operator asked for that width.
pub fn plan_cpu_pool_in(
    requested: usize,
    reserve_coordinator: bool,
    topology: &CpuTopology,
) -> CpuPoolPlan {
    let (mut num_threads, mut core_ids) = resolve_threads_and_affinity_in(requested, topology);
    let mut coordinator_cpu = None;
    if reserve_coordinator && requested == 0 && num_threads > 2 {
        coordinator_cpu = core_ids.pop();
        num_threads -= 1;
    }
    CpuPoolPlan {
        num_threads,
        core_ids,
        coordinator_cpu,
    }
}

/// [`plan_cpu_pool_in`] against this host's real topology.
pub fn plan_cpu_pool(requested: usize, reserve_coordinator: bool) -> CpuPoolPlan {
    plan_cpu_pool_in(requested, reserve_coordinator, &CpuTopology::detect())
}

/// The intra-op width a *second* thread pool may still use once `plan`
/// has claimed its cores: `physical_cores - workers - coordinator - 1`,
/// clamped into `1..=configured`. The trailing `-1` is the calling
/// thread itself, which is running the surrounding forward.
///
/// Without this clamp the framework's own pool defaults to the full core
/// count and each of its threads lands on a core a pinned, spinning
/// worker already owns — the pinned pool cannot yield, so the
/// oversubscription is paid as scheduler latency on the decode critical
/// path instead of buying parallelism anywhere. Never returns 0: a
/// zero-width intra-op pool is not a degraded configuration, it is a
/// broken one.
pub fn clamp_intra_op_threads(
    configured: usize,
    plan: &CpuPoolPlan,
    physical_cores: usize,
) -> usize {
    let coordinator = usize::from(plan.coordinator_cpu.is_some());
    let spare = physical_cores
        .saturating_sub(plan.num_threads)
        .saturating_sub(coordinator)
        .saturating_sub(1);
    configured.min(spare).max(1)
}

/// How many rayon workers to run: `FERROX_CPU_THREADS`, else
/// `RAYON_NUM_THREADS`, else [`perf_core_count`].
pub fn resolve_cpu_threads() -> usize {
    for key in ["FERROX_CPU_THREADS", "RAYON_NUM_THREADS"] {
        if let Ok(v) = std::env::var(key) {
            if let Ok(n) = v.trim().parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    perf_core_count()
}

/// Prefer serial when fork-join overhead exceeds the matvec work.
/// ~256k element-ops matches the previous `prefer_serial_matvec` gate.
pub fn should_parallelize(n_rows: usize, n_cols: usize) -> bool {
    n_rows > 1 && n_rows.saturating_mul(n_cols) >= 256_000
}

/// One output row per slot; parallel when [`should_parallelize`] says the
/// matvec is big enough to pay for the fork-join. Rows are never split.
pub fn for_each_row<F>(output: &mut [f32], n_rows: usize, n_cols: usize, row_fn: F)
where
    F: Fn(usize, &mut f32) + Send + Sync,
{
    let n = n_rows.min(output.len());
    if !should_parallelize(n, n_cols) {
        for (row, out) in output.iter_mut().enumerate().take(n) {
            row_fn(row, out);
        }
        return;
    }
    // Global rayon, the same pool act-quant uses. A second dedicated
    // matvec pool of P-core width was measured to regress CPU pp512 on
    // Host B (~40 -> ~14 tok/s), so there is only one pool.
    let rows = &mut output[..n];
    crate::par::items_mut(rows, 1, |row, out| row_fn(row, out));
}

/// Chunk-parallel sibling of [`for_each_row`]; chunks are never split.
pub fn for_each_chunk_init<S, I, F>(
    output: &mut [f32],
    chunk_len: usize,
    work_per_chunk: usize,
    init: I,
    f: F,
) where
    I: Fn() -> S + Send + Sync,
    S: Send,
    F: Fn(&mut S, usize, &mut [f32]) + Send + Sync,
{
    if chunk_len == 0 {
        return;
    }
    let n_chunks = output.len() / chunk_len;
    if !should_parallelize(n_chunks, work_per_chunk) {
        let mut state = init();
        for (i, chunk) in output[..n_chunks * chunk_len]
            .chunks_mut(chunk_len)
            .enumerate()
        {
            f(&mut state, i, chunk);
        }
        return;
    }
    let chunks = &mut output[..n_chunks * chunk_len];
    crate::par::chunks_mut_init(chunks, chunk_len, 1, init, |state, i, c| f(state, i, c));
}

/// Stack each rayon worker gets.
///
/// Rust's default for a spawned thread is 2 MiB, and that was fine while
/// workers only ever ran one matvec chunk. Since
/// [`crate::par::on_workers`] a WHOLE forward pass runs on a worker: the
/// decoder's layer loop, its Metal and CUDA arms, and every scratch
/// buffer any of them puts on the stack. The thread that used to run
/// that body is the process's main thread, which gets 8 MiB on macOS and
/// Linux, so this matches it rather than quietly halving the headroom of
/// code that never had to think about it.
const WORKER_STACK_BYTES: usize = 8 * 1024 * 1024;

/// Builds the global rayon pool with an explicit width, an explicit QoS
/// and an explicit stack, so none of the three depends on which thread
/// first touched rayon. Safe to call more than once and from either
/// binary; a pool that already exists is left alone.
///
/// Returns the thread count the pool was built with, or `None` if the
/// global pool already existed.
pub fn init_cpu_pool() -> Option<usize> {
    let threads = resolve_cpu_threads();
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(WORKER_STACK_BYTES)
        .start_handler(move |_idx| set_user_interactive_qos())
        .build_global()
        .is_ok();
    if built {
        Some(threads)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perf_core_count_is_at_least_one_and_no_more_than_logical_cores() {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let perf = perf_core_count();
        assert!(perf >= 1, "perf core count must be positive, got {perf}");
        assert!(
            perf <= logical,
            "perf cores ({perf}) cannot exceed logical cores ({logical})"
        );
    }

    #[test]
    fn resolved_thread_count_falls_back_to_perf_cores_without_env_overrides() {
        // `resolve_cpu_threads` reads process-global env, and tests share
        // a process, so assert the fallback only when nothing is set.
        if std::env::var_os("FERROX_CPU_THREADS").is_none()
            && std::env::var_os("RAYON_NUM_THREADS").is_none()
        {
            assert_eq!(resolve_cpu_threads(), perf_core_count());
        }
    }

    #[test]
    fn current_qos_name_is_reported_on_macos_and_absent_elsewhere() {
        let qos = current_qos_name();
        #[cfg(target_os = "macos")]
        assert!(qos.is_some(), "macOS must report a QoS class");
        #[cfg(not(target_os = "macos"))]
        assert!(qos.is_none(), "QoS is a macOS-only concept");
    }

    /// An 8-logical/4-physical SMT host, siblings numbered adjacently.
    /// Every CPU is in the affinity mask.
    fn smt_8t_4c() -> CpuTopology {
        CpuTopology::from_sibling_lists([
            (0, vec![0, 1]),
            (1, vec![0, 1]),
            (2, vec![2, 3]),
            (3, vec![2, 3]),
            (4, vec![4, 5]),
            (5, vec![4, 5]),
            (6, vec![6, 7]),
            (7, vec![6, 7]),
        ])
    }

    /// THE central test, and it FAILS against the pre-topology
    /// implementation: on this host `available_parallelism()` reports 8
    /// while there are only 4 physical cores, so the old
    /// `perf_core_count` fallback would have sized the pool to 8 and put
    /// two bandwidth-bound workers on every core. The auto width must be
    /// the physical-core count, and the chosen CPUs must be one per core.
    #[test]
    fn an_smt_host_is_sized_to_its_physical_cores_not_its_logical_cpus() {
        let topology = smt_8t_4c();
        assert_eq!(
            topology.len(),
            8,
            "the fixture must have twice as many logical CPUs as cores"
        );
        assert_eq!(
            physical_core_cpus_in(&topology),
            vec![0, 2, 4, 6],
            "one representative per physical core, lowest sibling first"
        );
        let (threads, core_ids) = resolve_threads_and_affinity_in(0, &topology);
        assert_eq!(threads, 4, "auto sizing must not count SMT siblings");
        assert_eq!(core_ids, vec![0, 2, 4, 6]);
    }

    #[test]
    fn siblings_numbered_apart_are_deduplicated_the_same_as_adjacent_ones() {
        // AMD/POWER style: all first siblings, then all second siblings.
        let topology = CpuTopology::from_sibling_lists([
            (0, vec![0, 64]),
            (1, vec![1, 65]),
            (64, vec![0, 64]),
            (65, vec![1, 65]),
        ]);
        assert_eq!(physical_core_cpus_in(&topology), vec![0, 1]);
    }

    #[test]
    fn thread_siblings_lists_parse_as_ranges_comma_lists_and_mixtures() {
        assert_eq!(parse_thread_siblings_list("0-1\n"), vec![0, 1]);
        assert_eq!(parse_thread_siblings_list("0,64\n"), vec![0, 64]);
        assert_eq!(parse_thread_siblings_list(" 3 "), vec![3]);
        assert_eq!(parse_thread_siblings_list("0-1,64-65"), vec![0, 1, 64, 65]);
        assert_eq!(parse_thread_siblings_list("2-4"), vec![2, 3, 4]);
        // Garbage tokens are dropped, not turned into CPU 0.
        assert_eq!(parse_thread_siblings_list(""), Vec::<usize>::new());
        assert_eq!(parse_thread_siblings_list("x,-,7"), vec![7]);
        // A descending range names nothing rather than panicking.
        assert_eq!(parse_thread_siblings_list("5-1"), Vec::<usize>::new());
    }

    #[test]
    fn a_host_without_sysfs_topology_degrades_to_one_worker_per_allowed_cpu() {
        // Empty sibling lists model an unreadable `topology/` directory:
        // the answer must be today's `available_parallelism`-shaped one.
        let topology =
            CpuTopology::from_sibling_lists((0..6).map(|cpu| (cpu, Vec::<usize>::new())));
        assert_eq!(physical_core_cpus_in(&topology), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(resolve_threads_and_affinity_in(0, &topology).0, 6);
    }

    #[test]
    fn an_empty_topology_still_yields_one_usable_cpu() {
        let topology = CpuTopology::default();
        assert!(topology.is_empty());
        assert_eq!(physical_core_cpus_in(&topology), vec![0]);
        assert_eq!(resolve_threads_and_affinity_in(0, &topology), (1, vec![0]));
        assert_eq!(
            resolve_threads_and_affinity_in(2, &topology),
            (2, vec![0, 0])
        );
    }

    #[test]
    fn cores_outside_the_affinity_mask_are_never_used_as_representatives() {
        // Same 8T/4C host, but only CPUs 1, 3, 4, 5 are allowed. Cores
        // {0,1} and {2,3} contribute their high sibling (the only allowed
        // one); core {4,5} contributes 4 once, not twice.
        let full = smt_8t_4c();
        let allowed = [1usize, 3, 4, 5];
        let topology = CpuTopology::from_sibling_lists(
            full.allowed_cpus()
                .into_iter()
                .filter(|cpu| allowed.contains(cpu))
                .map(|cpu| {
                    (
                        cpu,
                        parse_thread_siblings_list(&format!("{}-{}", cpu & !1, cpu | 1)),
                    )
                }),
        );
        assert_eq!(physical_core_cpus_in(&topology), vec![1, 3, 4]);
        assert_eq!(resolve_threads_and_affinity_in(0, &topology).0, 3);
    }

    #[test]
    fn an_explicit_count_fills_physical_cores_before_doubling_up_siblings() {
        let topology = smt_8t_4c();
        // Four workers land on four distinct cores...
        assert_eq!(
            resolve_threads_and_affinity_in(4, &topology),
            (4, vec![0, 2, 4, 6])
        );
        // ...and only the fifth onward touches a sibling.
        assert_eq!(
            resolve_threads_and_affinity_in(6, &topology),
            (6, vec![0, 2, 4, 6, 1, 3])
        );
        assert_eq!(
            resolve_threads_and_affinity_in(8, &topology),
            (8, vec![0, 2, 4, 6, 1, 3, 5, 7])
        );
    }

    #[test]
    fn an_explicit_count_larger_than_the_machine_wraps_instead_of_truncating() {
        let topology = smt_8t_4c();
        let (threads, core_ids) = resolve_threads_and_affinity_in(10, &topology);
        assert_eq!(threads, 10, "an explicit width is honoured exactly");
        assert_eq!(core_ids.len(), 10);
        assert_eq!(&core_ids[8..], &[0, 2], "wraps back to the representatives");
    }

    #[test]
    fn auto_sizing_donates_the_last_physical_core_to_the_coordinator() {
        let plan = plan_cpu_pool_in(0, true, &smt_8t_4c());
        assert_eq!(plan.num_threads, 3, "workers drop from N to N-1");
        assert_eq!(plan.core_ids, vec![0, 2, 4]);
        assert_eq!(plan.coordinator_cpu, Some(6));
        assert_eq!(plan.num_threads, plan.core_ids.len());
    }

    #[test]
    fn no_core_is_donated_without_a_coordinator_or_for_an_explicit_count() {
        let topology = smt_8t_4c();
        let no_coordinator = plan_cpu_pool_in(0, false, &topology);
        assert_eq!(no_coordinator.num_threads, 4);
        assert_eq!(no_coordinator.coordinator_cpu, None);

        // An operator-supplied width is never silently reduced.
        let explicit = plan_cpu_pool_in(4, true, &topology);
        assert_eq!(explicit.num_threads, 4);
        assert_eq!(explicit.coordinator_cpu, None);
    }

    #[test]
    fn a_pool_of_two_or_fewer_keeps_its_workers_rather_than_donating() {
        // 2 physical cores: donating would leave a single worker, so the
        // coordinator shares instead.
        let dual = CpuTopology::from_sibling_lists([
            (0, vec![0, 1]),
            (1, vec![0, 1]),
            (2, vec![2, 3]),
            (3, vec![2, 3]),
        ]);
        let plan = plan_cpu_pool_in(0, true, &dual);
        assert_eq!(plan.num_threads, 2);
        assert_eq!(plan.coordinator_cpu, None);
    }

    #[test]
    fn the_intra_op_clamp_leaves_a_core_for_the_calling_thread() {
        // 16 physical cores, 3 workers + coordinator -> 16-3-1-1 = 11.
        let plan = plan_cpu_pool_in(0, true, &smt_8t_4c());
        assert_eq!(clamp_intra_op_threads(16, &plan, 16), 11);
        // A configured width below the spare is left alone.
        assert_eq!(clamp_intra_op_threads(4, &plan, 16), 4);
        // A fully claimed machine still gets a usable intra-op width.
        assert_eq!(clamp_intra_op_threads(16, &plan, 4), 1);
        assert_eq!(clamp_intra_op_threads(16, &plan, 0), 1);
    }

    #[test]
    fn sibling_lists_are_read_from_a_sysfs_layout_on_disk() {
        let root = std::env::temp_dir().join(format!(
            "ferrox-threads-sysfs-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        // cpu0/cpu1 are siblings; cpu2 has no topology directory at all.
        for cpu in [0usize, 1] {
            let dir = root.join(format!("cpu{cpu}")).join("topology");
            std::fs::create_dir_all(&dir).expect("temp sysfs tree must be creatable");
            std::fs::write(dir.join("thread_siblings_list"), "0-1\n")
                .expect("temp sibling list must be writable");
        }
        let topology = CpuTopology::read_from(&root, &[0, 1, 2]);
        assert_eq!(topology.allowed_cpus(), vec![0, 1, 2]);
        assert_eq!(
            physical_core_cpus_in(&topology),
            vec![0, 2],
            "cpu2 is unreadable, so it counts as a core of its own"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn this_hosts_physical_core_count_is_positive_and_within_its_logical_cpus() {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let physical = physical_core_count();
        assert!(physical >= 1, "physical core count must be positive");
        assert!(
            physical <= logical,
            "physical cores ({physical}) cannot exceed logical cores ({logical})"
        );
        assert_eq!(
            physical_core_cpus().len().clamp(1, logical.max(1)),
            physical
        );
    }

    #[test]
    fn this_hosts_affinity_mask_is_non_empty_and_ascending() {
        let cpus = process_affinity_cpus();
        assert!(!cpus.is_empty(), "a running process may run somewhere");
        assert!(
            cpus.windows(2).all(|w| w[0] < w[1]),
            "affinity CPUs must be ascending and unique: {cpus:?}"
        );
    }

    #[test]
    fn for_each_row_parallel_matches_serial() {
        let n = 4097usize;
        let f = |row: usize| ((row % 97) as f32) * 0.25 - 3.0;

        let mut par = vec![0.0f32; n];
        for_each_row(&mut par, n, 4096, |row, slot| *slot = f(row));

        let mut serial = vec![0.0f32; n];
        for (row, slot) in serial.iter_mut().enumerate() {
            *slot = f(row);
        }
        assert_eq!(par, serial);
    }
}
