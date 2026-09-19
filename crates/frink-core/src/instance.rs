//! Who else is already running a model on this box.
//!
//! One frink process holding a model saturates the machine by design:
//! prefill is a dense GEMM across every core, and the decode pool spins.
//! Two of them do not run at half speed each -- they thrash, and the
//! numbers both of them report become meaningless. The same is true of a
//! `frink-server` left running in another terminal while a benchmark
//! starts.
//!
//! So a model-loading process registers itself here first, and by
//! default refuses to start when another live instance already holds
//! one. The registry is a directory of one small file per process:
//!
//! ```text
//! $FRINK_INSTANCE_DIR (default ~/.cache/frink/instances)/<pid>
//! ```
//!
//! **This is advisory, not a lock.** Two processes starting in the same
//! instant can each see the other and both refuse -- which is the safe
//! direction -- but nothing here prevents a determined caller from
//! running two models. It exists to stop the accident, not to enforce a
//! policy against the operator.

use std::io::Write;
use std::path::{Path, PathBuf};

/// Whether a second model-loading process may start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstancePolicy {
    /// Refuse to start while another live instance holds a model.
    Single,
    /// Start anyway. The caller has accepted the CPU contention.
    Multi,
}

impl InstancePolicy {
    /// `FRINK_ALLOW_MULTIPLE_INSTANCES=1|true|on` selects `Multi`.
    /// The CLI flag wins over the environment; this is the fallback.
    pub fn from_env_or(default: InstancePolicy) -> InstancePolicy {
        match std::env::var("FRINK_ALLOW_MULTIPLE_INSTANCES")
            .ok()
            .as_deref()
        {
            Some("1") | Some("true") | Some("on") => InstancePolicy::Multi,
            Some("0") | Some("false") | Some("off") => InstancePolicy::Single,
            _ => default,
        }
    }
}

/// One live frink process, as it described itself at startup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceInfo {
    pub pid: u32,
    /// Subcommand or binary: `run`, `bench`, `server`, …
    pub command: String,
    /// Model path, when the process named one.
    pub model: Option<String>,
    /// `cpu` / `metal` / `cuda`.
    pub backend: String,
    /// Seconds since the Unix epoch, or 0 if the clock would not say.
    pub started_unix: u64,
}

impl InstanceInfo {
    fn encode(&self) -> String {
        // Tab-separated so no dependency is needed to read it back, and
        // tabs are stripped from the free-text fields on the way in so a
        // path can never split a record.
        format!(
            "{}\t{}\t{}\t{}\t{}\n",
            self.pid,
            scrub(&self.command),
            scrub(self.model.as_deref().unwrap_or("")),
            scrub(&self.backend),
            self.started_unix,
        )
    }

    fn decode(line: &str) -> Option<InstanceInfo> {
        let mut f = line.trim_end_matches('\n').split('\t');
        let pid = f.next()?.parse().ok()?;
        let command = f.next()?.to_string();
        let model = match f.next()? {
            "" => None,
            m => Some(m.to_string()),
        };
        let backend = f.next()?.to_string();
        let started_unix = f.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        Some(InstanceInfo {
            pid,
            command,
            model,
            backend,
            started_unix,
        })
    }

    /// `bench pid 1234, Metal, models/foo.gguf`
    pub fn describe(&self) -> String {
        let model = self
            .model
            .as_deref()
            .map(|m| format!(", {m}"))
            .unwrap_or_default();
        format!(
            "{} pid {}, {}{}",
            self.command, self.pid, self.backend, model
        )
    }
}

fn scrub(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

/// Removes this process's registry entry when it drops.
///
/// A `SIGKILL` or a panic-abort skips this, which is why every read
/// prunes entries whose pid is gone rather than trusting the directory.
pub struct InstanceGuard {
    path: PathBuf,
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Another live instance already holds a model.
#[derive(Debug)]
pub struct InstanceConflict {
    pub others: Vec<InstanceInfo>,
}

impl std::fmt::Display for InstanceConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{} frink instance(s) are already running a model on this host:",
            self.others.len()
        )?;
        for o in &self.others {
            writeln!(f, "  - {}", o.describe())?;
        }
        write!(
            f,
            "Running several models at once does not share the machine -- it \
             thrashes it, and any timing either process reports is noise. Stop \
             the other instance, or pass --allow-multiple-instances (or set \
             FRINK_ALLOW_MULTIPLE_INSTANCES=1) to start anyway."
        )
    }
}

impl std::error::Error for InstanceConflict {}

/// The accelerator this process would actually use, for the registry
/// entry. Compiled-in features only decide what is *possible*; the
/// runtime toggles decide what is enabled, so both are consulted.
pub fn current_backend() -> &'static str {
    #[cfg(feature = "metal")]
    {
        if crate::weight_matrix::metal_dense_enabled() {
            return "metal";
        }
    }
    #[cfg(feature = "cuda")]
    {
        if crate::weight_matrix::cuda_dense_enabled() {
            return "cuda";
        }
    }
    "cpu"
}

/// Where the per-process files live. `FRINK_INSTANCE_DIR` overrides.
pub fn registry_dir() -> PathBuf {
    if let Ok(d) = std::env::var("FRINK_INSTANCE_DIR") {
        return PathBuf::from(d);
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache"))
        })
        .unwrap_or_else(std::env::temp_dir);
    base.join("frink").join("instances")
}

/// Every registered instance whose process is still alive, excluding
/// this one. Entries for dead processes are deleted as they are found.
pub fn live_instances() -> Vec<InstanceInfo> {
    live_instances_in(&registry_dir(), std::process::id())
}

fn live_instances_in(dir: &Path, self_pid: u32) -> Vec<InstanceInfo> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<(PathBuf, InstanceInfo)> = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        match InstanceInfo::decode(&body) {
            Some(info) if info.pid != self_pid => found.push((path, info)),
            // Unreadable or truncated: a half-written file from a
            // process that died mid-register. Not ours to interpret.
            Some(_) => {}
            None => {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    let alive = alive_pids(&found.iter().map(|(_, i)| i.pid).collect::<Vec<_>>());
    let mut out = Vec::new();
    for (path, info) in found {
        if alive.contains(&info.pid) {
            out.push(info);
        } else {
            let _ = std::fs::remove_file(&path);
        }
    }
    out.sort_by_key(|i| i.pid);
    out
}

/// Which of `pids` still exist. One `ps` call for the whole set, or
/// `/proc` on Linux where no child process is needed at all.
fn alive_pids(pids: &[u32]) -> Vec<u32> {
    if pids.is_empty() {
        return Vec::new();
    }
    if Path::new("/proc/self").exists() {
        return pids
            .iter()
            .copied()
            .filter(|p| Path::new(&format!("/proc/{p}")).exists())
            .collect();
    }
    let list = pids
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let Ok(out) = std::process::Command::new("ps")
        .args(["-p", &list, "-o", "pid="])
        .output()
    else {
        // If we cannot tell, assume they are alive: refusing to start is
        // recoverable, trampling a running model is not.
        return pids.to_vec();
    };
    parse_ps_pids(&String::from_utf8_lossy(&out.stdout))
}

fn parse_ps_pids(stdout: &str) -> Vec<u32> {
    stdout
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Registers this process and enforces `policy`.
///
/// Registration happens *before* the conflict check, so two processes
/// racing each other both see the other and both refuse. That is the
/// safe direction: a spurious refusal costs a retry, a spurious admit
/// costs a thrashed host and two meaningless measurements.
pub fn register(
    command: &str,
    model: Option<&str>,
    backend: &str,
    policy: InstancePolicy,
) -> Result<InstanceGuard, InstanceConflict> {
    let dir = registry_dir();
    let pid = std::process::id();
    let info = InstanceInfo {
        pid,
        command: command.to_string(),
        model: model.map(str::to_string),
        backend: backend.to_string(),
        started_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    let path = dir.join(pid.to_string());
    // A registry we cannot write is not a reason to refuse to run: the
    // guard degrades to a no-op rather than making an unwritable cache
    // directory fatal.
    let wrote = std::fs::create_dir_all(&dir).is_ok()
        && std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(info.encode().as_bytes()))
            .is_ok();
    let guard = InstanceGuard {
        path: if wrote { path } else { PathBuf::new() },
    };
    if policy == InstancePolicy::Multi {
        return Ok(guard);
    }
    let others = live_instances_in(&dir, pid);
    if others.is_empty() {
        Ok(guard)
    } else {
        // `guard` drops here, removing our own entry, so a refused start
        // does not leave a ghost behind for the next process to trip on.
        Err(InstanceConflict { others })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("frink-instance-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_record_round_trips_including_an_absent_model() {
        let i = InstanceInfo {
            pid: 42,
            command: "bench".into(),
            model: None,
            backend: "metal".into(),
            started_unix: 7,
        };
        assert_eq!(InstanceInfo::decode(&i.encode()), Some(i));
    }

    #[test]
    fn a_tab_in_a_model_path_cannot_split_the_record() {
        let i = InstanceInfo {
            pid: 42,
            command: "run".into(),
            model: Some("models/we\tird\nname.gguf".into()),
            backend: "cpu".into(),
            started_unix: 7,
        };
        let back = InstanceInfo::decode(&i.encode()).expect("still one record");
        assert_eq!(back.pid, 42);
        assert_eq!(back.backend, "cpu", "fields did not shift");
        assert_eq!(back.started_unix, 7);
        assert_eq!(back.model.as_deref(), Some("models/we ird name.gguf"));
    }

    #[test]
    fn a_dead_pid_is_pruned_rather_than_reported_as_a_conflict() {
        let d = tmpdir("dead");
        // pid 1 is alive on every unix; a pid this large is not in use.
        let dead = InstanceInfo {
            pid: 4_000_000_000,
            command: "server".into(),
            model: Some("m.gguf".into()),
            backend: "cpu".into(),
            started_unix: 1,
        };
        std::fs::write(d.join(dead.pid.to_string()), dead.encode()).unwrap();
        assert!(live_instances_in(&d, std::process::id()).is_empty());
        assert!(
            !d.join(dead.pid.to_string()).exists(),
            "the stale entry is deleted, not just skipped"
        );
    }

    #[test]
    fn a_live_pid_is_reported_and_its_entry_kept() {
        let d = tmpdir("live");
        // This test process is by definition alive; register it under a
        // different "self" so it counts as an other.
        let me = InstanceInfo {
            pid: std::process::id(),
            command: "run".into(),
            model: Some("m.gguf".into()),
            backend: "metal".into(),
            started_unix: 1,
        };
        std::fs::write(d.join(me.pid.to_string()), me.encode()).unwrap();
        let others = live_instances_in(&d, 0);
        assert_eq!(others.len(), 1);
        assert_eq!(others[0].pid, me.pid);
        assert!(d.join(me.pid.to_string()).exists());
    }

    #[test]
    fn a_process_never_conflicts_with_its_own_entry() {
        let d = tmpdir("self");
        let me = InstanceInfo {
            pid: std::process::id(),
            command: "run".into(),
            model: None,
            backend: "cpu".into(),
            started_unix: 1,
        };
        std::fs::write(d.join(me.pid.to_string()), me.encode()).unwrap();
        assert!(live_instances_in(&d, std::process::id()).is_empty());
    }

    #[test]
    fn ps_output_parses_to_pids_and_ignores_a_header() {
        assert_eq!(parse_ps_pids(" 1234\n 5678\n"), vec![1234, 5678]);
        assert_eq!(parse_ps_pids(""), Vec::<u32>::new());
    }

    #[test]
    fn the_conflict_message_names_every_other_instance_and_the_escape_hatch() {
        let c = InstanceConflict {
            others: vec![InstanceInfo {
                pid: 99,
                command: "server".into(),
                model: Some("models/a.gguf".into()),
                backend: "metal".into(),
                started_unix: 0,
            }],
        };
        let s = c.to_string();
        assert!(s.contains("server pid 99, metal, models/a.gguf"), "{s}");
        assert!(s.contains("--allow-multiple-instances"), "{s}");
        assert!(s.contains("FRINK_ALLOW_MULTIPLE_INSTANCES=1"), "{s}");
    }

    #[test]
    fn the_env_var_selects_a_policy_but_the_caller_default_wins_when_unset() {
        // Serialised implicitly: these three run in the same test and
        // restore the variable before returning.
        let prev = std::env::var("FRINK_ALLOW_MULTIPLE_INSTANCES").ok();
        std::env::set_var("FRINK_ALLOW_MULTIPLE_INSTANCES", "1");
        assert_eq!(
            InstancePolicy::from_env_or(InstancePolicy::Single),
            InstancePolicy::Multi
        );
        std::env::set_var("FRINK_ALLOW_MULTIPLE_INSTANCES", "0");
        assert_eq!(
            InstancePolicy::from_env_or(InstancePolicy::Multi),
            InstancePolicy::Single
        );
        std::env::remove_var("FRINK_ALLOW_MULTIPLE_INSTANCES");
        assert_eq!(
            InstancePolicy::from_env_or(InstancePolicy::Multi),
            InstancePolicy::Multi
        );
        if let Some(p) = prev {
            std::env::set_var("FRINK_ALLOW_MULTIPLE_INSTANCES", p);
        }
    }
}
