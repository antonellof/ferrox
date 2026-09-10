//! `ferrox bench --suite` / `--render`: the llama-bench-shaped ledger.
//!
//! Numbers are `pp<N>` / `tg<N>` from `ferrox bench` vs `llama-bench`,
//! with no HTTP, template, tokenizer, or sampler. That is what a kernel
//! change moves, and what [`benchmarks/RESULTS.md`](../../../benchmarks/RESULTS.md)
//! quotes.
//!
//! Each suite entry runs in a **fresh child process**. Backend selection
//! reads process-global environment and the rayon pool is built once, so
//! benchmarking several backends inside one process would silently
//! measure the first one's configuration for all of them.

use anyhow::Context;
use std::path::{Path, PathBuf};

/// One `models[]` entry of `benchmarks/suite.json`.
pub(crate) struct SuiteEntry {
    pub(crate) id: String,
    pub(crate) name: String,
    gguf: String,
    backends: Vec<String>,
    estimated_ram_gb: f64,
}

pub struct SuiteArgs {
    pub bench_dir: PathBuf,
    pub n_prompt: usize,
    pub n_gen: usize,
    pub reps: usize,
    pub only_id: Option<String>,
    pub only_backend: Option<String>,
    pub fit_host: bool,
    pub skip_missing: bool,
    /// Forwarded to every child `bench` run: the 1-minute load average
    /// above which a timed run refuses to start.
    pub max_load: f64,
}

/// A filesystem-safe short name for a host label.
///
/// Lowercase, non-alphanumerics collapsed to `-`, trimmed. Only used to
/// keep two machines' receipts from overwriting each other; the
/// authoritative label stays inside the receipt as `host_spec.label`.
fn host_slug(label: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for c in label.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// The machine a receipt describes, as one string: the host spec plus
/// the card, when it ran on one.
///
/// Both the section list and the rows themselves are keyed on this,
/// through this ONE function. They used to be two separate reads of
/// `host_spec.label`, which is a CPU string -- so a GTX 1080 row and an
/// RTX 3080 row taken on the same Xeon sorted into one section, which
/// is precisely the merge the grouping exists to prevent.
pub(crate) fn host_identity(receipt: &serde_json::Value) -> String {
    let label = receipt
        .get("host_spec")
        .and_then(|h| h.get("label"))
        .and_then(|v| v.as_str())
        .unwrap_or("unrecorded (receipt written before 0.13.0)");
    match receipt.get("accelerator").and_then(|v| v.as_str()) {
        Some(card) => format!("{label} + {card}"),
        None => label.to_string(),
    }
}

fn suite_path(bench_dir: &Path) -> PathBuf {
    bench_dir.join("suite.json")
}

pub(crate) fn engine_receipt_dir(bench_dir: &Path) -> PathBuf {
    bench_dir.join("receipts").join("engine")
}

pub(crate) fn load_suite(bench_dir: &Path) -> anyhow::Result<Vec<SuiteEntry>> {
    let path = suite_path(bench_dir);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading suite at {}", path.display()))?;
    let root: serde_json::Value = serde_json::from_str(&text)?;
    let models = root
        .get("models")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("suite.json has no `models` array"))?;
    Ok(models
        .iter()
        .filter_map(|m| {
            Some(SuiteEntry {
                id: m.get("id")?.as_str()?.to_string(),
                name: m.get("name")?.as_str()?.to_string(),
                gguf: m.get("gguf")?.as_str()?.to_string(),
                backends: m
                    .get("backends")?
                    .as_array()?
                    .iter()
                    .filter_map(|b| Some(b.as_str()?.to_string()))
                    .collect(),
                estimated_ram_gb: m
                    .get("estimated_ram_gb")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
            })
        })
        .collect())
}

/// Physical RAM in GiB, for `--fit-host`.
fn host_ram_gb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn sysctlbyname(
                name: *const std::os::raw::c_char,
                oldp: *mut std::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut std::ffi::c_void,
                newlen: usize,
            ) -> std::os::raw::c_int;
        }
        let key = std::ffi::CString::new("hw.memsize").unwrap();
        let mut out: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: `hw.memsize` returns a u64 and `out`/`len` describe one.
        let rc = unsafe {
            sysctlbyname(
                key.as_ptr(),
                &mut out as *mut u64 as *mut std::ffi::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && out > 0 {
            return out as f64 / (1024.0 * 1024.0 * 1024.0);
        }
    }
    0.0
}

pub fn run_suite(args: SuiteArgs) -> anyhow::Result<()> {
    let mut measured = 0usize;
    // The suite is the unit of truth for RESULTS.md, so check the host
    // once up front rather than discovering at model 9 of 13 that the
    // first eight rows were measured on a busy box. Children re-check
    // individually, because load can rise mid-suite, and the loop below
    // waits for the previous entry's own load to decay before starting
    // the next one so the suite does not lock itself out.
    crate::host_state::ensure_quiet_enough(args.max_load)?;
    let entries = load_suite(&args.bench_dir)?;
    let exe = std::env::current_exe()?;
    let ram = host_ram_gb();
    let out_dir = engine_receipt_dir(&args.bench_dir);
    std::fs::create_dir_all(&out_dir)?;

    for entry in &entries {
        if let Some(only) = &args.only_id {
            if &entry.id != only {
                continue;
            }
        }
        for backend in &entry.backends {
            if let Some(only) = &args.only_backend {
                if backend != only {
                    continue;
                }
            }
            if backend == "cuda" && cfg!(target_os = "macos") {
                eprintln!("skip {} {backend}: no CUDA on this host", entry.id);
                continue;
            }
            if backend == "metal" && !cfg!(feature = "metal") {
                eprintln!(
                    "skip {} metal: this binary was built without --features metal",
                    entry.id
                );
                continue;
            }
            // 75% of physical RAM headroom for OS + weights + KV.
            if args.fit_host && ram > 0.0 && entry.estimated_ram_gb > 0.75 * ram {
                eprintln!(
                    "skip {} {backend}: needs ~{:.0} GiB, host has {ram:.0} GiB",
                    entry.id, entry.estimated_ram_gb
                );
                continue;
            }
            // Total RAM says the model COULD fit this machine. Free RAM
            // says whether it fits right now. A 32 GiB box with 3.5 GiB
            // free accepts a 10 GiB model on the check above, then runs
            // it out of swap and reports a real-looking number for work
            // the disk did. Skipping keeps the previous receipt, which
            // is stale and says so, rather than replacing it with a
            // paged one that does not.
            if args.fit_host && args.max_load > 0.0 {
                if let Some(free) = crate::host_state::free_ram_gb() {
                    if entry.estimated_ram_gb + 2.0 > free {
                        eprintln!(
                            "skip {} {backend}: needs ~{:.0} GiB, only {free:.1} GiB free \
                             (it would run from swap)",
                            entry.id, entry.estimated_ram_gb
                        );
                        continue;
                    }
                }
            }
            let model_path = args.bench_dir.join("..").join(&entry.gguf);
            if !model_path.exists() {
                if args.skip_missing {
                    eprintln!("skip {} {backend}: {} not present", entry.id, entry.gguf);
                    continue;
                }
                anyhow::bail!("missing GGUF for {}: {}", entry.id, entry.gguf);
            }

            // The host belongs in the NAME, not only inside the file.
            // `{id}_{backend}.json` collides the moment a second
            // machine runs the same entry: the Xeon's `*_cpu.json`
            // silently replaces the laptop's, and the ledger loses a
            // host instead of gaining one. Discovered while adding the
            // first x86 and CUDA rows.
            let receipt = out_dir.join(format!(
                "{}_{backend}__{}.json",
                entry.id,
                host_slug(&crate::host_state::host_label(
                    &crate::host_state::host_spec()
                ))
            ));
            eprintln!("\n=== {} [{}] {backend} ===", entry.id, entry.name);
            // The previous entry's own benchmark is still in the
            // 1-minute average, and the child re-checks the bar. Let it
            // decay instead of letting the suite lock itself out.
            // Skip this entry rather than abandoning the suite. `?` here
            // meant one busy stretch killed the whole run and every
            // model after it never went, which is how a 12-model suite
            // stopped at 8 and left the table half old and half new.
            // A missing GGUF already skips; an unclearable host is the
            // same kind of "not now", and the previous receipt stands.
            if let Err(why) = crate::host_state::wait_until_quiet_enough(
                args.max_load,
                std::time::Duration::from_secs(180),
            ) {
                eprintln!("skip {} {backend}: {why}", entry.id);
                continue;
            }
            let status = std::process::Command::new(&exe)
                .arg("bench")
                .args(["-m", &entry.gguf])
                .args(["-p", &args.n_prompt.to_string()])
                .args(["-n", &args.n_gen.to_string()])
                .args(["-r", &args.reps.to_string()])
                .args(["--n-gpu-layers", if backend == "cpu" { "0" } else { "99" }])
                .arg("--compare")
                .args(["--suite-id", &entry.id])
                .args(["--backend-label", backend])
                .args(["--receipt", receipt.to_str().unwrap()])
                .args(["--max-load", &args.max_load.to_string()])
                .status()?;
            if status.success() {
                measured += 1;
            } else {
                eprintln!(
                    "!! {} {backend} failed ({status}); leaving previous receipt alone",
                    entry.id
                );
            }
        }
    }

    // A run that measured nothing must not republish the table.
    //
    // `render` reads whatever receipts are on disk, so a suite where
    // every entry failed or skipped would rewrite RESULTS.md from the
    // OLD receipts and print its usual success line. That happened: a
    // stray `ferrox` process from an earlier run held the instance
    // lock, all 21 entries refused, and the table was regenerated from
    // stale receipts anyway, mixing versions under one heading. The
    // table is only republished when this run actually produced a
    // number.
    if measured == 0 {
        eprintln!(
            "ferrox bench: no entry produced a measurement, so {} was left alone. \
             Nothing here is a result, and republishing the table would date it to \
             this run while its numbers came from earlier ones.",
            args.bench_dir.join("RESULTS.md").display()
        );
        return Ok(());
    }
    crate::bench_render::render(&args.bench_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two machines running the same entry must not share a filename.
    #[test]
    fn a_receipt_name_carries_its_host() {
        assert_eq!(
            host_slug("Apple M2 Pro (10c/6p) macOS 26.6.1"),
            "apple-m2-pro-10c-6p-macos-26-6-1"
        );
        assert_eq!(host_slug("Xeon E5-2630 v4"), "xeon-e5-2630-v4");
        assert_ne!(
            host_slug("Apple M2 Pro"),
            host_slug("Xeon E5-2630 v4"),
            "two hosts sharing a receipt name is how a ledger loses a machine"
        );
        assert_eq!(host_slug("  --  "), "");
    }
}

#[cfg(test)]
mod committed_receipt_tests {
    /// No committed receipt may claim one backend and record another.
    ///
    /// `bench_model::run` refuses to WRITE such a receipt (#126), but
    /// that only guards receipts this build produces. Receipts arrive
    /// by other routes: copied off a rented box, restored from a branch
    /// cut before the fix, or pulled with a glob that swept up
    /// neighbours. All three happened on 2026-09-04, and the last one
    /// silently reintroduced five Metal-measured rows under a `cpu`
    /// heading AFTER they had been deleted.
    ///
    /// So the repository asserts it too, over what is actually
    /// committed, which is the artifact readers trust.
    #[test]
    fn every_committed_receipt_ran_on_the_backend_it_claims() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/receipts/engine");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return; // not a checkout with receipts; nothing to assert
        };
        let mut wrong = Vec::new();
        let mut seen = 0usize;
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            let (Some(label), Some(active)) = (
                v.get("backend").and_then(|x| x.as_str()),
                v.get("backend_active").and_then(|x| x.as_str()),
            ) else {
                continue;
            };
            seen += 1;
            if !label.eq_ignore_ascii_case(active) {
                wrong.push(format!(
                    "{}: labelled `{label}` but ran on {active}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                ));
            }
        }
        assert!(seen > 0, "no receipts found under {}", dir.display());
        assert!(
            wrong.is_empty(),
            "receipts that misdescribe the backend they ran on:\n  {}",
            wrong.join("\n  ")
        );
    }
}

#[cfg(test)]
mod host_identity_tests {
    use super::host_identity;

    /// A receipt carrying just the fields `host_identity` reads.
    fn receipt(label: &str, accelerator: Option<&str>) -> serde_json::Value {
        let mut r = serde_json::json!({"host_spec": {"label": label}});
        if let Some(card) = accelerator {
            r["accelerator"] = serde_json::Value::String(card.to_string());
        }
        r
    }

    /// The failure this exists to stop: two GPUs in one box. Both rows
    /// carry the same CPU label, and grouping on that label alone put a
    /// Pascal gap and an Ampere gap under one heading -- two computers
    /// in one table, which is the thing the grouping was added to
    /// prevent.
    #[test]
    fn two_cards_in_one_host_are_two_hosts() {
        let xeon = "Intel(R) Xeon(R) CPU E5-2630 v4 (10c) Linux";
        let pascal = host_identity(&receipt(xeon, Some("NVIDIA GeForce GTX 1080")));
        let ampere = host_identity(&receipt(xeon, Some("NVIDIA GeForce RTX 3080")));
        assert_ne!(
            pascal, ampere,
            "same CPU, different card: these are different machines"
        );
        assert!(pascal.contains("GTX 1080") && ampere.contains("RTX 3080"));
    }

    /// A `cpu` row names no card, and must not grow a phantom one.
    #[test]
    fn a_cpu_row_is_identified_by_its_host_alone() {
        let label = "AMD Ryzen 9 7945HX (16c) Linux";
        assert_eq!(host_identity(&receipt(label, None)), label);
    }

    /// Same machine, same card: one section, not two.
    #[test]
    fn the_same_host_and_card_is_one_host() {
        let a = host_identity(&receipt("Xeon (10c)", Some("RTX 3070")));
        let b = host_identity(&receipt("Xeon (10c)", Some("RTX 3070")));
        assert_eq!(a, b);
    }

    /// Pre-0.13.0 receipts carry no spec; they group as one unknown
    /// host rather than each being waved through on its own.
    #[test]
    fn a_receipt_with_no_spec_is_one_named_unknown_host() {
        let id = host_identity(&serde_json::json!({}));
        assert!(id.contains("unrecorded"), "{id}");
        assert_eq!(id, host_identity(&serde_json::json!({})));
    }
}
