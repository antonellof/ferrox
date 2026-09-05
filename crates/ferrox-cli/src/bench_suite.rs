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
struct SuiteEntry {
    id: String,
    name: String,
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

fn suite_path(bench_dir: &Path) -> PathBuf {
    bench_dir.join("suite.json")
}

fn engine_receipt_dir(bench_dir: &Path) -> PathBuf {
    bench_dir.join("receipts").join("engine")
}

fn load_suite(bench_dir: &Path) -> anyhow::Result<Vec<SuiteEntry>> {
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
    render(&args.bench_dir)
}

/// Rewrites the engine table in `RESULTS.md` from the receipts on disk,
/// leaving the header and Open notes outside the markers untouched.
const BEGIN: &str = "<!-- BEGIN ENGINE TABLE (generated by `ferrox bench --render`) -->";
const END: &str = "<!-- END ENGINE TABLE -->";

pub fn render(bench_dir: &Path) -> anyhow::Result<()> {
    let dir = engine_receipt_dir(bench_dir);
    let mut receipts: Vec<serde_json::Value> = Vec::new();
    if dir.is_dir() {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        paths.sort();
        for p in paths {
            if let Ok(text) = std::fs::read_to_string(&p) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                    receipts.push(v);
                }
            }
        }
    }

    // No receipts at all means there is nothing to render. Writing the
    // table anyway would replace a real ledger with an empty one and
    // report success, which is worse than doing nothing.
    if receipts.is_empty() {
        anyhow::bail!(
            "no engine receipts under {}, so there is nothing to render. \
             Run `ferrox bench --suite` first.",
            dir.display()
        );
    }

    let suite = load_suite(bench_dir).unwrap_or_default();
    let name_of = |id: &str| {
        suite
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.name.clone())
            .unwrap_or_else(|| id.to_string())
    };

    #[derive(Clone)]
    struct Row {
        /// Which machine produced this row. Rows never merge across
        /// hosts; see the grouping below.
        host: String,
        model: String,
        backend: String,
        test: String,
        ferrox: Option<f64>,
        llama: Option<f64>,
        gap: Option<f64>,
    }

    // Rows from two machines are not one table.
    //
    // `render` reads every receipt in the directory, so the moment a
    // second host writes one, its numbers would sort in beside this
    // one's under a single heading with nothing saying so. A reader
    // comparing a 5.06x row against a 1.41x row would be comparing two
    // computers. Receipts written before 0.13.0 carry no spec at all,
    // and they group together as one unknown host rather than being
    // waved through individually.
    let mut hosts: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for r in &receipts {
        hosts.insert(
            r.get("host_spec")
                .and_then(|h| h.get("label"))
                .and_then(|v| v.as_str())
                .unwrap_or("unrecorded (receipt written before 0.13.0)")
                .to_string(),
        );
    }
    // More than one host used to be a hard refusal, on the grounds
    // that "one table cannot describe them". That reasoning is right
    // and is kept: a gap is only meaningful against the machine it was
    // measured on. What changed is the remedy. Refusing meant the
    // ledger could only ever describe whichever laptop happened to run
    // the suite, which is how it came to claim a CPU gap of 1.41x to
    // 5.06x while saying nothing about x86 or CUDA at all. Rows are now
    // grouped into one section per host, so two machines coexist
    // without any row being compared against the wrong one.

    let mut rows: Vec<Row> = Vec::new();
    for r in &receipts {
        let host_label = r
            .get("host_spec")
            .and_then(|h| h.get("label"))
            .and_then(|v| v.as_str())
            .unwrap_or("unrecorded (receipt written before 0.13.0)")
            .to_string();
        let id = r.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let backend = r
            .get("backend")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let Some(tests) = r.get("tests").and_then(|v| v.as_array()) else {
            continue;
        };
        for t in tests {
            rows.push(Row {
                host: host_label.clone(),
                model: name_of(id),
                backend: backend.clone(),
                test: t
                    .get("test")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string(),
                ferrox: t.get("ferrox_tps").and_then(|v| v.as_f64()),
                llama: t.get("llama_tps").and_then(|v| v.as_f64()),
                gap: t.get("gap").and_then(|v| v.as_f64()),
            });
        }
    }

    // Worst-first within (backend, test): high gap = ferrox farther behind.
    rows.sort_by(|a, b| {
        let backend_ord = |s: &str| match s {
            "metal" => 0,
            "cuda" => 1,
            "cpu" => 2,
            _ => 3,
        };
        let test_ord = |s: &str| {
            if s.starts_with("pp") {
                0
            } else if s.starts_with("tg") {
                1
            } else {
                2
            }
        };
        a.host
            .cmp(&b.host)
            .then_with(|| backend_ord(&a.backend).cmp(&backend_ord(&b.backend)))
            .then_with(|| test_ord(&a.test).cmp(&test_ord(&b.test)))
            .then_with(|| {
                b.gap
                    .partial_cmp(&a.gap)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.model.cmp(&b.model))
    });

    let mut table = String::new();
    table.push_str(BEGIN);
    table.push_str("\n\n## Engine (`ferrox bench` vs `llama-bench`)\n\n");
    // Which machine, stated in the generated block rather than in prose
    // above it, so it cannot drift away from the numbers it describes.
    {
        let n = hosts.len();
        if n == 1 {
            table.push_str(&format!(
                "Measured on: **{}**\n\n",
                hosts.iter().next().cloned().unwrap_or_default()
            ));
        } else if n > 1 {
            table.push_str(&format!(
                "Measured on **{n} hosts**, one section each. Rows are never \
                 compared across machines.\n\n"
            ));
        }
    }
    // A summary table, not prose. Generated from the same receipts as
    // the detail rows below, so it cannot drift away from them the way
    // a hand-written headline does. One line per host and backend: the
    // range is what a reader wants before any individual model.
    {
        use std::collections::BTreeMap;
        let mut by: BTreeMap<(String, String, bool), Vec<f64>> = BTreeMap::new();
        for r in &rows {
            if let Some(g) = r.gap {
                by.entry((r.host.clone(), r.backend.clone(), r.test.starts_with("pp")))
                    .or_default()
                    .push(g);
            }
        }
        if !by.is_empty() {
            table.push_str("### Summary\n\n");
            table.push_str("| Host | Backend | Prefill gap | Decode gap |\n");
            table.push_str("|---|---|---|---|\n");
            let mut seen: Vec<(String, String)> =
                by.keys().map(|(h, b, _)| (h.clone(), b.clone())).collect();
            seen.dedup();
            for (host, backend) in seen {
                let fmt = |pp: bool| -> String {
                    match by.get(&(host.clone(), backend.clone(), pp)) {
                        Some(v) if !v.is_empty() => {
                            let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
                            let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            if (hi - lo).abs() < 0.005 {
                                gap_cell(lo)
                            } else {
                                format!("{} to {}", gap_cell(lo), gap_cell(hi))
                            }
                        }
                        _ => "—".to_string(),
                    }
                };
                table.push_str(&format!(
                    "| {host} | {} | {} | {} |\n",
                    backend.to_uppercase(),
                    fmt(true),
                    fmt(false)
                ));
            }
            table.push('\n');
        }
    }

    fn push_section_at(table: &mut String, depth: &str, title: &str, rows: &[&Row]) {
        if rows.is_empty() {
            return;
        }
        table.push_str(&format!("{depth} {title}\n\n"));
        table.push_str("| Model | Test | ferrox tok/s | llama.cpp tok/s | Gap |\n");
        table.push_str("|---|---|---|---|---|\n");
        for r in rows {
            table.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                r.model,
                r.test,
                r.ferrox
                    .map(|v| format!("**{v:.2}**"))
                    .unwrap_or_else(|| "—".into()),
                r.llama
                    .map(|v| format!("**{v:.2}**"))
                    .unwrap_or_else(|| "—".into()),
                r.gap.map(gap_cell).unwrap_or_else(|| "—".into()),
            ));
        }
        table.push('\n');
    }

    let metal: Vec<&Row> = rows.iter().filter(|r| r.backend == "metal").collect();
    let cuda: Vec<&Row> = rows.iter().filter(|r| r.backend == "cuda").collect();
    let cpu: Vec<&Row> = rows.iter().filter(|r| r.backend == "cpu").collect();
    let other: Vec<&Row> = rows
        .iter()
        .filter(|r| !matches!(r.backend.as_str(), "metal" | "cuda" | "cpu"))
        .collect();

    if rows.is_empty() {
        table.push_str("| _no engine receipts yet_ | | | | | |\n\n");
    } else if hosts.len() <= 1 {
        push_section_at(&mut table, "###", "Metal", &metal);
        push_section_at(&mut table, "###", "CUDA", &cuda);
        push_section_at(&mut table, "###", "CPU", &cpu);
        push_section_at(&mut table, "###", "Other backends", &other);
    } else {
        // One section per machine. A reader scanning for a gap sees the
        // host before the number, which is the only order in which the
        // number means anything.
        for host in &hosts {
            let here: Vec<&Row> = rows.iter().filter(|r| &r.host == host).collect();
            table.push_str(&format!("### {host}\n\n"));
            for (title, backend) in [("Metal", "metal"), ("CUDA", "cuda"), ("CPU", "cpu")] {
                let sub: Vec<&Row> = here
                    .iter()
                    .copied()
                    .filter(|r| r.backend == backend)
                    .collect();
                push_section_at(&mut table, "####", title, &sub);
            }
            let sub: Vec<&Row> = here
                .iter()
                .copied()
                .filter(|r| !matches!(r.backend.as_str(), "metal" | "cuda" | "cpu"))
                .collect();
            push_section_at(&mut table, "####", "Other backends", &sub);
        }
    }

    table.push_str(END);

    let results = bench_dir.join("RESULTS.md");
    let existing = std::fs::read_to_string(&results).unwrap_or_default();
    let updated = splice(&existing, &table);
    std::fs::write(&results, updated)?;
    eprintln!(
        "ferrox bench: engine table written to {}",
        results.display()
    );
    Ok(())
}

fn gap_cell(g: f64) -> String {
    let marker = if g < 0.95 {
        "🟢"
    } else if g <= 1.05 {
        "⚪"
    } else {
        "🔴"
    };
    format!("{marker} **{g:.2}×**")
}

/// Replaces the marked block, or appends it if the markers are absent.
fn splice(existing: &str, block: &str) -> String {
    if let (Some(start), Some(end)) = (existing.find(BEGIN), existing.find(END)) {
        let mut out = String::with_capacity(existing.len() + block.len());
        out.push_str(&existing[..start]);
        out.push_str(block);
        out.push_str(&existing[end + END.len()..]);
        return out;
    }
    let mut out = existing.to_string();
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push('\n');
    out.push_str(block);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes one engine receipt for `host` into `dir`.
    fn receipt(dir: &Path, id: &str, host: &str, backend: &str, ferrox: f64, llama: f64) {
        let r = serde_json::json!({
            "schema": 2, "kind": "engine", "id": id,
            "backend": backend, "backend_active": backend,
            "host_spec": {"label": host},
            "tests": [{"test": "tg128", "ferrox_tps": ferrox, "llama_tps": llama,
                       "gap": llama / ferrox}],
        });
        std::fs::write(
            dir.join(format!("{id}_{backend}.json")),
            serde_json::to_string(&r).expect("json"),
        )
        .expect("write receipt");
    }

    /// Two machines render as two sections, and neither is dropped.
    ///
    /// This used to be a hard refusal ("one table cannot describe
    /// them"), which meant the ledger could only ever describe the
    /// laptop that happened to run the suite: it claimed a CPU gap of
    /// 1.41x to 5.06x while having no x86 or CUDA row at all. The
    /// refusal's REASON was right, so rows are separated rather than
    /// merged, and this pins that.
    #[test]
    fn two_hosts_render_as_two_sections_rather_than_an_error() {
        let dir = std::env::temp_dir().join(format!(
            "ferrox_render_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let engine = dir.join("receipts").join("engine");
        std::fs::create_dir_all(&engine).expect("mkdir");
        std::fs::write(
            dir.join("RESULTS.md"),
            format!("head\n{BEGIN}\nold\n{END}\ntail\n"),
        )
        .expect("seed");
        receipt(&engine, "m1", "Apple M2 Pro", "metal", 100.0, 90.0);
        receipt(&engine, "m1", "Rented Xeon", "cpu", 10.0, 20.0);

        render(&dir).expect("two hosts must render, not refuse");

        let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");
        assert!(out.contains("Apple M2 Pro"), "first host missing:\n{out}");
        assert!(out.contains("Rented Xeon"), "second host missing:\n{out}");
        assert!(
            out.contains("2 hosts"),
            "the reader is not told there are two machines:\n{out}"
        );
        // The whole point: a row is never presented without its host.
        let xeon = out.find("Rented Xeon").expect("host heading");
        let cpu_row = out.find("| 10.00 |").or_else(|| out.find("**10.00**"));
        if let Some(cpu_row) = cpu_row {
            assert!(
                cpu_row > xeon,
                "the Xeon's row appears before its host heading"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

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

    /// The summary is GENERATED, not written by hand.
    ///
    /// It replaced a hand-written headline table, which is a thing that
    /// drifts: the numbers above the fold stop matching the receipts
    /// below it and nobody notices, because nothing compares them. One
    /// row per host and backend, with the range taken from the same
    /// rows the detail tables use.
    #[test]
    fn the_summary_is_derived_from_the_same_rows_as_the_detail_tables() {
        let dir = std::env::temp_dir().join(format!(
            "ferrox_summary_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let engine = dir.join("receipts").join("engine");
        std::fs::create_dir_all(&engine).expect("mkdir");
        std::fs::write(
            dir.join("RESULTS.md"),
            format!("head\n{BEGIN}\nold\n{END}\ntail\n"),
        )
        .expect("seed");
        // Two gaps on one host+backend, so the summary must show a range.
        receipt(&engine, "a", "Box One", "cuda", 10.0, 20.0);
        receipt(&engine, "b", "Box One", "cuda", 10.0, 100.0);

        render(&dir).expect("render");
        let out = std::fs::read_to_string(dir.join("RESULTS.md")).expect("read");

        assert!(out.contains("### Summary"), "no summary table:\n{out}");
        let summary = &out[out.find("### Summary").expect("summary")..];
        let first_detail = summary.find("\n### ").unwrap_or(summary.len());
        let summary = &summary[..first_detail];
        assert!(
            summary.contains("2.00×") && summary.contains("10.00×"),
            "the summary must span the rows it describes:\n{summary}"
        );
        assert!(
            summary.contains("Box One"),
            "the summary must name the host:\n{summary}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gap_cell_colours_match_the_ledger_convention() {
        assert!(gap_cell(0.80).starts_with("🟢"));
        assert!(gap_cell(1.00).starts_with("⚪"));
        assert!(gap_cell(0.96).starts_with("⚪"));
        assert!(gap_cell(1.40).starts_with("🔴"));
    }

    #[test]
    fn splice_replaces_an_existing_block_and_keeps_the_surrounding_text() {
        let doc = format!("before\n{BEGIN}\nold\n{END}\nafter\n");
        let out = splice(&doc, &format!("{BEGIN}\nnew\n{END}"));
        assert!(out.contains("before"), "text before the block must survive");
        assert!(out.contains("after"), "text after the block must survive");
        assert!(out.contains("new"));
        assert!(!out.contains("old"), "the old block must be gone");
    }

    #[test]
    fn splice_appends_when_the_markers_are_missing() {
        let out = splice("just some prose\n", &format!("{BEGIN}\nfresh\n{END}"));
        assert!(out.starts_with("just some prose"));
        assert!(out.contains("fresh"));
    }

    /// A render with no receipts must refuse rather than publish an
    /// empty table over a real one.
    #[test]
    fn rendering_nothing_refuses_instead_of_emptying_the_ledger() {
        let dir = std::env::temp_dir().join(format!("ferrox-render-guard-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("receipts").join("engine")).unwrap();
        let results = dir.join("RESULTS.md");
        let original = "# Results\n\nreal numbers live here\n";
        std::fs::write(&results, original).unwrap();

        let err = render(&dir).unwrap_err().to_string();
        assert!(
            err.contains("nothing to render"),
            "expected a refusal naming the empty receipt dir, got: {err}"
        );
        assert_eq!(
            std::fs::read_to_string(&results).unwrap(),
            original,
            "the existing ledger must survive a render that had no receipts"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn splice_does_not_duplicate_the_block_on_a_second_render() {
        let block = format!("{BEGIN}\nv1\n{END}");
        let once = splice("doc\n", &block);
        let twice = splice(&once, &format!("{BEGIN}\nv2\n{END}"));
        assert_eq!(twice.matches(BEGIN).count(), 1, "exactly one engine block");
        assert!(twice.contains("v2") && !twice.contains("v1"));
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
