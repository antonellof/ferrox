//! How a batched-bench row is printed and recorded.
//!
//! The markdown table is `tools/batched-bench/batched-bench.cpp:128-129`
//! (header) and `:245` (row), reproduced width for width so a ferrox
//! table and a llama.cpp table paste side by side. The JSONL line is
//! `:238-243` minus the keys that would describe things ferrox does
//! not have (`n_batch`, `flash_attn`, `n_threads_batch`): a key with a
//! made-up value is worse than a missing one in a file meant for
//! comparison.

use super::workload::Measured;
use std::path::Path;

/// The run-wide values upstream prints before the table (`:126`).
#[derive(Debug, Clone, Copy)]
pub(super) struct Header {
    pub n_kv_max: usize,
    pub n_ubatch: usize,
    pub is_pp_shared: bool,
    pub is_tg_separate: bool,
    pub n_gpu_layers: usize,
    pub n_threads: usize,
    pub backend: &'static str,
}

/// `:126`, with ferrox's values and without the three fields that
/// have no ferrox meaning.
pub(super) fn header_line(h: &Header) -> String {
    format!(
        "ferrox batched-bench: n_kv_max = {}, n_ubatch = {}, is_pp_shared = {}, \
         is_tg_separate = {}, n_gpu_layers = {}, n_threads = {}, backend = {}",
        h.n_kv_max,
        h.n_ubatch,
        h.is_pp_shared as u8,
        h.is_tg_separate as u8,
        h.n_gpu_layers,
        h.n_threads,
        h.backend
    )
}

/// `:128-129`, byte for byte. Pinned by a test against the literal
/// C format so a width cannot drift.
pub(super) const TABLE_HEADER: [&str; 2] = [
    "|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |",
    "|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|",
];

/// `:245`: `|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |`.
pub(super) fn md_row(m: &Measured) -> String {
    format!(
        "|{:>6} | {:>6} | {:>4} | {:>6} | {:>8.3} | {:>8.2} | {:>8.3} | {:>8.2} | {:>8.3} | {:>8.2} |",
        m.combo.pp,
        m.combo.tg,
        m.combo.pl,
        m.n_kv,
        m.t_pp,
        m.speed_pp,
        m.t_tg,
        m.speed_tg,
        m.t,
        m.speed
    )
}

/// `:238-243`, with the same per-row keys and the header keys ferrox
/// can answer for.
pub(super) fn jsonl_row(h: &Header, m: &Measured) -> String {
    serde_json::json!({
        "n_kv_max": h.n_kv_max,
        "n_ubatch": h.n_ubatch,
        "is_pp_shared": h.is_pp_shared as u8,
        "is_tg_separate": h.is_tg_separate as u8,
        "n_gpu_layers": h.n_gpu_layers,
        "n_threads": h.n_threads,
        "backend": h.backend,
        "pp": m.combo.pp,
        "tg": m.combo.tg,
        "pl": m.combo.pl,
        "n_kv": m.n_kv,
        "t_pp": m.t_pp,
        "speed_pp": m.speed_pp,
        "t_tg": m.t_tg,
        "speed_tg": m.speed_tg,
        "t": m.t,
        "speed": m.speed,
        "workload_digest": m.digest.hex(),
    })
    .to_string()
}

/// The receipt: the shared envelope (already checked by
/// `bench_contract::receipt_common`) plus this run's rows.
pub(super) fn write_receipt(
    dest: &Path,
    mut receipt: serde_json::Map<String, serde_json::Value>,
    h: &Header,
    model: &str,
    arch: &str,
    rows: &[Measured],
) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let rows: Vec<serde_json::Value> = rows
        .iter()
        .map(|m| {
            serde_json::json!({
                "pp": m.combo.pp,
                "tg": m.combo.tg,
                "pl": m.combo.pl,
                "n_kv": m.n_kv,
                "t_pp": m.t_pp,
                "speed_pp": m.speed_pp,
                "t_tg": m.t_tg,
                "speed_tg": m.speed_tg,
                "t": m.t,
                "speed": m.speed,
                "workload_digest": m.digest.hex(),
            })
        })
        .collect();
    let serde_json::Value::Object(own) = serde_json::json!({
        "schema": 1,
        "kind": "batched",
        "model_path": model,
        "arch": arch,
        "n_kv_max": h.n_kv_max,
        "n_ubatch": h.n_ubatch,
        "is_pp_shared": h.is_pp_shared,
        "is_tg_separate": h.is_tg_separate,
        "n_gpu_layers": h.n_gpu_layers,
        "rows": rows,
    }) else {
        unreachable!("json! with braces is an object");
    };
    receipt.extend(own);
    std::fs::write(
        dest,
        serde_json::to_string_pretty(&serde_json::Value::Object(receipt))? + "\n",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::workload::Combo;
    use super::*;
    use crate::bench_guard::WorkloadDigest;

    fn sample() -> Measured {
        Measured {
            combo: Combo {
                pp: 128,
                tg: 128,
                pl: 2,
            },
            n_kv: 512,
            t_pp: 0.198,
            speed_pp: 1295.19,
            t_tg: 5.029,
            speed_tg: 50.90,
            t: 5.227,
            speed: 97.95,
            digest: WorkloadDigest::new(),
        }
    }

    /// The README's sample row for these exact values, from
    /// `tools/batched-bench/README.md`. Every column width is upstream's
    /// `%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f`.
    #[test]
    fn a_row_prints_in_upstream_widths() {
        assert_eq!(
            md_row(&sample()),
            "|   128 |    128 |    2 |    512 |    0.198 |  1295.19 |    5.029 |    50.90 |    5.227 |    97.95 |"
        );
    }

    /// The header and separator are upstream's `:128-129` expanded by
    /// hand; the row must be exactly as wide as both, or the columns
    /// do not line up when the two tables are pasted together.
    #[test]
    fn the_header_separator_and_rows_are_the_same_width() {
        let row = md_row(&sample());
        assert_eq!(TABLE_HEADER[0].len(), row.len());
        assert_eq!(TABLE_HEADER[1].len(), row.len());
        // Column boundaries: every `|` in the row sits under one in
        // the separator.
        for (i, (a, b)) in TABLE_HEADER[1].chars().zip(row.chars()).enumerate() {
            if a == '|' {
                assert_eq!(b, '|', "row column boundary drifted at byte {i}");
            }
        }
    }

    /// The header line is what `printf("|%6s | %6s | %4s | %6s | %8s |
    /// ...")` produces for the column names; pinned so a renamed
    /// column shows up as a diff against upstream.
    #[test]
    fn the_table_header_is_upstreams_format_string_expanded() {
        let expected = format!(
            "|{:>6} | {:>6} | {:>4} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |",
            "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s"
        );
        assert_eq!(TABLE_HEADER[0], expected);
        let sep = format!(
            "|{:>6}-|-{:>6}-|-{:>4}-|-{:>6}-|-{:>8}-|-{:>8}-|-{:>8}-|-{:>8}-|-{:>8}-|-{:>8}-|",
            "------",
            "------",
            "----",
            "------",
            "--------",
            "--------",
            "--------",
            "--------",
            "--------",
            "--------"
        );
        assert_eq!(TABLE_HEADER[1], sep);
    }

    fn header() -> Header {
        Header {
            n_kv_max: 2048,
            n_ubatch: 512,
            is_pp_shared: false,
            is_tg_separate: false,
            n_gpu_layers: 0,
            n_threads: 6,
            backend: "CPU",
        }
    }

    /// The per-row keys are upstream's (`:240`), so a jq pipeline
    /// written for `llama-batched-bench` output reads ferrox output.
    #[test]
    fn jsonl_rows_carry_upstreams_per_row_keys() {
        let v: serde_json::Value = serde_json::from_str(&jsonl_row(&header(), &sample())).unwrap();
        for key in [
            "pp",
            "tg",
            "pl",
            "n_kv",
            "t_pp",
            "speed_pp",
            "t_tg",
            "speed_tg",
            "t",
            "speed",
            "n_kv_max",
            "n_gpu_layers",
            "n_threads",
            "is_pp_shared",
        ] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        assert_eq!(v["pl"], 2);
        assert_eq!(v["n_kv"], 512);
        assert_eq!(v["is_pp_shared"], 0, "upstream prints the flag as an int");
        // What ferrox has no value for is absent, not invented.
        for key in ["n_batch", "flash_attn", "n_threads_batch"] {
            assert!(v.get(key).is_none(), "{key} would be a fabricated value");
        }
    }

    #[test]
    fn the_header_line_prints_flags_as_ints_like_upstream() {
        let line = header_line(&Header {
            is_pp_shared: true,
            ..header()
        });
        assert!(line.contains("is_pp_shared = 1"), "{line}");
        assert!(line.contains("is_tg_separate = 0"), "{line}");
        assert!(line.contains("n_kv_max = 2048"), "{line}");
    }

    #[test]
    fn a_receipt_keeps_the_envelope_and_adds_the_rows() {
        let dir = std::env::temp_dir().join(format!(
            "ferrox-batched-bench-receipt-{}",
            std::process::id()
        ));
        let dest = dir.join("nested").join("r.json");
        let mut envelope = serde_json::Map::new();
        envelope.insert("backend".into(), "cpu".into());
        write_receipt(&dest, envelope, &header(), "m.gguf", "llama", &[sample()]).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&dest).unwrap()).unwrap();
        assert_eq!(v["backend"], "cpu", "the envelope survives the merge");
        assert_eq!(v["kind"], "batched");
        assert_eq!(v["rows"][0]["n_kv"], 512);
        assert_eq!(v["rows"][0]["workload_digest"], WorkloadDigest::new().hex());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
