//! Detect `mmproj-*.gguf` vision towers next to a main GGUF (P7).
//!
//! Frink does not yet run VL e2e — this helper only discovers companion
//! projector files the way OpenAI-style multimodal loaders expect, so
//! serve/CLI can fail closed with a clear path once `vl_engine` lands.

use std::path::{Path, PathBuf};

use crate::capability::{resolve_profile, ArchScope};

/// Look for `mmproj*.gguf` (case-insensitive) in the same directory as
/// `main_gguf`. Returns the first match by name, if any.
pub fn find_mmproj_beside(main_gguf: &Path) -> Option<PathBuf> {
    let dir = main_gguf.parent()?;
    let mut matches: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| {
                        let lower = n.to_ascii_lowercase();
                        lower.starts_with("mmproj")
                    })
                    .unwrap_or(false)
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

/// Log a clear warning when a companion mmproj exists but VL is not implemented.
/// Known VL architectures get a stronger message (fail-closed hint for operators).
pub fn warn_mmproj_if_present(main_gguf: &Path, arch: Option<&str>) {
    let Some(mmproj) = find_mmproj_beside(main_gguf) else {
        return;
    };
    let vl_arch = arch.is_some_and(|a| {
        resolve_profile(a).is_some_and(|p| p.scope == ArchScope::DeferredMultimodal)
    });
    let arch_note = arch.map(|a| format!(" (arch={a})")).unwrap_or_default();
    if vl_arch {
        eprintln!(
            "frink: warning: mmproj {} beside {}{} — VL architecture but multimodal \
             generation not implemented (see docs/MODELS.md P7)",
            mmproj.display(),
            main_gguf.display(),
            arch_note
        );
    } else {
        eprintln!(
            "frink: warning: mmproj companion {} found beside {}{} — vision projector not \
             loaded; text-only path continues (see docs/MODELS.md P7)",
            mmproj.display(),
            main_gguf.display(),
            arch_note
        );
    }
}

/// CLI-facing stderr variant of [`warn_mmproj_if_present`].
pub fn eprint_mmproj_if_present(main_gguf: &Path, arch: Option<&str>) {
    let Some(mmproj) = find_mmproj_beside(main_gguf) else {
        return;
    };
    let arch_note = arch.map(|a| format!(" (arch={a})")).unwrap_or_default();
    eprintln!(
        "frink: warning: mmproj companion {} found beside {}{} — VL/projector not \
         implemented; text-only load continues (see docs/MODELS.md)",
        mmproj.display(),
        main_gguf.display(),
        arch_note
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn finds_mmproj_next_to_main() {
        let dir = std::env::temp_dir().join(format!("frink_mmproj_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let main = dir.join("model-Q4_K_M.gguf");
        let mm = dir.join("mmproj-f16.gguf");
        fs::File::create(&main).unwrap().write_all(b"x").unwrap();
        fs::File::create(&mm).unwrap().write_all(b"y").unwrap();
        let found = find_mmproj_beside(&main).expect("mmproj");
        assert_eq!(found, mm);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn none_when_no_mmproj() {
        let dir = std::env::temp_dir().join(format!("frink_nomm_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let main = dir.join("only.gguf");
        fs::File::create(&main).unwrap().write_all(b"x").unwrap();
        assert!(find_mmproj_beside(&main).is_none());
        let _ = fs::remove_dir_all(&dir);
    }
}
