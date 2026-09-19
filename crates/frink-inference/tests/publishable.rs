//! A publishable crate may not depend on an unpublishable one.
//!
//! Cargo requires every dependency of a published crate to be resolvable
//! on the registry, OPTIONAL ONES INCLUDED. So a workspace member marked
//! `publish = false` silently makes every crate that depends on it
//! unpublishable, and nothing says so until `cargo publish` fails
//! halfway through a release.
//!
//! That is exactly how v0.15.0 broke: `frink-vulkan` was `publish =
//! false` as a beachhead, `frink-core` depends on it, and four leaf
//! crates went up before `frink-core` failed with "no matching package
//! named `frink-vulkan`". crates.io versions are immutable, so the
//! recovery cost a whole version number.
//!
//! The pre-flight check that existed asked "is every publishable crate
//! listed in the publish order". Real, and not this. Two properties that
//! both have to hold, with only one of them checked.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// `(name, publishable, frink_deps)` for every workspace member.
fn members() -> Vec<(String, bool, Vec<String>)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(crates_dir()).expect("crates/ is readable") {
        let dir = entry.expect("readable entry").path();
        let manifest = dir.join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&manifest).expect("manifest is readable");
        let name = dir
            .file_name()
            .expect("a crate directory has a name")
            .to_string_lossy()
            .into_owned();

        // `publish = false` anywhere outside a comment. Deliberately
        // crude: this file is a guard, not a TOML parser, and a crate
        // that spells it in some exotic way should still be caught.
        let publishable = !text
            .lines()
            .map(str::trim)
            .any(|l| !l.starts_with('#') && l.replace(' ', "").starts_with("publish=false"));

        let deps = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| l.split_once('='))
            .map(|(k, _)| k.trim().to_string())
            .filter(|k| k.starts_with("frink-"))
            .collect();

        out.push((name, publishable, deps));
    }
    out.sort();
    out
}

#[test]
fn no_publishable_crate_depends_on_an_unpublishable_one() {
    let members = members();
    assert!(
        members.len() > 5,
        "expected to find the workspace members, found {}",
        members.len()
    );
    let publishable: HashMap<&str, bool> =
        members.iter().map(|(n, p, _)| (n.as_str(), *p)).collect();

    let mut broken = Vec::new();
    for (name, is_publishable, deps) in &members {
        if !is_publishable {
            continue;
        }
        for dep in deps {
            if publishable.get(dep.as_str()) == Some(&false) {
                broken.push(format!("{name} -> {dep}"));
            }
        }
    }

    assert!(
        broken.is_empty(),
        "these publishable crates depend on a crate marked `publish = false`, so \
         `cargo publish` will fail partway through a release and the versions \
         already uploaded cannot be reused: {broken:?}"
    );
}

/// The crate that broke it, pinned by name.
///
/// Not redundant with the test above: this one fails with a sentence
/// that says what to do, where the general test only names a pair.
#[test]
fn frink_vulkan_stays_publishable_because_core_depends_on_it() {
    let members = members();
    let vulkan = members
        .iter()
        .find(|(n, _, _)| n == "frink-vulkan")
        .expect("frink-vulkan is a workspace member");
    assert!(
        vulkan.1,
        "frink-vulkan is marked `publish = false`, but `frink-core` depends on it. \
         Cargo cannot publish a crate whose dependency is absent from the registry, \
         optional or not. Either publish it, or remove the dependency from frink-core."
    );
}

// A test asserting that the internal `version = "x"` pins match
// `[workspace.package] version` was written here and DELETED, because
// it could not fail. Cargo resolves a path dependency's version
// requirement against the crate at that path, so a stale pin does not
// build at all:
//
//     error: failed to select a version for the requirement
//     `frink-api = "^0.15.3"`
//     candidate versions found which didn't match: 0.16.0
//
// Recorded rather than silently dropped, because "the version is
// written in eleven places" looks exactly like this repo's dominant bug
// shape and the next person will reach for the same test. It is not an
// instance of it: the build system is the thing enforcing agreement.
// What DID cost a release was a `publish = false` crate, which the
// tests above cover, and that failure is invisible until publish time.
