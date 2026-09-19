#!/usr/bin/env bash
#
# The lint gate an Apple-silicon checkout cannot run by itself.
#
# Several kernels in frink-quant and frink-core exist twice: a NEON
# path under `#[cfg(target_arch = "aarch64")]` and a portable scalar
# fallback for everything else. On an M-series laptop `cargo clippy
# --workspace` never compiles the fallbacks, so a lint that only fires
# there is invisible locally and red on CI. That is not hypothetical:
# Rust 1.98 rejected `chunks_exact` with a compile-time-known size in 58
# places, the fix passed every local gate, and CI still went red on the
# sites that live in those fallbacks.
#
# Compiling the workspace for a second architecture locally is the whole
# fix. `x86_64-apple-darwin` is the cheap one to use for it: no cross
# linker, no sysroot, same libc, and it selects exactly the code paths
# an aarch64 build skips.
#
# Usage:
#   scripts/check-cross-target.sh [target]
#   FRINK_CROSS_TARGET=x86_64-unknown-linux-gnu scripts/check-cross-target.sh
#
# To run it on every push, call it from the local pre-push hook. That
# directory is gitignored and may already hold a hook, so append rather
# than replace -- see CONTRIBUTING.md.
set -euo pipefail

TARGET="${1:-${FRINK_CROSS_TARGET:-x86_64-apple-darwin}}"
HOST="$(rustc -vV | awk '/^host: /{print $2}')"

if [ "$HOST" = "$TARGET" ]; then
  echo "cross-target gate: host is already ${TARGET}, so the ordinary workspace"
  echo "build already compiles the code paths this gate exists to reach. Nothing to do."
  exit 0
fi

if ! rustup target list --installed | grep -qx "$TARGET"; then
  cat >&2 <<EOF
cross-target gate: the ${TARGET} standard library is not installed, so this
check cannot run and a lint confined to that architecture's code paths would
reach CI unseen. Install it once:

    rustup target add ${TARGET}
EOF
  exit 1
fi

echo "cross-target gate: clippy --workspace --all-targets --target ${TARGET}"
# clippy runs the compiler front end, so this subsumes a `cargo check`
# for the same target -- running both would compile the workspace twice
# to learn the same thing.
cargo clippy --workspace --all-targets --target "$TARGET" -- -D warnings
echo "cross-target gate: ok (${TARGET})"

# --- the second half, and the reason it exists -------------------------
#
# The gate above defaults to `x86_64-apple-darwin`, which selects the
# non-aarch64 kernel fallbacks but is STILL macOS. Two releases were
# broken by code that is dead only on LINUX and compiled fine here:
# v0.13.0 by a `sysctl` helper behind `#[cfg(target_os = "macos")]`, and
# v0.13.1 by `sum_footprints` and `ProbeCache::last`, orphans of a
# deleted supervisor. Neither is reachable by any Apple target.
#
# A full Linux build needs a C cross-compiler (`ring`, via the HTTP
# stack, will not build without one), so the whole workspace cannot be
# checked here. The COMPUTE crates have no such dependency and can be,
# which covers the `#[cfg(target_os)]` and dead-code classes in the code
# that actually differs per platform.
#
# Skipped rather than failed when the target is absent: this is an
# addition to the gate, not a new prerequisite for running it.
LINUX_TARGET="x86_64-unknown-linux-gnu"
if [ "$TARGET" != "$LINUX_TARGET" ] \
  && rustup target list --installed | grep -qx "$LINUX_TARGET"; then
  echo "cross-target gate: compute crates against ${LINUX_TARGET} (real non-Apple OS)"
  cargo clippy --target "$LINUX_TARGET" --all-targets \
    -p frink-quant -p frink-gguf -p frink-core -p frink-moe -p frink-models \
    -- -D warnings
  echo "cross-target gate: ok (${LINUX_TARGET}, compute crates)"
else
  echo "cross-target gate: skipping the ${LINUX_TARGET} pass (target not installed)"
fi
