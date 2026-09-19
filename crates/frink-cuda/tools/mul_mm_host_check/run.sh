#!/usr/bin/env bash
# Executes the mul_mm CUDA C on this host and compares it, bit for bit,
# against the Rust scalar twin. No GPU involved; see shim.h for what
# this does and does not establish.
#
#   crates/frink-cuda/tools/mul_mm_host_check/run.sh
#
# Needs a C++14 compiler. Every kind in `frink_cuda::mul_mm::KINDS` is
# covered automatically -- adding a quant kind needs no edit here.
#
# `-ffp-contract=off` is required, not cosmetic: contraction fuses
# `acc += a * b` into an FMA and the result stops being bit-comparable
# to the twin. A real GPU *does* contract, which is why the on-device
# test (`mul_mm_launch::tests`) compares with a relative tolerance
# instead of exactly.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
cxx="${CXX:-c++}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

cargo run -q --manifest-path "$root/Cargo.toml" -p frink-cuda \
    --example mul_mm_host_check -- "$out"

status=0
while read -r kind fn tag n_rows n_cols batch row_bytes; do
    bin="$out/run_$kind"
    if [ ! -x "$bin" ]; then
        { echo '#include "shim.h"'; cat "$out/$kind.cu"; cat "$here/main.cpp"; } > "$out/tu_$kind.cpp"
        "$cxx" -std=c++14 -O1 -ffp-contract=off -Wall -Wextra \
            -Wno-unused-function -I"$here" -DFX_KERNEL="$fn" \
            "$out/tu_$kind.cpp" -o "$bin"
    fi
    "$bin" "$out/$tag" "$n_rows" "$n_cols" "$batch" "$row_bytes" || status=1
done < "$out/manifest.txt"

if [ "$status" -eq 0 ]; then
    echo "OK: every emitted mul_mm kernel matches the scalar twin bit for bit"
else
    echo "FAIL: the emitted CUDA C diverges from the scalar twin" >&2
fi
exit "$status"
