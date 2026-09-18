#!/bin/bash
# Task: one-sync prefill stack. Hardware tests, verify, A/B main vs branch.
set -x
source $HOME/.cargo/env
cd /work/ferrox
[ -f /work/ferrox-main ] || cp target/release/ferrox /work/ferrox-main
echo "=== HW_TESTS"
cargo test -p ferrox-cuda --features cuda -- --ignored 2>&1 | grep -E "^test |test result|panicked|TC=|GPU=|error" | head -60
echo "=== BUILD_BRANCH"
cargo build -q --release -p ferrox-cli --features cuda 2>&1 | tail -5
F=target/release/ferrox
echo "=== VERIFY"
for m in llama32_3b_q4km llama32_1b_q4km qwen3_06b_q8; do
  echo "--- $m"
  FERROX_CUDA=1 $F verify -m /work/models/$m.gguf --backend cuda --prompt-tokens 64 2>&1 | tail -3
done
echo "=== AB_PP512"
for i in 1 2 3; do
  for B in /work/ferrox-main $F; do
    echo "--- $B run $i"
    FERROX_CUDA=1 $B bench -m /work/models/llama32_3b_q4km.gguf -p 512 -n 0 -r 2 --n-gpu-layers 99 --max-load 0 2>&1 | grep -iE "pp512" | head -3
  done
done
echo "=== AB_1B"
for B in /work/ferrox-main $F; do
  echo "--- $B"
  FERROX_CUDA=1 $B bench -m /work/models/llama32_1b_q4km.gguf -p 512 -n 0 -r 2 --n-gpu-layers 99 --max-load 0 2>&1 | grep -iE "pp512" | head -3
done
NSYS=$(command -v nsys || ls /usr/local/cuda/bin/nsys 2>/dev/null)
if [ -n "$NSYS" ]; then
  rm -f /root/os_pp512.sqlite /root/os_pp512stats*.csv
  echo "=== NSYS_PP512"
  FERROX_CUDA=1 $NSYS profile -t cuda -o /root/os_pp512 --force-overwrite true $F bench -m /work/models/llama32_3b_q4km.gguf -p 512 -n 0 -r 1 --n-gpu-layers 99 --max-load 0 >/dev/null 2>&1
  $NSYS stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv -o /root/os_pp512stats /root/os_pp512.nsys-rep >/dev/null 2>&1
  for f in /root/os_pp512stats*.csv; do echo "--- $f"; head -12 "$f"; done
fi
echo "=== TASK_DONE"
