# A recurrent layer that does not come back to the host

Status: **kernels landed and verified, not wired.** The wiring was
measured and is a LOSS until the state stays on the device.

## The measurement this plan exists for

Bonsai-2-27B on an M2 Pro, `ferrox bench -p 128 -n 32`, decode at 7.7
tok/s against PrismML's llama.cpp fork at 11.5, measured back to back on
a quiet box. Every Metal submission
is timed (`FERROX_METAL_GPU_TIMING=1`), so the token accounts for
itself:

| per decode token | |
|---|---|
| command buffers | 159 |
| GPU, summed | 66 ms |
| submission overhead beyond that GPU time | 34 ms |
| host compute | 41 ms |
| wall | 141 ms (130 ms on a quiet box) |

A `sample` of the decode thread agrees from the other side: **83% of it
sits in `waitUntilCompleted`**. The fork's token is 87 ms and it encodes
ONE graph. So the gap is not kernel speed -- 66 ms of GPU for 5.95 GB of
weights is ~90 GB/s of a 200 GB/s machine, and the trit decode is
arithmetic-bound in both engines -- it is that a Ferrox layer returns to
the host about three times.

## What a recurrent layer costs today

Per gated delta-net layer, at one token:

1. `qkv` + `z` projections, one submission, result downloaded.
2. conv step, SiLU, per-head l2 norms, **the delta rule**, the gated
   output norm: all host.
3. `ssm_out`, one submission, result downloaded.

Plus the FFN's one submission. 48 recurrent layers, 16 attention layers
and 64 FFNs is the 159.

## What landed

`ferrox-metal/src/gdn.rs`:

- `gdn_delta_step`: one threadgroup per value head, one thread per state
  row; decay in place, predict `k`, scale the error by beta, rank-one
  update, read out with the scaled query. Pinned against
  `ferrox_core::gdn::delta_step` on three shapes including Bonsai's
  (48 heads, 128 wide) by
  `gdn::tests::the_device_delta_step_matches_this_one`; swapping the
  head map in the kernel turns it red.
- `gdn_gated_norm`: `rms_norm(o, weight) * silu(z)` per head.
- `launch_gdn_tail`: delta step, gated norm, the folded rotation and
  `ssm_out` in ONE command buffer.

## Why it is not wired: three measurements, not one

Wired into `Gdn::forward_rows`, the fused tail is slower every way it
has been built, against a host baseline of **7.3 to 7.4 tok/s**:

| how the state travels | tg32 |
|---|---|
| copied to the device and back each token | 6.0 |
| wrapped in place, `newBufferWithBytesNoCopy` | 6.6 |
| wrapped in place, the wrapper cached by address | 6.9 |

The first says what everyone expects: Bonsai's state is
`48 heads x 128 x 128 x 4 B = 3.1 MB` per layer, so copying it both ways
is 300 MB a token, more traffic than the whole 5.95 GB weight read.

The second and third are the interesting ones. `RecurrentState::ssm` is
page-aligned (`ferrox_core::recurrent_state::AlignedF32`) precisely so
that Metal can wrap the host's own bytes with no copy at all, and
wrapping them still costs: mapping host pages for the GPU is not free,
which the cached wrapper then removes. With BOTH of those gone the fused
tail is STILL 6% behind the host recurrence.

So the remaining difference is the kernel, not the plumbing: one
threadgroup per head streaming a 128x128 state through registers is
simply not better than six CPU cores doing the same reduction out of
cache, when the layer's own work is only 3 MFLOP.

## What would close it

The answer is not "move this loop to the GPU". It is the CHUNKED delta
rule llama.cpp uses (`delta-net-base.cpp`): a block of C rows is
processed together with matrix products, so the state is read and
written once per CHUNK instead of once per row, and the arithmetic
intensity rises to where a GPU beats a cache. That is a different
algorithm with its own correctness story (a WY-style representation of
the rank-one updates), and it is the honest next step. Everything below
is what the plumbing around it should look like.

## The plumbing, once the algorithm is right

The state has to live on the device across tokens, with the host copy
updated only when a host consumer actually reads it. The consumers are
the ones that make a recurrent cache different from an attention cache
(`ferrox_core::recurrent_state`): `KvCache::truncate` refuses a middle
position, the prefix cache does not store such a cache, `--model-draft`
refuses such a model, and a slot file writes it out. So the shape is:

1. `RecurrentState` gains a device mirror and a `dirty` side: written by
   the GPU path, read back by `as_host_slice()`.
2. Every host reader goes through that accessor, so a reader that
   forgets it does not compile rather than reading a stale state. This
   is the seam the repo's dominant bug shape demands: two copies of one
   fact, with something enforcing agreement.
3. Only then does `launch_gdn_tail` pay, and the conv step and the l2
   norms should join it, which makes a recurrent layer ONE submission
   from projection to `ssm_out`.

Arithmetic for the prize, at today's numbers: 48 layers x (one
submission at ~0.2 ms + the host recurrence at ~0.15 ms) is about 17 ms
of a 141 ms token, and fusing the FFN in behind a device-side residual
and norm is about 10 ms more. That is 8.8 tok/s, not 11.5: the rest is
the remaining submissions and the 66 ms of GPU itself, which is where
the fused decode stack (`ferrox-metal/src/decode_dense.rs`, already
serving dense models) would have to take over for PTQ1_0 and for folded
weights.

## The prefill surprise

The recurrence is **23% of a prefill step** (a `sample` of `pp512`-shaped
work), and a batch amortises the state copy over all its rows: upload
once, `rows` dispatches in one command buffer against one state buffer,
read back once. `launch_delta_step_rows` does exactly that and is pinned
against the host stepping the same rows in order
(`the_device_delta_rows_step_in_order`).

It measured `pp128` **32.7 -> 28.5 tok/s**. The state's working set is
why: 3.1 MB per layer stays in the CPU's shared cache across a batch's
rows, while the GPU re-reads and re-writes it from memory for every row,
which is 38 GB of traffic for a 128-token prefill. So the batched entry
is also kernels-without-a-caller until the state is resident AND the
rows are chunked so a block of them shares one pass over it.

## Measured non-results, so they are not tried again

- The Hadamard rotation on the device for a prefill BATCH: `pp128` 33.96
  to 32.04 tok/s. `transform_rows` is already parallel across cores;
  the kernel serialises into the GEMM's own command buffer.
- Spin-then-block instead of `waitUntilCompleted`, aimed at the 0.166 ms
  of wake-up latency: 3.7 tok/s against 7.1. Polling the status through
  objc takes the core the host work needs.
- `commandBufferWithUnretainedReferences` on the matvec path: 7.00
  against 7.1, for an `unsafe` invariant somebody has to keep.
- 8 rows per simdgroup in the PTQ1_0 matvec instead of 4: 6.9 against
  7.1 end to end.
- The batched recurrence on the device, with the state copied once per
  BATCH rather than once per token: `pp128` 32.7 to 28.5 (above).
- The PTQ1_0 GEMM's dequant moved to the float pipe, the rewrite that
  bought the matvec 1.8x: `pp128` 32.8 to 31.9. The GEMM dequantizes a
  tile ONCE into threadgroup memory and the simdgroup matrix ops hide
  it, so the trit decode is not what that kernel waits on.

## Where the prefill gap actually is

Arithmetic, not a profile: 128 tokens through 26.9B parameters is
6.9 TFLOP of matmul. The fork's `pp128` of 66.8 tok/s is 1.92 s, so
3.6 TFLOP/s; ours of 32.9 is 3.89 s, so 1.8 TFLOP/s. An M2 Pro's GPU
peaks near 6.8 TFLOP/s in f32, so the fork runs its PTQ1_0 GEMM at 53%
of peak and ferrox at 26%. Both use llama.cpp's simdgroup-matrix
`mul_mm` shape with the tile dequantized once into threadgroup memory,
so the difference is in the tiling constants and the dequant's cost per
tile, not in the algorithm. That is a bounded kernel project and it is
the prefill half of this gap; the decode half is the chunked recurrence
above plus the submissions the fused stack would remove.
