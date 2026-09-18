# A recurrent layer that does not come back to the host

Status: **the chunked rule landed for prefill, and then took the device
with it.** `ferrox_core::gdn_chunk` is 2.1x on the step alone, and
`ferrox_metal::gdn_chunk` is 3.0x on top of THAT at prefill lengths, so
a batch's recurrence now runs on the GPU after three attempts that lost.
The row-at-a-time kernels in `ferrox-metal/src/gdn.rs` are still not
wired, and the three measurements below still say why: the thing that
changed is not the kernel, it is what the kernel is given to do.

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

## Why the decode step is not the kernel, measured at last

The three losses below were only ever measured END TO END, where they
read as "the kernel is not better than six cores" without saying why.
Measured on its own, at one row, Bonsai's shape, state wrapped in place
and nothing copied (`gdn_chunk::tests::device_chunk_against_host_throughput`):

| rows | host | device | |
|---|---|---|---|
| 1 | 0.26 ms | 1.01 ms | 0.25x |
| 2 | 0.58 ms | 0.95 ms | 0.61x |
| 8 | 0.81 ms | 1.08 ms | 0.75x |
| 32 | 3.47 ms | 1.45 ms | 2.40x |

One row and eight rows cost the DEVICE the same, so what it is paying
is not the recurrence: it is about 0.9 ms of fixed submission and
buffer setup, and the kernel is the small part. That is why no faster
recurrence kernel can win a decode token, and it is the measurement
that turns the three rows below from "the GPU is not better at this"
into "a decode token cannot afford a submission per layer". The fix is
the one the arithmetic at the end of this file already names -- fewer
command buffers, not a better kernel -- and `DEVICE_ROWS = 32` in
`ferrox_core::gdn_chunk` is where the two sides cross.

The access pattern was still wrong, and fixing it is what made the
prefill numbers above: `gdn_delta_step` gave each thread a whole state
ROW and walked it, so adjacent threads touched addresses `head_dim`
floats apart and every 128-byte transaction carried 4 useful bytes.
`GDN_DELTA_STEP_COALESCED_SRC` replaces it -- one threadgroup per
`(head, row)`, one thread per COLUMN, the dot products as threadgroup
reductions -- and is what `encode_delta_step` now encodes, so there is
one kernel rather than two that would have to agree. A head width that
is not a power of two is refused there, because the reduction halves
its stride from it.

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

The answer is not "move this loop to the GPU", and it is not a faster
kernel either. `delta_step_throughput_probe` measures the host step at
**36 GB/s of state**, which is the floor for reading and writing 3.1 MB
once per row: rewriting its two passes as one read pass plus a
streaming update -- algebraically exact, because `S_new . q` expands to
`decay (S_old . q) + d (k . q)` -- measured 197 us against 175, since
both passes already hit the same 512-byte row while it is hot in L1.

The prize is therefore sized, and it is the largest single one left.
A 128-token prefill moves `128 rows x 48 layers x 6.2 MB = 38 GB` of
state at that 36 GB/s, which is **1.05 s of a 3.9 s `pp128` run**, and
the profile agrees (23% of samples). The CHUNKED delta rule llama.cpp
uses (`delta-net-base.cpp`) processes a block of C rows together with
matrix products, so the state is read and written once per CHUNK rather
than once per row: at C = 16 that 1.05 s becomes about 0.07 s, which is
`pp128` 32.9 -> roughly 44 tok/s on arithmetic alone. That is a different
algorithm with its own correctness story (a WY-style representation of
the rank-one updates), and it is the honest next step. Everything below
is what the plumbing around it should look like.

## The decode token, priced

Measured on the real checkpoint, `FERROX_METAL_GPU_TIMING=1`, 24 decode
tokens after a short prompt. The submissions are two `matvec-fused` and
one `dense-ffn` per layer, 64 layers:

| per decode token | count | GPU each | latency each | GPU | latency |
|---|---|---|---|---|---|
| `matvec-fused` | 128 | 0.185 ms | 0.152 ms | 23.7 ms | 19.5 ms |
| `dense-ffn` | 64 | 0.748 ms | 0.162 ms | 47.9 ms | 10.4 ms |
| | **192** | | | **71.6 ms** | **29.9 ms** |

Host work is the remaining ~35 ms of a 137 ms token, and a `sample`
puts two thirds of it in `delta_step`.

Three things follow, and the third is why no increment closes this.

**The GPU work is already ahead.** 71.6 ms for a 5.95 GB weight read is
83 GB/s; the reference's WHOLE token is 87 ms, so its GPU cannot be
under about 68 GB/s. Nothing in the kernels is the gap.

**The latency per submission is irreducible.** 0.15 ms is the OS
wake-up from `waitUntilCompleted`, and every way around it has been
measured and lost: spin-then-block (3.7 against 7.1, polling takes the
core the host work needs) and
`commandBufferWithUnretainedReferences` (7.00 against 7.1). So the only
lever on 29.9 ms is the COUNT.

**And the count only falls to one per layer if the host has nothing to
do inside a layer.** Fusing the tail -- output projection, residual,
FFN norm, FFN, residual -- into the FFN's own command buffer removes
64 submissions, which is 64 x 0.152 = 9.7 ms, or 137 ms to 127: **7.3
to 7.9 tok/s**. Making the host recurrence hit 60 GB/s instead of the
25.8 it measures would be about 7 ms more, so **8.3**. Both together do
not reach 11.5, and each one adds a branch to the hottest, most
decorated code in the repo -- `ffn_block_row` alone applies a parallel
sum scale, a down scale, a post-FFN norm, a residual scale, a skip
stream and two FFN-input shapes, and this file's own history is eight
model features lost one at a time to exactly that kind of branch.

So the honest ceiling on increments is ~8.3 tok/s, and parity needs the
whole layer in one command buffer with NO host step inside it: the QKV
projection, the conv and gates, the recurrence, the output projection,
the residual, the norm and the FFN, with the hidden state resident
across layers. That is `ferrox-metal/src/decode_dense.rs` -- which
already does exactly this for dense models -- extended to PTQ1_0
weights, to the folded Hadamard rotation between matmuls, and to the
gated delta-net block. At one submission per layer the arithmetic is
64 x 0.15 = 9.6 ms of latency, 71.6 ms of GPU plus about 5 ms for the
recurrence on device, and no host time: **about 86 ms, which is the
reference's 87.**

That is the project. It is three pieces, each with its own correctness
story, and none of them is a faster kernel.

## The host recurrence is gone, and what that bought

`ferrox-metal/src/gdn_branch.rs` runs a recurrent layer's WHOLE branch
in the submission `ssm_out` already cost: the two gates, the causal
convolution with its SiLU, the per-head l2 norms
(`ferrox-metal/src/gdn_head.rs`), the delta rule, the gated norm, the
folded rotation and the output projection. A `sample` of a decode run
no longer has `delta_step` in it at all, where it had been two thirds
of the host time.

Interleaved A/B on one build, Bonsai, 220 decode tokens:

    device branch  7.27 / 7.27 / 7.30 / 7.26 tok/s
    host branch    7.13 / 7.10

**+2.2%, and that is the honest surprise**: removing what the profile
said was two thirds of the host time is worth 3 ms of a 137 ms token,
because a decode thread spends 83% of itself in `waitUntilCompleted`
and most of the host work was already inside somebody else's wait. The
profile named the biggest host cost correctly and the ledger says host
cost was never the gap.

Getting there took three findings, each worth more than the 2.2%:

- **`Q5_0` and `PTQ1_0` were unreachable from every fused Metal path in
  `ferrox-models`.** `ferrox_metal::gpu::MATVEC_KINDS` served both and
  a hand-written match in the decoder listed six kinds and neither.
  `QuantKind::metal_kind_name` is exhaustive with no `_` arm now, and
  `crate::metal_launch` asks the backend's own table by that name, so
  the two cannot drift again. This is the repo's dominant bug shape,
  found for the fifth time.
- **A launch that allocates its own scratch pays for it per call, and
  the GPU ledger cannot see that.** The first version of this branch
  ran 7.22 to 6.68 -- SLOWER -- while the ledger accounted for only 10
  ms of the 19 it had lost, because it made eleven fresh Metal buffers
  per layer per token, some 500 allocations a token. Resident
  constants, the convolution window wrapped in place, and
  `ferrox-metal/src/scratch_pool.rs` took it 6.68 to 6.98 to 7.11 to
  7.27.
- **`RecurrentState::conv` was not page-aligned** while `ssm` was, so
  the one state small enough to seem harmless was the one being copied
  both ways per layer per token.

## What the decode token looks like now

GPU 79 ms, submission latency 35 ms, host about 20 ms -- and the host
that is left is the SIXTEEN attention layers' row kernel, not the 48
recurrent ones. The token is submission-bound, and the count is
unchanged at 192 because this branch rides in a submission that already
existed.

So the next step is the one the arithmetic named before any of this was
built, and it is now the ONLY step: merge a layer's three submissions
into one. The branch ends with `ssm_out` and the FFN begins with a
residual add and a norm, with nothing host-side between them; the
pieces are all `encode_` functions already. What stands in the way is
not a kernel, it is that the FFN's weights live in the decoder's layer
loop while the branch's live in `Gdn`, so the fusion has to happen at a
call site that owns both. At one submission per layer the arithmetic is
64 x 0.15 = 9.6 ms of latency against today's 35, no host step, and
about 86 ms a token: the reference's 87.

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


## What reversed it

The three losses above share one cause, and it is not "the GPU is bad at
this": every one of them moved the state once per ROW. Bonsai's state is
3.1 MB a layer, so a row-at-a-time kernel reads and writes 38 GB for a
128-token prefill whatever it computes, and 3 MFLOP of work cannot hide
that. Chunking is what removes it -- the state is touched once per CHUNK
-- and only then is there anything for a GPU to be good AT.

`ferrox-metal/src/gdn_chunk.rs` is the same algebra as the host chunk:
one threadgroup per value head, one thread per state ROW, the `t` loop
kept sequential behind threadgroup barriers because that is the
recurrence, and the `j` loop spread across threads because it is not.
`CHUNK` is 16 there against the host's 32, since what bounds it is a
threadgroup's 32 KiB rather than cache, and the two agreeing about the
answer at different widths is the test
(`gdn_chunk::tests::the_device_chunk_matches_this_one`, both measured
against the sequential rule, which is the definition).

Measured on an M2 Pro at Bonsai's shape, device against host, both
warmed (`device_chunk_against_host_throughput`):

| rows | host chunk | device chunk | |
|---|---|---|---|
| 8 | 1.24 ms | 1.13 ms | 1.10x |
| 32 | 3.43 ms | 2.49 ms | 1.37x |
| 64 | 5.48 ms | 2.48 ms | 2.21x |
| 128 | 10.43 ms | 5.41 ms | 1.93x |
| 512 | 43.38 ms | 13.86 ms | 3.13x |
| 1201 | 96.10 ms | 32.15 ms | 2.99x |

The first version of that kernel read **1.05x at 512 rows**, and what
fixed it is worth more than the kernel: `m[t]` and `n[t]` are per-thread
arrays indexed by a loop whose trip count was the RUNTIME `c`, so the
compiler could not unroll it and spilled both out of registers into
device memory. Padding the chunk's tiles to the compile-time `CHUNK` and
running every hot loop to that constant is 50 GFLOP/s to 160, on a part
that peaks near 6.8 TFLOP/s -- so there is more there, and the next step
is the simdgroup-matrix form, since the two `S x S x C` products are
exactly a matmul.

End to end on the real checkpoint, a 2420-token prompt, interleaved
A/B/A on the same box, `--max-load 0` with `suggestd` held down:

    device  39.78 / 41.27 / 41.66 t/s
    host             36.49 / 36.60

so +13% of prefill, and parity stays MATCH through it (KL 2.03e-6
against the fork's libllama on a 256-token prompt, which is the same
path). A decode token still steps one row on the host, where there is
no traffic to amortise and the three losses below still hold.

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
  BATCH rather than once per token: `pp128` 32.7 to 28.5 (above). This
  is the one entry that was later REVERSED, and only by changing what
  the kernel was asked to do: chunked, the same idea is 3x ahead.
- The PTQ1_0 GEMM's dequant moved to the float pipe, the rewrite that
  bought the matvec 1.8x: `pp128` 32.8 to 31.9. The GEMM dequantizes a
  tile ONCE into threadgroup memory and the simdgroup matrix ops hide
  it, so the trit decode is not what that kernel waits on.

## What prefill spends itself on now

With the recurrence on the device the profile moved twice in one day,
and the second move is the one that matters. A `sample` of a 2420-token
Bonsai prefill, by top-of-stack:

| | samples |
|---|---|
| `attention::dot_f32` | 5640 |
| `attention::pv_tile` | 3295 |
| `attention::qk_tile` | 2243 |
| `hadamard::fwht_normalized` | 1504 |
| `mamba2::conv_step` | 769 |

`delta_chunk` is not in it. The attention rows are Bonsai's 16 softmax
layers, which the FUSED Metal attention block refuses because their Q
is gated, so they fell back to the Rayon host kernel while
`ferrox-metal`'s own `launch_gqa_prefill_host_ex` sat there with no
caller. Wiring it (`Decoder::prefill_attention_blocked`) is 41.5 to
43.1 tok/s.

That is much less than 11178 of ~18000 samples suggests, and the
`FERROX_METAL_GPU_TIMING` ledger says why: those samples are on worker
threads that were already overlapping the GPU. With both host costs
gone, prefill is the GEMM and nothing else -- 82 ms a submission over
some 500 submissions of a 56 s run, about 3.2 TFLOP/s against the
fork's 3.6 overall. So the prefill gap is now entirely the kernel
below, and the host side of it is finished.

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
