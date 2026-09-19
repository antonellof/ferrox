//! DSV4 paged KV: sizing four heterogeneous tiers out of one budget, and
//! the page-atomic allocator that hands out the window tier.
//!
//! A DSV4 layer stack does not have *a* KV cache. It has four pools with
//! four different growth laws, and the whole difficulty is that they are
//! bought with the same bytes:
//!
//! * the **window** tier -- the 128-position sliding KV ring, present on
//!   every layer, page-granular;
//! * the **compressed** tier -- one block per `ratio` full-history
//!   tokens, on `ratio > 0` layers;
//! * the **indexer** tier -- the Lightning-Indexer keys, one block per 4
//!   full-history tokens, on `ratio == 4` layers;
//! * the fp32 **compress-state rings** -- `ring_size` slots per *window*
//!   page, one for the attention compressor and (on ratio-4 layers) one
//!   for the indexer's own compressor.
//!
//! # The window tier is sized independently, and the rest is not
//!
//! The window tier reads only the last `P` positions, so it needs
//! `swa_ratio` of the history and no more. The compressed and indexer
//! tiers answer questions about the *whole* history, so they stay
//! anchored to it: `cmp_blocks = full_token / ratio`, with `full_token`
//! the budget anchor and `swa_ratio` nowhere in it.
//!
//! That split is why [`dsv4_pool_sizes`] takes the window size as an
//! explicit page count instead of deriving everything from one number,
//! and why the working-set floor is applied **once, in pages**, by the
//! caller. Applying it by raising `swa_ratio` until the window clears the
//! floor looks equivalent at one anchor and is not: the ratio keeps
//! scaling, so at every larger anchor the window is bigger than the floor
//! ever asked for, and the pool that was sized is not the pool that fits.
//!
//! # Why the page solve is a binary search
//!
//! [`dsv4_cache_per_page`] collapses all four tiers into one
//! bytes-per-`P`-tokens number, which is the right shape for a planner
//! but the wrong tool for the final answer. Dividing the budget by it
//! assumes every tier scales with the anchor -- and at a small budget the
//! window does not: it pins at its floor while the full anchor shrinks
//! underneath it. So [`dsv4_solve_num_pages`] binary-searches the exact
//! [`dsv4_pool_bytes`] instead. A division there over-sizes the anchor,
//! every tier inflates past the budget at once, and the failure lands at
//! the first allocation rather than at config time.
//!
//! # Why the window allocator is page-atomic
//!
//! The compress-state ring is addressed by
//! `state_loc = (ws / P) * ring_size + ws % ring_size`, **derived** from
//! the window slot and never stored. `ring_size` divides `P`, so distinct
//! window *pages* land on disjoint ring blocks -- but only while every
//! window slot handed out sits at a page base plus its in-page offset.
//! [`FreeListAllocator`] therefore hands out unit bases that are always
//! multiples of the page unit, and [`Dsv4WindowPool::alloc_swa`]
//! preserves in-page offsets. Break either and two full pages share a
//! ring block: one request reads another's carry state, with no error
//! anywhere.
//!
//! [`crate::window_pool`] is the generic per-token FIFO window pool --
//! correct for the pool it ports, and unable to serve a per-page ring,
//! because a slot it hands back is any free slot rather than a page base.
//!
//! # The conservation invariant
//!
//! `free units + bound pages == capacity units`, checked by
//! [`Dsv4WindowPool::check_integrity`]. An **equality**, not a bound: a
//! `<=` tolerates a leaked window page, which surfaces an hour later as a
//! pool that cannot admit anything, with nothing to point at.
//!
//! Ported 1:1 from FreeToken's `kvcache/dsv4_cost_model.py`,
//! `kvcache/dsv4_paged_pool.py` and the ring context in
//! `attention/dsv4_sparse.py` (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::BTreeMap;

/// KV, compressed KV and indexer keys are all `bf16`. The fp4/fp8 quant
/// is an in-place round-trip already baked into the `bf16` value, so
/// there is no narrower width to price here.
pub const BF16_BYTES: u64 = 2;

/// The compress-state rings are fp32 -- and only they are.
pub const FP32_BYTES: u64 = 4;

/// One `int64` per full-history token in the full -> window mapping.
pub const INT64_BYTES: u64 = 8;

/// `P`: the window page. It is the sliding window *and* the radix block
/// key, deliberately independent of the generic `page_size`.
pub const DEFAULT_WINDOW_PAGE: usize = 128;

/// Slack the auto planner reserves on top of the working set, absorbing
/// plan-vs-measured drift (observed ~265 MiB) and leaving a usable pool.
pub const AUTO_KV_SLACK_BYTES: u64 = 2 << 30;

/// The "no live window state" sentinel, in both the full -> window
/// mapping and every ring location derived from it.
pub const NO_WINDOW_SLOT: i64 = -1;

/// The model geometry the four tiers are priced from.
///
/// `compress_ratios` is per layer: `0` for a layer with no compressed
/// tier, `4` for a compressed + indexer layer, `128` for a
/// compressed-only layer. A checkpoint may ship more ratios than the
/// model has layers (44 for 43), so only the first `n_layers` are ever
/// read -- see [`Dsv4Args::ratios`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Dsv4Args {
    /// The MLA latent width: one slab per token, because V aliases K.
    pub head_dim: u64,
    /// The Lightning-Indexer key width.
    pub index_head_dim: u64,
    pub n_layers: usize,
    pub compress_ratios: Vec<u32>,
}

impl Dsv4Args {
    /// The ratios the model actually uses: the first `n_layers`.
    ///
    /// Truncating rather than cycling or padding, exactly as upstream:
    /// a checkpoint that ships fewer ratios than layers describes fewer
    /// layers, and inventing ratios for the rest would price tiers that
    /// are never allocated.
    pub fn ratios(&self) -> &[u32] {
        let end = self.n_layers.min(self.compress_ratios.len());
        &self.compress_ratios[..end]
    }
}

/// Compress-state ring slots per window page (non-speculative).
///
/// # Panics
///
/// On any ratio other than 4 or 128. The ring geometry is not derivable
/// from the ratio -- it is a fixed property of the two compressors the
/// stack ships -- so an unknown ratio is a configuration bug, and
/// guessing a ring size for it would silently mis-address every carry
/// state on that layer.
pub fn ring_size_for_ratio(ratio: u32) -> usize {
    match ratio {
        4 => 8,
        128 => 128,
        _ => panic!("no ring for ratio {ratio} (only 4 / 128)"),
    }
}

/// CSA's ratio. Overlapping blocks, a doubled compressor projection,
/// and a Lightning Indexer.
pub const CSA_RATIO: u32 = 4;

/// HCA's ratio. Non-overlapping blocks, a single-width compressor, and
/// no indexer.
pub const HCA_RATIO: u32 = 128;

/// Which compressor one layer runs, and everything that follows from it.
///
/// The ratio is not a tuning knob with a smooth range: it selects one of
/// three *different mechanisms*, and the parameters below are not
/// interpolations between them. Deriving them here, once, is what stops
/// a single scalar ratio from building an indexer on an HCA layer or
/// none on a CSA layer -- both of which run, produce numbers, and are
/// wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerCompressor {
    /// Ratio 0: no compressed tier at all. Attention sees the raw
    /// sliding window and nothing else.
    ///
    /// A real entry in the shipped schedule rather than a disabled
    /// state -- `(0, 0, 4, 128, 4, 128, 4, 0)` opens with two of them
    /// and closes with one -- so it must be executable, not skipped.
    None,
    /// Ratio 4: Compressed Sparse Attention. Overlapping blocks, a
    /// compressor projection twice as wide, and the Lightning Indexer
    /// restricting which compressed entries a query may see.
    Csa,
    /// Ratio 128: Heavily Compressed Attention. Non-overlapping blocks,
    /// a single-width compressor, and dense visibility over every
    /// compressed entry -- no indexer.
    Hca,
}

impl LayerCompressor {
    /// Reads one layer's ratio.
    ///
    /// Returns `None` for a ratio that is not 0, 4 or 128, rather than
    /// approximating it to the nearest mechanism: there is no nearest
    /// mechanism, and picking one would give that layer the wrong
    /// compressor width and the wrong indexer, silently.
    pub fn from_ratio(ratio: u32) -> Option<Self> {
        match ratio {
            0 => Some(LayerCompressor::None),
            CSA_RATIO => Some(LayerCompressor::Csa),
            HCA_RATIO => Some(LayerCompressor::Hca),
            _ => None,
        }
    }

    /// The ratio this compressor runs at; `0` for [`None`](Self::None).
    pub fn ratio(self) -> u32 {
        match self {
            LayerCompressor::None => 0,
            LayerCompressor::Csa => CSA_RATIO,
            LayerCompressor::Hca => HCA_RATIO,
        }
    }

    /// Whether this layer instantiates the Lightning Indexer.
    ///
    /// CSA only. On an HCA layer every compressed entry is visible, so
    /// there is nothing for a top-k selector to select; building one
    /// there costs its own compressor, its own keys and its own tier of
    /// device memory to answer a question with a fixed answer.
    pub fn has_indexer(self) -> bool {
        matches!(self, LayerCompressor::Csa)
    }

    /// How many times wider this layer's raw per-token compressor
    /// projection is than one head.
    ///
    /// `2` for CSA, and this is the detail a single scalar ratio gets
    /// wrong on half the stack. Each raw token is projected *twice*:
    /// once for its role as the tail of the block ending at it, and
    /// once as the head of the next, overlapping block -- two different
    /// learned projections of the same token, not one reused twice
    /// (llama.cpp `load_arch_tensors`' `coff = ratio == 4 ? 2 : 1`, and
    /// `build_overlap_compressed_kv_from_state`'s
    /// `GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head)`).
    pub fn projection_width_multiple(self) -> usize {
        match self {
            LayerCompressor::Csa => 2,
            LayerCompressor::None | LayerCompressor::Hca => 1,
        }
    }

    /// Whether consecutive compression blocks share raw positions.
    pub fn overlapping(self) -> bool {
        matches!(self, LayerCompressor::Csa)
    }

    /// How many compressed entries a query at position `pos` may see.
    ///
    /// `(pos + 1) / ratio`: ratio-derived, like everything else here,
    /// and zero on a layer with no compressor. The `+1` is because
    /// `pos` is an index and the count of tokens through it is one
    /// more -- without it the query at the last position of a block
    /// cannot see the block it just completed, which is off by exactly
    /// one entry for the whole of the sequence.
    pub fn visible_compressed(self, pos: usize) -> usize {
        match self.ratio() {
            0 => 0,
            r => (pos + 1) / r as usize,
        }
    }
}

/// The schedule a ratio that is not 0, 4 or 128 could not be read into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownCompressRatio {
    pub layer: usize,
    pub ratio: u32,
}

impl std::fmt::Display for UnknownCompressRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { layer, ratio } = self;
        write!(
            f,
            "layer {layer} has compress ratio {ratio}, which is none of 0 (no compressor), \
             {CSA_RATIO} (CSA) or {HCA_RATIO} (HCA); these are three different mechanisms, \
             so there is no nearest one to fall back to"
        )
    }
}

impl std::error::Error for UnknownCompressRatio {}

impl Dsv4Args {
    /// The compressor each layer runs, in layer order.
    ///
    /// Reads the same `compress_ratios` the four KV tiers are priced
    /// from, so what a layer is *sized* for and what it *executes* can
    /// never drift apart -- which they would the moment the execution
    /// side kept a scalar of its own.
    ///
    /// Refuses the whole schedule on the first unreadable ratio rather
    /// than dropping that layer: a stack missing one layer's compressor
    /// still runs, and answers with the wrong attention on it.
    pub fn compressors(&self) -> Result<Vec<LayerCompressor>, UnknownCompressRatio> {
        self.ratios()
            .iter()
            .enumerate()
            .map(|(layer, &ratio)| {
                LayerCompressor::from_ratio(ratio).ok_or(UnknownCompressRatio { layer, ratio })
            })
            .collect()
    }
}

/// Window pages the sliding pool must always keep for the concurrent
/// working set.
///
/// Each running request's decode transients (2 per request, plus the
/// dummy's) and, in radix mode, per concurrent request one locked
/// live-tail page plus a retained soft-pinned prompt-end window -- two
/// pages, because the retention gap page-aligns to a whole extra page at
/// `P == window == 128`, so a distinct follow-up per running request can
/// re-lock two each. Plus the reserved dummy page itself.
///
/// The upstream docstring reads "2 per req + dummy", which is
/// `2 * mr + 1`; the code is `2 * (mr + 1) + 3 * mr + 1`. The **code** is
/// ported, because it is the one the engine's window floor and the
/// manager's prefill chunk budget both reserve against -- they have to
/// agree with each other, not with the prose.
pub fn dsv4_reserved_window_pages(max_running_req: usize, radix: bool) -> usize {
    2 * (max_running_req + 1) + if radix { 3 * max_running_req } else { 0 } + 1
}

/// The only hard floor on DSV4 KV sizing: the live sliding working set
/// the window pool must always hold.
///
/// One prefill chunk's reach, capped at 8 pages (1024 tokens -- chunked
/// prefill bounds the rest), plus
/// [`dsv4_reserved_window_pages`]. Everything above it is purely
/// memory-derived, and a request longer than the anchor is gated
/// gracefully by the pool's available size rather than by this.
///
/// Below the floor a full batch cannot get its window pages at all and
/// admission deadlocks -- which is why [`dsv4_solve_num_pages`] refuses
/// at config time instead of letting the pool boot and die at the first
/// allocation.
///
/// `radix` keys on "not naive": DSV4 config resolution rewrites the cache
/// type to `swa_radix`, so testing for the literal `radix` was always
/// false upstream.
pub fn dsv4_window_floor_pages(
    max_seq_len: usize,
    max_running_req: usize,
    radix: bool,
    page: usize,
) -> usize {
    assert!(page > 0, "a window page holds at least one position");
    let prefill_reach_pages = max_seq_len.div_ceil(page);
    prefill_reach_pages.min(8) + dsv4_reserved_window_pages(max_running_req, radix)
}

fn kv_bytes(args: &Dsv4Args) -> u64 {
    args.head_dim * BF16_BYTES
}

fn index_bytes(args: &Dsv4Args) -> u64 {
    args.index_head_dim * BF16_BYTES
}

/// One attention compress-state row: `kv | score`, each
/// `(1 + overlap) * head_dim` wide, fp32. Ratio-4 layers overlap.
fn state_bytes(args: &Dsv4Args, ratio: u32) -> u64 {
    let overlap = u64::from(ratio == 4);
    2 * (1 + overlap) * args.head_dim * FP32_BYTES
}

/// One indexer compress-state row: the same ring geometry keyed on
/// `index_head_dim`, with overlap always on. Its own pool -- it never
/// shares slots with the attention ring.
fn idx_state_bytes(args: &Dsv4Args) -> u64 {
    2 * 2 * args.index_head_dim * FP32_BYTES
}

/// `round(ratio * count)` as a count, clamped at zero.
fn scaled(ratio: f64, count: usize) -> usize {
    assert!(
        ratio.is_finite() && ratio >= 0.0,
        "swa_ratio must be a non-negative fraction of the full history, got {ratio}"
    );
    frink_core::placement::round_half_even(ratio * count as f64).max(0) as usize
}

/// Bytes per `P`-token page across **all** tiers, summed over the layers.
///
/// This is the one number a budget *division* would use, and it exists
/// for the affine planner ([`dsv4_auto_cost_model`]) rather than for the
/// final sizing -- see the module docs on why the solve does not divide.
///
/// The window term is scaled by `swa_ratio` and exists on every layer
/// (all-sliding); the compressed tier is `P / ratio` blocks; the indexer
/// tier `P / 4` blocks on ratio-4 layers. The two state rings are scaled
/// by `swa_ratio` **as well**, because they are sized off the *window*
/// pages (`state_slots = n_win_pages * ring_size`) and not off the full
/// pages -- charging `ring_size` slots to every full page prices a ring
/// nobody allocates.
pub fn dsv4_cache_per_page(args: &Dsv4Args, swa_ratio: f64, page: usize) -> u64 {
    assert!(page > 0, "a window page holds at least one position");
    let kv_b = kv_bytes(args);
    let idx_b = index_bytes(args);

    let mut total = 0u64;
    for ratio in args.ratios().iter().copied() {
        // The window tier exists on EVERY layer.
        total += scaled(swa_ratio, page) as u64 * kv_b;
        if ratio == 0 {
            continue;
        }
        total += (page as u64 / u64::from(ratio)) * kv_b;
        if ratio == 4 {
            total += (page as u64 / 4) * idx_b;
            total += scaled(swa_ratio, ring_size_for_ratio(4)) as u64 * idx_state_bytes(args);
        }
        total += scaled(swa_ratio, ring_size_for_ratio(ratio)) as u64 * state_bytes(args, ratio);
    }
    total
}

/// FULL-tier bytes per full-history token: compressed KV, indexer KV and
/// the full -> window mapping.
///
/// Independent of `swa_ratio` on purpose -- these tiers scale with the
/// anchor (`cmp_blocks = full_token / ratio`), so their per-token cost
/// does not move when the window does. The window pool and its rings are
/// **not** here; see [`dsv4_window_unit_bytes`].
///
/// Rounded **up** to whole bytes per token: this is the divisor a cache
/// slider hands a user, and a conservative maximum is the only kind that
/// cannot promise a pool the budget will not buy.
pub fn dsv4_kv_unit_bytes(args: &Dsv4Args, page: usize) -> u64 {
    assert!(page > 0, "a window page holds at least one position");
    let kv_b = kv_bytes(args);
    let idx_b = index_bytes(args);
    // The full_to_window map: one int64 slot per full token.
    let mut per_page = page as u64 * INT64_BYTES;
    for ratio in args.ratios().iter().copied() {
        if ratio == 0 {
            continue;
        }
        per_page += (page as u64 / u64::from(ratio)) * kv_b;
        if ratio == 4 {
            per_page += (page as u64 / 4) * idx_b;
        }
    }
    per_page.div_ceil(page as u64)
}

/// WINDOW-tier bytes per *window* token: the sliding KV pool on every
/// layer, plus both state rings (sized off the window pages).
///
/// Also independent of `swa_ratio`: the ratio sets how many window tokens
/// exist, never what one costs.
pub fn dsv4_window_unit_bytes(args: &Dsv4Args, page: usize) -> u64 {
    assert!(page > 0, "a window page holds at least one position");
    let kv_b = kv_bytes(args);
    let ratios = args.ratios();
    // Window KV: P slots per page, every layer.
    let mut per_page = ratios.len() as u64 * page as u64 * kv_b;
    for ratio in ratios.iter().copied() {
        if ratio == 0 {
            continue;
        }
        per_page += ring_size_for_ratio(ratio) as u64 * state_bytes(args, ratio);
        if ratio == 4 {
            per_page += ring_size_for_ratio(4) as u64 * idx_state_bytes(args);
        }
    }
    per_page.div_ceil(page as u64)
}

/// One layer's tier sizes. Absent (`None` on
/// [`Dsv4PoolSizes::layers`]) for a `ratio == 0` layer, which has a
/// window tier and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dsv4LayerSizes {
    pub ratio: u32,
    /// Attention compress-state ring slots per window page.
    pub ring_size: usize,
    /// Compressed KV blocks: anchored to the FULL history.
    pub cmp_blocks: usize,
    /// Indexer KV blocks, on ratio-4 layers only. Also full-anchored.
    pub idx_blocks: Option<usize>,
    /// Attention ring slots: `n_win_pages * ring_size` -- window-anchored.
    pub state_slots: usize,
    /// Indexer ring slots, on ratio-4 layers only. Window-anchored.
    pub idx_state_slots: Option<usize>,
}

/// Per-tier slot counts derived from the budget anchor `full_token`.
#[derive(Debug, Clone, PartialEq)]
pub struct Dsv4PoolSizes {
    /// `P`, the window page.
    pub page: usize,
    pub swa_ratio: f64,
    /// `num_pages * P` -- the budget anchor.
    pub full_token: usize,
    /// Global window pool rows.
    pub n_win_slots: usize,
    /// `n_win_slots / P`.
    pub n_win_pages: usize,
    /// One entry per layer, in layer order.
    pub layers: Vec<Option<Dsv4LayerSizes>>,
}

impl Dsv4PoolSizes {
    /// The anchor in pages: `full_token / P`.
    pub fn num_pages(&self) -> usize {
        self.full_token / self.page
    }
}

/// Size every tier from `num_pages`, with the window tier sized
/// independently.
///
/// `n_win_pages` is the window in whole pages. `None` means "take
/// `swa_ratio` of the full history, rounded **up** to whole pages";
/// `Some` is the caller's own count -- the working-set floor, a pinned
/// window, or a rebuild's target. Either way it is capped at `num_pages`,
/// because a window longer than the history it slides over is bytes
/// nobody can address.
///
/// The floor belongs in that `Some`, applied exactly once, in pages. It
/// must never be applied by raising `swa_ratio`: the ratio multiplies the
/// anchor, so a ratio raised until it clears the floor at one anchor
/// overshoots it at every larger one, and the compressed and indexer
/// tiers -- which stay anchored to the full history here -- then get sized
/// against a window that was never budgeted.
///
/// # Panics
///
/// When a non-zero ratio does not divide `page`: `P / ratio` blocks per
/// page is the compressed tier's whole addressing scheme, and a ragged
/// division rounds some layer's blocks to zero.
pub fn dsv4_pool_sizes(
    num_pages: usize,
    args: &Dsv4Args,
    swa_ratio: f64,
    page: usize,
    n_win_pages: Option<usize>,
) -> Dsv4PoolSizes {
    assert!(page > 0, "a window page holds at least one position");
    let full_token = num_pages * page;

    let n_win_pages = match n_win_pages {
        Some(pages) => pages,
        None => scaled(swa_ratio, full_token).div_ceil(page),
    };
    let n_win_pages = n_win_pages.min(num_pages);
    let n_win_slots = n_win_pages * page;

    let mut layers = Vec::with_capacity(args.ratios().len());
    for ratio in args.ratios().iter().copied() {
        if ratio == 0 {
            layers.push(None);
            continue;
        }
        assert!(
            page.is_multiple_of(ratio as usize),
            "P={page} must be divisible by ratio {ratio}"
        );
        let ring_size = ring_size_for_ratio(ratio);
        layers.push(Some(Dsv4LayerSizes {
            ratio,
            ring_size,
            // Full-anchored: the compressed tier answers for the whole
            // history, whatever fraction of it the window holds.
            cmp_blocks: full_token / ratio as usize,
            idx_blocks: (ratio == 4).then_some(full_token / 4),
            // Window-anchored: one ring block per window page.
            state_slots: n_win_pages * ring_size,
            idx_state_slots: (ratio == 4).then(|| n_win_pages * ring_size_for_ratio(4)),
        }));
    }

    Dsv4PoolSizes {
        page,
        swa_ratio,
        full_token,
        n_win_slots,
        n_win_pages,
        layers,
    }
}

/// The exact bytes a pool built from `sizes` allocates.
///
/// Every scratch and sentinel row is counted, because every one of them
/// is allocated: `n_scratch` rows per compressed and indexer pool (one
/// per running request row, so a decode whose block did not complete this
/// step scatters to its own discarded row instead of colliding), one
/// scratch row per ring, and one sentinel row on the full -> window map.
/// Leaving them out prices a pool a few rows smaller than the one that
/// gets built, which is exactly the shortfall a byte-exact solve exists
/// to avoid.
///
/// # Panics
///
/// When `sizes` was not built from `args`: the layer counts must agree,
/// or the window term is summed over a different stack than the tiers.
pub fn dsv4_pool_bytes(sizes: &Dsv4PoolSizes, args: &Dsv4Args, n_scratch: usize) -> u64 {
    let ratios = args.ratios();
    assert_eq!(
        sizes.layers.len(),
        ratios.len(),
        "these sizes were built for a {}-layer stack, not a {}-layer one",
        sizes.layers.len(),
        ratios.len()
    );
    let kv_b = kv_bytes(args);
    let idx_b = index_bytes(args);
    let n_scratch = n_scratch as u64;

    // The window pool, on every layer.
    let mut total = ratios.len() as u64 * sizes.n_win_slots as u64 * kv_b;
    // full_to_window, plus its permanent -1 sentinel row.
    total += (sizes.full_token as u64 + 1) * INT64_BYTES;
    for layer in sizes.layers.iter().flatten() {
        total += (layer.cmp_blocks as u64 + n_scratch) * kv_b;
        total += (layer.state_slots as u64 + 1) * state_bytes(args, layer.ratio);
        if layer.ratio == 4 {
            let idx_blocks = layer
                .idx_blocks
                .expect("a ratio-4 layer has an indexer tier");
            let idx_state = layer
                .idx_state_slots
                .expect("a ratio-4 layer has an indexer ring");
            total += (idx_blocks as u64 + n_scratch) * idx_b;
            total += (idx_state as u64 + 1) * idx_state_bytes(args);
        }
    }
    total
}

/// The budget cannot pay for the smallest pool that could serve
/// anything. Refused at config time, before a pool exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dsv4BudgetTooSmall {
    pub available_bytes: u64,
    /// What the minimal pool costs.
    pub needed_bytes: u64,
    /// The minimal anchor, in `P`-pages.
    pub min_pages: usize,
    /// The window working-set floor inside it.
    pub floor_win_pages: usize,
}

impl std::fmt::Display for Dsv4BudgetTooSmall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DSV4 KV budget {} bytes cannot fit the minimal pool ({} pages incl. the window \
             working-set floor {}, needing {} bytes); raise memory_ratio or lower \
             max_running_req/max_seq_len",
            self.available_bytes, self.min_pages, self.floor_win_pages, self.needed_bytes
        )
    }
}

impl std::error::Error for Dsv4BudgetTooSmall {}

/// The largest budget-respecting pool: the greatest `num_pages` whose
/// exact [`dsv4_pool_bytes`] still fits, with the window at
/// `max(floor_win_pages, ceil(swa_ratio * num_pages))`.
///
/// A **binary search over the exact bytes**, not a division. The two are
/// not the same function: `available / cache_per_page` assumes every tier
/// scales with the anchor, and at a small budget the window does not --
/// it pins at its floor while the full anchor shrinks underneath it. The
/// division therefore returns an anchor at which every tier is inflated
/// past the budget at once, and the first allocation, not this call, is
/// where that is discovered.
///
/// The floor is honoured in **pages**, once, per
/// [`dsv4_pool_sizes`]. Below it a full batch cannot get its window pages
/// and admission deadlocks, so a budget that cannot reach it is an error
/// here rather than a pool that boots and dies.
pub fn dsv4_solve_num_pages(
    available_bytes: u64,
    args: &Dsv4Args,
    swa_ratio: f64,
    floor_win_pages: usize,
    page: usize,
    n_scratch: usize,
) -> Result<Dsv4PoolSizes, Dsv4BudgetTooSmall> {
    let sizes_at = |num: usize| -> Dsv4PoolSizes {
        let win = floor_win_pages.max(scaled(swa_ratio, num * page).div_ceil(page));
        dsv4_pool_sizes(num, args, swa_ratio, page, Some(win))
    };

    // The full history must at least cover the window working set.
    let lo0 = floor_win_pages.max(2);
    let needed = dsv4_pool_bytes(&sizes_at(lo0), args, n_scratch);
    if needed > available_bytes {
        return Err(Dsv4BudgetTooSmall {
            available_bytes,
            needed_bytes: needed,
            min_pages: lo0,
            floor_win_pages,
        });
    }

    let mut lo = lo0;
    // A cheap upper bracket: cache_per_page at ratio 0 undercounts the
    // window term, so the quotient over-estimates -- then double until it
    // genuinely does not fit.
    let mut hi = lo.max((available_bytes / dsv4_cache_per_page(args, 0.0, page).max(1)) as usize);
    while dsv4_pool_bytes(&sizes_at(hi), args, n_scratch) <= available_bytes {
        hi *= 2;
    }
    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if dsv4_pool_bytes(&sizes_at(mid), args, n_scratch) <= available_bytes {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(sizes_at(lo))
}

/// The affine price of a DSV4 geometry for the MoE-first auto planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Dsv4AutoCost {
    /// Exact marginal cost of one more `P`-page, across all tiers plus
    /// the full -> window mapping.
    pub cache_per_page: u64,
    /// The intercept, anchored at the minimal viable pool.
    pub fixed_cache_size: u64,
    /// The reserve floor: the window working set plus slack for
    /// plan-vs-measured drift, in tokens.
    pub min_reserve_tokens: usize,
}

/// The `(per page, fixed, reserve)` triple the MoE-first planner splits
/// VRAM with.
///
/// Conservative at the shipped `swa_ratio`; an extreme ratio can dip
/// under 1% below exact, which is harmless because `num_pages` is
/// re-solved byte-exactly by [`dsv4_solve_num_pages`] from the *measured*
/// memory afterwards. This is the estimate the split is planned with, not
/// the number the pool is built from.
pub fn dsv4_auto_cost_model(
    args: &Dsv4Args,
    swa_ratio: f64,
    floor_win_pages: usize,
    page: usize,
    n_scratch: usize,
) -> Dsv4AutoCost {
    let per_page = dsv4_cache_per_page(args, swa_ratio, page) + page as u64 * INT64_BYTES;
    let n0 = floor_win_pages.max(2);
    let win0 = floor_win_pages.max(scaled(swa_ratio, n0 * page).div_ceil(page));
    let base = dsv4_pool_bytes(
        &dsv4_pool_sizes(n0, args, swa_ratio, page, Some(win0)),
        args,
        n_scratch,
    );
    let slack_pages = AUTO_KV_SLACK_BYTES.div_ceil(per_page.max(1)) as usize;
    Dsv4AutoCost {
        cache_per_page: per_page,
        // Saturating: an intercept below zero is a geometry whose marginal
        // page costs more than the minimal pool does, and a wrapped-around
        // enormous fixed term would refuse every split.
        fixed_cache_size: base.saturating_sub(n0 as u64 * per_page),
        min_reserve_tokens: (n0 + slack_pages) * page,
    }
}

// ---------------------------------------------------------------------
// The paged window allocator
// ---------------------------------------------------------------------

/// The free list had fewer units than the request needed. Nothing was
/// taken: the check happens before the first unit moves, so a refused
/// allocation leaves the allocator exactly as it was and the caller can
/// evict and retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeListExhausted {
    pub needed_units: usize,
    pub available_units: usize,
    pub capacity: usize,
    pub page_unit: usize,
}

impl std::fmt::Display for FreeListExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "window free list out of slots: requested {} units, have {} (capacity {}, unit {})",
            self.needed_units, self.available_units, self.capacity, self.page_unit
        )
    }
}

impl std::error::Error for FreeListExhausted {}

/// A LIFO free list over **unit bases**, each a multiple of `page_unit`.
///
/// The window tier allocates one unit of `page_unit == P` slots at a
/// time, so every base it hands out is a page base. That is not a
/// convenience: the compress-state ring block a window slot maps to is
/// `(ws / P) * ring_size`, so a base that is not a multiple of `P` puts
/// two full pages' states in one ring block, where each silently
/// overwrites the other's carry. Keeping the free list in units rather
/// than slots makes that invariant unbreakable rather than merely
/// checked.
///
/// LIFO -- pop from the tail -- so a page just freed is the next one
/// handed out, keeping the live window pages clustered.
///
/// The compressed and indexer tiers have no allocator at all: their rows
/// are `full_loc / ratio`, pure arithmetic.
#[derive(Debug, Clone)]
pub struct FreeListAllocator {
    capacity: usize,
    page_unit: usize,
    free: Vec<usize>,
}

impl FreeListAllocator {
    /// A free list over `capacity` slots in `page_unit`-slot units.
    ///
    /// # Panics
    ///
    /// When the capacity is not a whole number of units. A ragged tail
    /// would either be handed out as a short unit or silently lost.
    pub fn new(capacity: usize, page_unit: usize) -> Self {
        assert!(page_unit > 0, "a unit spans at least one slot");
        assert!(
            capacity.is_multiple_of(page_unit),
            "capacity {capacity} must be a multiple of page_unit {page_unit}"
        );
        let mut allocator = FreeListAllocator {
            capacity,
            page_unit,
            free: Vec::new(),
        };
        allocator.reset();
        allocator
    }

    /// `n_units` unit bases, each a multiple of `page_unit`.
    ///
    /// All or nothing: the capacity check precedes the first pop.
    pub fn alloc(&mut self, n_units: usize) -> Result<Vec<usize>, FreeListExhausted> {
        if n_units > self.free.len() {
            return Err(FreeListExhausted {
                needed_units: n_units,
                available_units: self.free.len(),
                capacity: self.capacity,
                page_unit: self.page_unit,
            });
        }
        // Take the tail, in ascending order, exactly as the reference
        // slices it: the caller pairs unit `i` with its `i`-th page.
        Ok(self.free.split_off(self.free.len() - n_units))
    }

    /// Return previously-allocated unit bases.
    ///
    /// # Panics
    ///
    /// On anything that is not a unit base inside the capacity. A base
    /// that is not page-aligned poisons every later allocation -- and the
    /// ring aliasing it causes has no symptom other than wrong logits.
    /// Asserted here rather than assumed, because the callers derive
    /// these bases from a mapping they may have mutated.
    pub fn free(&mut self, units: &[usize]) {
        for base in units.iter().copied() {
            assert!(
                base.is_multiple_of(self.page_unit) && base < self.capacity,
                "{base} is not a unit base of a {}-slot unit inside a capacity of {}",
                self.page_unit,
                self.capacity
            );
            self.free.push(base);
        }
    }

    /// Free capacity in **slots**.
    pub fn available(&self) -> usize {
        self.free.len() * self.page_unit
    }

    /// Free capacity in units.
    pub fn free_units(&self) -> usize {
        self.free.len()
    }

    /// Total slots, allocated or not.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn page_unit(&self) -> usize {
        self.page_unit
    }

    /// Drop every outstanding allocation.
    pub fn reset(&mut self) {
        let n_units = self.capacity / self.page_unit;
        self.free = (0..n_units).map(|unit| unit * self.page_unit).collect();
    }
}

/// The layer-invariant ring context for one decode step of one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dsv4WindowCtx {
    /// Where this step's own KV is written.
    pub window_slot: i64,
    /// The previous position's slot, for the compressor's carry.
    pub prev_window_slot: i64,
    /// One entry per ring slot `j`: the window slot holding the latest
    /// position `p <= pos` with `p % win == j`, or
    /// [`NO_WINDOW_SLOT`] where the sequence has not reached it.
    pub window_slots_topk: Vec<i64>,
}

/// Which position occupies ring slot `j` at decode position `pos`.
///
/// `p = pos - ((pos - j) % win)` -- the latest position at or before
/// `pos` that is congruent to `j`. `None` before the sequence has reached
/// the slot (`j > pos`, or `p < 0` early in a decode), which the caller
/// renders as [`NO_WINDOW_SLOT`] and the sparse kernel masks.
///
/// The modulo is **euclidean**: Rust's `%` keeps the sign of the
/// dividend, so for `j > pos` a truncated remainder names a position
/// *after* `pos` -- a slot the request has not written yet, read as
/// though it had.
pub fn window_ring_position(pos: i64, j: usize, win: usize) -> Option<i64> {
    assert!(win > 0, "a ring holds at least one slot");
    if (j as i64) > pos {
        return None;
    }
    let p = pos - (pos - j as i64).rem_euclid(win as i64);
    (p >= 0).then_some(p)
}

/// The window tier's page-atomic bookkeeping: which window page backs
/// each full page, and which window pages are still free.
///
/// Token-faced on the outside (the generic cache manager speaks tokens),
/// page-atomic inside. Window pages are bound 1:1 to full pages and the
/// per-page state ring requires exactly that, so the page-completeness of
/// every call is asserted here rather than assumed.
#[derive(Debug, Clone)]
pub struct Dsv4WindowPool {
    page: usize,
    full_token: usize,
    n_win_slots: usize,
    /// Full loc -> window slot, `-1` unbound. One row longer than the
    /// history: the trailing row is a permanent `-1` sentinel, so a
    /// gather at `-1` reads it and returns `-1` instead of faulting.
    full_to_window: Vec<i64>,
    allocator: FreeListAllocator,
    chunk_budget: usize,
}

impl Dsv4WindowPool {
    /// Build the pool-owned free list and the tail dummy binding.
    ///
    /// The **last** full page and the **last** window page are the
    /// reserved dummy region: the page table's dummy row points at
    /// `full_token - P`, permanently bound, so graph-padded rows scatter
    /// to a real slot instead of a negative index. That page is bound
    /// outside the free list -- the allocator's capacity is
    /// `n_win_slots - P` -- which is also what makes the usable window
    /// count `swa_num_tokens - 1`, the same capacity convention the
    /// generic pool gets from reserving slot 0.
    pub fn new(sizes: &Dsv4PoolSizes, max_running_req: usize, radix: bool) -> Self {
        let page = sizes.page;
        assert!(
            sizes.full_token >= page && sizes.full_token.is_multiple_of(page),
            "the full anchor must be whole pages and hold the dummy page"
        );
        assert!(
            sizes.n_win_slots >= page && sizes.n_win_slots.is_multiple_of(page),
            "the window pool must be whole pages and hold the dummy page"
        );
        let mut pool = Dsv4WindowPool {
            page,
            full_token: sizes.full_token,
            n_win_slots: sizes.n_win_slots,
            full_to_window: vec![NO_WINDOW_SLOT; sizes.full_token + 1],
            allocator: FreeListAllocator::new(sizes.n_win_slots - page, page),
            chunk_budget: 0,
        };
        pool.bind_window_pages(sizes.full_token - page, sizes.n_win_slots - page);

        // A batched prefill holds the whole chunk's window live at once
        // (sliding frees only between chunks, so the peak is ~2x the
        // chunk): reserve the concurrent working set and halve the rest.
        let n_win_pages = (sizes.n_win_slots / page) - 1;
        let reserved = dsv4_reserved_window_pages(max_running_req, radix);
        pool.chunk_budget = page.max(n_win_pages.saturating_sub(reserved) / 2 * page);
        pool
    }

    /// `P`: the window page, the sliding window, and the radix block key.
    pub fn page_size(&self) -> usize {
        self.page
    }

    /// The window pool in the generic capacity convention: allocatable
    /// slots plus one. The dummy page is already excluded from the free
    /// list, so the `+1` re-encodes the same `capacity == tokens - 1`
    /// the generic pool gets from its slot-0 sentinel.
    pub fn swa_num_tokens(&self) -> usize {
        (self.n_win_slots - self.page) + 1
    }

    /// Window slots that can still be handed out.
    pub fn swa_available_size(&self) -> usize {
        self.allocator.available()
    }

    /// The largest prefill chunk this pool can hold the window for.
    pub fn prefill_chunk_budget(&self) -> usize {
        self.chunk_budget
    }

    /// Permanently bind one full page to one window page, offsets
    /// preserved.
    ///
    /// # Panics
    ///
    /// On a base that is not page-aligned: the ring block layout is keyed
    /// on the page base, so an unaligned binding aliases two pages onto
    /// one ring block.
    pub fn bind_window_pages(&mut self, full_page_base: usize, window_page_base: usize) {
        assert!(
            full_page_base.is_multiple_of(self.page) && window_page_base.is_multiple_of(self.page),
            "window bindings are page-aligned: full {full_page_base}, window {window_page_base}"
        );
        for offset in 0..self.page {
            self.full_to_window[full_page_base + offset] = (window_page_base + offset) as i64;
        }
    }

    /// Drop the binding of these full locs, returning nothing to the
    /// free list. Negative locs are ignored.
    pub fn unbind_window_pages(&mut self, full_locs: &[i64]) {
        for loc in full_locs.iter().copied().filter(|loc| *loc >= 0) {
            self.full_to_window[loc as usize] = NO_WINDOW_SLOT;
        }
    }

    /// Bind one window page per incoming full page.
    ///
    /// `full_indices` must be whole contiguous ascending pages -- the
    /// page-to-token expansion the caller already performs. The in-page
    /// offsets are **preserved** (`window_slot = wbase + pos % P`), which
    /// is what the state ring's page-block layout requires: a slot
    /// permuted inside its page still lands in the right ring block, but
    /// on the wrong slot of it.
    ///
    /// All or nothing: the pages are allocated before the first mapping
    /// row is written, so an exhausted pool leaves the mapping untouched.
    ///
    /// # Panics
    ///
    /// On a partial, unaligned or non-ascending page. Upstream gets these
    /// from the page expansion by construction; asserted here, not
    /// assumed.
    pub fn alloc_swa(&mut self, full_indices: &[i64]) -> Result<(), FreeListExhausted> {
        if full_indices.is_empty() {
            return Ok(());
        }
        let page = self.page;
        assert!(
            full_indices.len().is_multiple_of(page),
            "alloc_swa needs whole pages, got {} slots",
            full_indices.len()
        );
        let mut bases = Vec::with_capacity(full_indices.len() / page);
        for chunk in full_indices.chunks(page) {
            let base = chunk[0];
            assert!(
                base >= 0 && (base as usize).is_multiple_of(page),
                "alloc_swa pages start at a page base, got {base}"
            );
            for (offset, loc) in chunk.iter().copied().enumerate() {
                assert_eq!(
                    loc,
                    base + offset as i64,
                    "alloc_swa pages must be contiguous ascending"
                );
            }
            debug_assert_eq!(
                self.full_to_window[base as usize], NO_WINDOW_SLOT,
                "full page {base} already holds a window page; binding over it would leak it"
            );
            bases.push(base as usize);
        }

        let wbases = self.allocator.alloc(bases.len())?;
        for (fbase, wbase) in bases.into_iter().zip(wbases) {
            for offset in 0..page {
                self.full_to_window[fbase + offset] = (wbase + offset) as i64;
            }
        }
        Ok(())
    }

    /// Return the window pages backing these full locs and unbind them.
    ///
    /// Page-atomic: the locs must cover every touched page completely.
    /// Idempotent over pages that are already unbound -- a slide, a
    /// tombstone and an eviction pass may all name the same page, and
    /// only the first hands its window page back.
    ///
    /// # Panics
    ///
    /// On a partially covered page. Freeing half a page would return a
    /// window page whose other half is still mapped, so the next
    /// allocation gets a page two full pages read through.
    pub fn free_swa(&mut self, full_indices: &[i64]) {
        let page = self.page;
        let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
        for loc in full_indices.iter().copied().filter(|loc| *loc >= 0) {
            *counts.entry(loc as usize / page * page).or_insert(0) += 1;
        }
        if counts.is_empty() {
            return;
        }
        let partial: Vec<(usize, usize)> = counts
            .iter()
            .filter(|(_, count)| **count != page)
            .map(|(base, count)| (*base, *count))
            .take(4)
            .collect();
        assert!(
            partial.is_empty(),
            "free_swa got partial pages (base, count): {partial:?}"
        );

        let mut freed = Vec::with_capacity(counts.len());
        for base in counts.into_keys() {
            let window_slot = self.full_to_window[base];
            for offset in 0..page {
                self.full_to_window[base + offset] = NO_WINDOW_SLOT;
            }
            if window_slot >= 0 {
                freed.push(window_slot as usize / page * page);
            }
        }
        self.allocator.free(&freed);
    }

    /// The window slot holding `full_loc`, or [`NO_WINDOW_SLOT`].
    ///
    /// A negative loc reads the permanent sentinel row and returns `-1`,
    /// so callers carrying "no such position" as `-1` need no special
    /// case -- upstream gets the same effect from negative tensor
    /// indexing into the trailing row.
    pub fn translate(&self, full_loc: i64) -> i64 {
        if full_loc < 0 {
            return NO_WINDOW_SLOT;
        }
        self.full_to_window[full_loc as usize]
    }

    /// The compress-state ring location for a window slot -- **derived**,
    /// never stored.
    ///
    /// `(ws / P) * ring_size + ws % ring_size`: the page picks the ring
    /// block, the in-page offset picks the slot inside it. `ring_size`
    /// divides `P`, so distinct window pages land on disjoint blocks.
    ///
    /// Deriving it on every use is the whole safety property. A stored
    /// `state_loc` outlives the binding it was derived from: free a
    /// window page and the next request to take it inherits the same ring
    /// block, so a cached location reads another request's carry state --
    /// no error, no fault, just wrong numbers.
    ///
    /// # Panics
    ///
    /// When `ring_size` does not divide `page`, which is what makes the
    /// blocks disjoint in the first place.
    pub fn state_loc(window_slot: i64, ring_size: usize, page: usize) -> i64 {
        assert!(
            ring_size > 0 && page.is_multiple_of(ring_size),
            "ring_size {ring_size} must divide P={page}, or two pages share a ring block"
        );
        if window_slot < 0 {
            return NO_WINDOW_SLOT;
        }
        let pages = window_slot / page as i64;
        pages * ring_size as i64 + window_slot % ring_size as i64
    }

    /// The ring context for one decode step: this position's slot, the
    /// previous position's, and the whole `P`-slot ring.
    ///
    /// `full_locs` is the request's snapshot -- full loc per position --
    /// read instead of the live mapping so a concurrent allocation cannot
    /// redirect an in-flight step. Computed fresh every call: caching it
    /// freezes a replay at the capture-time ring slots, `-1` fills
    /// included.
    pub fn window_ctx(&self, pos: usize, full_locs: &[i64]) -> Dsv4WindowCtx {
        let win = self.page;
        let window_slots_topk = (0..win)
            .map(|j| match window_ring_position(pos as i64, j, win) {
                Some(p) => self.translate(full_locs[p as usize]),
                None => NO_WINDOW_SLOT,
            })
            .collect();
        Dsv4WindowCtx {
            window_slot: self.translate(full_locs[pos]),
            prev_window_slot: self.translate(full_locs[pos.saturating_sub(1)]),
            window_slots_topk,
        }
    }

    /// Every window page is either free or bound to exactly one full
    /// page, and every binding is page-atomic.
    ///
    /// The count is an **equality**: free units plus bound pages is the
    /// capacity exactly. A `<=` would catch a double free and tolerate a
    /// leak, and a leaked window page is the failure that shows up an
    /// hour later as a pool that admits nothing.
    pub fn check_integrity(&self) {
        let page = self.page;
        let n_full_pages = self.full_token / page;
        let mut seen = vec![false; self.n_win_slots / page];
        let mut bound = 0usize;

        // The dummy page (the last one) is permanently bound outside the
        // free list, so it is neither free nor counted.
        assert_eq!(
            self.full_to_window[self.full_token - page],
            (self.n_win_slots - page) as i64,
            "the reserved dummy page lost its permanent binding"
        );
        assert_eq!(
            self.full_to_window[self.full_token], NO_WINDOW_SLOT,
            "the trailing sentinel row was written"
        );

        for full_page in 0..n_full_pages - 1 {
            let base = full_page * page;
            let window_slot = self.full_to_window[base];
            for offset in 0..page {
                let expected = if window_slot < 0 {
                    NO_WINDOW_SLOT
                } else {
                    window_slot + offset as i64
                };
                assert_eq!(
                    self.full_to_window[base + offset],
                    expected,
                    "full page {base} is bound partially or out of order at offset {offset}"
                );
            }
            if window_slot < 0 {
                continue;
            }
            assert!(
                (window_slot as usize).is_multiple_of(page),
                "window base {window_slot} is not page-aligned; its ring block aliases"
            );
            let index = window_slot as usize / page;
            assert!(
                !std::mem::replace(&mut seen[index], true),
                "window page {window_slot} is bound to two full pages"
            );
            bound += 1;
        }

        for base in self.free_bases() {
            assert!(
                !std::mem::replace(&mut seen[base / page], true),
                "window page {base} is both free and bound, or free twice"
            );
        }

        let capacity_units = self.allocator.capacity() / page;
        assert_eq!(
            self.allocator.free_units() + bound,
            capacity_units,
            "window pages leaked or double-freed: {} free + {bound} bound != {capacity_units}",
            self.allocator.free_units()
        );
    }

    fn free_bases(&self) -> Vec<usize> {
        self.allocator.free.clone()
    }
}

#[cfg(test)]
mod tests {

    /// The shipped schedule, and the whole reason a scalar ratio is
    /// wrong: one array holds all three mechanisms, so any single value
    /// applied uniformly is right for at most one kind of layer.
    #[test]
    fn the_shipped_schedule_reads_as_three_different_mechanisms() {
        let args = Dsv4Args {
            head_dim: 64,
            index_head_dim: 32,
            n_layers: 8,
            compress_ratios: vec![0, 0, 4, 128, 4, 128, 4, 0],
        };
        assert_eq!(
            args.compressors().unwrap(),
            vec![
                LayerCompressor::None,
                LayerCompressor::None,
                LayerCompressor::Csa,
                LayerCompressor::Hca,
                LayerCompressor::Csa,
                LayerCompressor::Hca,
                LayerCompressor::Csa,
                LayerCompressor::None,
            ]
        );
    }

    /// The two properties a uniform ratio gets wrong on half the stack:
    /// an indexer built where every entry is already visible, and a
    /// compressor projection of the wrong width. Both run and produce
    /// numbers, which is why they are derived from the mechanism rather
    /// than configured.
    #[test]
    fn the_indexer_and_the_projection_width_follow_from_the_mechanism() {
        assert!(LayerCompressor::Csa.has_indexer());
        assert!(!LayerCompressor::Hca.has_indexer());
        assert!(!LayerCompressor::None.has_indexer());

        assert_eq!(LayerCompressor::Csa.projection_width_multiple(), 2);
        assert_eq!(LayerCompressor::Hca.projection_width_multiple(), 1);
        assert_eq!(LayerCompressor::None.projection_width_multiple(), 1);

        assert!(LayerCompressor::Csa.overlapping());
        assert!(!LayerCompressor::Hca.overlapping());
    }

    /// Visibility is ratio-derived, and the `+1` is load-bearing: the
    /// query at the last position of a block must see the block it just
    /// completed. Without it every layer is short exactly one
    /// compressed entry, for the whole sequence.
    #[test]
    fn a_query_sees_one_compressed_entry_per_completed_block() {
        let csa = LayerCompressor::Csa;
        assert_eq!(csa.visible_compressed(0), 0, "no block is complete yet");
        assert_eq!(csa.visible_compressed(2), 0);
        assert_eq!(csa.visible_compressed(3), 1, "the first block just closed");
        assert_eq!(csa.visible_compressed(7), 2);
        assert_eq!(csa.visible_compressed(8), 2);

        let hca = LayerCompressor::Hca;
        assert_eq!(hca.visible_compressed(126), 0);
        assert_eq!(hca.visible_compressed(127), 1);
        assert_eq!(hca.visible_compressed(255), 2);

        // A layer with no compressor has nothing to see, at any
        // position -- not "all of them", which a ratio of zero would
        // produce as a divide by zero rather than an answer.
        for pos in [0, 1, 127, 1_000_000] {
            assert_eq!(LayerCompressor::None.visible_compressed(pos), 0);
        }
    }

    /// A ratio that is none of the three is refused, naming its layer.
    /// There is no nearest mechanism to round to, and picking one gives
    /// that layer the wrong compressor width and the wrong indexer with
    /// nothing to point at afterwards.
    #[test]
    fn an_unknown_ratio_is_refused_and_names_its_layer() {
        assert_eq!(LayerCompressor::from_ratio(7), None);
        assert_eq!(LayerCompressor::from_ratio(64), None);

        let args = Dsv4Args {
            head_dim: 64,
            index_head_dim: 32,
            n_layers: 3,
            compress_ratios: vec![0, 64, 128],
        };
        let err = args.compressors().unwrap_err();
        assert_eq!(
            err,
            UnknownCompressRatio {
                layer: 1,
                ratio: 64
            }
        );
        assert!(err.to_string().contains("layer 1"));
    }

    /// The schedule reads the same array the tiers are priced from, and
    /// truncates the same way -- a checkpoint shipping 44 ratios for 43
    /// layers describes 43 layers. Sizing a tier this stack never
    /// executes, or executing one it never sized, are the two failures
    /// sharing the array prevents.
    #[test]
    fn the_schedule_and_the_sizing_read_the_same_truncated_array() {
        let args = Dsv4Args {
            head_dim: 64,
            index_head_dim: 32,
            n_layers: 3,
            compress_ratios: vec![4, 128, 0, 4],
        };
        assert_eq!(args.ratios(), &[4, 128, 0]);
        let compressors = args.compressors().unwrap();
        assert_eq!(compressors.len(), args.ratios().len());
        for (c, &r) in compressors.iter().zip(args.ratios()) {
            assert_eq!(c.ratio(), r);
        }
    }

    /// Every ratio the ring geometry accepts is one the schedule can
    /// read, and vice versa for the compressed tiers. The two tables
    /// are separate functions over the same three values, and a ratio
    /// only one of them knows is a layer that is sized without being
    /// executable or the reverse.
    #[test]
    fn the_ring_table_and_the_compressor_table_agree_on_which_ratios_exist() {
        for ratio in [CSA_RATIO, HCA_RATIO] {
            assert!(LayerCompressor::from_ratio(ratio).is_some());
            assert!(ring_size_for_ratio(ratio) > 0);
        }
        // Ratio 0 is readable as a mechanism but has no ring, because a
        // layer with no compressor has no carry state to address.
        assert_eq!(LayerCompressor::from_ratio(0), Some(LayerCompressor::None));
        assert!(std::panic::catch_unwind(|| ring_size_for_ratio(0)).is_err());
    }
    use super::*;

    const P: usize = DEFAULT_WINDOW_PAGE;

    /// head_dim 8 -> kv 16 B; index_head_dim 4 -> idx 8 B.
    /// Ratio-4 state rows 2*2*8*4 = 128 B, ratio-128 rows 2*1*8*4 = 64 B,
    /// indexer state rows 2*2*4*4 = 64 B.
    fn args() -> Dsv4Args {
        Dsv4Args {
            head_dim: 8,
            index_head_dim: 4,
            n_layers: 4,
            compress_ratios: vec![0, 4, 128, 4],
        }
    }

    /// `dsv4_pool_bytes` for [`args`], worked out by hand:
    /// window 4 layers x 16 B x P = 8192 B per window page, the two
    /// ratio-4 layers' rings 2 x (8x128 + 8x64) = 3072 B per window page,
    /// the ratio-128 layer's ring 128x64 = 8192 B per window page;
    /// per full page 8 x 128 mapping + 2 x (32x16 + 32x8) + 1x16 = 2576 B.
    fn expected_bytes(num_pages: usize, win_pages: usize, n_scratch: usize) -> u64 {
        19456 * win_pages as u64
            + 2576 * num_pages as u64
            // scratch rows: 2 x (cmp 16 B + idx 8 B) + 1 x cmp 16 B
            + 64 * n_scratch as u64
            // ring scratch rows 2x(128+64) + 1x64, mapping sentinel 8
            + 456
    }

    // ---- the tiered cost model ----

    #[test]
    fn ring_sizes_are_fixed_per_ratio() {
        assert_eq!(ring_size_for_ratio(4), 8);
        assert_eq!(ring_size_for_ratio(128), 128);
    }

    /// The ring geometry is a property of the two compressors that ship,
    /// not a function of the ratio -- so an unknown ratio is refused
    /// rather than given a guessed ring.
    #[test]
    #[should_panic(expected = "no ring for ratio 8")]
    fn an_unsupported_ratio_has_no_ring() {
        ring_size_for_ratio(8);
    }

    /// The docstring says "2 per req + dummy"; the code reserves
    /// `2 * (mr + 1) + 3 * mr + 1`. The code is what the engine floor and
    /// the chunk budget both reserve against, so the code is what is
    /// ported.
    #[test]
    fn the_reserved_window_pages_follow_the_code_not_the_docstring() {
        assert_eq!(dsv4_reserved_window_pages(2, true), 2 * 3 + 3 * 2 + 1);
        assert_eq!(dsv4_reserved_window_pages(2, false), 2 * 3 + 1);
        // Not the docstring's 2 * mr + 1.
        assert_ne!(dsv4_reserved_window_pages(2, false), 2 * 2 + 1);
    }

    #[test]
    fn the_window_floor_caps_the_prefill_reach_at_eight_pages() {
        // 2048 tokens is 16 pages of reach, capped at 8.
        assert_eq!(
            dsv4_window_floor_pages(2048, 2, true, P),
            8 + dsv4_reserved_window_pages(2, true)
        );
        // A short context pays only its own reach.
        assert_eq!(
            dsv4_window_floor_pages(256, 2, true, P),
            2 + dsv4_reserved_window_pages(2, true)
        );
    }

    #[test]
    fn the_per_page_cost_sums_every_tier_over_the_layers() {
        // window 4 x round(0.5 x 128) x 16 = 4096
        // ratio-4 layers x2: 32x16 + 32x8 + round(0.5x8)x64 + round(0.5x8)x128
        // ratio-128 layer: 1x16 + round(0.5x128)x64
        assert_eq!(
            dsv4_cache_per_page(&args(), 0.5, P),
            4096 + 2 * (512 + 256 + 256 + 512) + (16 + 4096)
        );
        // At ratio 0 the window tier and both rings vanish, leaving only
        // the full-anchored tiers -- which is the undercounting bracket
        // the page solve starts its search from.
        assert_eq!(dsv4_cache_per_page(&args(), 0.0, P), 2 * (512 + 256) + 16);
    }

    /// The sizing rounds halves to even (Python's `round`), while the
    /// unit costs round bytes-per-token up. `f64::round` would take
    /// 0.5 -> 1 here and buy a ring slot per page the budget never
    /// priced.
    #[test]
    fn the_ratio_scaling_rounds_halves_to_even() {
        let args = args();
        // The ratio-4 rings: round(0.0625 x 8) == round(0.5) == 0, not 1,
        // so those two layers buy no ring slots at all on this page.
        let half_down = dsv4_cache_per_page(&args, 0.0625, P);
        assert_eq!(half_down, 4 * 8 * 16 + 2 * (512 + 256) + (16 + 8 * 64));
        // round(0.1875 x 8) == round(1.5) == 2, up to the even neighbour.
        let half_up = dsv4_cache_per_page(&args, 0.1875, P);
        assert_eq!(
            half_up,
            4 * 24 * 16 + 2 * (512 + 256 + 2 * 64 + 2 * 128) + (16 + 24 * 64)
        );
    }

    #[test]
    fn the_unit_costs_round_bytes_per_token_up() {
        // full tier: 1024 mapping + 2 x 768 + 16 = 2576 per page -> 20.125
        assert_eq!(dsv4_kv_unit_bytes(&args(), P), 21);
        // window tier: 8192 + 2 x 1536 + 8192 = 19456 per page -> exact
        assert_eq!(dsv4_window_unit_bytes(&args(), P), 152);
    }

    /// The window tier is `swa_ratio` of the history; the compressed and
    /// indexer tiers stay anchored to the FULL history whatever the
    /// window does.
    #[test]
    fn the_window_tier_is_sized_independently_of_the_full_anchor() {
        let args = args();
        let sizes = dsv4_pool_sizes(64, &args, 0.1, P, None);
        assert_eq!(sizes.full_token, 64 * P);
        // round(0.1 x 8192) = 819 slots -> ceil to 7 whole pages.
        assert_eq!(sizes.n_win_pages, 7);
        assert_eq!(sizes.n_win_slots, 7 * P);

        assert_eq!(sizes.layers[0], None, "a ratio-0 layer has no tiers");
        let ratio4 = sizes.layers[1].unwrap();
        assert_eq!(ratio4.cmp_blocks, 64 * P / 4, "full-anchored");
        assert_eq!(ratio4.idx_blocks, Some(64 * P / 4));
        assert_eq!(ratio4.state_slots, 7 * 8, "window-anchored");
        assert_eq!(ratio4.idx_state_slots, Some(7 * 8));
        let ratio128 = sizes.layers[2].unwrap();
        assert_eq!(ratio128.cmp_blocks, 64 * P / 128);
        assert_eq!(ratio128.idx_blocks, None);
        assert_eq!(ratio128.state_slots, 7 * 128);
    }

    #[test]
    fn an_explicit_window_is_capped_at_the_full_history() {
        let sizes = dsv4_pool_sizes(8, &args(), 0.5, P, Some(64));
        assert_eq!(
            sizes.n_win_pages, 8,
            "a window past the history is bytes nobody can address"
        );
        // And a ratio over 1.0 is capped the same way.
        assert_eq!(dsv4_pool_sizes(8, &args(), 4.0, P, None).n_win_pages, 8);
    }

    #[test]
    fn the_pool_bytes_are_the_sum_of_every_allocated_row() {
        let args = args();
        let sizes = dsv4_pool_sizes(64, &args, 0.1, P, Some(21));
        assert_eq!(dsv4_pool_bytes(&sizes, &args, 1), expected_bytes(64, 21, 1));
        // Each scratch row is really allocated, so each is really priced.
        assert_eq!(
            dsv4_pool_bytes(&sizes, &args, 3) - dsv4_pool_bytes(&sizes, &args, 1),
            2 * 64
        );
    }

    /// The naive solve -- `available / cache_per_page` -- returns an
    /// anchor whose exact bytes are OVER the budget, because the window
    /// tier does not scale with the anchor once it pins at its floor.
    /// The binary search returns the largest anchor that actually fits.
    #[test]
    fn dividing_the_budget_by_the_per_page_cost_overshoots_it() {
        let args = args();
        let floor = 21;
        let available = 500_000;
        let solved = dsv4_solve_num_pages(available, &args, 0.5, floor, P, 1).expect("fits");

        let naive =
            (available / (dsv4_cache_per_page(&args, 0.5, P) + P as u64 * INT64_BYTES)) as usize;
        assert_eq!(naive, 40);
        let naive_sizes = dsv4_pool_sizes(naive, &args, 0.5, P, Some(floor.max(naive.div_ceil(2))));
        assert!(
            dsv4_pool_bytes(&naive_sizes, &args, 1) > available,
            "the division must be the one that overshoots"
        );

        assert_eq!(solved.num_pages(), 35);
        assert!(dsv4_pool_bytes(&solved, &args, 1) <= available);
    }

    /// At a small budget the window pins at its floor in PAGES and the
    /// full anchor shrinks underneath it. Honouring the floor by
    /// inflating `swa_ratio` instead -- the ratio that yields the floor at
    /// the minimal pool -- keeps the window scaling with the anchor, so
    /// at the solved anchor it buys far more window than the floor asked
    /// for and blows the budget.
    #[test]
    fn a_small_budget_pins_the_window_at_its_floor_and_shrinks_the_full_anchor() {
        let args = args();
        let floor = 21;
        let available = 900_000;
        let solved = dsv4_solve_num_pages(available, &args, 0.1, floor, P, 1).expect("fits");

        assert_eq!(solved.n_win_pages, floor, "the window pinned at its floor");
        assert_eq!(solved.num_pages(), 190, "the full anchor took the rest");
        assert!(solved.num_pages() > solved.n_win_pages * 4);
        // The full-anchored tiers still cover the whole history.
        assert_eq!(solved.layers[1].unwrap().cmp_blocks, 190 * P / 4);

        let inflated = floor as f64 / floor.max(2) as f64; // 1.0
        let inflated_sizes = dsv4_pool_sizes(solved.num_pages(), &args, inflated, P, None);
        assert!(inflated_sizes.n_win_pages > solved.n_win_pages);
        assert!(
            dsv4_pool_bytes(&inflated_sizes, &args, 1) > available,
            "inflating swa_ratio to carry the floor busts the budget"
        );
    }

    #[test]
    fn the_solved_pool_is_the_largest_that_fits() {
        let args = args();
        let available = 900_000;
        let solved = dsv4_solve_num_pages(available, &args, 0.1, 21, P, 1).expect("fits");
        let bytes = dsv4_pool_bytes(&solved, &args, 1);
        assert!(bytes <= available);

        let one_more = dsv4_pool_sizes(
            solved.num_pages() + 1,
            &args,
            0.1,
            P,
            Some(21.max(scaled(0.1, (solved.num_pages() + 1) * P).div_ceil(P))),
        );
        assert!(dsv4_pool_bytes(&one_more, &args, 1) > available);
    }

    /// Below the floor a full batch cannot get its window pages and
    /// admission deadlocks -- so the budget is refused at config time
    /// rather than at the first allocation.
    #[test]
    fn a_budget_below_the_minimal_pool_is_refused_at_config_time() {
        let args = args();
        let err = dsv4_solve_num_pages(100_000, &args, 0.1, 21, P, 1).unwrap_err();
        assert_eq!(err.min_pages, 21);
        assert_eq!(err.floor_win_pages, 21);
        assert!(err.needed_bytes > err.available_bytes);
        assert!(err.to_string().contains("working-set floor 21"));
    }

    #[test]
    fn the_auto_cost_model_is_affine_through_the_minimal_pool() {
        let args = args();
        let floor = 21;
        let cost = dsv4_auto_cost_model(&args, 0.1, floor, P, 1);
        assert_eq!(
            cost.cache_per_page,
            dsv4_cache_per_page(&args, 0.1, P) + P as u64 * INT64_BYTES
        );

        // The intercept is anchored at the minimal pool: the affine price
        // reproduces its exact bytes there, as an equality.
        let n0 = floor.max(2);
        let base = dsv4_pool_bytes(&dsv4_pool_sizes(n0, &args, 0.1, P, Some(floor)), &args, 1);
        assert_eq!(
            cost.fixed_cache_size + n0 as u64 * cost.cache_per_page,
            base
        );

        let slack_pages = AUTO_KV_SLACK_BYTES.div_ceil(cost.cache_per_page) as usize;
        assert_eq!(cost.min_reserve_tokens, (n0 + slack_pages) * P);
    }

    // ---- the paged window allocator ----

    fn pool() -> Dsv4WindowPool {
        // 8 full pages, 5 window pages -> 4 allocatable, 1 dummy.
        let sizes = dsv4_pool_sizes(8, &args(), 0.5, P, Some(5));
        Dsv4WindowPool::new(&sizes, 2, true)
    }

    fn page_locs(full_page: usize) -> Vec<i64> {
        let base = (full_page * P) as i64;
        (0..P as i64).map(|offset| base + offset).collect()
    }

    /// A base that is not a multiple of the page unit puts two full pages
    /// in one ring block. The free list is kept in units precisely so
    /// that cannot be expressed.
    #[test]
    fn every_unit_base_is_a_multiple_of_the_page_unit() {
        let mut allocator = FreeListAllocator::new(8 * P, P);
        assert_eq!(allocator.available(), 8 * P);
        let taken = allocator.alloc(3).expect("three of eight");
        assert_eq!(taken.len(), 3);
        assert!(taken.iter().all(|base| base.is_multiple_of(P)), "{taken:?}");
        // LIFO from the tail, ascending inside the slice.
        assert_eq!(taken, vec![5 * P, 6 * P, 7 * P]);
        assert_eq!(allocator.available(), 5 * P);

        allocator.free(&taken);
        assert_eq!(allocator.available(), 8 * P);
        // The just-freed pages come straight back.
        assert_eq!(allocator.alloc(1).unwrap(), vec![7 * P]);
    }

    #[test]
    #[should_panic(expected = "is not a unit base")]
    fn returning_a_base_that_is_not_a_unit_base_is_refused() {
        let mut allocator = FreeListAllocator::new(8 * P, P);
        allocator.free(&[P + 1]);
    }

    #[test]
    fn an_oversized_allocation_takes_nothing() {
        let mut allocator = FreeListAllocator::new(4 * P, P);
        let err = allocator.alloc(5).unwrap_err();
        assert_eq!(err.needed_units, 5);
        assert_eq!(err.available_units, 4);
        assert_eq!(allocator.available(), 4 * P, "nothing was taken");
        allocator.alloc(4).expect("the free list is intact");
    }

    /// A refused window allocation must leave the mapping untouched: the
    /// caller's next move is to evict and retry, not to undo a partial
    /// binding.
    #[test]
    fn an_exhausted_window_pool_binds_nothing() {
        let mut pool = pool();
        let mut locs = Vec::new();
        for full_page in 0..5 {
            locs.extend(page_locs(full_page));
        }
        let err = pool.alloc_swa(&locs).unwrap_err();
        assert_eq!(err.needed_units, 5);
        assert_eq!(err.available_units, 4);
        assert_eq!(pool.translate(0), NO_WINDOW_SLOT);
        pool.check_integrity();
    }

    /// The state ring's block layout is keyed on the in-page offset, so a
    /// binding that permutes inside its page lands on the wrong ring slot.
    #[test]
    fn alloc_swa_preserves_in_page_offsets() {
        let mut pool = pool();
        pool.alloc_swa(&page_locs(0)).expect("one of four");
        let base = pool.translate(0);
        assert!(base >= 0 && (base as usize).is_multiple_of(P));
        for offset in 0..P as i64 {
            assert_eq!(pool.translate(offset), base + offset);
        }
        pool.check_integrity();
    }

    #[test]
    #[should_panic(expected = "whole pages")]
    fn alloc_swa_refuses_a_partial_page() {
        let mut pool = pool();
        let _ = pool.alloc_swa(&page_locs(0)[..P - 1]);
    }

    #[test]
    #[should_panic(expected = "contiguous ascending")]
    fn alloc_swa_refuses_a_page_that_is_not_contiguous_ascending() {
        let mut pool = pool();
        let mut locs = page_locs(0);
        locs.swap(3, 9);
        let _ = pool.alloc_swa(&locs);
    }

    /// Freeing half a page returns a window page whose other half is
    /// still mapped: the next allocation gets a page two full pages read
    /// through.
    #[test]
    #[should_panic(expected = "partial pages")]
    fn free_swa_refuses_partial_pages() {
        let mut pool = pool();
        pool.alloc_swa(&page_locs(0)).unwrap();
        pool.free_swa(&page_locs(0)[..P - 1]);
    }

    #[test]
    fn freeing_the_same_page_twice_is_a_no_op() {
        let mut pool = pool();
        pool.alloc_swa(&page_locs(1)).unwrap();
        assert_eq!(pool.swa_available_size(), 3 * P);
        pool.free_swa(&page_locs(1));
        assert_eq!(pool.swa_available_size(), 4 * P);
        assert_eq!(pool.translate(P as i64), NO_WINDOW_SLOT);
        // A slide, a tombstone and an eviction pass may all name it.
        pool.free_swa(&page_locs(1));
        assert_eq!(
            pool.swa_available_size(),
            4 * P,
            "the second free took nothing"
        );
        pool.free_swa(&page_locs(2));
        assert_eq!(pool.swa_available_size(), 4 * P);
        pool.check_integrity();
    }

    #[test]
    fn a_negative_or_unbound_loc_translates_to_the_sentinel() {
        let pool = pool();
        assert_eq!(pool.translate(-1), NO_WINDOW_SLOT);
        assert_eq!(pool.translate(-99), NO_WINDOW_SLOT);
        assert_eq!(pool.translate(0), NO_WINDOW_SLOT);
        // The trailing sentinel row is addressable and permanently -1.
        assert_eq!(pool.translate((8 * P) as i64), NO_WINDOW_SLOT);
    }

    /// The dummy page is bound so graph-padded rows scatter to a real
    /// slot, and excluded from the free list so it can never be handed to
    /// a request.
    #[test]
    fn the_dummy_page_is_bound_outside_the_free_list() {
        let pool = pool();
        assert_eq!(pool.translate((7 * P) as i64), (4 * P) as i64);
        assert_eq!(pool.swa_available_size(), 4 * P);
        assert_eq!(pool.swa_num_tokens(), 4 * P + 1);
        pool.check_integrity();
    }

    /// The cap is halved because a batched prefill holds a whole chunk's
    /// window live at once, and the concurrent working set comes off
    /// first.
    #[test]
    fn the_prefill_chunk_cap_halves_what_is_left_after_the_reserve() {
        let sizes = dsv4_pool_sizes(128, &args(), 0.5, P, Some(41));
        let pool = Dsv4WindowPool::new(&sizes, 2, true);
        let reserved = dsv4_reserved_window_pages(2, true); // 13
        assert_eq!(pool.prefill_chunk_budget(), (40 - reserved) / 2 * P);
        // A pool that cannot cover its own reserve still admits one page.
        assert_eq!(pool.prefill_chunk_budget(), 13 * P);
        let tiny = Dsv4WindowPool::new(&dsv4_pool_sizes(8, &args(), 0.5, P, Some(5)), 2, true);
        assert_eq!(tiny.prefill_chunk_budget(), P);
    }

    /// `ring_size | P`, so two window pages never share a ring block --
    /// and the location is DERIVED from the live binding. Storing it is
    /// the naive way, and this shows what it costs: once a window page is
    /// recycled, a `state_loc` cached from its previous owner addresses
    /// the new owner's carry state, silently.
    #[test]
    fn state_loc_is_derived_so_two_pages_never_share_a_ring_block() {
        let ring = 8;
        // Page 0 covers ring block 0..8, page 1 covers 8..16.
        for offset in 0..P as i64 {
            assert_eq!(Dsv4WindowPool::state_loc(offset, ring, P), offset % 8);
            assert_eq!(
                Dsv4WindowPool::state_loc(P as i64 + offset, ring, P),
                8 + offset % 8
            );
        }
        assert_eq!(Dsv4WindowPool::state_loc(NO_WINDOW_SLOT, ring, P), -1);

        let mut pool = pool();
        pool.alloc_swa(&page_locs(0)).unwrap();
        pool.alloc_swa(&page_locs(1)).unwrap();
        // What a caller that STORED the location would keep holding.
        let stored = Dsv4WindowPool::state_loc(pool.translate(0), ring, P);

        pool.free_swa(&page_locs(0));
        pool.alloc_swa(&page_locs(2)).unwrap();
        let recycled = Dsv4WindowPool::state_loc(pool.translate((2 * P) as i64), ring, P);
        assert_eq!(
            stored, recycled,
            "the recycled page inherits the ring block, so a stored state_loc reads its carry"
        );
        // Derived from the live binding, page 0 now has no state at all.
        assert_eq!(
            Dsv4WindowPool::state_loc(pool.translate(0), ring, P),
            NO_WINDOW_SLOT
        );
        pool.check_integrity();
    }

    #[test]
    #[should_panic(expected = "must divide")]
    fn a_ring_that_does_not_divide_the_page_is_refused() {
        Dsv4WindowPool::state_loc(0, 7, P);
    }

    /// `p = pos - ((pos - j) % win)` names the position in ring slot `j`.
    /// The modulo is euclidean: a truncated remainder would name a
    /// position AFTER `pos` for `j > pos`, read as though the request had
    /// already written it.
    #[test]
    fn the_ring_names_the_latest_position_congruent_to_each_slot() {
        let win = 4;
        // Mid-sequence: every slot holds one of the last `win` positions.
        let held: Vec<i64> = (0..win)
            .map(|j| window_ring_position(10, j, win).unwrap())
            .collect();
        assert_eq!(held, vec![8, 9, 10, 7]);

        // Early decode: the slots the sequence has not reached are masked.
        assert_eq!(window_ring_position(1, 0, win), Some(0));
        assert_eq!(window_ring_position(1, 1, win), Some(1));
        assert_eq!(window_ring_position(1, 2, win), None);
        assert_eq!(window_ring_position(1, 3, win), None);
        // A truncated `%` would have said 1 - ((1 - 3) % 4) == 3 here.
        assert_ne!(window_ring_position(1, 3, win), Some(3));
    }

    #[test]
    fn the_window_context_reads_the_ring_through_the_live_mapping() {
        let mut pool = pool();
        pool.alloc_swa(&page_locs(0)).unwrap();
        let full_locs: Vec<i64> = (0..P as i64).collect();

        let ctx = pool.window_ctx(3, &full_locs);
        assert_eq!(ctx.window_slot, pool.translate(3));
        assert_eq!(ctx.prev_window_slot, pool.translate(2));
        assert_eq!(ctx.window_slots_topk.len(), P);
        // Only the first four slots have been reached.
        for (j, slot) in ctx.window_slots_topk.iter().enumerate() {
            let expected = if j <= 3 {
                pool.translate(j as i64)
            } else {
                NO_WINDOW_SLOT
            };
            assert_eq!(*slot, expected, "ring slot {j}");
        }

        // Position 0 has no predecessor: the clamp reads itself, never -1.
        let first = pool.window_ctx(0, &full_locs);
        assert_eq!(first.prev_window_slot, pool.translate(0));
    }

    /// A long-running page-granular workload must conserve exactly,
    /// however many times the window slides.
    #[test]
    fn the_window_pool_conserves_every_page() {
        let sizes = dsv4_pool_sizes(64, &args(), 0.5, P, Some(9));
        let mut pool = Dsv4WindowPool::new(&sizes, 1, true);
        let live_pages = 4;
        for full_page in 0..48 {
            pool.alloc_swa(&page_locs(full_page % 60))
                .unwrap_or_else(|err| panic!("page {full_page}: {err}"));
            if full_page >= live_pages {
                pool.free_swa(&page_locs((full_page - live_pages) % 60));
            }
            pool.check_integrity();
        }
        // Exactly the live window is held: 8 allocatable pages, 4 bound.
        assert_eq!(pool.swa_available_size(), (8 - live_pages) * P);
    }

    /// The equality is what catches a leak; a `<=` would tolerate it.
    #[test]
    #[should_panic(expected = "leaked or double-freed")]
    fn the_invariant_catches_a_leaked_window_page() {
        let mut pool = pool();
        pool.alloc_swa(&page_locs(0)).unwrap();
        // A caller that dropped the binding without handing the page back.
        pool.unbind_window_pages(&page_locs(0));
        pool.check_integrity();
    }

    /// And a window page bound to two full pages -- the aliasing the
    /// page-atomic free list exists to prevent.
    #[test]
    #[should_panic(expected = "bound to two full pages")]
    fn the_invariant_catches_a_window_page_bound_twice() {
        let mut pool = pool();
        pool.alloc_swa(&page_locs(0)).unwrap();
        let window_base = pool.translate(0) as usize;
        pool.bind_window_pages(P, window_base);
        pool.check_integrity();
    }
}
