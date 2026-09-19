//! MiniMax-M3's block-sparse attention selection: which 128-token KV
//! blocks a query may look at.
//!
//! Ported from FreeToken's `models/minimax_m3/args.py` selection rule.
//! This is the *decision* half -- which blocks are visible. The
//! attention itself is then an ordinary masked pass over the positions
//! those blocks cover, which
//! [`crate::attention::causal_mla_attention_sparse`] already does.
//!
//! # Four rules, and one of them is what stops a NaN
//!
//! - **A block's score is the MAX over its positions**, not the mean
//!   and not the sum. A block earns its place on its single best
//!   match: one strongly-related token in a block of 128 is exactly the
//!   case sparse attention exists to catch, and a mean would average it
//!   away against 127 unrelated neighbours.
//!
//! - **No softmax scale.** The raw dot product is used directly,
//!   because only the ORDER of the scores is consumed. Dividing by
//!   `sqrt(d)` would scale every score by the same positive constant
//!   and change nothing, so the reference does not, and neither does
//!   this.
//!
//! - **Selection is per KV head, with no cross-head reduction.** One
//!   index head scores for one KV head, and that KV head's whole GQA
//!   group reads the blocks it picked. Reducing across heads first --
//!   by summing or maxing the scores -- would give every group the same
//!   block set, which is the opposite of what per-head selection is
//!   for.
//!
//! - **The newest `local_blocks` and the first `init_blocks` are
//!   force-included**, before any scoring. This is not a quality
//!   heuristic bolted on top: it is what guarantees the selection is
//!   never empty. A query early in a sequence, or one whose scores are
//!   all equally poor, would otherwise select zero blocks, and
//!   attention over zero positions is a softmax over an empty set --
//!   which is a NaN that propagates through the whole forward pass and
//!   surfaces as garbage output, not as an error.
//!
//! # The block size is an ABI, not a tuning knob
//!
//! [`MINIMAX_BLOCK_SIZE`] is 128 and also pins the KV page size: the
//! selection hands back block indices, and a pager whose page is a
//! different size cannot honour them without splitting or merging
//! pages, which is exactly the bookkeeping block-sparse attention
//! exists to avoid.

/// MiniMax-M3's KV block, in tokens. Also the KV page size -- see the
/// module docs.
pub const MINIMAX_BLOCK_SIZE: usize = 128;

/// How many blocks a query may see, and which are free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSparseConfig {
    /// Tokens per block. [`MINIMAX_BLOCK_SIZE`] for a real checkpoint;
    /// configurable only so the rules can be tested at a readable size.
    pub block_size: usize,
    /// Total blocks visible per query, force-included ones counted.
    pub top_blocks: usize,
    /// Blocks at the start of the sequence that are always visible.
    pub init_blocks: usize,
    /// Blocks nearest the query that are always visible.
    pub local_blocks: usize,
}

impl Default for BlockSparseConfig {
    fn default() -> Self {
        BlockSparseConfig {
            block_size: MINIMAX_BLOCK_SIZE,
            top_blocks: 32,
            init_blocks: 1,
            local_blocks: 2,
        }
    }
}

impl BlockSparseConfig {
    /// Blocks covering positions `0..=query_pos`, the last one possibly
    /// partial.
    pub fn causal_blocks(&self, query_pos: usize) -> usize {
        if self.block_size == 0 {
            return 0;
        }
        query_pos / self.block_size + 1
    }
}

/// The blocks each KV head may read for one query position, ascending.
///
/// `index_q` is one index head per KV head (`[n_kv_heads]
/// [index_head_dim]`); `index_k` is one index key per position
/// (`[n_positions][index_head_dim]`), shared across heads. Returns one
/// selection per KV head, in the same order.
///
/// Never returns an empty selection for a query at a valid position --
/// see the module docs on why that matters.
pub fn block_sparse_select(
    index_q: &[Vec<f32>],
    index_k: &[Vec<f32>],
    query_pos: usize,
    cfg: &BlockSparseConfig,
) -> Vec<Vec<usize>> {
    let n_blocks = cfg.causal_blocks(query_pos).min(
        // A query cannot see past the keys that exist, even if its
        // position claims otherwise.
        if cfg.block_size == 0 {
            0
        } else {
            index_k.len().div_ceil(cfg.block_size)
        },
    );
    if n_blocks == 0 {
        return vec![Vec::new(); index_q.len()];
    }

    index_q
        .iter()
        .map(|q| select_for_head(q, index_k, query_pos, n_blocks, cfg))
        .collect()
}

fn select_for_head(
    q: &[f32],
    index_k: &[Vec<f32>],
    query_pos: usize,
    n_blocks: usize,
    cfg: &BlockSparseConfig,
) -> Vec<usize> {
    let mut forced = vec![false; n_blocks];
    for slot in forced.iter_mut().take(cfg.init_blocks.min(n_blocks)) {
        *slot = true;
    }
    for slot in forced
        .iter_mut()
        .skip(n_blocks.saturating_sub(cfg.local_blocks))
    {
        *slot = true;
    }

    let budget = cfg.top_blocks.max(forced.iter().filter(|f| **f).count());
    let mut chosen: Vec<usize> = (0..n_blocks).filter(|&b| forced[b]).collect();
    if chosen.len() >= budget || chosen.len() == n_blocks {
        chosen.sort_unstable();
        return chosen;
    }

    // Score the rest. MAX over the block's causally-visible positions,
    // raw dot product, no scale.
    let mut scored: Vec<(usize, f32)> = (0..n_blocks)
        .filter(|&b| !forced[b])
        .map(|b| {
            let start = b * cfg.block_size;
            let end = ((b + 1) * cfg.block_size)
                .min(index_k.len())
                .min(query_pos + 1);
            let best = (start..end)
                .map(|p| dot(q, &index_k[p]))
                .fold(f32::NEG_INFINITY, f32::max);
            (b, best)
        })
        .collect();

    // Ties break toward the LOWER block index, deterministically: a
    // selection that varied run to run would make a cached prefix
    // disagree with the run that produced it.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    for (b, _) in scored.into_iter().take(budget - chosen.len()) {
        chosen.push(b);
    }
    chosen.sort_unstable();
    chosen
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// The positions a block selection covers, ascending and
/// causally clipped -- the form
/// [`crate::attention::causal_mla_attention_sparse`] takes.
pub fn positions_of_blocks(
    blocks: &[usize],
    query_pos: usize,
    n_positions: usize,
    cfg: &BlockSparseConfig,
) -> Vec<usize> {
    let mut out = Vec::new();
    for &b in blocks {
        let start = b * cfg.block_size;
        let end = ((b + 1) * cfg.block_size)
            .min(n_positions)
            .min(query_pos + 1);
        out.extend(start..end);
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(n: usize, hot: &[usize]) -> Vec<Vec<f32>> {
        (0..n)
            .map(|p| {
                if hot.contains(&p) {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 0.01]
                }
            })
            .collect()
    }

    fn cfg(block: usize, top: usize, init: usize, local: usize) -> BlockSparseConfig {
        BlockSparseConfig {
            block_size: block,
            top_blocks: top,
            init_blocks: init,
            local_blocks: local,
        }
    }

    /// The rule that stops a NaN. A query whose scores are all equally
    /// poor -- and one at the very start of a sequence -- still gets at
    /// least one block, because the forced ones are added before any
    /// scoring. Attention over zero positions is a softmax over an
    /// empty set, which propagates NaN through the whole forward pass
    /// and surfaces as garbage rather than as an error.
    #[test]
    fn a_selection_is_never_empty_however_poor_the_scores() {
        let c = cfg(4, 1, 1, 1);
        // Every key identical and orthogonal to the query: no block can
        // win on score.
        let k: Vec<Vec<f32>> = (0..64).map(|_| vec![0.0, 0.0]).collect();
        let q = vec![vec![1.0, 0.0]];
        for pos in [0usize, 1, 3, 4, 17, 63] {
            let sel = block_sparse_select(&q, &k, pos, &c);
            assert!(
                !sel[0].is_empty(),
                "position {pos} selected nothing, which is a NaN downstream"
            );
            let covered = positions_of_blocks(&sel[0], pos, k.len(), &c);
            assert!(!covered.is_empty(), "position {pos} covers no positions");
            assert!(
                covered.iter().all(|&p| p <= pos),
                "position {pos} saw the future"
            );
        }
    }

    /// A block earns its place on its single best position, not its
    /// average. One strongly-related token among 127 unrelated ones is
    /// exactly what block-sparse attention exists to find, and a mean
    /// would average it away.
    #[test]
    fn a_block_is_scored_by_its_best_position_not_its_average() {
        // Four blocks of four. Block 1 holds a single hot key; block 2
        // holds none. With one free slot, block 1 must win.
        let c = cfg(4, 3, 1, 1);
        let k = keys(16, &[5]);
        let q = vec![vec![1.0, 0.0]];
        let sel = block_sparse_select(&q, &k, 15, &c);
        assert!(
            sel[0].contains(&1),
            "the block holding the one hot key must be chosen, got {:?}",
            sel[0]
        );
    }

    /// Selection is per KV head with no cross-head reduction: two heads
    /// looking for different things get different blocks. Reducing
    /// across heads first would hand every GQA group one shared block
    /// set, which is the opposite of what per-head selection is for.
    #[test]
    fn each_kv_head_selects_for_itself() {
        let c = cfg(4, 3, 1, 1);
        let mut k: Vec<Vec<f32>> = (0..16).map(|_| vec![0.0, 0.0]).collect();
        k[5] = vec![1.0, 0.0]; // block 1, only head 0 wants it
        k[9] = vec![0.0, 1.0]; // block 2, only head 1 wants it
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let sel = block_sparse_select(&q, &k, 15, &c);
        assert_eq!(sel.len(), 2, "one selection per KV head");
        assert!(sel[0].contains(&1), "head 0: {:?}", sel[0]);
        assert!(sel[1].contains(&2), "head 1: {:?}", sel[1]);
        assert_ne!(sel[0], sel[1], "the heads must not be reduced together");
    }

    /// The first `init_blocks` and the newest `local_blocks` are in
    /// before anything is scored, even when they score worst.
    #[test]
    fn the_first_and_newest_blocks_are_included_before_scoring() {
        let c = cfg(4, 3, 1, 1);
        // The hot keys are all in the middle blocks, so on score alone
        // blocks 0 and 3 would both lose.
        let k = keys(16, &[4, 5, 8, 9]);
        let q = vec![vec![1.0, 0.0]];
        let sel = block_sparse_select(&q, &k, 15, &c);
        assert!(sel[0].contains(&0), "init block missing: {:?}", sel[0]);
        assert!(sel[0].contains(&3), "local block missing: {:?}", sel[0]);
    }

    /// Only the ORDER of the scores is consumed, so no softmax scale is
    /// applied. Scaling every index key by a positive constant must
    /// leave the selection identical -- if a scale were being applied
    /// and compared against anything absolute, it would not.
    #[test]
    fn the_selection_depends_only_on_the_order_of_the_scores() {
        let c = cfg(4, 3, 1, 1);
        let k = keys(16, &[5]);
        let scaled: Vec<Vec<f32>> = k
            .iter()
            .map(|row| row.iter().map(|v| v * 1000.0).collect())
            .collect();
        let q = vec![vec![1.0, 0.0]];
        assert_eq!(
            block_sparse_select(&q, &k, 15, &c),
            block_sparse_select(&q, &scaled, 15, &c)
        );
    }

    /// A query never sees a block that starts after it, and the block
    /// containing it is clipped to its own position.
    #[test]
    fn selection_is_causal_and_the_current_block_is_clipped() {
        let c = cfg(4, 8, 1, 1);
        let k = keys(16, &[]);
        let q = vec![vec![1.0, 0.0]];
        let sel = block_sparse_select(&q, &k, 6, &c);
        assert!(
            sel[0].iter().all(|&b| b <= 1),
            "block 2 starts at position 8, after the query at 6: {:?}",
            sel[0]
        );
        let covered = positions_of_blocks(&sel[0], 6, k.len(), &c);
        assert_eq!(covered, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    /// The budget is a ceiling, and the forced blocks are inside it --
    /// not added on top of it, which would silently widen every query's
    /// working set beyond what the KV pool was sized for.
    #[test]
    fn the_budget_bounds_the_selection_including_the_forced_blocks() {
        let c = cfg(4, 3, 1, 1);
        let k = keys(64, &[]);
        let q = vec![vec![1.0, 0.0]];
        let sel = block_sparse_select(&q, &k, 63, &c);
        assert_eq!(sel[0].len(), 3, "budget of 3: {:?}", sel[0]);

        // A budget smaller than the forced set cannot drop a forced
        // block -- doing so would reintroduce the empty selection --
        // so it widens to hold exactly them.
        let tight = cfg(4, 1, 2, 2);
        let sel = block_sparse_select(&q, &k, 63, &tight);
        assert_eq!(sel[0], vec![0, 1, 14, 15]);
    }

    /// Ties break toward the lower block index rather than by whatever
    /// order the sort happened to see, so a cached prefix and the run
    /// that produced it cannot disagree.
    #[test]
    fn ties_break_deterministically_toward_the_lower_block() {
        let c = cfg(4, 3, 0, 0);
        // Every key identical: every block ties.
        let k: Vec<Vec<f32>> = (0..16).map(|_| vec![1.0, 0.0]).collect();
        let q = vec![vec![1.0, 0.0]];
        let first = block_sparse_select(&q, &k, 15, &c);
        assert_eq!(first[0], vec![0, 1, 2]);
        for _ in 0..8 {
            assert_eq!(block_sparse_select(&q, &k, 15, &c), first);
        }
    }

    /// A query past the end of the keys is clipped to what exists,
    /// rather than naming blocks the pager never allocated.
    #[test]
    fn a_query_beyond_the_keys_is_clipped_to_what_exists() {
        let c = cfg(4, 8, 1, 1);
        let k = keys(6, &[]);
        let q = vec![vec![1.0, 0.0]];
        let sel = block_sparse_select(&q, &k, 100, &c);
        assert_eq!(sel[0], vec![0, 1], "only two blocks of keys exist");
        let covered = positions_of_blocks(&sel[0], 100, k.len(), &c);
        assert_eq!(covered, vec![0, 1, 2, 3, 4, 5]);
    }

    /// The real geometry: 128-token blocks, and the block size is the
    /// KV page size too.
    #[test]
    fn the_real_block_size_is_the_kv_page_size() {
        assert_eq!(MINIMAX_BLOCK_SIZE, 128);
        let c = BlockSparseConfig::default();
        assert_eq!(c.block_size, MINIMAX_BLOCK_SIZE);
        assert_eq!(c.causal_blocks(0), 1);
        assert_eq!(c.causal_blocks(127), 1);
        assert_eq!(c.causal_blocks(128), 2);
    }
}
