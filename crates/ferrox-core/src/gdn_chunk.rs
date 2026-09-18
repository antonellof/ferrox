//! The gated delta rule over a CHUNK of tokens, which is the same
//! recurrence as [`crate::gdn::delta_step`] with the state read once
//! per chunk instead of once per row.
//!
//! # Why
//!
//! `crate::gdn::tests::delta_step_throughput_probe` measures the
//! sequential step at 36 GB/s of state on an M2 Pro, which is the floor
//! for touching a 3.1 MB state once per row: a 128-token Bonsai prefill
//! moves `128 rows x 48 layers x 6.2 MB = 38 GB` that way, 1.05 s of a
//! 3.9 s `pp128` run, and the profile agrees at 23% of samples. Nothing
//! about the loop is slow; the traffic is the cost, and the only way to
//! cut it is to stop paying it per row.
//!
//! # The algebra
//!
//! Per value head, with `S_t` the state after row `t`, `a_t` the row's
//! decay, `k_t` / `q_t` the key and query, `v_t` the value and `b_t`
//! the gate:
//!
//! ```text
//! d_t   = b_t (v_t - a_t (S_{t-1} k_t))          (a vector over value index j)
//! S_t   = a_t S_{t-1} + d_t k_t^T                 (rank one)
//! out_t = (S_t q_t) / sqrt(S)
//! ```
//!
//! Unrolling the rank-one updates across a chunk gives every quantity
//! in terms of `S_0` and the `d_u` already computed:
//!
//! ```text
//! a_t (S_{t-1} k_t) = A_t (S_0 k_t) + sum_{u<t} R[t][u] (k_u . k_t) d_u
//! out_t             = A_t (S_0 q_t) + sum_{u<=t} R[t][u] (k_u . q_t) d_u
//! S_C               = A_C S_0 + sum_t R[C][t] d_t k_t^T
//! ```
//!
//! where `A_t = a_1 ... a_t` and `R[t][u] = a_{u+1} ... a_t`. Every
//! ratio is a product of decays with no division in it, so a long chunk
//! of small decays underflows to zero (the state's contribution really
//! has vanished) instead of dividing by it, which is what the textbook
//! `d_u / A_u` form does and why that form needs a normalisation step.
//!
//! So a chunk costs three `S x S x C` products (the state against `K`
//! and `Q`, and the state's update) plus a `C x C` forward
//! substitution, which is 1.5x the multiply-adds of the sequential form
//! and `1/C` of its state traffic. The step is bandwidth-bound, so that
//! is the trade worth making.

use crate::gdn::DeltaDims;

/// Rows processed against one read of the state.
///
/// Measured on Bonsai's shape (48 value heads of 128, 128 rows) against
/// the sequential step: C=16 is 1.96x, C=32 is 2.09x, C=64 is 1.66x.
/// The turn is the `C x C` forward substitution, whose cost per row
/// grows with C while the state traffic it saves is already down by
/// 32x, so 32 is the measured optimum here rather than llama.cpp's 64.
pub const CHUNK: usize = 32;

/// `rows` consecutive tokens of one sequence through the delta rule,
/// advancing `state` in place, writing `[rows][n_v_heads * head_dim]`
/// into `out`.
///
/// Identical in exact arithmetic to calling [`crate::gdn::delta_step`]
/// once per row, up to float association; the test module pins the two
/// against each other across shapes, chunk boundaries and decays.
#[allow(clippy::too_many_arguments)]
pub fn delta_chunk(
    dims: DeltaDims,
    rows: usize,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    out: &mut [f32],
) {
    let DeltaDims {
        n_k_heads,
        n_v_heads,
        head_dim: s,
        map,
    } = dims;
    assert_eq!(state.len(), dims.state_len());
    assert_eq!(q.len(), rows * n_k_heads * s);
    assert_eq!(k.len(), rows * n_k_heads * s);
    assert_eq!(v.len(), rows * n_v_heads * s);
    assert_eq!(g.len(), rows * n_v_heads);
    assert_eq!(beta.len(), rows * n_v_heads);
    assert_eq!(out.len(), rows * n_v_heads * s);
    let scale = 1.0 / (s as f32).sqrt();
    let key_row = n_k_heads * s;
    let val_row = n_v_heads * s;

    /// The `[row][head][head_dim]` output, shared across the per-head
    /// tasks below.
    ///
    /// A head's slots are disjoint from every other head's, but they
    /// interleave with them once per row, so they cannot be handed out
    /// as `&mut` chunks the way the state can. Each task writes only
    /// the slots of the head it owns, which is what makes this
    /// race-free; the wrapper carries the pointer across the
    /// `Send`/`Sync` boundary and nothing else.
    #[derive(Clone, Copy)]
    struct HeadOut(*mut f32);
    // SAFETY: see above; the pointer is only ever written at indices
    // belonging to the writing task's head.
    unsafe impl Send for HeadOut {}
    unsafe impl Sync for HeadOut {}

    let out_ptr = HeadOut(out.as_mut_ptr());
    // Heads are independent, as in the sequential step, so the parallel
    // region is over them and each owns its `S x S` state.
    crate::par::chunks_mut(state, s * s, 1, |h, st| {
        // The whole wrapper, not its field: a 2021 closure captures
        // disjoint fields, and the bare `*mut f32` is exactly what the
        // `Send`/`Sync` impls above do not cover.
        let out_ptr = { out_ptr };
        let kh = map.k_head(h, n_k_heads, n_v_heads);
        // Scratch that lives across the chunks of one head.
        let mut m = vec![0.0f32; s * CHUNK]; // S_0 K^T, [j][t]
        let mut n = vec![0.0f32; s * CHUNK]; // S_0 Q^T, [j][t]
        let mut d = vec![0.0f32; CHUNK * s]; // the per-row updates, [t][j]
        let mut gram = vec![0.0f32; CHUNK * CHUNK]; // k_u . k_t
        let mut qk = vec![0.0f32; CHUNK * CHUNK]; // k_u . q_t
        let mut ratio = vec![0.0f32; CHUNK * CHUNK]; // R[t][u]
        let mut a = [0.0f32; CHUNK]; // A_t

        let mut start = 0;
        while start < rows {
            let c = CHUNK.min(rows - start);
            let krow = |t: usize| &k[(start + t) * key_row + kh * s..][..s];
            let qrow = |t: usize| &q[(start + t) * key_row + kh * s..][..s];

            // The decay products. `R[t][u]` for `u <= t` is a product of
            // decays and never a quotient, so it cannot overflow and an
            // underflow to zero is the state's contribution really
            // having vanished.
            for t in 0..c {
                let decay = g[(start + t) * n_v_heads + h].exp();
                ratio[t * CHUNK + t] = 1.0;
                for u in (0..t).rev() {
                    ratio[t * CHUNK + u] =
                        ratio[t * CHUNK + u + 1] * g[(start + u + 1) * n_v_heads + h].exp();
                }
                a[t] = if t == 0 { decay } else { a[t - 1] * decay };
            }

            // The state against this chunk's keys and queries: the two
            // `S x S x C` products that replace one pass over the state
            // per row.
            //
            // Four `t` at a time against one read of the row, because
            // this is the loop the step is now bound BY: chunking
            // traded 1.5x the multiply-adds for a 32nd of the traffic,
            // so the arithmetic is what is left, and a row dotted
            // against one vector at a time leaves the pipeline waiting
            // on the load rather than on the multiply.
            for j in 0..s {
                let row = &st[j * s..(j + 1) * s];
                let mut t = 0;
                while t + 4 <= c {
                    let (m0, m1, m2, m3) =
                        dot4(row, krow(t), krow(t + 1), krow(t + 2), krow(t + 3));
                    m[j * CHUNK + t] = m0;
                    m[j * CHUNK + t + 1] = m1;
                    m[j * CHUNK + t + 2] = m2;
                    m[j * CHUNK + t + 3] = m3;
                    let (n0, n1, n2, n3) =
                        dot4(row, qrow(t), qrow(t + 1), qrow(t + 2), qrow(t + 3));
                    n[j * CHUNK + t] = n0;
                    n[j * CHUNK + t + 1] = n1;
                    n[j * CHUNK + t + 2] = n2;
                    n[j * CHUNK + t + 3] = n3;
                    t += 4;
                }
                while t < c {
                    m[j * CHUNK + t] = dot(row, krow(t));
                    n[j * CHUNK + t] = dot(row, qrow(t));
                    t += 1;
                }
            }
            // The two triangles over the chunk's own vectors.
            for t in 0..c {
                for u in 0..=t {
                    gram[t * CHUNK + u] = dot(krow(t), krow(u));
                    qk[t * CHUNK + u] = dot(qrow(t), krow(u));
                }
            }

            // Forward substitution: row `t` needs every `d_u` before it,
            // which is the sequential dependency, but it is now a
            // `C x C` triangle against `C x S` rather than a pass over
            // the state.
            for t in 0..c {
                let b = beta[(start + t) * n_v_heads + h];
                let (d_head, d_tail) = d.split_at_mut(t * s);
                let d_t = &mut d_tail[..s];
                // SAFETY: `HeadOut` wraps the caller's `out`, which is
                // `rows * val_row` long and borrowed mutably for this
                // call; this task writes only head `h`'s `s` slots of
                // row `start + t`, which no other task touches.
                let o_t = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.0.add((start + t) * val_row + h * s), s)
                };
                let v_t = &v[(start + t) * val_row + h * s..][..s];
                for j in 0..s {
                    let mut pred = a[t] * m[j * CHUNK + t];
                    let mut read = a[t] * n[j * CHUNK + t];
                    for u in 0..t {
                        let w = ratio[t * CHUNK + u];
                        let du = d_head[u * s + j];
                        pred += w * gram[t * CHUNK + u] * du;
                        read += w * qk[t * CHUNK + u] * du;
                    }
                    let dj = b * (v_t[j] - pred);
                    d_t[j] = dj;
                    // `u == t`: `R[t][t]` is 1 and `d_t` is this value.
                    o_t[j] = (read + qk[t * CHUNK + t] * dj) * scale;
                }
            }

            // The state, updated once for the whole chunk.
            let last = c - 1;
            for j in 0..s {
                let row = &mut st[j * s..(j + 1) * s];
                for x in row.iter_mut() {
                    *x *= a[last];
                }
                for t in 0..c {
                    let w = ratio[last * CHUNK + t] * d[t * s + j];
                    if w == 0.0 {
                        continue;
                    }
                    let kt = krow(t);
                    for (x, kx) in row.iter_mut().zip(kt) {
                        *x += w * kx;
                    }
                }
            }
            start += c;
        }
    });
}

/// Four dots against one vector, from one pass over it: the four
/// accumulator sets keep the multiply pipeline busy where a single dot
/// waits on `row`'s loads.
#[inline]
fn dot4(row: &[f32], a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> (f32, f32, f32, f32) {
    let mut acc = [[0.0f32; 4]; 4];
    let (rb, rt) = row.as_chunks::<4>();
    let (ab, at) = a.as_chunks::<4>();
    let (bb, bt) = b.as_chunks::<4>();
    let (cb, ct) = c.as_chunks::<4>();
    let (db, dt) = d.as_chunks::<4>();
    for ((((r, x), y), z), w) in rb.iter().zip(ab).zip(bb).zip(cb).zip(db) {
        for l in 0..4 {
            acc[0][l] += r[l] * x[l];
            acc[1][l] += r[l] * y[l];
            acc[2][l] += r[l] * z[l];
            acc[3][l] += r[l] * w[l];
        }
    }
    let mut tail = [0.0f32; 4];
    for ((((r, x), y), z), w) in rt.iter().zip(at).zip(bt).zip(ct).zip(dt) {
        tail[0] += *r * *x;
        tail[1] += *r * *y;
        tail[2] += *r * *z;
        tail[3] += *r * *w;
    }
    let sum = |v: [f32; 4], t: f32| v[0] + v[1] + v[2] + v[3] + t;
    (
        sum(acc[0], tail[0]),
        sum(acc[1], tail[1]),
        sum(acc[2], tail[2]),
        sum(acc[3], tail[3]),
    )
}

/// A dot with four accumulators, as the sequential step's reductions
/// use: a single-accumulator float sum cannot be reordered without
/// fast-math, so the compiler will not vectorise it.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 4];
    let (ab, at) = a.as_chunks::<4>();
    let (bb, bt) = b.as_chunks::<4>();
    for (x, y) in ab.iter().zip(bb) {
        for l in 0..4 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in at.iter().zip(bt) {
        tail += *x * *y;
    }
    acc[0] + acc[1] + acc[2] + acc[3] + tail
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn::{delta_step, HeadMap};

    /// The chunked rule IS the sequential one. Shapes chosen so the
    /// chunk boundary falls in different places: shorter than a chunk,
    /// exactly one, one and a bit, several.
    #[test]
    fn a_chunk_is_the_sequential_recurrence() {
        for (n_k, n_v, s, rows, map) in [
            (2usize, 4usize, 8usize, 3usize, HeadMap::Tiled),
            (2, 4, 8, CHUNK, HeadMap::Tiled),
            (2, 4, 8, CHUNK + 5, HeadMap::Grouped),
            (4, 8, 16, 2 * CHUNK + 1, HeadMap::Tiled),
            (1, 3, 32, 7, HeadMap::Grouped),
        ] {
            let dims = DeltaDims {
                n_k_heads: n_k,
                n_v_heads: n_v,
                head_dim: s,
                map,
            };
            let mut seed = 7u32 + rows as u32;
            let mut draw = |n: usize, scale: f32| -> Vec<f32> {
                (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((seed >> 8) as f32 / 8388608.0 - 1.0) * scale
                    })
                    .collect()
            };
            let state0 = draw(dims.state_len(), 0.4);
            let q = draw(rows * n_k * s, 1.0);
            let k = draw(rows * n_k * s, 1.0);
            let v = draw(rows * n_v * s, 1.0);
            // A real layer's decay is `exp(g)` with `g < 0`; the spread
            // here covers "barely decays" and "forgets almost at once".
            let g: Vec<f32> = draw(rows * n_v, 3.0).iter().map(|x| -x.abs()).collect();
            let beta: Vec<f32> = draw(rows * n_v, 1.0)
                .iter()
                .map(|x| 0.5 + 0.3 * x)
                .collect();

            let mut seq_state = state0.clone();
            let mut seq_out = vec![0.0f32; rows * n_v * s];
            for r in 0..rows {
                let mut o = vec![0.0f32; n_v * s];
                delta_step(
                    dims,
                    &mut seq_state,
                    &q[r * n_k * s..(r + 1) * n_k * s],
                    &k[r * n_k * s..(r + 1) * n_k * s],
                    &v[r * n_v * s..(r + 1) * n_v * s],
                    &g[r * n_v..(r + 1) * n_v],
                    &beta[r * n_v..(r + 1) * n_v],
                    &mut o,
                );
                seq_out[r * n_v * s..(r + 1) * n_v * s].copy_from_slice(&o);
            }

            let mut chunk_state = state0.clone();
            let mut chunk_out = vec![0.0f32; rows * n_v * s];
            delta_chunk(
                dims,
                rows,
                &mut chunk_state,
                &q,
                &k,
                &v,
                &g,
                &beta,
                &mut chunk_out,
            );

            let tol = 2e-4;
            for (i, (a, b)) in chunk_out.iter().zip(seq_out.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol * b.abs().max(1.0),
                    "{n_v}x{s} rows={rows} out[{i}]: chunk={a} seq={b}"
                );
            }
            for (i, (a, b)) in chunk_state.iter().zip(seq_state.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol * b.abs().max(1.0),
                    "{n_v}x{s} rows={rows} state[{i}]: chunk={a} seq={b}"
                );
            }
        }
    }

    /// Chunked against sequential on Bonsai's shape, printed rather
    /// than asserted: run with `--nocapture`. The sequential step is
    /// bandwidth-bound at 36 GB/s of state, so the question this
    /// answers is whether trading 1.5x the multiply-adds for `1/CHUNK`
    /// of the traffic actually pays on this host.
    #[test]
    #[ignore = "a measurement, not an assertion; run with --nocapture"]
    fn chunked_against_sequential_throughput() {
        let dims = DeltaDims {
            n_k_heads: 4,
            n_v_heads: 48,
            head_dim: 128,
            map: HeadMap::Tiled,
        };
        let rows = 128;
        let s = 128;
        let state0 = vec![0.01f32; dims.state_len()];
        let q: Vec<f32> = (0..rows * 4 * s).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..rows * 4 * s).map(|i| (i as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..rows * 48 * s)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();
        let g = vec![-0.05f32; rows * 48];
        let beta = vec![0.5f32; rows * 48];

        let mut st = state0.clone();
        let mut out = vec![0.0f32; rows * 48 * s];
        let t = std::time::Instant::now();
        for r in 0..rows {
            let mut o = vec![0.0f32; 48 * s];
            delta_step(
                dims,
                &mut st,
                &q[r * 4 * s..(r + 1) * 4 * s],
                &k[r * 4 * s..(r + 1) * 4 * s],
                &v[r * 48 * s..(r + 1) * 48 * s],
                &g[r * 48..(r + 1) * 48],
                &beta[r * 48..(r + 1) * 48],
                &mut o,
            );
            out[r * 48 * s..(r + 1) * 48 * s].copy_from_slice(&o);
        }
        let seq = t.elapsed().as_secs_f64();

        let mut st2 = state0.clone();
        let mut out2 = vec![0.0f32; rows * 48 * s];
        let t = std::time::Instant::now();
        delta_chunk(dims, rows, &mut st2, &q, &k, &v, &g, &beta, &mut out2);
        let chunked = t.elapsed().as_secs_f64();

        eprintln!(
            "{rows} rows: sequential {:.1} ms, chunked(C={CHUNK}) {:.1} ms, {:.2}x",
            seq * 1e3,
            chunked * 1e3,
            seq / chunked
        );
    }

    /// A decay small enough that `A_t` underflows inside a chunk is the
    /// case the textbook `d_u / A_u` form divides by: here it is a
    /// product that reaches zero, and the answer stays the sequential
    /// one (the state really has been forgotten).
    #[test]
    fn a_vanishing_decay_is_forgetting_and_not_a_division() {
        let dims = DeltaDims {
            n_k_heads: 1,
            n_v_heads: 2,
            head_dim: 8,
            map: HeadMap::Tiled,
        };
        let rows = CHUNK + 3;
        let state0: Vec<f32> = (0..dims.state_len())
            .map(|i| 0.1 + i as f32 * 0.01)
            .collect();
        let q = vec![0.3f32; rows * 8];
        let k = vec![0.2f32; rows * 8];
        let v = vec![0.7f32; rows * 2 * 8];
        // exp(-90) underflows a product of a few of these to zero.
        let g = vec![-90.0f32; rows * 2];
        let beta = vec![0.5f32; rows * 2];

        let mut seq_state = state0.clone();
        let mut seq_out = vec![0.0f32; rows * 2 * 8];
        for r in 0..rows {
            let mut o = vec![0.0f32; 2 * 8];
            delta_step(
                dims,
                &mut seq_state,
                &q[r * 8..(r + 1) * 8],
                &k[r * 8..(r + 1) * 8],
                &v[r * 16..(r + 1) * 16],
                &g[r * 2..(r + 1) * 2],
                &beta[r * 2..(r + 1) * 2],
                &mut o,
            );
            seq_out[r * 16..(r + 1) * 16].copy_from_slice(&o);
        }
        let mut chunk_state = state0.clone();
        let mut chunk_out = vec![0.0f32; rows * 2 * 8];
        delta_chunk(
            dims,
            rows,
            &mut chunk_state,
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut chunk_out,
        );
        for (a, b) in chunk_out.iter().zip(seq_out.iter()) {
            assert!((a - b).abs() <= 1e-5, "chunk={a} seq={b}");
            assert!(a.is_finite(), "no division by a vanished decay");
        }
        for (a, b) in chunk_state.iter().zip(seq_state.iter()) {
            assert!((a - b).abs() <= 1e-5 && a.is_finite());
        }
    }
}
