//! The reranker classification head: `cls`, `cls.output`, `cls.norm`
//! and `{arch}.classifier.output_labels`.
//!
//! Transcribed from llama.cpp `llm_graph_context::build_pooling`'s
//! `LLAMA_POOLING_TYPE_RANK` arm (`src/llama-graph.cpp`), which is a
//! **classification head and not a pooling rule** — the reason
//! [`crate::pooling::PoolingType::Rank`] still refuses in
//! [`crate::pooling::pool`] and always will. `pool` sees hidden states
//! and a width; it cannot see these matrices, so RANK is not a
//! question it can answer. The rank path is CLS pooling *followed by*
//! this head, and this module is the "followed by".
//!
//! # The graph, for the `bert` shape
//!
//! ```text
//! cur = hidden[CLS]                      (row 0 — see below)
//! if cls:      cur = cls · cur + cls_b
//!              cur = tanh(cur)
//!              if cls_norm: cur = LayerNorm(cur, cls_norm, no bias)
//! if cls_out:  cur = cls_out · cur + cls_out_b
//! score = cur[0]
//! ```
//!
//! Three of upstream's branches are deliberately **not** here, and each
//! is refused rather than approximated, because each belongs to an
//! architecture [`crate::bert_gguf_loader`] already refuses by name:
//!
//! * `modern-bert` pools with MEAN rather than CLS and uses GELU in
//!   place of the `tanh`.
//! * `qwen3` / `qwen3vl` take the **last** token rather than row 0
//!   (`build_inp_cls`'s `last` flag, `llama-graph.cpp:296-299`) and
//!   append a softmax over the outputs.
//! * `jina-reranker-v1-tiny-en` is the checkpoint upstream cites for
//!   the `cls_out`-absent case; it is `jina-bert-v2`, which this crate
//!   does not load.
//!
//! Since only `bert` reaches here, row 0 is the CLS row unconditionally
//! and there is no softmax. If another architecture is ever admitted,
//! this module has to grow its branch — it must not inherit `bert`'s.
//!
//! # The pooler, and the two score scales (issue #82)
//!
//! `cls` IS HuggingFace's `bert.pooler.dense`: llama.cpp's tensor
//! mapping renames it, and the `tanh` above is
//! `BertPooler.activation`. A `BertForSequenceClassification` reranker
//! was trained as `classifier(tanh(pooler(cls_hidden)))`, so the
//! `dense` branch is not an optional flourish — it is most of the
//! head's calibration.
//!
//! Every reranker GGUF in circulation is missing it, because
//! llama.cpp's converter deletes it by name (`conversion/bert.py`,
//! `BertModel.filter_tensors`: "we are only using BERT for embeddings
//! so we don't need the pooling layer"). This module does the one
//! thing an engine can do about that: it **runs the pooler when the
//! file carries it and refuses to invent one when it does not**. There
//! is no identity stand-in and no zero-filled `cls` — a made-up pooler
//! is a made-up score, and the direct-projection shape is legitimate
//! for `jina-reranker-v1-tiny-en`, so the absence cannot be refused
//! either.
//!
//! What it must never do is leave the difference invisible. The two
//! regimes differ by roughly a factor of 50 in score magnitude
//! (about ±11 vs about ±0.2 on `ms-marco-MiniLM-L6-v2`), which changes
//! nothing for a caller that sorts and everything for a caller that
//! thresholds. [`RankHead::has_pooler`] and [`RankHead::graph`] are the
//! machine-readable answer, `/v1/rerank` reports the second one on
//! every response, and [`missing_pooler_note`] says it once at load for
//! whoever is reading the server's log rather than its JSON.
//!
//! # Which float is the score
//!
//! `send_rerank` (`tools/server/server-context.cpp`) reads `embd[0]`:
//! the FIRST of `n_cls_out` outputs, whatever the rest are. That is
//! fine for a one-output relevance head and is a silent choice for a
//! many-output classifier, so [`load_rank_head`] refuses a multi-output
//! head that does not name its labels — see [`RankHead::labels`].

use frink_core::matmul::layer_norm;
use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::{GgufValue, ShardedGguf, TensorSource};

use crate::loader::{load_f32_vec_optional, load_weight_matrix, LoadError};

/// `LLM_TENSOR_CLS` / `CLS_OUT` / `CLS_NORM`, by their GGUF names
/// (`llama-arch.cpp:428-430`).
const CLS_W: &str = "cls.weight";
const CLS_B: &str = "cls.bias";
const CLS_OUT_W: &str = "cls.output.weight";
const CLS_OUT_B: &str = "cls.output.bias";
const CLS_NORM_W: &str = "cls.norm.weight";
const CLS_NORM_B: &str = "cls.norm.bias";

/// A dense projection with an optional bias: `W · x (+ b)`.
struct Dense {
    w: WeightMatrix,
    b: Option<Vec<f32>>,
}

impl Dense {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.w.apply(x);
        if let Some(b) = &self.b {
            for (o, bv) in out.iter_mut().zip(b.iter()) {
                *o += bv;
            }
        }
        out
    }
}

/// The classification head a reranker checkpoint carries on top of the
/// encoder.
pub struct RankHead {
    /// `cls` + `cls.bias`, followed by `tanh`. Optional upstream: some
    /// checkpoints project straight to the output.
    dense: Option<Dense>,
    /// `cls.norm`, applied after the `tanh`. Upstream calls
    /// `build_norm(cur, cls_norm, NULL, LLM_NORM, -1)` — LayerNorm with
    /// a weight and **no bias**, so a `cls.norm.bias` in a checkpoint
    /// would be a tensor this graph does not apply and is refused.
    norm: Option<Vec<f32>>,
    /// `cls.output` + `cls.output.bias`, the projection to `n_cls_out`.
    out: Option<Dense>,
    /// `{arch}.classifier.output_labels`, verbatim. Empty only when the
    /// head has exactly one output, where a name adds nothing and
    /// upstream's own default (`hparams.n_cls_out = 1`) is unambiguous.
    labels: Vec<String>,
    eps: f32,
}

impl RankHead {
    /// How many floats [`Self::apply`] returns.
    pub fn n_cls_out(&self) -> usize {
        self.labels.len().max(1)
    }

    /// `{arch}.classifier.output_labels`, in output order. Empty for a
    /// single-output head that named none.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Every output of the head, for `pooled` — which must already be
    /// the CLS row, not the whole hidden-state matrix.
    pub fn apply(&self, pooled: &[f32]) -> Vec<f32> {
        let mut cur = pooled.to_vec();
        if let Some(dense) = &self.dense {
            cur = dense.apply(&cur);
            for v in cur.iter_mut() {
                *v = v.tanh();
            }
            if let Some(w) = &self.norm {
                // No bias: `build_norm(..., NULL, ...)` upstream.
                let zeros = vec![0.0f32; w.len()];
                cur = layer_norm(&cur, w, &zeros, self.eps);
            }
        }
        if let Some(out) = &self.out {
            cur = out.apply(&cur);
        }
        cur
    }

    /// The single relevance score `/v1/rerank` reports: output 0, which
    /// is what upstream's `send_rerank` sends (`res->score = embd[0]`).
    pub fn score(&self, pooled: &[f32]) -> f32 {
        self.apply(pooled).first().copied().unwrap_or(0.0)
    }

    /// Whether `cls` — HuggingFace's `bert.pooler.dense` — is in this
    /// checkpoint, and therefore in every score this head produces.
    ///
    /// The one bit that decides which of the two score scales in the
    /// module docs a caller is reading. Everything else that reports
    /// the regime is derived from it, so nothing can disagree with the
    /// weights that are actually loaded.
    pub fn has_pooler(&self) -> bool {
        self.dense.is_some()
    }

    /// The composition [`Self::apply`] runs, as a formula over the CLS
    /// row: `classifier(tanh(pooler(cls)))` for a head with its pooler,
    /// `classifier(cls)` for one without.
    ///
    /// Built from the same three `Option`s [`Self::apply`] branches on,
    /// in the same order, rather than restated as a table of the four
    /// shapes — two structures that must agree about one thing is this
    /// repo's dominant bug shape, and a *description* that has drifted
    /// from the graph is worse than none. The test
    /// `the_graph_names_a_tanh_exactly_when_a_tanh_is_applied` closes
    /// the loop by checking the label against the arithmetic.
    pub fn graph(&self) -> String {
        let mut g = "cls".to_string();
        if self.dense.is_some() {
            g = format!("tanh(pooler({g}))");
        }
        if self.norm.is_some() {
            g = format!("norm({g})");
        }
        if self.out.is_some() {
            g = format!("classifier({g})");
        }
        g
    }
}

fn refuse(what: String) -> LoadError {
    LoadError::UnsupportedFeature("bert".to_string(), what)
}

/// Reads `{arch}.classifier.output_labels`, an array of strings.
///
/// Absent is `Ok(vec![])`, which is only *allowed* for a one-output
/// head — [`load_rank_head`] enforces that. A key of the wrong type is
/// an error rather than a silent empty: a checkpoint that says
/// something about its labels and is not understood must stop the load.
fn read_labels(file: &impl TensorSource, arch: &str) -> Result<Vec<String>, LoadError> {
    let key = format!("{arch}.classifier.output_labels");
    let Some(value) = file.metadata(&key) else {
        return Ok(Vec::new());
    };
    match value {
        GgufValue::Array(items) => items
            .iter()
            .map(|v| match v {
                GgufValue::String(s) => Ok(s.clone()),
                other => Err(refuse(format!(
                    "{key} contains a non-string entry {other:?}; it must be an array of \
                     label names"
                ))),
            })
            .collect(),
        other => Err(refuse(format!(
            "{key} is {other:?}, but it must be an array of strings"
        ))),
    }
}

/// The load-time NOTE for a sequence-classification head that arrived
/// without its pooler — issue #82, and **not** a refusal.
///
/// Fires on exactly one combination: `cls.output` present, `cls`
/// absent, and `{arch}.classifier.output_labels` declared. That triple
/// is what a HuggingFace `BertForSequenceClassification` looks like
/// after llama.cpp's converter has deleted `pooler.dense` by name. It
/// is deliberately narrower than "no pooler":
/// `jina-reranker-v1-tiny-en` is the direct-projection shape upstream
/// documents as CORRECT, it names no labels, and warning about it
/// would train the reader to ignore this line.
///
/// A refusal is not available here. The file does not carry anything
/// that separates "trained without a pooler" from "converted without
/// one", so refusing would also refuse the checkpoints where the shape
/// is right — see the module docs. A note is what is left, and it is
/// why the regime is *also* on every `/v1/rerank` response: an operator
/// reads a log, a thresholding client reads JSON, and only the second
/// one is in a position to act on it.
///
/// A free function taking three booleans rather than an `if` inside
/// [`load_rank_head`], so that the condition can be shown to fire — and
/// shown not to fire on the two shapes next to it — without three GGUFs.
fn missing_pooler_note(has_dense: bool, has_out: bool, labels: &[String]) -> Option<String> {
    if has_dense || !has_out || labels.is_empty() {
        return None;
    }
    Some(format!(
        "frink: NOTE this reranker head runs classifier(cls) — it carries {CLS_OUT_W} and \
         labels ({}) but no {CLS_W}. A HuggingFace BertForSequenceClassification was trained \
         as classifier(tanh(pooler(cls))), and llama.cpp's converter deletes pooler.dense by \
         name (conversion/bert.py, \"we are only using BERT for embeddings so we don't need \
         the pooling layer\"), so this file cannot carry it. frink runs the pooler whenever \
         a file DOES carry it and will not invent one. The ranking is the checkpoint's; the \
         score SCALE is not (about ±0.2 rather than about ±11 on \
         ms-marco-MiniLM-L6-v2), so an absolute relevance threshold will not fire. See \
         frink issue #82.",
        labels.join(", "),
    ))
}

/// Builds the head a `bert` checkpoint carries, or `None` when it
/// carries none (a plain embedding model).
///
/// Refuses, rather than loading something that would score wrongly:
///
/// * a `cls.norm.bias`, which upstream never applies;
/// * labels whose count disagrees with `cls.output.weight`'s row count,
///   because then one of the two is describing a different model;
/// * a multi-output head with no labels, because
///   [`RankHead::score`] would silently pick output 0 out of several
///   that nothing has named.
pub fn load_rank_head(
    file: &ShardedGguf,
    arch: &str,
    n_embd: usize,
    eps: f32,
) -> Result<Option<RankHead>, LoadError> {
    let has_any = [CLS_W, CLS_OUT_W, CLS_NORM_W]
        .iter()
        .any(|n| file.find_tensor(n).is_some());
    let labels = read_labels(file, arch)?;
    if !has_any {
        // Labels without a head is a checkpoint describing a classifier
        // whose weights are not here. Loading it as a plain embedding
        // model would quietly drop what the file says it is.
        if !labels.is_empty() {
            return Err(refuse(format!(
                "{arch}.classifier.output_labels names {} label(s) ({}) but the checkpoint \
                 carries no {CLS_W}, {CLS_OUT_W} or {CLS_NORM_W} — the classification head \
                 the labels describe is not in this file",
                labels.len(),
                labels.join(", "),
            )));
        }
        return Ok(None);
    }

    let dense = match file.find_tensor(CLS_W) {
        Some(_) => {
            let w = load_weight_matrix(file, CLS_W)?;
            if w.cols() != n_embd {
                return Err(refuse(format!(
                    "{CLS_W} takes {} inputs but the encoder is {n_embd} wide",
                    w.cols()
                )));
            }
            Some(Dense {
                b: load_f32_vec_optional(file, CLS_B)?,
                w,
            })
        }
        None => None,
    };

    if file.find_tensor(CLS_NORM_B).is_some() {
        return Err(refuse(format!(
            "checkpoint carries {CLS_NORM_B}, but upstream's head norm is \
             build_norm(cur, cls_norm, NULL, ...) — a weight and no bias. Applying the \
             weight and dropping the bias would score wrongly and silently"
        )));
    }
    let norm = load_f32_vec_optional(file, CLS_NORM_W)?;
    if norm.is_some() && dense.is_none() {
        return Err(refuse(format!(
            "checkpoint carries {CLS_NORM_W} but no {CLS_W}; upstream applies the head norm \
             only inside the `cls` branch, so this norm would never run"
        )));
    }

    let out = match file.find_tensor(CLS_OUT_W) {
        Some(_) => {
            let w = load_weight_matrix(file, CLS_OUT_W)?;
            let want = dense.as_ref().map(|d| d.w.rows()).unwrap_or(n_embd);
            if w.cols() != want {
                return Err(refuse(format!(
                    "{CLS_OUT_W} takes {} inputs but the value reaching it is {want} wide",
                    w.cols()
                )));
            }
            Some(Dense {
                b: load_f32_vec_optional(file, CLS_OUT_B)?,
                w,
            })
        }
        None => None,
    };

    // How many scores this head really produces, from the weights.
    let n_out = match &out {
        Some(d) => d.w.rows(),
        None => match &dense {
            Some(d) => d.w.rows(),
            // Unreachable while `has_any` is true and norm-without-dense
            // is refused, but stated rather than assumed.
            None => n_embd,
        },
    };
    if !labels.is_empty() && labels.len() != n_out {
        return Err(refuse(format!(
            "{arch}.classifier.output_labels names {} label(s) ({}) but the head produces \
             {n_out} output(s) — the metadata and the weights describe different models",
            labels.len(),
            labels.join(", "),
        )));
    }
    if labels.is_empty() && n_out > 1 {
        return Err(refuse(format!(
            "this classification head produces {n_out} outputs and the checkpoint carries no \
             {arch}.classifier.output_labels, so nothing says which one is the relevance \
             score. Refusing rather than reporting output 0 as if it were named"
        )));
    }

    if let Some(note) = missing_pooler_note(dense.is_some(), out.is_some(), &labels) {
        eprintln!("{note}");
    }

    Ok(Some(RankHead {
        dense,
        norm,
        out,
        labels,
        eps,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::tensor::Tensor;

    fn dense(rows: usize, cols: usize, fill: f32, bias: Option<f32>) -> Dense {
        let data: Vec<f32> = (0..rows * cols).map(|i| fill * (i as f32 + 1.0)).collect();
        Dense {
            w: WeightMatrix::F32(Tensor::new(data, vec![rows, cols])),
            b: bias.map(|b| vec![b; rows]),
        }
    }

    /// The head's ORDER is the whole content of it: dense, tanh, norm,
    /// output. Applying the same pieces in another order still returns
    /// a plausible float, which is why this is checked against a
    /// hand-computed value rather than against "it ran".
    #[test]
    fn the_head_applies_dense_then_tanh_then_output() {
        let head = RankHead {
            // 1x2, weights [1, 2], bias 0 -> 1*x0 + 2*x1
            dense: Some(Dense {
                w: WeightMatrix::F32(Tensor::new(vec![1.0, 2.0], vec![1, 2])),
                b: Some(vec![0.5]),
            }),
            norm: None,
            // 1x1, weight [3], bias -1 -> 3*y - 1
            out: Some(Dense {
                w: WeightMatrix::F32(Tensor::new(vec![3.0], vec![1, 1])),
                b: Some(vec![-1.0]),
            }),
            labels: vec![],
            eps: 1e-12,
        };
        // dense: 1*0.25 + 2*0.5 + 0.5 = 1.75; tanh(1.75) = 0.94138...
        // out:   3*0.94138 - 1 = 1.82414...
        let want = 3.0 * 1.75f32.tanh() - 1.0;
        let got = head.score(&[0.25, 0.5]);
        assert!(
            (got - want).abs() < 1e-6,
            "head produced {got}, hand-computed {want}"
        );
        assert_eq!(head.n_cls_out(), 1);
    }

    /// A head with no `cls` is the direct-projection shape upstream
    /// documents for `jina-reranker-v1-tiny-en`: no tanh anywhere.
    #[test]
    fn a_head_with_no_dense_does_not_apply_a_tanh() {
        let head = RankHead {
            dense: None,
            norm: None,
            out: Some(Dense {
                w: WeightMatrix::F32(Tensor::new(vec![2.0, 0.0], vec![1, 2])),
                b: None,
            }),
            labels: vec![],
            eps: 1e-12,
        };
        // 2 * 5.0 = 10.0. A stray tanh would make this 1.0.
        assert!((head.score(&[5.0, 1.0]) - 10.0).abs() < 1e-6);
    }

    /// The label a caller reads off a response must be the arithmetic
    /// the head performed, so this checks the two against each other
    /// rather than checking [`RankHead::graph`] against a string
    /// constant — a constant would still agree with itself after
    /// [`RankHead::apply`] stopped applying the `tanh`.
    ///
    /// The probe is linearity. A bias-free `classifier(cls)` head is a
    /// matrix, so `f(2x) == 2·f(x)` exactly; a `tanh` in the middle is
    /// the head's ONLY non-linearity, so the same equality fails as
    /// soon as one runs. `graph()` naming a `tanh` and `apply` being
    /// linear (or the reverse) is the drift this is here for.
    #[test]
    fn the_graph_names_a_tanh_exactly_when_a_tanh_is_applied() {
        let out = || {
            Some(Dense {
                w: WeightMatrix::F32(Tensor::new(vec![1.0, -0.5], vec![1, 2])),
                b: None,
            })
        };
        let pooler = || {
            Some(Dense {
                w: WeightMatrix::F32(Tensor::new(vec![0.3, 0.7, -0.2, 0.4], vec![2, 2])),
                b: None,
            })
        };
        let x = [0.5f32, 0.25];
        let two_x = [1.0f32, 0.5];

        for (head, want_graph, want_pooler) in [
            (
                RankHead {
                    dense: pooler(),
                    norm: None,
                    out: out(),
                    labels: vec![],
                    eps: 1e-12,
                },
                "classifier(tanh(pooler(cls)))",
                true,
            ),
            (
                RankHead {
                    dense: None,
                    norm: None,
                    out: out(),
                    labels: vec![],
                    eps: 1e-12,
                },
                "classifier(cls)",
                false,
            ),
        ] {
            assert_eq!(head.graph(), want_graph);
            assert_eq!(head.has_pooler(), want_pooler);
            // `graph()` is derived from the same `Option`s, so it can
            // never disagree with `has_pooler` -- but the point is that
            // neither may disagree with the numbers.
            assert_eq!(head.graph().contains("tanh"), head.has_pooler());

            let linear = (head.score(&two_x) - 2.0 * head.score(&x)).abs() < 1e-6;
            assert_eq!(
                linear,
                !want_pooler,
                "{want_graph} was {} but its graph says otherwise: f(x)={}, f(2x)={}",
                match linear {
                    true => "linear",
                    false => "non-linear",
                },
                head.score(&x),
                head.score(&two_x),
            );
        }
    }

    /// The NOTE must fire on the shape llama.cpp's converter produces
    /// and on NOTHING else, because a note that also fires on the
    /// legitimate direct-projection head is a note operators learn to
    /// skip. All eight combinations, so the condition is pinned from
    /// both sides rather than demonstrated once.
    #[test]
    fn the_missing_pooler_note_fires_only_on_a_labelled_head_with_no_pooler() {
        let labelled = vec!["LABEL_0".to_string()];
        for has_dense in [false, true] {
            for has_out in [false, true] {
                for labels in [&[][..], &labelled[..]] {
                    let got = missing_pooler_note(has_dense, has_out, labels);
                    // `ms-marco-MiniLM-L6-v2` after conversion, and
                    // only that: cls.output, labels, no cls.
                    let want = !has_dense && has_out && !labels.is_empty();
                    assert_eq!(
                        got.is_some(),
                        want,
                        "dense={has_dense} out={has_out} labels={labels:?} produced {got:?}"
                    );
                }
            }
        }
        // And it says which head it is talking about, so a log line is
        // actionable without reading this file.
        let note = missing_pooler_note(false, true, &labelled).expect("the note fires");
        assert!(note.contains("LABEL_0"), "{note}");
        assert!(note.contains("#82"), "{note}");
    }

    /// `n_cls_out` follows the labels, and `score` is output 0 of
    /// however many there are -- upstream's `embd[0]`.
    #[test]
    fn score_is_the_first_output_and_labels_size_the_head() {
        let head = RankHead {
            dense: None,
            norm: None,
            out: Some(dense(3, 2, 1.0, None)),
            labels: vec!["a".into(), "b".into(), "c".into()],
            eps: 1e-12,
        };
        assert_eq!(head.n_cls_out(), 3);
        assert_eq!(head.labels(), ["a", "b", "c"]);
        let all = head.apply(&[1.0, 1.0]);
        assert_eq!(all.len(), 3);
        assert_eq!(head.score(&[1.0, 1.0]), all[0]);
        assert_ne!(all[0], all[1], "the fixture must distinguish the outputs");
    }
}
