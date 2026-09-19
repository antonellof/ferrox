//! `{arch}.pooling_type`: how a sequence of hidden states becomes one
//! embedding vector.
//!
//! This is llama.cpp's `enum llama_pooling_type` and the switch in
//! `llm_graph_context::build_pooling` (`src/llama-graph.cpp`),
//! transcribed. The key is written by the HF converter
//! (`gguf_writer.add_pooling_type`) as a **uint32** holding that enum's
//! value, so the wire numbers below are load-bearing and are not
//! frink's own invention.
//!
//! Its own module rather than a section of [`crate::bert_encoder`]
//! because pooling is not a BERT fact: `llama-embed` and
//! `gemma-embedding` are decoders that carry the same key, and
//! `/v1/embeddings` needs to honour it for whatever produced the hidden
//! states.
//!
//! # What is implemented, and what refuses
//!
//! `NONE`, `MEAN`, `CLS` and `LAST` are here. `RANK` is **not**, and
//! never will be: it is not a pooling rule at all but a classification
//! head — upstream runs `cls`/`cls_out` matrices, a `tanh`, and an
//! optional head norm over the pooled row, and reports the result
//! through `/v1/rerank` against `classifier.output_labels`. That head
//! is [`crate::rank_head`] and that route exists, but neither is
//! reachable from here: [`pool`] sees hidden states and a width, so the
//! head's matrices are not a thing it *could* apply. So
//! [`PoolingType::Rank`] parses (which is how the refusal can name it)
//! and [`pool`] returns [`PoolingError::Unimplemented`] rather than
//! quietly handing back a CLS row that means something else. An
//! `/v1/embeddings` request against a reranker checkpoint refuses here,
//! deliberately, and that is what sends the caller to `/v1/rerank`.

use thiserror::Error;

/// `enum llama_pooling_type`, by its wire values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingType {
    /// No pooling: every token's hidden state is returned.
    None,
    /// Arithmetic mean over the sequence.
    Mean,
    /// The first token's row. What BERT/BGE checkpoints use, because
    /// their `[CLS]` position is the one the model was trained to put
    /// the sentence representation in.
    Cls,
    /// The last token's row. Decoder-style embedding models.
    Last,
    /// Not pooling — a reranker classification head. See module docs.
    Rank,
}

#[derive(Debug, Error)]
pub enum PoolingError {
    #[error(
        "{key} = {value} is not one of llama.cpp's llama_pooling_type values \
         (-1 unspecified, 0 NONE, 1 MEAN, 2 CLS, 3 LAST, 4 RANK)"
    )]
    UnknownWireValue { key: String, value: i64 },
    #[error("{key} is present but is not an integer: {value}")]
    NotAnInteger { key: String, value: String },
    #[error(
        "pooling type RANK is a reranker classification head (cls / cls_out / \
         classifier.output_labels), not a pooling rule: pooling sees hidden states and a \
         width, and cannot reach those matrices. POST /v1/rerank with a query and \
         documents, which runs the head this checkpoint carries. Refusing rather than \
         returning a CLS row that is not what RANK means"
    )]
    Unimplemented,
    #[error("cannot pool an empty sequence")]
    EmptySequence,
    #[error("hidden states are {len} floats, which is not a whole number of {n_embd}-wide rows")]
    RaggedHiddenStates { len: usize, n_embd: usize },
}

impl PoolingType {
    /// The name upstream prints, so a refusal names the same thing the
    /// user's `llama-embedding --pooling` flag does.
    pub fn name(self) -> &'static str {
        match self {
            PoolingType::None => "NONE",
            PoolingType::Mean => "MEAN",
            PoolingType::Cls => "CLS",
            PoolingType::Last => "LAST",
            PoolingType::Rank => "RANK",
        }
    }

    /// `None` for `-1` (`LLAMA_POOLING_TYPE_UNSPECIFIED`), which means
    /// "the caller decides" and is not an error.
    fn from_wire(key: &str, value: i64) -> Result<Option<Self>, PoolingError> {
        Ok(match value {
            -1 => None,
            0 => Some(PoolingType::None),
            1 => Some(PoolingType::Mean),
            2 => Some(PoolingType::Cls),
            3 => Some(PoolingType::Last),
            4 => Some(PoolingType::Rank),
            other => {
                return Err(PoolingError::UnknownWireValue {
                    key: key.to_string(),
                    value: other,
                })
            }
        })
    }

    /// Reads `{arch}.pooling_type`. `Ok(None)` means the key is absent
    /// or explicitly unspecified — the caller picks a default and says
    /// so, rather than this function inventing one.
    pub fn from_gguf(
        file: &impl frink_gguf::TensorSource,
        arch: &str,
    ) -> Result<Option<Self>, PoolingError> {
        let key = format!("{arch}.pooling_type");
        let Some(value) = file.metadata(&key) else {
            return Ok(None);
        };
        let n = match value {
            frink_gguf::GgufValue::U8(v) => i64::from(*v),
            frink_gguf::GgufValue::I8(v) => i64::from(*v),
            frink_gguf::GgufValue::U16(v) => i64::from(*v),
            frink_gguf::GgufValue::I16(v) => i64::from(*v),
            frink_gguf::GgufValue::U32(v) => i64::from(*v),
            frink_gguf::GgufValue::I32(v) => i64::from(*v),
            frink_gguf::GgufValue::U64(v) => *v as i64,
            frink_gguf::GgufValue::I64(v) => *v,
            other => {
                return Err(PoolingError::NotAnInteger {
                    key,
                    value: format!("{other:?}"),
                })
            }
        };
        Self::from_wire(&key, n)
    }
}

/// Pools `hidden` (`n_tokens` rows of `n_embd` floats, in row order)
/// down to one vector — except for [`PoolingType::None`], which returns
/// every row unchanged.
///
/// No L2 normalization happens here, because none happens in
/// `build_pooling` either: upstream normalizes in the *caller*
/// (`common_embd_normalize`, chosen by `llama-embedding --embd-normalize`
/// and by the server's `/v1/embeddings`), and folding it in here would
/// make MEAN and CLS silently return something the graph did not.
pub fn pool(hidden: &[f32], n_embd: usize, ty: PoolingType) -> Result<Vec<f32>, PoolingError> {
    if n_embd == 0 || hidden.is_empty() {
        return Err(PoolingError::EmptySequence);
    }
    if !hidden.len().is_multiple_of(n_embd) {
        return Err(PoolingError::RaggedHiddenStates {
            len: hidden.len(),
            n_embd,
        });
    }
    let n_tokens = hidden.len() / n_embd;
    Ok(match ty {
        PoolingType::None => hidden.to_vec(),
        PoolingType::Cls => hidden[..n_embd].to_vec(),
        PoolingType::Last => hidden[(n_tokens - 1) * n_embd..].to_vec(),
        PoolingType::Mean => {
            let mut out = vec![0.0f32; n_embd];
            for row in hidden.chunks_exact(n_embd) {
                for (o, v) in out.iter_mut().zip(row) {
                    *o += *v;
                }
            }
            let inv = 1.0 / n_tokens as f32;
            for o in out.iter_mut() {
                *o *= inv;
            }
            out
        }
        PoolingType::Rank => return Err(PoolingError::Unimplemented),
    })
}

/// L2-normalizes in place, which is what every BGE/E5/GTE consumer
/// expects of an embedding and what `common_embd_normalize`'s default
/// (`p == 2`) does. A zero vector is left alone, exactly as upstream
/// leaves it (it divides by `norm > 0 ? 1/norm : 0`, i.e. it zeroes —
/// and a zero vector is already zero).
pub fn l2_normalize(v: &mut [f32]) {
    let sum: f32 = v.iter().map(|x| x * x).sum();
    if sum <= 0.0 {
        return;
    }
    let inv = 1.0 / sum.sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cls_takes_the_first_row_and_last_takes_the_last() {
        let hidden = vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0];
        assert_eq!(pool(&hidden, 2, PoolingType::Cls).unwrap(), vec![1.0, 2.0]);
        assert_eq!(
            pool(&hidden, 2, PoolingType::Last).unwrap(),
            vec![100.0, 200.0]
        );
        assert_eq!(
            pool(&hidden, 2, PoolingType::Mean).unwrap(),
            vec![37.0, 74.0]
        );
        assert_eq!(pool(&hidden, 2, PoolingType::None).unwrap(), hidden);
    }

    /// RANK must refuse. If this ever starts returning a vector, the
    /// caller is getting a CLS row labelled as a rerank score.
    #[test]
    fn rank_refuses_by_name() {
        let err = pool(&[1.0, 2.0], 2, PoolingType::Rank).unwrap_err();
        assert!(matches!(err, PoolingError::Unimplemented));
        assert!(err.to_string().contains("RANK"));
    }

    #[test]
    fn empty_and_ragged_inputs_refuse() {
        assert!(matches!(
            pool(&[], 4, PoolingType::Cls),
            Err(PoolingError::EmptySequence)
        ));
        assert!(matches!(
            pool(&[1.0, 2.0, 3.0], 2, PoolingType::Cls),
            Err(PoolingError::RaggedHiddenStates { len: 3, n_embd: 2 })
        ));
    }

    #[test]
    fn wire_values_match_llama_pooling_type() {
        let k = "bert.pooling_type";
        assert_eq!(PoolingType::from_wire(k, -1).unwrap(), None);
        assert_eq!(
            PoolingType::from_wire(k, 0).unwrap(),
            Some(PoolingType::None)
        );
        assert_eq!(
            PoolingType::from_wire(k, 1).unwrap(),
            Some(PoolingType::Mean)
        );
        assert_eq!(
            PoolingType::from_wire(k, 2).unwrap(),
            Some(PoolingType::Cls)
        );
        assert_eq!(
            PoolingType::from_wire(k, 3).unwrap(),
            Some(PoolingType::Last)
        );
        assert_eq!(
            PoolingType::from_wire(k, 4).unwrap(),
            Some(PoolingType::Rank)
        );
        assert!(PoolingType::from_wire(k, 5).is_err());
    }

    #[test]
    fn l2_normalize_makes_a_unit_vector_and_leaves_zero_alone() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut z = vec![0.0f32; 3];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0, 0.0, 0.0]);
    }
}
