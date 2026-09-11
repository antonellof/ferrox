//! `/v1/embeddings`, over either of the two things that can produce a
//! vector for a piece of text.
//!
//! # The two sources, and why one route serves both
//!
//! * A **decoder** (`FERROX_MODEL_PATH`). Its final hidden states get
//!   pooled. This is what the route has always done, and it is a
//!   best-effort reading of a model that was not trained to put a
//!   sentence representation anywhere in particular.
//! * A **real embedding model**, i.e. a BERT/BGE encoder loaded through
//!   [`ferrox_models::EmbeddingModel`]. This one has a `[CLS]` position
//!   it was trained to use and a `pooling_type` in its own metadata
//!   saying so. It reaches the server either as the *loaded model*
//!   (`FERROX_MODEL_PATH` pointing at an encoder-only GGUF, or
//!   `/admin/models/load` swapping one in -- see
//!   [`crate::loaded::Loaded`]) or as a side-car beside a generative
//!   model (`FERROX_EMBEDDING_MODEL_PATH`).
//!
//! When an embedding model is present it wins, because it is the one
//! that was asked for. The decoder path is unchanged underneath it.
//!
//! Both go through [`ferrox_models::pooling::pool`]. That matters: the
//! mean and last-token arms used to be written out here as a private
//! `pool_hidden`, and adding CLS would have meant a second copy of the
//! same three arms varying in one place.
//!
//! # Where the two paths deliberately differ
//!
//! * **Default pooling.** The decoder path defaults to `mean`, as
//!   before. The encoder path defaults to whatever the checkpoint's own
//!   `{arch}.pooling_type` says — CLS for every BGE — because the file
//!   knows and the caller usually does not.
//! * **Normalization.** The encoder path L2-normalizes, which is what
//!   llama.cpp's server does by default and what every BGE/E5 consumer
//!   assumes of a returned embedding. The decoder path does not, and is
//!   left alone: changing what an existing caller gets back would be a
//!   silent behaviour change, not a feature.
//! * **Accepted `embedding_type`.** `mean` and `last` on both. `cls` on
//!   the encoder path only — row 0 of a decoder's hidden states is its
//!   BOS position and means nothing in particular.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;

use ferrox_models::pooling::{l2_normalize, pool, PoolingType};
use ferrox_models::tokenizer::SpecialTokens;
use ferrox_models::EmbeddingModel;

use crate::openai_extra::Call;
use crate::{join_error_response, unsupported_feature, ApiError, AppState};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingsRequest {
    input: EmbeddingInput,
    #[serde(default)]
    model: Option<String>,
    /// Only `"float"` is supported (OpenAI also has `base64`).
    #[serde(default)]
    encoding_format: Option<String>,
    /// Pooling over token hidden states. `mean` (default) or `last` on
    /// the decoder path; `mean`, `last` or `cls` on the encoder path,
    /// where omitting it uses the checkpoint's own `pooling_type`.
    #[serde(default)]
    embedding_type: Option<String>,
}

fn bad_request(message: String) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": { "message": message } })),
    )
}

/// Maps the request's `embedding_type` onto a [`PoolingType`], refusing
/// anything outside `allowed` by name.
///
/// `NONE` is not in either path's `allowed` set and could not be: it
/// returns one vector per *token*, and this response shape carries one
/// vector per *input*. `RANK` is refused one layer down, by
/// [`ferrox_models::pooling::pool`] itself.
fn requested_pooling(
    embedding_type: Option<&str>,
    allowed: &[(&str, PoolingType)],
    default: PoolingType,
) -> Result<PoolingType, ApiError> {
    let Some(name) = embedding_type else {
        return Ok(default);
    };
    allowed
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, ty)| *ty)
        .ok_or_else(|| {
            let names: Vec<&str> = allowed.iter().map(|(n, _)| *n).collect();
            bad_request(format!(
                "embedding_type must be one of {names:?}, got {name:?}"
            ))
        })
}

const DECODER_POOLING: &[(&str, PoolingType)] =
    &[("mean", PoolingType::Mean), ("last", PoolingType::Last)];

const ENCODER_POOLING: &[(&str, PoolingType)] = &[
    ("mean", PoolingType::Mean),
    ("last", PoolingType::Last),
    ("cls", PoolingType::Cls),
];

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let call = Call::new(&headers);
    let result = embeddings_inner(&state, req).await;
    // Embeddings *do* run the model, so the prompt tokens they paid for
    // are real prompt tokens and are recorded as such. There is no
    // decode loop, so `decode_ms` stays null rather than borrowing the
    // total -- the same rule the two duration columns exist for.
    let usage = result
        .as_ref()
        .ok()
        .map(|(_, prompt_tokens)| ferrox_api::Usage::new(*prompt_tokens, 0));
    call.record(
        &state,
        ferrox_api::routes::V1_EMBEDDINGS,
        state.embedding_model_name(),
        &result,
        usage.as_ref(),
    );
    result.map(|(body, _)| Json(body))
}

async fn embeddings_inner(
    state: &AppState,
    req: EmbeddingsRequest,
) -> Result<(serde_json::Value, usize), ApiError> {
    if let Some(fmt) = req.encoding_format.as_deref() {
        if fmt != "float" {
            return Err(bad_request(format!(
                "encoding_format {fmt:?} is not supported (only \"float\")"
            )));
        }
    }

    let inputs: Vec<String> = match req.input {
        EmbeddingInput::One(s) => vec![s],
        EmbeddingInput::Many(v) => v,
    };
    if inputs.is_empty() {
        return Err(bad_request(
            "input must be a non-empty string or array of strings".to_string(),
        ));
    }

    let (data, prompt_tokens, default_name) = match state.embedding_model() {
        Some(encoder) => {
            let pooling = requested_pooling(
                req.embedding_type.as_deref(),
                ENCODER_POOLING,
                encoder.pooling_type(),
            )?;
            let name = encoder.name().to_string();
            let (data, tokens) = encoder_embeddings(encoder, pooling, inputs).await?;
            (data, tokens, name)
        }
        None => {
            let pooling = requested_pooling(
                req.embedding_type.as_deref(),
                DECODER_POOLING,
                PoolingType::Mean,
            )?;
            let active_model = state.require_model()?;
            // Fail fast for non-GGUF engines before paying encode cost.
            if active_model.embed_tokens(&[]).is_none() {
                return Err(unsupported_feature(
                    "embeddings engine not yet available for this model. Load a real \
                     embedding checkpoint (a `bert` GGUF such as BGE) instead: either as \
                     the served model (FERROX_MODEL_PATH), or beside this one \
                     (FERROX_EMBEDDING_MODEL_PATH)",
                ));
            }
            let name = active_model.name().to_string();
            let (data, tokens) = decoder_embeddings(active_model, pooling, inputs).await?;
            (data, tokens, name)
        }
    };

    let model_name = req.model.unwrap_or(default_name);
    Ok((
        serde_json::json!({
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            }
        }),
        prompt_tokens,
    ))
}

fn entry(index: usize, embedding: Vec<f32>) -> serde_json::Value {
    serde_json::json!({
        "object": "embedding",
        "index": index,
        "embedding": embedding,
    })
}

/// A real encoder: the checkpoint's own tokenizer, its own `[CLS]`/`[SEP]`
/// wrapping, its own pooling, and an L2-normalized result.
async fn encoder_embeddings(
    encoder: Arc<EmbeddingModel>,
    pooling: PoolingType,
    inputs: Vec<String>,
) -> Result<(Vec<serde_json::Value>, usize), ApiError> {
    tokio::task::spawn_blocking(move || {
        let mut out = Vec::with_capacity(inputs.len());
        let mut prompt_tokens = 0usize;
        for (i, text) in inputs.iter().enumerate() {
            let ids = encoder.token_ids(text);
            prompt_tokens += ids.len();
            let hidden = encoder
                .hidden_states(text)
                .map_err(|e| bad_request(e.to_string()))?;
            let mut embedding =
                pool(&hidden, encoder.n_embd(), pooling).map_err(|e| bad_request(e.to_string()))?;
            l2_normalize(&mut embedding);
            out.push(entry(i, embedding));
        }
        Ok::<_, ApiError>((out, prompt_tokens))
    })
    .await
    .map_err(join_error_response)?
}

/// The pre-existing path: pool a decoder's last-layer hidden states.
async fn decoder_embeddings(
    model: Arc<crate::Model>,
    pooling: PoolingType,
    inputs: Vec<String>,
) -> Result<(Vec<serde_json::Value>, usize), ApiError> {
    tokio::task::spawn_blocking(move || {
        let mut out = Vec::with_capacity(inputs.len());
        let mut prompt_tokens = 0usize;
        for (i, text) in inputs.iter().enumerate() {
            // `Parse`, as llama.cpp's `handle_embeddings_impl` does.
            let tokens = model.encode(text, SpecialTokens::Parse);
            prompt_tokens += tokens.len();
            if let Some(vocab) = model.vocab_size() {
                if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab) {
                    return Err(bad_request(format!(
                        "token id {bad} is outside this model's vocabulary of {vocab}"
                    )));
                }
            }
            let hiddens = model.embed_tokens(&tokens).ok_or_else(|| {
                unsupported_feature("embeddings engine not yet available for this model")
            })?;
            if hiddens.is_empty() {
                return Err(bad_request(
                    "input encoded to zero tokens; cannot embed empty sequence".to_string(),
                ));
            }
            let n_embd = hiddens[0].len();
            let flat: Vec<f32> = hiddens.concat();
            let embedding = pool(&flat, n_embd, pooling).map_err(|e| bad_request(e.to_string()))?;
            out.push(entry(i, embedding));
        }
        Ok::<_, ApiError>((out, prompt_tokens))
    })
    .await
    .map_err(join_error_response)?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_embedding_type_takes_the_paths_own_default() {
        assert_eq!(
            requested_pooling(None, DECODER_POOLING, PoolingType::Mean).unwrap(),
            PoolingType::Mean
        );
        // The encoder path's default is the checkpoint's own key, so
        // whatever is passed in is what comes back.
        assert_eq!(
            requested_pooling(None, ENCODER_POOLING, PoolingType::Cls).unwrap(),
            PoolingType::Cls
        );
    }

    /// `cls` is meaningful for an encoder and not for a decoder, and
    /// the refusal must say which names *are* accepted.
    #[test]
    fn cls_is_accepted_only_on_the_encoder_path() {
        assert_eq!(
            requested_pooling(Some("cls"), ENCODER_POOLING, PoolingType::Cls).unwrap(),
            PoolingType::Cls
        );
        let (status, body) =
            requested_pooling(Some("cls"), DECODER_POOLING, PoolingType::Mean).unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let msg = body.0["error"]["message"].as_str().unwrap().to_string();
        assert!(msg.contains("\"mean\""), "{msg}");
        assert!(msg.contains("\"last\""), "{msg}");
        assert!(msg.contains("\"cls\""), "{msg}");
    }

    /// `none` and `rank` are real `llama_pooling_type` values that this
    /// response shape cannot carry / this build does not implement.
    /// Neither may be silently reinterpreted as something else.
    #[test]
    fn none_and_rank_are_refused_on_both_paths() {
        for allowed in [DECODER_POOLING, ENCODER_POOLING] {
            for name in ["none", "rank"] {
                assert!(
                    requested_pooling(Some(name), allowed, PoolingType::Mean).is_err(),
                    "{name} was accepted"
                );
            }
        }
    }
}
