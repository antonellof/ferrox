//! Sampler knobs a caller may send that this server does not implement,
//! refused BY NAME rather than dropped.
//!
//! Serde drops an undeclared field silently, and a caller cannot tell
//! that apart from having had it honoured: they get a 200 and an answer
//! computed under different rules than they asked for, which is worse
//! than an error. So the field is deserialized purely in order to be
//! refused.
//!
//! **One implementation, two routes.** `/v1/chat/completions` and
//! `/v1/completions` cannot share a request struct -- their bodies are
//! genuinely different shapes -- but they decide here, so they cannot
//! disagree about the same field again. `/v1/completions` refused
//! `logit_bias` from the start; `/v1/chat/completions` did not even
//! declare it, and that split is the defect this module closes.

use ferrox_models::sampler_order::{SamplerOrder, SamplerOrderError};
use serde_json::Value;

use crate::{invalid_request, unsupported_feature, ApiError};

/// Parse llama.cpp's `samplers` request field into a validated chain, or
/// refuse it BY NAME.
///
/// **One implementation, three routes.** `/v1/chat/completions`,
/// `/v1/completions` and llama.cpp's native `/completion` all take this
/// field, and a per-route copy of "which samplers exist" is the defect
/// this module already closed once for `logit_bias`. The names
/// themselves are not restated here at all: they come from
/// [`ferrox_models::sampler_order`], which is also what `ferrox run
/// --samplers` parses, so the CLI and the server cannot disagree about
/// what a sampler is called or which ones exist.
///
/// ## The wire shape
///
/// llama.cpp's server accepts either a JSON array of names or the one
/// `;`-separated string its command line takes, so both are accepted
/// here. An EMPTY array is read as "the caller said nothing", matching
/// how every other optional list on `/completion` treats empty and
/// keeping clients that send `"samplers": []` as a default working; an
/// empty *string* is refused, because a caller who typed a value meant
/// something by it.
///
/// ## Why a refusal and not a filtered chain
///
/// ferrox implements llama.cpp's whole default chain but not `mirostat`
/// or `infill`. Building the chain out of the names it recognises and
/// dropping the rest would answer a request naming `mirostat` with a
/// chain that has none, a 200, and no way for the caller to tell -- the
/// same silence `logit_bias` is refused for.
///
/// The status codes distinguish the two failures a caller can have:
/// a name llama.cpp does not define either is a **400** (a typo in the
/// request), while a real upstream sampler ferrox lacks is a **501**
/// (a valid request this server cannot serve). Collapsing them would
/// tell someone their working llama.cpp flag was a spelling mistake.
pub(crate) fn parse_sampler_order(
    value: Option<&Value>,
    route: &str,
) -> Result<Option<SamplerOrder>, ApiError> {
    let names: Vec<String> = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(items)) => {
            if items.is_empty() {
                return Ok(None);
            }
            items
                .iter()
                .map(|item| match item {
                    Value::String(s) => Ok(s.clone()),
                    other => Err(invalid_request(
                        &format!("`samplers` must be a list of sampler names; got {other}"),
                        "samplers",
                    )),
                })
                .collect::<Result<_, _>>()?
        }
        Some(Value::String(s)) => s.split(';').map(str::to_string).collect(),
        Some(other) => {
            return Err(invalid_request(
                &format!(
                    "`samplers` must be a list of sampler names or a `;`-separated \
                     string; got {other}"
                ),
                "samplers",
            ))
        }
    };
    match SamplerOrder::from_names(names) {
        Ok(order) => Ok(Some(order)),
        Err(err @ SamplerOrderError::Unimplemented { .. }) => Err(unsupported_feature(&format!(
            "`samplers` on {route}: {err}"
        ))),
        Err(err) => Err(invalid_request(
            &format!("`samplers` on {route}: {err}"),
            "samplers",
        )),
    }
}

/// Refuse `logit_bias` by name, or accept a bias that would move
/// nothing.
///
/// ## Why refused rather than implemented
///
/// [`ferrox_models::sampling::Sampler::sample_with_mask`] already takes
/// an arbitrary `FnMut(&mut [f32])`, so *the sampler* could apply a bias
/// with no change to `sampling.rs`. Honouring it end to end is the part
/// that is not cheap, and each piece left out would be a fresh instance
/// of the same silent-wrong bug:
///
/// - The whole-response cache keys on the sampler settings
///   (`ChatCompletionRequest::cache_key`). A bias outside that key means
///   two requests differing only in their bias share one cached answer.
/// - The continuous-batching worker samples through its own call site,
///   so a bias wired into the private decode loop alone would be honoured
///   or ignored depending on `FERROX_CONTINUOUS_BATCHING`.
/// - On Metal at `temperature <= 0` the decoder folds `lm_head` and
///   argmax into the GPU stack and returns a *one-element* logits vector
///   holding the chosen id (see `generate::generate`'s greedy guard).
///   There is no vocabulary-shaped logit vector left to bias.
///
/// A refusal is coverage. Implementing it is tracked as its own item
/// (`docs/plans/llama-cpp-gap-inventory.md`, the `logit_bias` sampler
/// row), not smuggled in half-done.
///
/// `{}` is accepted: several OpenAI clients send an empty map on every
/// request as a default, and there is no token whose logit it would
/// have moved. Refusing it would be a false refusal.
pub(crate) fn refuse_logit_bias(value: Option<&Value>, route: &str) -> Result<(), ApiError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_null() || value.as_object().is_some_and(|m| m.is_empty()) {
        return Ok(());
    }
    Err(unsupported_feature(&format!(
        "`logit_bias` is not implemented on {route} (see docs/API.md). \
         It is refused rather than ignored: a dropped bias is \
         indistinguishable from an honoured one."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn a_real_bias_is_refused_by_name() {
        let bias = serde_json::json!({"50256": -100.0});
        let (status, body) =
            refuse_logit_bias(Some(&bias), "/v1/chat/completions").expect_err("refused");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let message = body["error"]["message"].as_str().expect("message");
        assert!(message.contains("logit_bias"), "{message}");
        assert!(message.contains("/v1/chat/completions"), "{message}");
    }

    /// An absent bias and a bias that would move nothing are the same
    /// request as far as the sampler is concerned. Refusing `{}` would
    /// reject clients that send it as a default on every call.
    #[test]
    fn an_empty_or_absent_bias_is_served() {
        assert!(refuse_logit_bias(None, "/v1/completions").is_ok());
        assert!(refuse_logit_bias(Some(&serde_json::json!({})), "/v1/completions").is_ok());
    }

    /// A chain of samplers ferrox has is honoured, in the order sent,
    /// from either wire shape llama.cpp's server accepts.
    #[test]
    fn a_supported_chain_is_honoured_from_a_list_or_from_a_string() {
        let from_list = parse_sampler_order(
            Some(&serde_json::json!(["penalties", "temperature", "top_k"])),
            "/completion",
        )
        .expect("supported")
        .expect("present");
        let from_string = parse_sampler_order(
            Some(&serde_json::json!("penalties;temperature;top_k")),
            "/completion",
        )
        .expect("supported")
        .expect("present");
        assert_eq!(from_list, from_string);
        assert_eq!(from_list.to_string(), "penalties;temperature;top_k");
    }

    /// An absent field, and an empty list, are both "the caller said
    /// nothing": the default chain, which is what ferrox always ran.
    ///
    /// The empty LIST is served rather than refused for the same reason
    /// `logit_bias: {}` is -- clients send it as a default on every
    /// request and there is no order it would have changed. An empty
    /// STRING is a value the caller typed, so it is refused.
    #[test]
    fn an_absent_or_empty_list_means_the_default_chain() {
        for silence in [
            None,
            Some(serde_json::json!(null)),
            Some(serde_json::json!([])),
        ] {
            assert!(
                parse_sampler_order(silence.as_ref(), "/completion")
                    .expect("silence is served")
                    .is_none(),
                "{silence:?} must resolve to the default chain"
            );
        }
        let (status, _) = parse_sampler_order(Some(&serde_json::json!("")), "/completion")
            .expect_err("an empty string is a value, not silence");
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// A sampler llama.cpp HAS and ferrox does not is refused by name,
    /// with the reason, as a 501.
    ///
    /// This is the whole point of the field being parsed rather than
    /// dropped: a caller who sent llama.cpp's own default chain and was
    /// served a four-sampler subset of it got a 200 computed under rules
    /// they did not ask for.
    #[test]
    fn a_sampler_ferrox_lacks_is_refused_by_name_as_not_implemented() {
        for name in ["mirostat", "infill"] {
            let body = serde_json::json!([name, "temperature"]);
            let (status, message) = parse_sampler_order(Some(&body), "/completion")
                .expect_err("{name} must be refused, not skipped");
            assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{name}");
            let text = message["error"]["message"].as_str().expect("message");
            assert!(text.contains(name), "{text}");
            assert!(text.contains("/completion"), "{text}");
        }
    }

    /// A name llama.cpp does not define either is a 400, not a 501: the
    /// two verdicts answer different questions, and collapsing them
    /// would tell a caller their working upstream flag was a typo.
    #[test]
    fn an_unknown_sampler_is_a_client_error_naming_the_name() {
        let (status, message) = parse_sampler_order(
            Some(&serde_json::json!(["top_k", "top_kk", "temperature"])),
            "/v1/completions",
        )
        .expect_err("no such sampler");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let text = message["error"]["message"].as_str().expect("message");
        assert!(text.contains("top_kk"), "{text}");

        // And a wire shape that is neither a list nor a string.
        let (status, _) = parse_sampler_order(Some(&serde_json::json!(7)), "/v1/completions")
            .expect_err("a number is not a chain");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = parse_sampler_order(
            Some(&serde_json::json!(["top_k", 7, "temperature"])),
            "/v1/completions",
        )
        .expect_err("a list of not-names is not a chain");
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// The three routes that take `samplers` answer one body the same
    /// way, asserted through the real request types.
    ///
    /// This is the `logit_bias` defect's shape: `/v1/completions`
    /// refused a field the chat route did not even declare, so the same
    /// body got a 501 on one route and a 200 on the other. A shared
    /// helper only helps if every route calls it.
    #[test]
    fn every_route_answers_a_sampler_chain_the_same_way() {
        for (samplers, expected_status) in [
            (serde_json::json!(["top_k", "temperature"]), None),
            (serde_json::json!([]), None),
            (
                serde_json::json!(["mirostat", "temperature"]),
                Some(StatusCode::NOT_IMPLEMENTED),
            ),
            // llama.cpp's own default chain, which every one of these
            // routes must now accept: this list is the whole reason the
            // four missing samplers were ported.
            (
                serde_json::json!([
                    "penalties",
                    "dry",
                    "top_n_sigma",
                    "top_k",
                    "typ_p",
                    "top_p",
                    "min_p",
                    "xtc",
                    "temperature"
                ]),
                None,
            ),
            (
                serde_json::json!(["top_kk", "temperature"]),
                Some(StatusCode::BAD_REQUEST),
            ),
            (serde_json::json!(["top_k"]), Some(StatusCode::BAD_REQUEST)),
        ] {
            let chat: crate::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "samplers": samplers,
            }))
            .expect("chat request");
            let completions: crate::openai_extra::CompletionsRequest =
                serde_json::from_value(serde_json::json!({
                    "prompt": "hi",
                    "samplers": samplers,
                }))
                .expect("completions request");
            let native: crate::completion::CompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "prompt": "hi",
                    "samplers": samplers,
                }))
                .expect("completion request");

            let chat_status = chat.validate_supported_fields().err().map(|(s, _)| s);
            let completions_status = completions.validate().err().map(|(s, _)| s);
            let native_status = native.validate(false).err().map(|(s, _)| s);
            assert_eq!(
                chat_status, completions_status,
                "chat and /v1/completions disagree about {samplers}"
            );
            assert_eq!(
                chat_status, native_status,
                "chat and /completion disagree about {samplers}"
            );
            assert_eq!(chat_status, expected_status, "wrong verdict for {samplers}");
        }
    }

    /// A bias that is not an object at all is still a bias the caller
    /// meant; it must not fall through the empty-map hole.
    #[test]
    fn a_malformed_bias_is_refused_rather_than_read_as_empty() {
        assert!(refuse_logit_bias(Some(&serde_json::json!([])), "/v1/completions").is_err());
        assert!(refuse_logit_bias(Some(&serde_json::json!("none")), "/v1/completions").is_err());
    }

    /// The defect this module exists for: `/v1/completions` refused
    /// `logit_bias` and `/v1/chat/completions` did not even declare it,
    /// so the SAME body got a 501 on one route and a 200 with unbiased
    /// output on the other. Asserted through both real request types,
    /// because a shared helper only helps if both routes call it.
    #[test]
    fn both_routes_answer_a_logit_bias_the_same_way() {
        for (bias, expected_refusal) in [
            (serde_json::json!({"50256": -100.0}), true),
            (serde_json::json!({}), false),
        ] {
            let chat: crate::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "logit_bias": bias,
            }))
            .expect("chat request");
            let completion: crate::openai_extra::CompletionsRequest =
                serde_json::from_value(serde_json::json!({
                    "prompt": "hi",
                    "logit_bias": bias,
                }))
                .expect("completions request");

            let chat_status = chat.validate_supported_fields().err().map(|(s, _)| s);
            let completion_status = completion.validate().err().map(|(s, _)| s);
            assert_eq!(
                chat_status, completion_status,
                "the two routes disagree about logit_bias {bias}"
            );
            assert_eq!(
                chat_status.is_some(),
                expected_refusal,
                "wrong verdict for logit_bias {bias}"
            );
        }
    }
}
