//! The response and SSE shapes llama.cpp's native `/completion`
//! speaks, and nothing else.
//!
//! Kept apart from the request half next door because they are two
//! different jobs: that side decides what this server will agree to
//! do, this side decides what the answer looks like on the wire. Both
//! are transcribed from `tools/server/server-task.cpp` --
//! `:368-390` for the terminal object, `:1077-1099` for a partial,
//! `:244-265` for `timings`, `:279-286` for `stop_type` -- rather than
//! inferred from the OpenAI shape.

use axum::response::sse::Event;
use serde_json::{json, Value};

use crate::generate::{FinishReason, GenerationParams};

/// `stop_type`, llama.cpp's vocabulary
/// (`tools/server/server-task.cpp:279-286`).
///
/// `Cancelled` has no upstream counterpart -- none of `eos`, `word` or
/// `limit` is true of an interrupted answer, and `none` means "still
/// generating" -- so it gets its own value rather than being folded
/// into one that would read as a normal finish.
///
/// **A known inaccuracy, recorded rather than hidden.** A caller stop
/// string that is exactly ONE token in the model's vocabulary is caught
/// by layer 1 of [`crate::stop`], which matches on the token id before
/// detokenizing and does not carry back which string it was: that
/// arrives here as `FinishReason::Stop` and is reported as `"eos"`
/// where llama.cpp would say `"word"`. It is not local to this
/// function -- `resolve_stop_tokens` discards the id-to-string mapping,
/// so `/v1/messages` loses its `stop_sequence` attribution for the same
/// inputs -- and fixing it means giving `StopMatcher` that mapping.
/// Multi-token stop strings, which is every stop a `/completion` caller
/// normally sets, go through layer 2 and are reported correctly.
pub(super) fn stop_type(finish: &FinishReason) -> &'static str {
    match finish {
        FinishReason::Stop => "eos",
        FinishReason::StopSequence(_) => "word",
        FinishReason::Length => "limit",
        FinishReason::Cancelled => "cancelled",
    }
}

/// The stop string that ended this generation, or `""` as upstream
/// spells "nothing did".
pub(super) fn stopping_word(finish: &FinishReason) -> &str {
    match finish {
        FinishReason::StopSequence(word) => word,
        _ => "",
    }
}

/// llama.cpp's `timings` object, from frink's `Usage`.
///
/// The two agree on more than they disagree: `prompt_per_second` and
/// `predicted_per_second` are already llama.cpp's own names in
/// `frink_api::Usage`. A timing this server did not measure is `null`
/// rather than `0`, which would read as instantaneous.
pub(super) fn timings(usage: &frink_api::Usage) -> Value {
    let per_token = |ms: Option<f64>, n: usize| match (ms, n) {
        (Some(ms), n) if n > 0 => json!(ms / n as f64),
        _ => Value::Null,
    };
    json!({
        // Upstream's sentinel for "not known", which is exactly what
        // `cached_tokens: None` means: no prefix cache is configured.
        "cache_n": usage.cached_tokens.map(|n| n as i64).unwrap_or(-1),
        "prompt_n": usage.prompt_tokens,
        "prompt_ms": usage.prompt_eval_duration_ms,
        "prompt_per_token_ms": per_token(usage.prompt_eval_duration_ms, usage.prompt_tokens),
        "prompt_per_second": usage.prompt_per_second,
        "predicted_n": usage.completion_tokens,
        "predicted_ms": usage.generation_duration_ms,
        "predicted_per_token_ms":
            per_token(usage.generation_duration_ms, usage.completion_tokens),
        "predicted_per_second": usage.predicted_per_second,
    })
}

/// What upstream echoes back as `generation_settings`: the options as
/// this server actually resolved them, which is the point of the field
/// -- a caller compares it against what it sent.
pub(super) fn generation_settings(params: &GenerationParams, model: &str) -> Value {
    let s = &params.sampling;
    json!({
        "model": model,
        "n_predict": params.max_tokens,
        "seed": params.seed,
        "temperature": s.temperature,
        "top_p": s.top_p,
        "min_p": s.min_p,
        "top_k": s.top_k,
        "repeat_penalty": s.repetition_penalty,
        "repeat_last_n": s.penalty_last_n,
        "presence_penalty": s.presence_penalty,
        "frequency_penalty": s.frequency_penalty,
        "stop": params.stop,
        "ignore_eos": params.ignore_eos,
        "grammar": params.grammar.is_some(),
    })
}

/// The terminal object, shared by the buffered response and the last
/// frame of a stream -- upstream sends the same shape in both
/// (`server-task.cpp:368`), and two copies of it here would be two
/// answers to drift apart.
pub(super) fn final_body(
    content: &str,
    finish: &FinishReason,
    usage: &frink_api::Usage,
    params: &GenerationParams,
    model: &str,
    prompt: &str,
) -> Value {
    json!({
        "index": 0,
        "content": content,
        // `return_tokens` is refused, so this is always empty --
        // upstream's own behaviour when it is false.
        "tokens": Vec::<u32>::new(),
        // This server has no slots; -1 is upstream's "no slot".
        "id_slot": -1,
        "stop": true,
        "model": model,
        "tokens_predicted": usage.completion_tokens,
        "tokens_evaluated": usage.prompt_tokens,
        "generation_settings": generation_settings(params, model),
        "prompt": prompt,
        "has_new_line": content.contains('\n'),
        // Always false, and that is a statement rather than a
        // placeholder: frink refuses a request that does not fit its
        // context (`crate::budget`) instead of discarding tokens to
        // make it fit, so a served answer was never truncated.
        "truncated": false,
        "stop_type": stop_type(finish),
        "stopping_word": stopping_word(finish),
        "tokens_cached": usage.cached_tokens.unwrap_or(0),
        "timings": timings(usage),
    })
}

/// One streamed token, in the shape upstream documents for a partial:
/// "only `content`, `tokens` and `stop` will be returned until end of
/// completion" (`tools/server/README.md`).
pub(super) fn partial_body(content: &str) -> Value {
    json!({
        "index": 0,
        "content": content,
        "tokens": Vec::<u32>::new(),
        "stop": false,
        "id_slot": -1,
    })
}

pub(super) fn frame(body: &Value) -> Event {
    Event::default().data(body.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four `FinishReason`s, each in llama.cpp's own vocabulary --
    /// plus the one it has no word for, which must not be reported as
    /// a normal finish.
    #[test]
    fn stop_type_speaks_llama_cpps_vocabulary() {
        assert_eq!(stop_type(&FinishReason::Stop), "eos");
        assert_eq!(stop_type(&FinishReason::Length), "limit");
        assert_eq!(stop_type(&FinishReason::StopSequence("END".into())), "word");
        assert_eq!(stop_type(&FinishReason::Cancelled), "cancelled");
        assert_eq!(
            stopping_word(&FinishReason::StopSequence("END".into())),
            "END"
        );
        assert_eq!(stopping_word(&FinishReason::Stop), "");
    }

    /// A timing that was not measured is `null`, never `0`: zero
    /// milliseconds reads as an instantaneous prefill.
    #[test]
    fn an_unmeasured_timing_is_null_rather_than_zero() {
        let mut usage = frink_api::Usage::new(10, 5);
        let untimed = timings(&usage);
        assert!(untimed["prompt_ms"].is_null());
        assert!(untimed["prompt_per_token_ms"].is_null());
        assert!(untimed["predicted_per_second"].is_null());
        // And `cache_n` is upstream's -1 when no prefix cache exists,
        // which is a different statement from "the cache missed".
        assert_eq!(untimed["cache_n"], -1);
        assert_eq!(untimed["prompt_n"], 10);
        assert_eq!(untimed["predicted_n"], 5);

        usage.prompt_eval_duration_ms = Some(100.0);
        usage.generation_duration_ms = Some(50.0);
        usage.cached_tokens = Some(0);
        let timed = timings(&usage);
        assert_eq!(timed["prompt_per_token_ms"], 10.0);
        assert_eq!(timed["predicted_per_token_ms"], 10.0);
        assert_eq!(timed["cache_n"], 0);
    }

    /// A partial frame carries only what upstream documents it to
    /// carry, and says `stop: false`; a client watches that field to
    /// know the stream ended.
    #[test]
    fn a_partial_frame_is_the_documented_shape() {
        let body = partial_body("tok");
        assert_eq!(body["content"], "tok");
        assert_eq!(body["stop"], false);
        assert!(body["tokens"].as_array().unwrap().is_empty());
        // No `timings`, no `generation_settings`: those belong to the
        // terminal frame only.
        assert!(body.get("timings").is_none());
        assert!(body.get("generation_settings").is_none());
    }
}
