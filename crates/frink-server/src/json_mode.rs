//! Best-effort JSON-object response mode: mask logits to JSON-safe token
//! pieces during sampling, then validate the final text parses as a JSON
//! object. Not grammar-complete — see docs/API.md.

use axum::http::StatusCode;
use axum::Json;

use crate::ApiError;

/// Characters allowed in a decoded token piece when `json_object` mode is on.
fn json_safe_char(c: char) -> bool {
    matches!(
        c,
        '{' | '}'
            | '['
            | ']'
            | '"'
            | ':'
            | ','
            | '.'
            | '-'
            | '+'
            | ' '
            | '\t'
            | '\n'
            | '\r'
            | '\\'
            | '/'
    ) || c.is_ascii_digit()
        || c.is_ascii_alphabetic()
        || c == '_'
}

pub(crate) fn piece_is_json_safe(piece: &str) -> bool {
    piece.chars().all(json_safe_char)
}

/// Zero logits for vocab entries whose decoded piece is not JSON-safe.
pub(crate) fn mask_logits_for_json(logits: &mut [f32], decode_token: impl Fn(usize) -> String) {
    for (i, s) in logits.iter_mut().enumerate() {
        if !piece_is_json_safe(&decode_token(i)) {
            *s = f32::NEG_INFINITY;
        }
    }
}

pub(crate) fn validate_json_object_output(text: &str) -> Result<(), ApiError> {
    let trimmed = text.trim();
    let value: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "response_format json_object: model output is not valid JSON ({e})"
                    )
                }
            })),
        )
    })?;
    if !value.is_object() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "response_format json_object: model output must be a JSON object"
                }
            })),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_safe_chars_allow_object_skeleton() {
        assert!(piece_is_json_safe("{\"key\": 1}"));
        assert!(!piece_is_json_safe("<html>"));
    }
}
