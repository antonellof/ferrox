//! Who a request came from, as far as this server can honestly tell.
//!
//! The API Monitor's job is to make an external caller (someone's
//! editor, pointed at this server all day) distinguishable from the
//! UI's own traffic. Two facts are available without inventing
//! anything, and they are recorded separately because they are worth
//! exactly different amounts:
//!
//! - **Which key served the request.** Real, in the sense that it was
//!   presented and checked. Recorded as a *fingerprint*: never the key,
//!   never anything the key can be recovered from. See
//!   [`key_fingerprint`] for why it is salted per process.
//! - **What the caller says it is.** The `X-Frink-Client` header.
//!   Frink Studio sends `frink-studio`; so could anything else.
//!   Recorded as a claim and documented as one, because a label a
//!   client volunteers is still better evidence than a guess assembled
//!   from timings, and because the alternative -- inferring "this must
//!   have been the UI" from request shape -- would be a heuristic that
//!   is wrong silently.
//!
//! What is deliberately *not* here: the key itself. `docs/plans/`
//! records one reference product returning its server's plaintext key
//! in a stats payload, listed under "do not inherit". `/admin/stats`
//! is behind the same key it would leak, but a payload gets pasted into
//! bug reports, and a fingerprint pastes harmlessly.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::OnceLock;

use axum::http::{header::AUTHORIZATION, HeaderMap};

/// The header a client uses to name itself. Lowercase because that is
/// how HTTP/2 and `HeaderMap` both spell it.
pub(crate) const CLIENT_HEADER: &str = "x-frink-client";

/// Longest self-declared label kept. Long enough for `frink-studio` or
/// a version suffix, short enough that a hostile client cannot push a
/// kilobyte of text into 200 ring entries.
const MAX_CLIENT_LEN: usize = 32;

/// Per-process salt for [`key_fingerprint`].
///
/// `RandomState::new()` seeds itself from the OS, so this is a fresh
/// secret every start. That is the point: without it the fingerprint
/// would be a pure function of the key, and anyone holding a stats
/// payload could confirm a guessed key offline by hashing it. With it,
/// the fingerprint answers only the question the monitor actually asks
/// -- "same key as that other row?" -- and answers nothing across a
/// restart.
fn salt() -> &'static RandomState {
    static SALT: OnceLock<RandomState> = OnceLock::new();
    SALT.get_or_init(RandomState::new)
}

/// A short, stable, non-reversible name for one bearer key.
pub(crate) fn key_fingerprint(key: &str) -> String {
    let mut hasher = salt().build_hasher();
    hasher.write(key.as_bytes());
    format!("key-{:08x}", hasher.finish() as u32)
}

/// Keeps a self-declared label to plain label characters.
///
/// Not a security boundary -- this value is JSON, and the UI renders it
/// as text, never as markup -- but a monitor column is not the place
/// for control characters or a paragraph, so anything that is not
/// alphanumeric or one of `-_.:/ ` is dropped and the rest is capped.
fn sanitize_client(raw: &str) -> Option<String> {
    let cleaned: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/' | ' '))
        .take(MAX_CLIENT_LEN)
        .collect();
    let trimmed = cleaned.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// What one request says about its origin.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Attribution {
    /// Fingerprint of the presented bearer key; `None` when the request
    /// carried no `Authorization: Bearer` header.
    pub(crate) via_api_key: Option<String>,
    /// The `X-Frink-Client` label, sanitized. A claim.
    pub(crate) client: Option<String>,
}

impl Attribution {
    pub(crate) fn from_headers(headers: &HeaderMap) -> Self {
        let via_api_key = headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(key_fingerprint);
        let client = headers
            .get(CLIENT_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(sanitize_client);
        Attribution {
            via_api_key,
            client,
        }
    }
}

// Built from an `axum::http::HeaderMap` extractor at each handler
// rather than as a `FromRequestParts` implementation of its own. There
// is nothing to reject -- an unattributable request is a fact to
// record, not an error -- and a bespoke extractor would add a trait
// impl whose only behaviour is "read two headers, never fail". Auth
// itself is `limits::require_api_key`'s job and runs in a layer above
// every handler, so what reaches here is only ever "which accepted
// key" or "none was needed".

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    /// The whole point of the field: the same key is the same row
    /// label, a different key is a different one.
    #[test]
    fn the_same_key_fingerprints_the_same_and_a_different_key_does_not() {
        assert_eq!(key_fingerprint("sk-abc"), key_fingerprint("sk-abc"));
        assert_ne!(key_fingerprint("sk-abc"), key_fingerprint("sk-abd"));
    }

    /// A stats payload that carried the key would be a credential in a
    /// bug report. It must not contain it in any form a reader could
    /// undo.
    #[test]
    fn a_fingerprint_never_contains_the_key_it_names() {
        let key = "sk-super-secret-value";
        let fp = key_fingerprint(key);
        assert!(!fp.contains(key));
        assert!(!fp.contains("secret"));
        assert!(fp.starts_with("key-"));
        assert_eq!(fp.len(), "key-".len() + 8);
    }

    #[test]
    fn a_bearer_header_is_recorded_as_a_fingerprint_and_nothing_else() {
        let attr = Attribution::from_headers(&headers(&[("authorization", "Bearer sk-abc")]));
        assert_eq!(attr.via_api_key, Some(key_fingerprint("sk-abc")));
        assert_eq!(attr.client, None);
    }

    /// `null` means "no key was presented", which on a server started
    /// without FRINK_API_KEY is every request. It must not be confused
    /// with a key that fingerprinted to something.
    #[test]
    fn no_authorization_header_means_no_fingerprint() {
        assert_eq!(
            Attribution::from_headers(&HeaderMap::new()).via_api_key,
            None
        );
        let empty = Attribution::from_headers(&headers(&[("authorization", "Bearer   ")]));
        assert_eq!(empty.via_api_key, None, "an empty key is not a key");
        let basic = Attribution::from_headers(&headers(&[("authorization", "Basic abc")]));
        assert_eq!(
            basic.via_api_key, None,
            "only Bearer is this server's scheme"
        );
    }

    #[test]
    fn a_client_label_is_kept_verbatim_when_it_is_already_a_label() {
        let attr = Attribution::from_headers(&headers(&[("x-frink-client", "frink-studio")]));
        assert_eq!(attr.client.as_deref(), Some("frink-studio"));
    }

    #[test]
    fn a_hostile_client_label_is_cut_down_to_a_label() {
        let attr = Attribution::from_headers(&headers(&[(
            "x-frink-client",
            "<img src=x onerror=alert(1)>",
        )]));
        let client = attr.client.expect("something survives");
        assert!(!client.contains('<'), "{client}");
        assert!(!client.contains('>'), "{client}");
        assert!(!client.contains('='), "{client}");
        assert!(!client.contains('('), "{client}");

        let long = "a".repeat(4096);
        let attr = Attribution::from_headers(&headers(&[("x-frink-client", &long)]));
        assert_eq!(attr.client.map(|c| c.len()), Some(MAX_CLIENT_LEN));
    }

    #[test]
    fn a_label_that_is_only_junk_is_absent_rather_than_empty() {
        let attr = Attribution::from_headers(&headers(&[("x-frink-client", "<<<>>>")]));
        assert_eq!(attr.client, None);
        let attr = Attribution::from_headers(&headers(&[("x-frink-client", "   ")]));
        assert_eq!(attr.client, None);
    }
}
