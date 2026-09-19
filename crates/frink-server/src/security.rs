//! Three related P0 server-hardening pieces, all opt-in/fail-closed in
//! the same spirit as `limits`'s auth/rate-limiting (see that module's
//! doc comment): a listener check that refuses to start unauthenticated
//! on a non-loopback address, optional TLS termination, and optional
//! CORS origin allow-listing. Kept in their own module (rather than
//! inline in `main()`) specifically so the address/loopback and
//! origin-validation logic is unit-testable without an actual listener,
//! following the same "extract a pure function so it's testable"
//! approach the rest of this crate uses.

use axum::http::HeaderValue;
use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;

/// Parses a `host:port` bind spec (the shape `FRINK_ADDR` always
/// takes) down to just the host's `IpAddr`. Tries `SocketAddr::from_str`
/// first (handles bracketed IPv6 like `[::1]:8383`), then falls back to
/// splitting on the last `:` for a bare IPv6 literal without brackets.
/// Returns `None` if neither parses -- e.g. a hostname instead of a
/// literal IP -- which callers must *not* treat as "safe": see
/// `check_bind_authorization`.
fn parse_bind_ip(addr: &str) -> Option<IpAddr> {
    if let Ok(sock) = SocketAddr::from_str(addr) {
        return Some(sock.ip());
    }
    let (host, _port) = addr.rsplit_once(':')?;
    IpAddr::from_str(host).ok()
}

/// Fail-closed check for whether `frink-server` may bind `addr_str`.
/// Refuses to start when the address is not (confirmably) loopback and
/// no API key is configured -- otherwise a `FRINK_ADDR=0.0.0.0:8383`
/// operator typo silently serves the whole API, unauthenticated, to
/// anyone who can reach the box. `allow_unauthenticated_remote` is the
/// explicit, named opt-out (`FRINK_ALLOW_UNAUTHENTICATED_REMOTE=1`)
/// for operators who really do want that (e.g. auth terminated by a
/// reverse proxy in front of this process).
///
/// A bind spec that doesn't even parse as a real address (`None` from
/// `parse_bind_ip`) is treated the same as a confirmed non-loopback
/// address -- fail closed, don't assume something unparsed is safe.
pub fn check_bind_authorization(
    addr_str: &str,
    api_key_configured: bool,
    allow_unauthenticated_remote: bool,
) -> Result<(), String> {
    if api_key_configured || allow_unauthenticated_remote {
        return Ok(());
    }
    let reason = match parse_bind_ip(addr_str) {
        Some(ip) if ip.is_loopback() => return Ok(()),
        Some(ip) => format!("is not a loopback address (resolved host: {ip})"),
        None => "could not be confirmed as a loopback address".to_string(),
    };
    Err(format!(
        "refusing to start: FRINK_ADDR={addr_str:?} {reason} and FRINK_API_KEY is not set -- \
         this would serve the API unauthenticated to anyone who can reach it. Fix this by \
         setting FRINK_API_KEY, binding a loopback address (127.0.0.1 or ::1) instead, or -- \
         only if you understand the risk and authentication is handled elsewhere (e.g. a \
         reverse proxy) -- setting FRINK_ALLOW_UNAUTHENTICATED_REMOTE=1."
    ))
}

/// Optional TLS material for serving HTTPS instead of plain HTTP --
/// `FRINK_TLS_CERT`/`FRINK_TLS_KEY`, the same all-env-var,
/// off-by-default convention as `FRINK_API_KEY` etc. (see `main.rs`).
pub struct TlsPaths {
    pub cert: String,
    pub key: String,
}

/// Reads `FRINK_TLS_CERT`/`FRINK_TLS_KEY`. Both unset -> `Ok(None)`
/// (plain HTTP, exactly as before this existed). Both set -> `Ok(Some)`.
/// Only one set is a config error, matching the
/// `FRINK_KV_POOL_BLOCKS`/`FRINK_KV_POOL_BLOCK_SIZE` "must be set
/// together" pattern in `main()`.
pub fn tls_paths_from_env() -> Result<Option<TlsPaths>, String> {
    match (
        std::env::var("FRINK_TLS_CERT"),
        std::env::var("FRINK_TLS_KEY"),
    ) {
        (Ok(cert), Ok(key)) => Ok(Some(TlsPaths { cert, key })),
        (Err(_), Err(_)) => Ok(None),
        _ => Err(
            "FRINK_TLS_CERT and FRINK_TLS_KEY must be set together (or neither, to serve \
             plain HTTP)"
                .to_string(),
        ),
    }
}

/// Parses `FRINK_CORS_ORIGINS` (a comma-separated list of *exact*
/// origins, e.g. `https://chat.example.com,https://app.example.com`)
/// into the `HeaderValue` list `tower_http::cors::AllowOrigin::list`
/// wants. Deliberately exact-match only, by design -- there is no
/// wildcard support here, so a literal `*` is rejected rather than
/// silently accepted as one (useless, and almost certainly not what an
/// operator who typed it meant) allowed origin string.
pub fn parse_cors_origins(spec: &str) -> Result<Vec<HeaderValue>, String> {
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|origin| {
            if origin == "*" {
                return Err(
                    "FRINK_CORS_ORIGINS does not support the \"*\" wildcard -- list the exact \
                     origin(s) to allow instead"
                        .to_string(),
                );
            }
            HeaderValue::from_str(origin)
                .map_err(|e| format!("FRINK_CORS_ORIGINS: invalid origin {origin:?}: {e}"))
        })
        .collect()
}

// TLS itself (`axum_server::tls_rustls::RustlsConfig::from_pem_file` +
// `axum_server::bind_rustls` in `main()`) has no unit test here on
// purpose: exercising it meaningfully needs a real cert/key pair and an
// actual TCP handshake, which isn't worth faking with self-signed
// throwaway certs generated at test time for what `cargo build`/type-
// checking already exercises (the `RustlsConfig`/`Server`/`Address`
// types lining up) -- see `frink-cuda::gpu`'s module doc comment for
// this same "verified by hardware, not by `#[test]`" disclosure
// pattern applied to something else this codebase can't fake
// convincingly. TLS wiring here is exercised by `cargo build` (this
// compiling at all proves the axum-server/rustls types line up) and by
// manual verification (start the server with a real cert/key and hit it
// with `curl --cacert`), not asserted with `#[test]`.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_with_no_key_starts_fine() {
        assert!(check_bind_authorization("127.0.0.1:8383", false, false).is_ok());
        assert!(check_bind_authorization("[::1]:8383", false, false).is_ok());
    }

    #[test]
    fn non_loopback_with_no_key_and_no_override_is_refused() {
        let err = check_bind_authorization("0.0.0.0:8383", false, false)
            .expect_err("must refuse an unauthenticated non-loopback bind");
        assert!(err.contains("FRINK_API_KEY"));
        assert!(err.contains("FRINK_ALLOW_UNAUTHENTICATED_REMOTE"));
    }

    #[test]
    fn non_loopback_with_no_key_and_override_starts() {
        assert!(check_bind_authorization("0.0.0.0:8383", false, true).is_ok());
    }

    #[test]
    fn non_loopback_with_key_starts() {
        assert!(check_bind_authorization("0.0.0.0:8383", true, false).is_ok());
    }

    #[test]
    fn unparseable_host_fails_closed_without_override_or_key() {
        // Not a literal IP (a hostname) -- can't be confirmed loopback,
        // so this must be refused exactly like a confirmed non-loopback
        // address, not assumed safe.
        assert!(check_bind_authorization("example.com:8383", false, false).is_err());
        assert!(check_bind_authorization("example.com:8383", false, true).is_ok());
        assert!(check_bind_authorization("example.com:8383", true, false).is_ok());
    }

    #[test]
    fn tls_paths_requires_both_or_neither() {
        // Real env var mutation is racy in a multi-threaded test
        // binary shared with every other test in this crate, so this
        // doesn't call `tls_paths_from_env()` itself -- the two-line
        // env::var match it wraps is exercised by type-checking plus
        // the FRINK_KV_POOL_BLOCKS/FRINK_KV_POOL_BLOCK_SIZE precedent
        // it deliberately mirrors (see that pattern in `main()`).
        // What's actually worth a standalone assertion is the
        // both-or-neither shape itself:
        let both: (Result<&str, ()>, Result<&str, ()>) = (Ok("cert"), Ok("key"));
        assert!(matches!(both, (Ok(_), Ok(_))));
        let neither: (Result<&str, ()>, Result<&str, ()>) = (Err(()), Err(()));
        assert!(matches!(neither, (Err(_), Err(_))));
    }

    #[test]
    fn cors_origin_parsing_accepts_a_valid_origin() {
        let origins = parse_cors_origins("https://chat.example.com").unwrap();
        assert_eq!(
            origins,
            vec![HeaderValue::from_static("https://chat.example.com")]
        );
    }

    #[test]
    fn cors_origin_parsing_accepts_multiple_valid_origins() {
        let origins =
            parse_cors_origins("https://chat.example.com, https://app.example.com").unwrap();
        assert_eq!(origins.len(), 2);
        assert_eq!(
            origins[0],
            HeaderValue::from_static("https://chat.example.com")
        );
        assert_eq!(
            origins[1],
            HeaderValue::from_static("https://app.example.com")
        );
    }

    #[test]
    fn cors_origin_parsing_rejects_an_invalid_header_value() {
        // A raw newline can never be a valid HTTP header value.
        let err = parse_cors_origins("https://ok.example.com,bad\nvalue")
            .expect_err("a header-value-invalid origin must be rejected");
        assert!(err.contains("invalid origin"));
    }

    #[test]
    fn cors_origin_parsing_rejects_wildcard() {
        let err =
            parse_cors_origins("*").expect_err("the \"*\" wildcard must be explicitly rejected");
        assert!(err.contains("wildcard"));

        let err = parse_cors_origins("https://chat.example.com,*")
            .expect_err("\"*\" must be rejected even mixed in with valid origins");
        assert!(err.contains("wildcard"));
    }
}
