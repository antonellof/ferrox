//! Optional bearer-token auth and a simple rate limiter for
//! frink-server. Both are off by default -- unset `FRINK_API_KEY` /
//! `FRINK_RATE_LIMIT_PER_MINUTE` and the server behaves exactly as
//! before -- following the same opt-in convention llama.cpp's server
//! uses for `--api-key`, so existing deployments and tests aren't
//! affected unless explicitly configured.

use axum::{
    body::Body,
    extract::State,
    http::{
        header::{AUTHORIZATION, RETRY_AFTER},
        HeaderValue, Request, StatusCode,
    },
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone)]
pub struct AuthConfig {
    pub api_key: Arc<String>,
}

/// The key a request presents, by either accepted spelling.
///
/// `Authorization: Bearer` is the OpenAI convention and was the only
/// one accepted. But this server also serves `/v1/messages`, and the
/// Anthropic SDKs send `x-api-key` instead, so a stock Anthropic client
/// pointed at a keyed deployment got a 401 no matter what key it held.
/// Both are read here; the Authorization header wins when a client
/// sends both, because it is the more specific request.
fn presented_key(headers: &axum::http::HeaderMap) -> Option<&str> {
    headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
}

/// Rejects any request that does not present the configured key, as
/// either `Authorization: Bearer <key>` or `x-api-key: <key>`. Only
/// ever added to the router when `FRINK_API_KEY` is set (see
/// `main.rs`) -- there is no "disabled" state to check here, keeping
/// the common unauthenticated path free of any per-request branch.
pub async fn require_api_key(
    State(config): State<AuthConfig>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let provided = presented_key(req.headers());

    if provided == Some(config.api_key.as_str()) {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": {"message": "invalid or missing API key"}})),
        )
            .into_response()
    }
}

struct Bucket {
    tokens: f64,
    capacity: f64,
    refill_per_sec: f64,
    last_refill: Instant,
}

/// A simple token-bucket rate limiter, shared across all requests
/// (global, not per-client): `capacity` tokens, continuously refilled
/// at `refill_per_sec` tokens/second, one token consumed per request
/// that passes through it. A reasonable minimum for a single-model
/// self-hosted deployment; a real multi-tenant deployment would want
/// per-API-key buckets instead of one global bucket, a possible
/// follow-on.
pub struct RateLimiter {
    inner: Mutex<Bucket>,
}

impl RateLimiter {
    pub fn new(capacity: u32, refill_per_sec: f64) -> Self {
        RateLimiter {
            inner: Mutex::new(Bucket {
                tokens: capacity as f64,
                capacity: capacity as f64,
                refill_per_sec,
                last_refill: Instant::now(),
            }),
        }
    }

    pub fn per_minute(requests_per_minute: u32) -> Self {
        // Bucket capacity equals a full minute's allowance, so a burst
        // after idle time isn't punished, refilled continuously at the
        // equivalent per-second rate.
        Self::new(requests_per_minute, requests_per_minute as f64 / 60.0)
    }

    /// Attempts to consume one token. Returns `false` (and consumes
    /// nothing) if none are available right now.
    pub fn try_acquire(&self) -> bool {
        let mut b = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let now = Instant::now();
        let elapsed = now.duration_since(b.last_refill).as_secs_f64();
        b.tokens = (b.tokens + elapsed * b.refill_per_sec).min(b.capacity);
        b.last_refill = now;
        if b.tokens >= 1.0 {
            b.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

pub async fn rate_limit(
    State(limiter): State<Arc<RateLimiter>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if limiter.try_acquire() {
        next.run(req).await
    } else {
        (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({"error": {"message": "rate limit exceeded"}})),
        )
            .into_response()
    }
}

/// Seconds a client is told to wait before retrying a 503.
///
/// A fixed short hint rather than a computed one: the honest estimate
/// would be "queue depth divided by throughput", and this server does
/// not know a caller's throughput. One second is short enough that a
/// legitimate client recovers quickly and long enough that a retry
/// storm loses its coordination.
pub const RETRY_AFTER_SECONDS: u64 = 1;

/// Stamps `Retry-After` on any 503 that does not already carry one.
///
/// Lives in a layer rather than at each rejection site because the
/// header is a property of the *status*, not of any one handler: 503
/// means "temporarily unavailable" (RFC 9110), so every 503 this server
/// emits -- queue full, KV pool exhausted, no model loaded -- is by
/// definition worth retrying, and a client that has to guess how long
/// to wait will guess "immediately". An existing header is never
/// overwritten, so a handler that knows better keeps its own value.
pub async fn retry_after(req: Request<Body>, next: Next) -> Response {
    let mut response = next.run(req).await;
    if response.status() == StatusCode::SERVICE_UNAVAILABLE
        && !response.headers().contains_key(RETRY_AFTER)
    {
        response
            .headers_mut()
            .insert(RETRY_AFTER, HeaderValue::from(RETRY_AFTER_SECONDS));
    }
    response
}

#[cfg(test)]
mod tests {
    use super::presented_key;

    fn headers(pairs: &[(&str, &str)]) -> axum::http::HeaderMap {
        let mut map = axum::http::HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                value.parse().unwrap(),
            );
        }
        map
    }

    /// An Anthropic SDK sends `x-api-key`, and this server answers
    /// `/v1/messages`, so refusing that header made the Anthropic
    /// surface unreachable on any deployment that set a key.
    #[test]
    fn both_the_openai_and_the_anthropic_spellings_are_accepted() {
        assert_eq!(
            presented_key(&headers(&[("authorization", "Bearer sk-abc")])),
            Some("sk-abc")
        );
        assert_eq!(
            presented_key(&headers(&[("x-api-key", "sk-abc")])),
            Some("sk-abc"),
            "a stock Anthropic client sends only this header"
        );
        assert_eq!(presented_key(&headers(&[])), None);
        assert_eq!(
            presented_key(&headers(&[("authorization", "sk-abc")])),
            None,
            "Bearer is still required when the Authorization header is used"
        );
        assert_eq!(
            presented_key(&headers(&[
                ("authorization", "Bearer from-auth"),
                ("x-api-key", "from-x"),
            ])),
            Some("from-auth"),
            "the more specific header wins rather than the lookup order deciding"
        );
    }

    use super::*;
    use axum::{routing::get, Router};
    use tower::ServiceExt;

    async fn call(app: Router, path: &str) -> Response {
        app.oneshot(
            Request::builder()
                .uri(path)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response")
    }

    fn retry_after_router() -> Router {
        Router::new()
            .route("/busy", get(|| async { StatusCode::SERVICE_UNAVAILABLE }))
            .route("/fine", get(|| async { StatusCode::OK }))
            .route(
                "/busy-with-hint",
                get(|| async { ([(RETRY_AFTER, "30")], StatusCode::SERVICE_UNAVAILABLE) }),
            )
            .layer(axum::middleware::from_fn(retry_after))
    }

    /// A 503 without a retry hint tells a client nothing except "try
    /// again", which clients read as "try again now" -- the retry storm
    /// the queue cap exists to survive.
    #[tokio::test]
    async fn a_503_gets_a_retry_after_header() {
        let response = call(retry_after_router(), "/busy").await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers().get(RETRY_AFTER).expect("header present"),
            &HeaderValue::from(RETRY_AFTER_SECONDS)
        );
    }

    #[tokio::test]
    async fn a_successful_response_is_left_alone() {
        let response = call(retry_after_router(), "/fine").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get(RETRY_AFTER).is_none());
    }

    #[tokio::test]
    async fn an_existing_retry_after_is_not_overwritten() {
        let response = call(retry_after_router(), "/busy-with-hint").await;
        assert_eq!(
            response.headers().get(RETRY_AFTER).expect("header present"),
            "30",
            "a handler that knows a better value keeps it"
        );
    }

    #[test]
    fn rate_limiter_allows_up_to_capacity_then_blocks() {
        let limiter = RateLimiter::new(3, 0.0); // no refill, isolate the capacity check
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(
            !limiter.try_acquire(),
            "a 4th request within the same instant must be rejected"
        );
    }

    #[test]
    fn rate_limiter_refills_over_time() {
        let limiter = RateLimiter::new(1, 1000.0); // fast refill for a quick test
        assert!(limiter.try_acquire());
        assert!(!limiter.try_acquire());
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(
            limiter.try_acquire(),
            "tokens must refill over time, not stay exhausted forever"
        );
    }

    #[test]
    fn per_minute_constructor_sets_a_full_minute_of_burst_capacity() {
        let limiter = RateLimiter::per_minute(60);
        for _ in 0..60 {
            assert!(limiter.try_acquire());
        }
        assert!(!limiter.try_acquire());
    }
}
