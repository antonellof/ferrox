//! Two-tier cancellation for in-flight generations.
//!
//! A dropped TCP connection does not reliably stop a decode loop. The
//! socket may take a long time to be noticed, a reverse proxy may
//! swallow the client's abort entirely, and until this module existed
//! `frink-server` ignored the failure of every `blocking_send` on the
//! SSE channel -- so a browser tab closed mid-answer left a CPU core
//! generating tokens into a receiver nobody held.
//!
//! Hence two tiers, both ending at the same flag:
//!
//! 1. **The socket.** The streaming handler now checks whether the SSE
//!    receiver is still there. Once it is gone, the flag is set and the
//!    decode loop stops at its next token.
//! 2. **An explicit request.** `POST /v1/cancel` with the `request_id`
//!    the server already stated on the first chunk. A browser can send
//!    it with `keepalive: true` so it survives the page unload that
//!    killed the stream, which is the case tier 1 is worst at.
//!
//! Cancellation is cooperative, and the honesty rule from the task
//! contract applies here too: the endpoint reports whether a *live*
//! generation was signalled, never a blanket success. An id that has
//! already finished, or was never issued, answers `404` -- "there is
//! nothing to stop" is a different fact from "stopping".
//!
//! What it cannot interrupt: a prefill already inside a batched forward
//! pass -- the flag is read between decoded tokens, so a long prompt is
//! still charged before the first check.
//!
//! A continuous-batching request decodes on the shared batcher thread
//! rather than through the polling loop, so it is reached a third way:
//! [`CancelToken::on_cancel`] hands the scheduler a callback, and the
//! callback only *enqueues an id*. The scheduler drains that queue at a
//! step boundary and does the batch mutation itself. That indirection
//! is not ceremony -- removing a sequence means slicing KV buffers that
//! a forward pass may be reading, so the mutation has to happen on the
//! thread that owns the step, between steps, and never from whichever
//! HTTP handler happened to receive the cancel.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A callback run once, when a token is cancelled.
type CancelHook = Box<dyn FnOnce() + Send>;

/// The shared stop flag one generation reads and any number of
/// cancellers may set.
///
/// Cheap to clone (one `Arc`) and cheap to poll (one relaxed atomic
/// load per decoded token), which is what lets the check sit in the
/// innermost loop without being measurable.
#[derive(Clone, Default)]
pub(crate) struct CancelToken(Arc<TokenInner>);

#[derive(Default)]
struct TokenInner {
    cancelled: AtomicBool,
    /// Callbacks to run on cancellation, taken (not merely read) so
    /// each runs at most once however many times `cancel` is called.
    hooks: Mutex<Vec<CancelHook>>,
}

impl CancelToken {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Ask the generation to stop at its next token.
    ///
    /// Idempotent: repeat cancels are the same fact stated twice, not a
    /// state machine, and the hooks still run exactly once.
    pub(crate) fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::Relaxed);
        let hooks = std::mem::take(&mut *self.0.hooks.lock().unwrap_or_else(|p| p.into_inner()));
        for hook in hooks {
            hook();
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Relaxed)
    }

    /// Runs `hook` when this token is cancelled -- or immediately, if
    /// it already has been.
    ///
    /// The immediate case is the whole reason this is not just "check
    /// the flag later": a cancel that lands between a request being
    /// submitted and its hook being registered would otherwise be lost,
    /// and the request would run to completion after the client asked
    /// it to stop.
    ///
    /// A hook runs on whichever thread called `cancel` -- an HTTP
    /// handler, usually -- so it must be cheap and must not touch
    /// engine state. Enqueueing an id is exactly the right amount of
    /// work for one.
    pub(crate) fn on_cancel(&self, hook: impl FnOnce() + Send + 'static) {
        if self.is_cancelled() {
            hook();
            return;
        }
        let mut hooks = self.0.hooks.lock().unwrap_or_else(|p| p.into_inner());
        // Re-check under the lock: `cancel` takes the vector, so a hook
        // pushed after that take would never run.
        if self.is_cancelled() {
            drop(hooks);
            hook();
            return;
        }
        hooks.push(Box::new(hook));
    }
}

/// The live generations that can currently be cancelled by id.
///
/// Registration is scoped by [`CancelGuard`] rather than by the
/// handler remembering to clean up: a panicking decode thread must not
/// leave an id in here forever, because a later request could not reuse
/// it and `/v1/cancel` would answer `200` for something that is not
/// running.
#[derive(Default)]
pub(crate) struct CancelRegistry {
    live: Mutex<HashMap<String, CancelToken>>,
}

impl CancelRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Registers `request_id` and hands back the token its decode loop
    /// should poll. The registration lasts as long as the returned
    /// guard.
    pub(crate) fn register(self: &Arc<Self>, request_id: &str) -> (CancelToken, CancelGuard) {
        let token = CancelToken::new();
        self.live
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(request_id.to_string(), token.clone());
        (
            token,
            CancelGuard {
                registry: Arc::clone(self),
                request_id: request_id.to_string(),
            },
        )
    }

    /// Signals the generation behind `request_id`.
    ///
    /// Returns whether one was live. A repeat cancel of a still-running
    /// generation is `true` again -- it is idempotent, not a state
    /// machine -- but an id that has already finished is `false`.
    pub(crate) fn cancel(&self, request_id: &str) -> bool {
        match self
            .live
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(request_id)
        {
            Some(token) => {
                token.cancel();
                true
            }
            None => false,
        }
    }

    /// How many generations could be cancelled right now. Surfaced as
    /// `generating_now` on `/admin/stats`.
    pub(crate) fn live_count(&self) -> usize {
        self.live.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    fn deregister(&self, request_id: &str) {
        self.live
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(request_id);
    }
}

/// Removes a generation from the registry when it ends, however it
/// ends -- including by panic.
pub(crate) struct CancelGuard {
    registry: Arc<CancelRegistry>,
    request_id: String,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.registry.deregister(&self.request_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_registered_generation_can_be_cancelled_by_id() {
        let registry = Arc::new(CancelRegistry::new());
        let (token, _guard) = registry.register("chatcmpl-1");
        assert!(!token.is_cancelled());
        assert!(registry.cancel("chatcmpl-1"));
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancelling_an_unknown_id_reports_that_nothing_was_running() {
        let registry = Arc::new(CancelRegistry::new());
        // The distinction the endpoint's 404 rests on: a client that
        // cancels twice, or cancels after the answer arrived, must be
        // able to tell "stopped it" from "there was nothing to stop".
        assert!(!registry.cancel("chatcmpl-never-issued"));
    }

    #[test]
    fn the_registration_ends_with_the_generation() {
        let registry = Arc::new(CancelRegistry::new());
        {
            let (_token, _guard) = registry.register("chatcmpl-2");
            assert_eq!(registry.live_count(), 1);
        }
        assert_eq!(registry.live_count(), 0);
        assert!(!registry.cancel("chatcmpl-2"));
    }

    #[test]
    fn a_panicking_generation_does_not_leak_its_id() {
        let registry = Arc::new(CancelRegistry::new());
        let for_thread = Arc::clone(&registry);
        let handle = std::thread::spawn(move || {
            let (_token, _guard) = for_thread.register("chatcmpl-3");
            panic!("decode thread died");
        });
        assert!(handle.join().is_err());
        assert_eq!(registry.live_count(), 0);
    }

    #[test]
    fn a_hook_runs_when_the_token_is_cancelled() {
        let token = CancelToken::new();
        let fired = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&fired);
        token.on_cancel(move || flag.store(true, Ordering::Relaxed));
        assert!(!fired.load(Ordering::Relaxed));
        token.cancel();
        assert!(fired.load(Ordering::Relaxed));
    }

    /// The race the immediate-fire path exists for: a cancel that lands
    /// before the scheduler has registered its hook must not be lost,
    /// or the request runs to completion after the client asked it to
    /// stop.
    #[test]
    fn a_hook_registered_after_the_cancel_still_runs() {
        let token = CancelToken::new();
        token.cancel();
        let fired = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&fired);
        token.on_cancel(move || flag.store(true, Ordering::Relaxed));
        assert!(
            fired.load(Ordering::Relaxed),
            "a hook on an already-cancelled token must fire at once"
        );
    }

    /// Cancellation is idempotent, and so are its hooks: a client that
    /// cancels twice must not make the scheduler act twice.
    #[test]
    fn a_hook_runs_at_most_once_however_often_cancel_is_called() {
        let token = CancelToken::new();
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&count);
        token.on_cancel(move || {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        token.cancel();
        token.cancel();
        token.cancel();
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_hook_on_a_registry_token_fires_through_the_endpoint() {
        let registry = Arc::new(CancelRegistry::new());
        let (token, _guard) = registry.register("chatcmpl-hooked");
        let fired = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&fired);
        token.on_cancel(move || flag.store(true, Ordering::Relaxed));
        assert!(registry.cancel("chatcmpl-hooked"));
        assert!(fired.load(Ordering::Relaxed));
    }

    #[test]
    fn concurrent_generations_are_cancelled_independently() {
        let registry = Arc::new(CancelRegistry::new());
        let (first, _g1) = registry.register("chatcmpl-a");
        let (second, _g2) = registry.register("chatcmpl-b");
        assert!(registry.cancel("chatcmpl-a"));
        assert!(first.is_cancelled());
        assert!(
            !second.is_cancelled(),
            "cancelling one chat must not stop another"
        );
    }
}
