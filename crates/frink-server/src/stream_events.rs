//! The two rules the Responses and Anthropic streams share, held in
//! one place.
//!
//! Both protocols carry their own `GenEvent`, and they genuinely
//! differ: Anthropic's `Done` reports the client's own matched stop
//! sequence and its `Failed` carries a `StatusCode`, where the
//! Responses one carries a static error code. So the thing to share is
//! not the type -- it is the two rules that were written twice against
//! it, and that a reader would have to diff two files to confirm still
//! agree.
//!
//! # The rule that matters: an empty piece is not an event
//!
//! [`map_tool_event`] drops an empty text run and an empty argument
//! fragment instead of forwarding them. A parser emits both routinely
//! at boundaries -- the text before a tool call opener is empty when
//! the call starts the message, and the first argument fragment is
//! empty when the opener and the first argument land in different
//! chunks. Forwarding them puts a zero-length delta frame on the wire,
//! which is not merely noise: a client that treats a content delta as
//! "the model is producing text" starts a text block for a message
//! that has none. Two copies of that rule is two chances for one
//! protocol to keep it and the other to lose it.
//!
//! # The keepalive, and why the receiver rides in the state
//!
//! [`with_keepalive`] turns a silent generator into a stream that still
//! emits, so an idle proxy does not close a connection that is only
//! waiting on a long prefill. The receiver lives in the `unfold`
//! state rather than being captured by the closure, because a future
//! the closure returns may not borrow it.

use std::time::Duration;

use futures_util::Stream;

/// What a protocol's own event enum must be able to construct for the
/// shared rules to build it.
///
/// Deliberately only the variants both protocols share. A trait wide
/// enough to cover `Done` would have to model the two protocols'
/// different terminal payloads, which is the part that is genuinely
/// not shared.
pub(crate) trait StreamEvent: Send + 'static {
    /// Injected by [`with_keepalive`], never by a generator.
    fn keepalive() -> Self;
    fn content(text: String) -> Self;
    fn call_start(index: usize, name: String) -> Self;
    fn call_arguments(index: usize, fragment: String) -> Self;
    fn call_end(index: usize, arguments: String) -> Self;
}

/// Maps one parser event to zero or more protocol events.
///
/// Zero is the interesting case -- see the module docs on why an empty
/// text run or argument fragment must not become a frame.
pub(crate) fn map_tool_event<E: StreamEvent>(
    event: crate::policy::parser::ToolCallEvent,
) -> Vec<E> {
    match event {
        crate::policy::parser::ToolCallEvent::Text(text) if text.is_empty() => Vec::new(),
        crate::policy::parser::ToolCallEvent::Text(text) => vec![E::content(text)],
        crate::policy::parser::ToolCallEvent::CallStart { index, name } => {
            vec![E::call_start(index, name)]
        }
        crate::policy::parser::ToolCallEvent::CallArguments { fragment, .. }
            if fragment.is_empty() =>
        {
            Vec::new()
        }
        crate::policy::parser::ToolCallEvent::CallArguments { index, fragment } => {
            vec![E::call_arguments(index, fragment)]
        }
        crate::policy::parser::ToolCallEvent::CallEnd { index, arguments } => {
            vec![E::call_end(index, arguments)]
        }
    }
}

/// Emits a keepalive whenever the generator has produced nothing for
/// `interval`.
pub(crate) fn with_keepalive<E: StreamEvent>(
    events: tokio::sync::mpsc::Receiver<E>,
    interval: Duration,
) -> impl Stream<Item = E> {
    // The receiver rides in the unfold's *state* rather than in the
    // closure: a future returned by the closure may not borrow it.
    futures_util::stream::unfold(events, move |mut events| async move {
        match tokio::time::timeout(interval, events.recv()).await {
            Err(_elapsed) => Some((E::keepalive(), events)),
            Ok(Some(event)) => Some((event, events)),
            Ok(None) => None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal event type standing in for both protocols' enums, so
    /// the shared rules are asserted once rather than twice.
    #[derive(Debug, PartialEq)]
    enum Ev {
        Keepalive,
        Content(String),
        CallStart(usize, String),
        CallArguments(usize, String),
        CallEnd(usize, String),
    }

    impl StreamEvent for Ev {
        fn keepalive() -> Self {
            Ev::Keepalive
        }
        fn content(text: String) -> Self {
            Ev::Content(text)
        }
        fn call_start(index: usize, name: String) -> Self {
            Ev::CallStart(index, name)
        }
        fn call_arguments(index: usize, fragment: String) -> Self {
            Ev::CallArguments(index, fragment)
        }
        fn call_end(index: usize, arguments: String) -> Self {
            Ev::CallEnd(index, arguments)
        }
    }

    /// The rule both protocols depend on: an empty text run and an
    /// empty argument fragment produce NO event. A parser emits both at
    /// boundaries, and a zero-length content delta tells a client the
    /// model is producing text when it is not.
    #[test]
    fn an_empty_text_run_or_argument_fragment_produces_no_event() {
        let none: Vec<Ev> =
            map_tool_event(crate::policy::parser::ToolCallEvent::Text(String::new()));
        assert!(none.is_empty(), "an empty text run is not an event");

        let none: Vec<Ev> = map_tool_event(crate::policy::parser::ToolCallEvent::CallArguments {
            index: 0,
            fragment: String::new(),
        });
        assert!(none.is_empty(), "an empty fragment is not an event");
    }

    /// Everything non-empty maps one-to-one, carrying its index.
    #[test]
    fn every_non_empty_parser_event_maps_to_exactly_one_protocol_event() {
        assert_eq!(
            map_tool_event::<Ev>(crate::policy::parser::ToolCallEvent::Text("hi".into())),
            vec![Ev::Content("hi".into())]
        );
        assert_eq!(
            map_tool_event::<Ev>(crate::policy::parser::ToolCallEvent::CallStart {
                index: 2,
                name: "search".into()
            }),
            vec![Ev::CallStart(2, "search".into())]
        );
        assert_eq!(
            map_tool_event::<Ev>(crate::policy::parser::ToolCallEvent::CallArguments {
                index: 2,
                fragment: "{\"q\"".into()
            }),
            vec![Ev::CallArguments(2, "{\"q\"".into())]
        );
        assert_eq!(
            map_tool_event::<Ev>(crate::policy::parser::ToolCallEvent::CallEnd {
                index: 2,
                arguments: "{}".into()
            }),
            vec![Ev::CallEnd(2, "{}".into())]
        );
    }

    /// A silent generator still produces frames, and a live one is
    /// passed through untouched rather than interleaved with
    /// keepalives it did not earn.
    #[tokio::test]
    async fn a_silent_generator_still_emits_and_a_live_one_is_untouched() {
        use futures_util::StreamExt;

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let mut stream = Box::pin(with_keepalive(rx, Duration::from_millis(10)));
        tx.send(Ev::Content("a".into())).await.unwrap();
        assert_eq!(stream.next().await, Some(Ev::Content("a".into())));
        // Nothing sent: the interval elapses and a keepalive lands.
        assert_eq!(stream.next().await, Some(Ev::Keepalive));
        drop(tx);
        // A closed channel ends the stream rather than keeping it alive
        // on keepalives forever.
        assert_eq!(stream.next().await, None);
    }
}
