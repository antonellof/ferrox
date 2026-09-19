//! Server-side conversation history: a
//! client sends `session_id` plus only its newest message, and this
//! reassembles the full history so the model still sees the whole
//! conversation. Deliberately unrelated to
//! `frink_models::PrefixCache`: this is about *which messages* form a
//! conversation, `PrefixCache` is about *KV-state reuse* for whatever
//! prompt a request happens to render to -- orthogonal concerns kept
//! orthogonal, not merged into one.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::ChatMessage;

#[derive(Default, Clone)]
pub struct SessionStore {
    sessions: Arc<Mutex<HashMap<String, Vec<ChatMessage>>>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `new_messages` to `id`'s stored history (creating an
    /// empty one if this is the session's first request) and returns
    /// the full, now-accumulated history a prompt should be rendered
    /// from.
    pub fn extend_and_get(&self, id: &str, new_messages: &[ChatMessage]) -> Vec<ChatMessage> {
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        let history = sessions.entry(id.to_string()).or_default();
        history.extend_from_slice(new_messages);
        history.clone()
    }

    /// Appends the assistant's reply to `id`'s stored history, so the
    /// *next* request (which sends only its own new message) sees it
    /// too.
    pub fn store_reply(&self, id: &str, reply: ChatMessage) {
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        sessions.entry(id.to_string()).or_default().push(reply);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(crate::MessageContent::Text(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn extend_and_get_accumulates_across_calls() {
        let store = SessionStore::new();
        let first = store.extend_and_get("s1", &[msg("user", "hi")]);
        assert_eq!(first.len(), 1);

        store.store_reply("s1", msg("assistant", "hello"));

        let second = store.extend_and_get("s1", &[msg("user", "how are you")]);
        assert_eq!(second.len(), 3);
        assert_eq!(second[0].role, "user");
        assert_eq!(second[1].role, "assistant");
        assert_eq!(second[2].role, "user");
    }

    #[test]
    fn different_session_ids_do_not_share_history() {
        let store = SessionStore::new();
        store.extend_and_get("a", &[msg("user", "hi")]);
        let b = store.extend_and_get("b", &[msg("user", "yo")]);
        assert_eq!(b.len(), 1, "session b must not see session a's history");
    }
}
