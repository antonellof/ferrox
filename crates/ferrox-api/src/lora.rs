//! Wire shapes for `GET /lora-adapters`, `POST /lora-adapters` and the
//! per-request `lora` field, llama.cpp's spellings.
//!
//! The list `GET` returns is `server_task_result_get_lora::to_json`
//! (`tools/server/server-task.cpp:1608-1626`) minus the aLoRA fields,
//! which ferrox refuses to load. `POST` takes the same `[{id, scale}]`
//! array `parse_lora_request` reads (`server-common.cpp:131-142`) and
//! answers `{"success": true}` (`server-task.cpp:1632-1634`). The
//! per-request `lora` field on a completion is that array again, and
//! it means what `construct_lora_list` makes it mean
//! (`server-context.cpp:1721-1732`): every adapter named gets its
//! scale, every adapter NOT named gets `0` for that request.

use serde::{Deserialize, Serialize};

/// One loaded adapter, as `GET /lora-adapters` lists it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraAdapterInfo {
    /// Its index in the load order: what `POST` and the per-request
    /// `lora` field address it by.
    pub id: usize,
    pub path: String,
    /// The scale currently applied to every request that does not
    /// override it.
    pub scale: f32,
    /// `adapter.lora.task_name` from the file, or empty.
    pub task_name: String,
    /// `adapter.lora.prompt_prefix` from the file, or empty.
    pub prompt_prefix: String,
}

/// One entry of the `[{id, scale}]` array `POST /lora-adapters` and a
/// request's `lora` field carry.
///
/// `scale` defaults to `0` when absent, as `json_value(entry, "scale",
/// 0.0f)` reads it upstream; an entry without an `id` is refused here
/// where upstream reads `-1` and silently matches nothing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LoraScaleRequest {
    pub id: usize,
    #[serde(default)]
    pub scale: f32,
}

/// The answer to `POST /lora-adapters`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoraApplyResponse {
    pub success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scale_request_without_a_scale_reads_zero() {
        let r: LoraScaleRequest = serde_json::from_str(r#"{"id": 1}"#).unwrap();
        assert_eq!(r, LoraScaleRequest { id: 1, scale: 0.0 });
        assert!(serde_json::from_str::<LoraScaleRequest>(r#"{"scale": 1.0}"#).is_err());
    }
}
