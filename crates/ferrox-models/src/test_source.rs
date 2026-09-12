//! A `TensorSource` for unit tests that answers only "does this tensor
//! exist" and "what is this key": no bytes behind it.
//!
//! Five modules had each written their own copy of this stub (a
//! names-only one in `parallel_dense_ffn`, metadata-only ones in
//! `swa_layers`, `mtp_blocks`, `act_layers`, `tokenizer`), which is the
//! repo's dominant bug shape in miniature even in test code. New tests
//! use this one; the old copies move here as their modules are next
//! touched.

use std::sync::Arc;

use ferrox_gguf::{GgmlType, GgufError, GgufValue, MmapHandle, TensorInfo, TensorSource};

/// Tensor names and metadata, nothing else.
pub(crate) struct StubSource {
    tensors: Vec<TensorInfo>,
    metadata: Vec<(String, GgufValue)>,
}

impl StubSource {
    /// A file holding exactly these tensor names and no metadata.
    pub(crate) fn with_tensors(names: &[&str]) -> Self {
        Self {
            tensors: names
                .iter()
                .map(|n| TensorInfo {
                    name: (*n).to_string(),
                    shape: Vec::new(),
                    dtype: GgmlType::F32,
                    offset: 0,
                })
                .collect(),
            metadata: Vec::new(),
        }
    }

    /// Adds one metadata key.
    pub(crate) fn with_key(mut self, key: &str, value: GgufValue) -> Self {
        self.metadata.push((key.to_string(), value));
        self
    }
}

impl TensorSource for StubSource {
    fn metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        Err(GgufError::TensorNotFound(name.to_string()))
    }
    fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<MmapHandle>, std::ops::Range<usize>), GgufError> {
        Err(GgufError::TensorNotFound(name.to_string()))
    }
}
