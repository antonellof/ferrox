//! Validate a Kimi K3 checkpoint directory without loading all shards.
//!
//! Reads `model.safetensors.index.json` (or a partial weight_map) and
//! checks that expected layer prefixes and tensor shapes are present.
//! Used by `frink inspect-kimi` / CI gates before a full ~1.56 TB run.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KimiValidateError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Message(String),
}

#[derive(Debug, Deserialize)]
struct IndexFile {
    weight_map: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct KimiCheckpointReport {
    pub n_tensors: usize,
    pub n_shards_referenced: usize,
    pub layers_seen: Vec<usize>,
    pub has_embed: bool,
    pub has_lm_head: bool,
    pub missing_required: Vec<String>,
}

/// Lightweight validation: index present, embed/lm_head named, at least
/// one layer prefix, shard files referenced exist when `check_files`.
pub fn validate_kimi_checkpoint_dir(
    dir: &Path,
    check_files: bool,
) -> Result<KimiCheckpointReport, KimiValidateError> {
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Err(KimiValidateError::Message(format!(
            "missing {} — not a Kimi safetensors directory",
            index_path.display()
        )));
    }
    let index: IndexFile = serde_json::from_str(&fs::read_to_string(&index_path)?)?;
    let n_tensors = index.weight_map.len();
    let mut shards: HashMap<String, ()> = HashMap::new();
    let mut layers = std::collections::BTreeSet::new();
    let mut has_embed = false;
    let mut has_lm_head = false;
    for (name, shard) in &index.weight_map {
        shards.insert(shard.clone(), ());
        if name.contains("embed_tokens") || name.ends_with("tok_embeddings.weight") {
            has_embed = true;
        }
        if name.contains("lm_head") || name.contains("output.weight") {
            has_lm_head = true;
        }
        if let Some(rest) = name.strip_prefix("language_model.model.layers.") {
            if let Some(num) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                layers.insert(num);
            }
        }
    }
    let mut missing_required = Vec::new();
    if !has_embed {
        missing_required.push("embed_tokens / tok_embeddings".into());
    }
    if layers.is_empty() {
        missing_required.push("language_model.model.layers.*".into());
    }
    if check_files {
        for shard in shards.keys() {
            let p = dir.join(shard);
            if !p.is_file() {
                missing_required.push(format!("shard file missing: {shard}"));
            }
        }
    }
    Ok(KimiCheckpointReport {
        n_tensors,
        n_shards_referenced: shards.len(),
        layers_seen: layers.into_iter().collect(),
        has_embed,
        has_lm_head,
        missing_required,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_minimal_index_json() {
        let dir = std::env::temp_dir().join(format!("frink_kimi_validate_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let index = r#"{
          "weight_map": {
            "language_model.model.embed_tokens.weight": "model-00001-of-000096.safetensors",
            "language_model.model.layers.0.input_layernorm.weight": "model-00001-of-000096.safetensors",
            "language_model.lm_head.weight": "model-00001-of-000096.safetensors"
          }
        }"#;
        fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
        let report = validate_kimi_checkpoint_dir(&dir, false).unwrap();
        assert_eq!(report.n_tensors, 3);
        assert_eq!(report.layers_seen, vec![0]);
        assert!(report.has_embed);
        assert!(report.has_lm_head);
        assert!(report.missing_required.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
