//! MCP tool-server config stub (roadmap P9). Parses `--mcp-config` JSON and
//! exposes server names in `/v1/models` metadata; invocation is not wired.

use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct McpServerEntry {
    pub name: String,
    #[serde(default)]
    #[allow(dead_code)] // parsed for future MCP attach
    pub command: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpConfigFile {
    #[serde(default)]
    pub servers: Vec<McpServerEntry>,
}

#[derive(Debug, Clone)]
pub struct LoadedMcpConfig {
    pub path: String,
    pub servers: Vec<McpServerEntry>,
}

pub fn load_mcp_config(path: &Path) -> anyhow::Result<LoadedMcpConfig> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read MCP config {}: {e}", path.display()))?;
    let parsed: McpConfigFile = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("invalid MCP config JSON in {}: {e}", path.display()))?;
    Ok(LoadedMcpConfig {
        path: path.display().to_string(),
        servers: parsed.servers,
    })
}

impl LoadedMcpConfig {
    pub fn models_metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "status": "planned",
            "message": "MCP tool servers are not invoked yet; config loaded for future attach",
            "config_path": self.path,
            "servers": self.servers.iter().map(|s| &s.name).collect::<Vec<_>>(),
        })
    }
}
