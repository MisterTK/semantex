use crate::search::adaptive::AdaptiveConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// ColBERT model selection.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColbertModelChoice {
    /// LateOn-Code-edge: 48d per-token, INT8 quantized, ~17MB
    #[default]
    LateOnCodeEdge,
}

/// Global sage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SageConfig {
    /// Maximum number of search results
    pub max_count: usize,
    /// Show content snippets in output
    pub content: bool,
    /// Lines of context around matches
    pub context_lines: usize,
    /// Enable cross-encoder reranking
    pub rerank: bool,
    /// Maximum file size to index (bytes)
    pub max_file_size: u64,
    /// Maximum number of files to index
    pub max_file_count: usize,
    /// Text chunk size in tokens
    pub chunk_size: usize,
    /// Chunk overlap in tokens
    pub chunk_overlap: usize,
    /// RRF fusion constant k
    pub rrf_k: f32,
    /// Number of candidates to fetch before reranking
    pub rerank_candidates: usize,
    /// Custom model directory override
    pub model_dir: Option<PathBuf>,
    /// Enable adaptive result sizing (dynamically adjust result count based on score distribution)
    pub adaptive_sizing: bool,
    /// Enable per-file deduplication (keep only top chunk per file unless scores are close)
    pub adaptive_dedup: bool,
    /// Score gap threshold for keeping multiple chunks from same file (0.0-1.0)
    pub adaptive_dedup_gap: f32,
    /// Minimum confidence score for Identifier queries (relative to top score, 0.0-1.0)
    pub min_score_identifier: f32,
    /// Minimum confidence score for Keyword queries (relative to top score, 0.0-1.0)
    pub min_score_keyword: f32,
    /// Minimum confidence score for Semantic queries (relative to top score, 0.0-1.0)
    pub min_score_semantic: f32,
    /// Minimum confidence score for Mixed queries (relative to top score, 0.0-1.0)
    pub min_score_mixed: f32,
    /// ColBERT model choice
    pub colbert_model: ColbertModelChoice,
    /// PLAID quantization bits (2 or 4, default: 4)
    pub plaid_nbits: usize,
}

impl Default for SageConfig {
    fn default() -> Self {
        Self {
            max_count: 15,
            content: false,
            context_lines: 0,
            rerank: false,
            max_file_size: 1_048_576, // 1MB
            max_file_count: 50_000,
            chunk_size: 512,
            chunk_overlap: 128,
            rrf_k: 30.0,
            rerank_candidates: 100,
            model_dir: None,
            // Adaptive result sizing
            adaptive_sizing: true,
            adaptive_dedup: true,
            adaptive_dedup_gap: 0.10,
            min_score_identifier: 0.08,
            min_score_keyword: 0.15,
            min_score_semantic: 0.10,
            min_score_mixed: 0.10,
            colbert_model: ColbertModelChoice::default(),
            plaid_nbits: 4,
        }
    }
}

impl SageConfig {
    /// Load config from file, falling back to defaults
    pub fn load(project_path: Option<&Path>) -> Result<Self> {
        let mut config = Self::default();

        // Load global config
        let global_config = Self::global_config_path();
        if global_config.exists() {
            let content = std::fs::read_to_string(&global_config)
                .with_context(|| format!("Failed to read {}", global_config.display()))?;
            config = serde_yml::from_str(&content)
                .with_context(|| format!("Failed to parse {}", global_config.display()))?;
        }

        // Overlay project config
        if let Some(project) = project_path {
            let project_config = project.join(".sagerc.yaml");
            if project_config.exists() {
                let content = std::fs::read_to_string(&project_config)
                    .with_context(|| format!("Failed to read {}", project_config.display()))?;
                let project_cfg: SageConfig = serde_yml::from_str(&content)
                    .with_context(|| format!("Failed to parse {}", project_config.display()))?;
                config = project_cfg;
            }
        }

        // Environment variable overrides
        if let Ok(v) = std::env::var("SAGE_MAX_COUNT") {
            config.max_count = v.parse().unwrap_or(config.max_count);
        }
        if let Ok(v) = std::env::var("SAGE_CONTENT") {
            config.content = v == "1" || v.to_lowercase() == "true";
        }
        if let Ok(v) = std::env::var("SAGE_RERANK") {
            config.rerank = v != "0" && v.to_lowercase() != "false";
        }
        if let Ok(v) = std::env::var("SAGE_MAX_FILE_SIZE") {
            config.max_file_size = v.parse().unwrap_or(config.max_file_size);
        }
        if let Ok(v) = std::env::var("SAGE_MODEL_DIR") {
            config.model_dir = Some(PathBuf::from(v));
        }

        Ok(config)
    }

    /// Default sage home directory
    pub fn sage_home() -> PathBuf {
        dirs_or_default("SAGE_HOME", ".sage")
    }

    /// Models directory
    pub fn models_dir(&self) -> PathBuf {
        self.model_dir
            .clone()
            .unwrap_or_else(|| Self::sage_home().join("models"))
    }

    /// Compute project index directory: `<project>/.sage/`
    pub fn project_index_dir(project_path: &Path) -> PathBuf {
        project_path.join(".sage")
    }

    /// Build an AdaptiveConfig from the current settings
    pub fn adaptive_config(&self) -> AdaptiveConfig {
        AdaptiveConfig {
            enabled: self.adaptive_sizing,
            min_score_identifier: self.min_score_identifier,
            min_score_keyword: self.min_score_keyword,
            min_score_semantic: self.min_score_semantic,
            min_score_mixed: self.min_score_mixed,
            deduplicate: self.adaptive_dedup,
            dedup_score_gap: self.adaptive_dedup_gap,
        }
    }

    /// Global config file path
    fn global_config_path() -> PathBuf {
        let config_dir = std::env::var("XDG_CONFIG_HOME").map_or_else(
            |_| dirs_or_default("HOME", "").join(".config"),
            PathBuf::from,
        );
        config_dir.join("sage").join("config.yaml")
    }
}

fn dirs_or_default(env_key: &str, suffix: &str) -> PathBuf {
    if let Ok(val) = std::env::var(env_key) {
        PathBuf::from(val)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(suffix)
    } else {
        PathBuf::from("/tmp").join("sage")
    }
}
