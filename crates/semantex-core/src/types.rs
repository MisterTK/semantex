use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::chunking::structured_meta::StructuredChunkMeta;

/// A chunk of text extracted from a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk ID within the index
    pub id: u64,
    /// Source file path (relative to project root)
    pub file_path: PathBuf,
    /// Starting line number (1-based)
    pub start_line: u32,
    /// Ending line number (1-based, inclusive)
    pub end_line: u32,
    /// The actual text content of this chunk
    pub content: String,
    /// How this chunk was created
    pub chunk_type: ChunkType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkType {
    /// AST-aware chunk: a function, method, class, etc.
    AstNode {
        name: String,
        kind: AstNodeKind,
        language: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        structured_meta: Option<Box<StructuredChunkMeta>>,
    },
    /// Sliding window text chunk
    TextWindow { window_index: u32 },
    /// PDF page chunk
    PdfPage { page_number: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AstNodeKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface,
    Module,
    Other(String),
}

impl AstNodeKind {
    /// Human-readable label for this AST node kind.
    pub fn label(&self) -> &str {
        match self {
            Self::Function => "function",
            Self::Method => "method",
            Self::Class => "class",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Interface => "interface",
            Self::Module => "module",
            Self::Other(_) => "definition",
        }
    }
}

impl std::fmt::Display for AstNodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Function => "fn",
            Self::Method => "method",
            Self::Class => "class",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Interface => "interface",
            Self::Module => "module",
            Self::Other(s) => s.as_str(),
        };
        write!(f, "{s}")
    }
}

/// A search result with score and provenance
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
    pub source: SearchSource,
    /// Per-channel scores from fusion (0.0 if not from fusion)
    pub score_dense: f32,
    pub score_sparse: f32,
    pub score_exact: f32,
    /// Per-result confidence label derived from channel agreement and score gap.
    /// See `Confidence` for derivation rules.
    pub confidence: Confidence,
    /// Numeric confidence in [0.0, 1.0] — channels-found / channels-fired.
    /// 0.0 when no fusion was performed.
    pub confidence_score: f32,
}

/// Per-result confidence classification (E6).
///
/// Derived from channel-agreement and score-gap during fusion:
/// - `Extracted`: all active channels found this result (highest confidence)
/// - `Inferred`: result found by only a subset of channels (single-channel hit common)
/// - `Ambiguous`: result score is within 5% of the next result's score (low margin)
///
/// `Inferred` is the default for results without explicit channel-source tracking.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    /// All active retrieval channels agree on this result.
    Extracted,
    /// Only a subset of channels surfaced this result.
    #[default]
    Inferred,
    /// Score gap to the next result is too small to discriminate confidently.
    Ambiguous,
}

impl Confidence {
    /// Lowercase label suitable for JSON/protocol serialization or display.
    pub fn label(self) -> &'static str {
        match self {
            Self::Extracted => "extracted",
            Self::Inferred => "inferred",
            Self::Ambiguous => "ambiguous",
        }
    }
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchSource {
    Dense,
    Sparse,
    Hybrid,
    Reranked,
    GraphExpanded,
}

/// Index metadata stored alongside the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    pub schema_version: u32,
    pub project_path: PathBuf,
    pub created_at: String,
    pub updated_at: String,
    pub file_count: u64,
    pub chunk_count: u64,
    pub embedding_model: String,
    pub embedding_dim: u32,
}

impl IndexMeta {
    /// Bumped from 7 → 8 when v0.3 added three auxiliary SQLite tables
    /// (`chunk_annotations`, `pattern_matches`, `chunk_centrality`).
    /// Pre-v0.3 indexes (schema_version=7) are missing these tables, so
    /// `state::detect` returns `Stale` and the MCP/CLI layer triggers a rebuild.
    pub const CURRENT_SCHEMA_VERSION: u32 = 8;
}

/// File metadata for incremental indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: PathBuf,
    pub hash: u64,
    pub size: u64,
    pub mtime: i64,
}

/// Scored item for internal ranking operations
#[derive(Debug, Clone, Default)]
pub struct ScoredChunkId {
    pub chunk_id: u64,
    pub score: f32,
    /// Per-channel normalized scores (populated by triple_cc_fuse, zeroed elsewhere)
    pub score_dense: f32,
    pub score_sparse: f32,
    pub score_exact: f32,
}

impl ScoredChunkId {
    /// Create a `ScoredChunkId` with per-channel scores defaulted to zero.
    pub fn new(chunk_id: u64, score: f32) -> Self {
        Self {
            chunk_id,
            score,
            score_dense: 0.0,
            score_sparse: 0.0,
            score_exact: 0.0,
        }
    }
}

/// File-type filter for scoping search results by extension
#[derive(Debug, Clone, Default)]
pub struct FileFilter {
    /// Only include files with these extensions (e.g. "rs", "py")
    pub include_extensions: Vec<String>,
    /// Exclude files with these extensions
    pub exclude_extensions: Vec<String>,
}

impl FileFilter {
    /// Extensions excluded by the --code-only flag
    pub const NON_CODE_EXTENSIONS: &[&str] = &[
        "md", "json", "yaml", "yml", "toml", "txt", "log", "cfg", "ini", "env", "pdf", "ipynb",
        "lock",
    ];

    /// Check whether a file path passes this filter
    pub fn matches(&self, path: &std::path::Path) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !self.include_extensions.is_empty() && !self.include_extensions.iter().any(|e| e == &ext)
        {
            return false;
        }

        if self.exclude_extensions.iter().any(|e| e == &ext) {
            return false;
        }

        true
    }

    pub fn is_active(&self) -> bool {
        !self.include_extensions.is_empty() || !self.exclude_extensions.is_empty()
    }
}
