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
    pub const CURRENT_SCHEMA_VERSION: u32 = 7;
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
#[derive(Debug, Clone)]
pub struct ScoredChunkId {
    pub chunk_id: u64,
    pub score: f32,
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
