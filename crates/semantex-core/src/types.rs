use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::chunking::structured_meta::StructuredChunkMeta;

/// Reserved sentinel for a positional dense backend's doc→chunk map.
///
/// A positional dense backend (one whose `positional_chunk_ids()` returns a
/// `mapping[doc_id] = chunk_id` vector) marks a deleted slot with this value
/// rather than truncating or shifting the vector: the backend may still
/// reference the doc_id internally, and later positions must stay positionally
/// correct. Subset construction in `hybrid.rs` (keyed off the
/// `positional_chunk_ids()` seam) skips any slot equal to this sentinel instead
/// of mapping it to a phantom chunk. `u64::MAX` is reserved because real chunk
/// IDs are SQLite AUTOINCREMENT row ids starting at 1.
///
/// No built-in backend keeps a positional map today (coderank-hnsw returns
/// `None`); this is seam infrastructure for a future positional backend.
pub const DENSE_TOMBSTONE: u64 = u64::MAX;

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

impl Chunk {
    /// Returns the symbol name for AST-aware chunks, or `None` for text-window
    /// / PDF-page chunks. Used by the v0.4 definition-boost ranking signal
    /// (see `search/path_signals.rs`, spec §7.4.2).
    pub fn symbol_name(&self) -> Option<&str> {
        match &self.chunk_type {
            ChunkType::AstNode { name, .. } => Some(name.as_str()),
            ChunkType::TextWindow { .. } | ChunkType::PdfPage { .. } => None,
        }
    }
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
///
/// v9 layout (v0.4.1 W-Index #4) adds `use_bm25_stemmer` so the daemon can
/// detect at open time that an index was built with a different stemmer
/// setting than the running config. Older v8 meta.json files lack the field
/// and fail to deserialize — `state::detect` then returns `Stale` via the
/// "unparseable meta -> Stale" rule, forcing a clean rebuild.
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
    /// Snowball-stemmer flag that was passed to `SparseIndex::create` when
    /// this index was built. Open-time code re-validates against the runtime
    /// `SemantexConfig.use_bm25_stemmer` and refuses to load on mismatch
    /// (since tantivy stores the analyzer state implicitly via the index
    /// schema). v0.4.1 W-Index #4.
    pub use_bm25_stemmer: bool,
    /// Dense backend identity this index was built with (e.g.
    /// `"coderank-hnsw"`). Open-time code re-validates against the runtime
    /// resolved backend and refuses to load on mismatch — the dense graph/index
    /// layout is backend-specific. An index stamped with a removed backend (e.g.
    /// a straggler `"colbert-plaid"`) trips the mismatch guard → clean rebuild
    /// guidance, never a panic. S1.
    pub dense_backend: String,
    /// Fingerprint of the embedder spec this dense index was built with
    /// (id+dims+pooling+quant+norm+prefix, xxh64). Open-time code compares it to
    /// the active embedder's fingerprint; a mismatch means the vector space
    /// changed, so the index is rebuilt under a new versioned dir and the active
    /// pointer is flipped atomically (S8 — zero-downtime). An older meta.json
    /// lacking this field fails to deserialize → `state::detect` returns `Stale`
    /// (same mechanism as the v9→v10 `dense_backend` add — no extra schema bump).
    pub embedder_fingerprint: String,
}

impl IndexMeta {
    /// Bumped from 7 → 8 when v0.3 added three auxiliary SQLite tables
    /// (`chunk_annotations`, `pattern_matches`, `chunk_centrality`).
    /// Pre-v0.3 indexes (schema_version=7) are missing these tables, so
    /// `state::detect` returns `Stale` and the MCP/CLI layer triggers a rebuild.
    ///
    /// v9: postcard wire format for `plaid_mapping.bin` (was bincode); also
    /// persists `use_bm25_stemmer` (see field below). v8 indexes written
    /// before the postcard switch fail to decode the mapping file and the
    /// missing-field deserialize blocks v8 meta.json from parsing as v9 —
    /// both surface as `Stale` via `state::detect`, triggering a rebuild.
    ///
    /// v10 (S1): persists `dense_backend` so the daemon can detect at open time
    /// that an index was built with a different dense backend than the running
    /// config. Older v9 meta.json files lack the field and fail to deserialize
    /// — `state::detect` then returns `Stale`, forcing a clean rebuild.
    ///
    /// S8: adds `embedder_fingerprint`. No version bump beyond S1's 10 — an older
    /// meta.json lacking the field fails the strict deserialize and is treated as
    /// `Stale` (same mechanism as the v9→v10 `dense_backend` add).
    ///
    /// v11 (S2): the single-vector dense backend (`coderank-hnsw`) introduces a
    /// new on-disk layout (`dense/coderank-hnsw/{index.bin,store.vecs}`). Bumping
    /// forces a clean reindex so an old PLAID-only index isn't half-read by the
    /// new path.
    ///
    /// v12 (D4): bumped when the ColBERT/PLAID dense backend was briefly removed
    /// (coderank-hnsw the sole backend), to force any straggler index stamped
    /// `dense_backend:"colbert-plaid"` `Stale` for a clean reindex. colbert-plaid
    /// was later RESTORED as the default backend (2026-06-02 cutover), but needs
    /// NO further schema bump — a change of active embedder/backend is caught by
    /// the per-embedder `embedder_fingerprint` staleness (auto-rebuild into the
    /// versioned dense dir) + the open-time backend-mismatch guard in `hybrid.rs`.
    ///
    /// v13 (Wave 0 spine): storage layout v13 — per-branch index directories
    /// under `.semantex/indexes/<branch_key>/`, a versioned top-level
    /// `.semantex/meta.json` (project_id/default_branch), `history.db` /
    /// `memory.db` schema creation, and registry v2. `IndexMeta` itself is
    /// UNCHANGED (see `index/layout.rs` module doc for why the branch/
    /// head_commit metadata lives in a sidecar rather than as new fields
    /// here) — the bump exists purely to force any pre-v13 index `Stale` so
    /// it picks up the new layout on its next build.
    pub const CURRENT_SCHEMA_VERSION: u32 = 13;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// v0.4.1 W-Index #1: the on-disk schema version must be 9 — the bump from
    /// 8 records both the postcard wire format for `plaid_mapping.bin` and the
    /// addition of the persisted `use_bm25_stemmer` field. Older v8 indexes
    /// stay incompatible (rebuild via the stale-detection path).
    /// S1: schema bumped 9 → 10 to add the persisted `dense_backend` field.
    /// Older v9 indexes (which lack the field) become `Stale` and rebuild.
    /// S2: schema bumped 10 → 11 for the single-vector dense on-disk layout
    /// (`dense/coderank-hnsw/{index.bin,store.vecs}`). Older indexes become
    /// `Stale`.
    /// D4: bumped 11 → 12 when the ColBERT/PLAID backend was removed; any
    /// straggler colbert-plaid index becomes `Stale` and rebuilds.
    /// Wave 0 spine: bumped 12 → 13 for storage layout v13 (per-branch index
    /// dirs, registry v2 — see `index/layout.rs`). Any pre-v13 index becomes
    /// `Stale` and rebuilds, migrating into the new layout on the way.
    #[test]
    fn current_schema_version_is_13() {
        assert_eq!(IndexMeta::CURRENT_SCHEMA_VERSION, 13);
    }

    #[test]
    fn index_meta_round_trips_dense_backend() {
        let meta = IndexMeta {
            schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: std::path::PathBuf::from("/x"),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 1,
            chunk_count: 2,
            embedding_model: "CodeRankEmbed".to_string(),
            embedding_dim: 768,
            use_bm25_stemmer: true,
            dense_backend: "coderank-hnsw".to_string(),
            embedder_fingerprint: "fp".to_string(),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let back: IndexMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dense_backend, "coderank-hnsw");
        assert_eq!(back.schema_version, IndexMeta::CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn index_meta_round_trips_embedder_fingerprint() {
        let meta = IndexMeta {
            schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: std::path::PathBuf::from("/x"),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 1,
            chunk_count: 2,
            embedding_model: "CodeRankEmbed".to_string(),
            embedding_dim: 768,
            use_bm25_stemmer: true,
            dense_backend: "coderank-hnsw".to_string(),
            embedder_fingerprint: "abc123".to_string(),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let back: IndexMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.embedder_fingerprint, "abc123");
    }

    /// Synthetic v8 meta.json must be detected as `Stale` by `state::detect`
    /// after the v9 bump, so the MCP/CLI rebuild path triggers a force-rebuild.
    #[test]
    fn synthetic_v8_meta_is_stale() {
        use crate::index::state;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let semantex_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).expect("create .semantex dir");
        let meta = IndexMeta {
            schema_version: 8,
            project_path: tmp.path().to_path_buf(),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 0,
            chunk_count: 0,
            embedding_model: "test".to_string(),
            embedding_dim: 48,
            use_bm25_stemmer: true,
            dense_backend: "colbert-plaid".to_string(),
            embedder_fingerprint: "fp".to_string(),
        };
        let meta_json = serde_json::to_string(&meta).expect("serialize meta");
        std::fs::write(semantex_dir.join("meta.json"), meta_json).expect("write meta");
        assert_eq!(state::detect(tmp.path()), state::IndexState::Stale);
    }

    /// D4: an old index stamped with the removed `colbert-plaid` backend (at the
    /// PRE-D4 schema 11) must be detected `Stale` after the v12 bump and rebuilt
    /// cleanly — never opened/panicked. This is the graceful-degradation path for
    /// stragglers that predate the backend removal.
    #[test]
    fn old_colbert_plaid_meta_is_stale_after_v12_bump() {
        use crate::index::state;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let semantex_dir = tmp.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).expect("create .semantex dir");
        let meta = IndexMeta {
            schema_version: 11, // pre-D4 schema
            project_path: tmp.path().to_path_buf(),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 0,
            chunk_count: 0,
            embedding_model: "LateOn-Code-edge".to_string(),
            embedding_dim: 48,
            use_bm25_stemmer: true,
            dense_backend: "colbert-plaid".to_string(),
            embedder_fingerprint: "fp".to_string(),
        };
        let meta_json = serde_json::to_string(&meta).expect("serialize meta");
        std::fs::write(semantex_dir.join("meta.json"), meta_json).expect("write meta");
        assert_eq!(state::detect(tmp.path()), state::IndexState::Stale);
    }
}
