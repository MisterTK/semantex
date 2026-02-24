pub mod ast_chunker;
pub mod call_graph;
pub mod doc_parser;
pub mod import_resolver;
pub mod pdf_chunker;
pub mod semantic_role;
pub mod structured_meta;
pub mod text_chunker;

use crate::types::Chunk;
use anyhow::Result;
use std::path::Path;

/// Trait for all chunking strategies
pub trait Chunker: Send + Sync {
    /// Split file content into chunks
    fn chunk(&self, path: &Path, content: &str) -> Result<Vec<Chunk>>;
}
