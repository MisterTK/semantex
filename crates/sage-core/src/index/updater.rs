use crate::config::SageConfig;
use crate::index::builder::{IndexBuilder, IndexStats};
use anyhow::Result;
use std::path::Path;

/// Incremental index updater.
/// Currently delegates to IndexBuilder which already handles incremental updates
/// via file hash comparison. This module exists for future optimization.
pub struct IndexUpdater;

impl IndexUpdater {
    /// Update the index for a project, only re-indexing changed files.
    /// This is currently equivalent to `IndexBuilder::build` since the builder
    /// already implements incremental logic via xxhash64 file comparison.
    pub fn update(project_path: &Path, config: &SageConfig) -> Result<IndexStats> {
        let builder = IndexBuilder::new(config)?;
        builder.build(project_path)
    }
}
