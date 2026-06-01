use crate::chunking::structured_meta::StructuredChunkMeta;
use crate::index::file_classifier::FileRole;
use crate::types::{Chunk, ChunkType, FileEntry};
use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── E8(a): mmap'd index file helper (Unix only — Windows uses read fallback) ──
//
// Per spec risk T5, mmap is feature-gated to non-Windows platforms. On Windows
// we fall back to `std::fs::read`. This is a generic single-file mapper used by
// the cold-start prefetch path.
#[cfg(not(target_os = "windows"))]
use memmap2::Mmap;
#[cfg(not(target_os = "windows"))]
use std::fs::File;

/// Statistics for the code graph tables (symbol_defs, type_refs, type_hierarchy, module_edges).
#[derive(Debug, Default)]
pub struct GraphStats {
    pub calls_resolved: usize,
    pub types_resolved: usize,
    pub hierarchy_resolved: usize,
    pub module_edges_count: usize,
    pub symbol_defs_count: usize,
}

/// SQLite-backed chunk and file metadata storage
pub struct ChunkStore {
    conn: Connection,
}

impl ChunkStore {
    /// Open or create the chunk store database (full PRAGMAs for indexing throughput)
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)?;

        // SQLite performance optimizations (tuned for indexing throughput)
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -8000;
            PRAGMA mmap_size = 67108864;
            PRAGMA temp_store = MEMORY;
            PRAGMA page_size = 4096;
            ",
        )?;

        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    /// Open the chunk store in search mode with minimal memory footprint.
    /// Search queries are short-lived; we rely on OS page cache for mmap.
    pub fn open_for_search(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;

        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -2000;
            PRAGMA mmap_size = 8388608;
            ",
        )?;

        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                file_hash INTEGER NOT NULL,
                file_mtime INTEGER NOT NULL,
                structured_meta TEXT
            );

            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash INTEGER NOT NULL,
                size INTEGER NOT NULL,
                mtime INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
            ",
        )?;

        // v3 Phase 1: Call graph edges
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS call_graph (
                caller_chunk_id INTEGER NOT NULL,
                callee_name TEXT NOT NULL,
                callee_chunk_id INTEGER,
                FOREIGN KEY (caller_chunk_id) REFERENCES chunks(id)
            );
            CREATE INDEX IF NOT EXISTS idx_call_graph_caller ON call_graph(caller_chunk_id);
            CREATE INDEX IF NOT EXISTS idx_call_graph_callee ON call_graph(callee_chunk_id);
            CREATE INDEX IF NOT EXISTS idx_call_graph_callee_name ON call_graph(callee_name);
            ",
        )?;

        // v3 Phase 3: File role classification
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS file_metadata (
                path TEXT PRIMARY KEY,
                role TEXT NOT NULL DEFAULT 'Unknown',
                imports_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_file_metadata_role ON file_metadata(role);
            ",
        )?;

        // v7: Symbol definitions, type references, module edges, type hierarchy
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS symbol_defs (
                chunk_id    INTEGER NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_kind TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            );
            CREATE INDEX IF NOT EXISTS idx_symbol_defs_name ON symbol_defs(symbol_name);
            CREATE INDEX IF NOT EXISTS idx_symbol_defs_chunk ON symbol_defs(chunk_id);

            CREATE TABLE IF NOT EXISTS type_refs (
                chunk_id       INTEGER NOT NULL,
                type_name      TEXT NOT NULL,
                ref_context    TEXT NOT NULL,
                defining_chunk INTEGER,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id),
                FOREIGN KEY (defining_chunk) REFERENCES chunks(id)
            );
            CREATE INDEX IF NOT EXISTS idx_type_refs_chunk ON type_refs(chunk_id);
            CREATE INDEX IF NOT EXISTS idx_type_refs_type ON type_refs(type_name);
            CREATE INDEX IF NOT EXISTS idx_type_refs_defining ON type_refs(defining_chunk);

            CREATE TABLE IF NOT EXISTS module_edges (
                importer_path TEXT NOT NULL,
                imported_path TEXT NOT NULL,
                import_stmt   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_module_edges_importer ON module_edges(importer_path);
            CREATE INDEX IF NOT EXISTS idx_module_edges_imported ON module_edges(imported_path);

            CREATE TABLE IF NOT EXISTS type_hierarchy (
                child_name  TEXT NOT NULL,
                parent_name TEXT NOT NULL,
                relation    TEXT NOT NULL,
                child_chunk INTEGER,
                parent_chunk INTEGER,
                FOREIGN KEY (child_chunk) REFERENCES chunks(id),
                FOREIGN KEY (parent_chunk) REFERENCES chunks(id)
            );
            CREATE INDEX IF NOT EXISTS idx_type_hierarchy_child ON type_hierarchy(child_name);
            CREATE INDEX IF NOT EXISTS idx_type_hierarchy_parent ON type_hierarchy(parent_name);
            ",
        )?;

        Ok(())
    }

    /// Insert a chunk and return its assigned ID
    pub fn insert_chunk(&self, chunk: &Chunk, file_hash: u64, file_mtime: i64) -> Result<u64> {
        let chunk_type_json =
            serde_json::to_string(&chunk.chunk_type).context("Failed to serialize chunk_type")?;

        // Serialize structured_meta into its own column for direct SQL access
        let structured_meta_json: Option<String> = match &chunk.chunk_type {
            ChunkType::AstNode {
                structured_meta: Some(meta),
                ..
            } => Some(
                serde_json::to_string(meta.as_ref())
                    .context("Failed to serialize structured_meta")?,
            ),
            _ => None,
        };

        self.conn.execute(
            "INSERT INTO chunks (file_path, start_line, end_line, content, chunk_type, file_hash, file_mtime, structured_meta)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                chunk.file_path.to_string_lossy().as_ref(),
                chunk.start_line,
                chunk.end_line,
                chunk.content,
                chunk_type_json,
                file_hash as i64,
                file_mtime,
                structured_meta_json,
            ],
        )?;

        Ok(self.conn.last_insert_rowid() as u64)
    }

    /// Get the structured metadata for a chunk directly from its dedicated column.
    pub fn get_structured_meta(&self, chunk_id: u64) -> Result<Option<StructuredChunkMeta>> {
        let json: Option<String> = self
            .conn
            .query_row(
                "SELECT structured_meta FROM chunks WHERE id = ?1",
                params![chunk_id as i64],
                |row| row.get(0),
            )
            .optional()?
            .flatten();

        Ok(json.as_deref().and_then(|j| serde_json::from_str(j).ok()))
    }

    /// Get a single chunk by ID
    pub fn get_chunk(&self, id: u64) -> Result<Chunk> {
        let chunk = self.conn.query_row(
            "SELECT id, file_path, start_line, end_line, content, chunk_type FROM chunks WHERE id = ?1",
            params![id as i64],
            |row| {
                let chunk_type_json: String = row.get(5)?;
                Ok(Chunk {
                    id: row.get::<_, i64>(0)? as u64,
                    file_path: PathBuf::from(row.get::<_, String>(1)?),
                    start_line: row.get::<_, i64>(2)? as u32,
                    end_line: row.get::<_, i64>(3)? as u32,
                    content: row.get(4)?,
                    chunk_type: serde_json::from_str(&chunk_type_json).unwrap_or(ChunkType::TextWindow { window_index: 0 }),
                })
            },
        )?;
        Ok(chunk)
    }

    /// Get multiple chunks by IDs
    pub fn get_chunks(&self, ids: &[u64]) -> Result<Vec<Chunk>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
        let query = format!(
            "SELECT id, file_path, start_line, end_line, content, chunk_type FROM chunks WHERE id IN ({})",
            placeholders.join(",")
        );

        let mut stmt = self.conn.prepare(&query)?;
        let id_params: Vec<Box<dyn rusqlite::types::ToSql>> = ids
            .iter()
            .map(|id| Box::new(*id as i64) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            id_params.iter().map(std::convert::AsRef::as_ref).collect();

        let chunks = stmt
            .query_map(param_refs.as_slice(), |row| {
                let chunk_type_json: String = row.get(5)?;
                Ok(Chunk {
                    id: row.get::<_, i64>(0)? as u64,
                    file_path: PathBuf::from(row.get::<_, String>(1)?),
                    start_line: row.get::<_, i64>(2)? as u32,
                    end_line: row.get::<_, i64>(3)? as u32,
                    content: row.get(4)?,
                    chunk_type: serde_json::from_str(&chunk_type_json)
                        .unwrap_or(ChunkType::TextWindow { window_index: 0 }),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(chunks)
    }

    /// Delete all chunks for a given file, returning deleted chunk IDs
    pub fn delete_chunks_for_file(&self, file_path: &Path) -> Result<Vec<u64>> {
        let path_str = file_path.to_string_lossy();

        let mut stmt = self
            .conn
            .prepare("SELECT id FROM chunks WHERE file_path = ?1")?;
        let ids: Vec<u64> = stmt
            .query_map(params![path_str.as_ref()], |row| {
                Ok(row.get::<_, i64>(0)? as u64)
            })?
            .collect::<Result<Vec<_>, _>>()?;

        self.conn.execute(
            "DELETE FROM chunks WHERE file_path = ?1",
            params![path_str.as_ref()],
        )?;

        Ok(ids)
    }

    /// Get the stored hash for a file
    pub fn get_file_hash(&self, file_path: &Path) -> Result<Option<u64>> {
        let path_str = file_path.to_string_lossy();
        let hash = self
            .conn
            .query_row(
                "SELECT hash FROM files WHERE path = ?1",
                params![path_str.as_ref()],
                |row| Ok(row.get::<_, i64>(0)? as u64),
            )
            .optional()?;
        Ok(hash)
    }

    /// Insert or update a file entry
    pub fn set_file_entry(&self, entry: &FileEntry) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO files (path, hash, size, mtime) VALUES (?1, ?2, ?3, ?4)",
            params![
                entry.path.to_string_lossy().as_ref(),
                entry.hash as i64,
                entry.size as i64,
                entry.mtime,
            ],
        )?;
        Ok(())
    }

    /// Remove a file entry
    pub fn remove_file_entry(&self, file_path: &Path) -> Result<()> {
        let path_str = file_path.to_string_lossy();
        self.conn.execute(
            "DELETE FROM files WHERE path = ?1",
            params![path_str.as_ref()],
        )?;
        Ok(())
    }

    /// Get all indexed file paths
    pub fn get_all_file_paths(&self) -> Result<Vec<PathBuf>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files")?;
        let paths = stmt
            .query_map([], |row| Ok(PathBuf::from(row.get::<_, String>(0)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(paths)
    }

    /// Get total chunk count
    pub fn chunk_count(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        Ok(count as u64)
    }

    /// Get total file count
    pub fn file_count(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?;
        Ok(count as u64)
    }

    /// Get a sample of file paths with their stored mtimes (for validation).
    /// Returns up to `limit` entries as (path, stored_mtime).
    pub fn get_file_mtimes_sample(&self, limit: usize) -> Result<Vec<(PathBuf, i64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, mtime FROM files LIMIT ?1")?;
        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok((
                    PathBuf::from(row.get::<_, String>(0)?),
                    row.get::<_, i64>(1)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Case-insensitive substring search over chunk content using SQLite LIKE.
    /// Replaces the in-memory ExactIndex to avoid duplicating all chunk content in RAM.
    pub fn search_exact(&self, query: &str, limit: usize) -> Result<Vec<u64>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }
        // Escape LIKE special characters
        let escaped = query.replace('%', "\\%").replace('_', "\\_");
        let pattern = format!("%{escaped}%");
        let mut stmt = self.conn.prepare(
            "SELECT id FROM chunks WHERE content LIKE ?1 ESCAPE '\\' COLLATE NOCASE LIMIT ?2",
        )?;
        let ids = stmt
            .query_map(params![pattern, limit as i64], |row| {
                Ok(row.get::<_, i64>(0)? as u64)
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    /// Iterate over all chunk IDs and content (for building exact index)
    pub fn for_each_chunk_content(&self, mut f: impl FnMut(u64, &str)) -> Result<()> {
        let mut stmt = self.conn.prepare("SELECT id, content FROM chunks")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let content: String = row.get(1)?;
            f(id as u64, &content);
        }
        Ok(())
    }

    /// Return all chunk IDs ordered by ID, for batched dense index rebuilds.
    pub fn get_all_chunk_ids(&self) -> Result<Vec<u64>> {
        let mut stmt = self.conn.prepare("SELECT id FROM chunks ORDER BY id")?;
        let ids = stmt
            .query_map([], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids.into_iter().map(|id| id as u64).collect())
    }

    /// Fetch persisted NL annotations (E3 `chunk_annotations`) for the given
    /// chunk ids, as a `chunk_id → annotation` map. Only ids that HAVE a stored
    /// annotation appear. Used by the coderank-hnsw `SEMANTEX_DENSE_CONTEXT` A/B
    /// to embed `format!("{annotation}\n{code}")`. Returns an empty map if the
    /// `chunk_annotations` table is absent (older index) or no ids match.
    pub fn get_annotations(&self, ids: &[u64]) -> Result<std::collections::HashMap<u64, String>> {
        let mut out = std::collections::HashMap::new();
        if ids.is_empty() || !self.table_exists("chunk_annotations")? {
            return Ok(out);
        }
        // Chunk into SQLite-variable-safe batches (mirrors get_chunks).
        for batch in ids.chunks(900) {
            let placeholders = batch.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT chunk_id, nl_annotation FROM chunk_annotations WHERE chunk_id IN ({placeholders})"
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let params: Vec<i64> = batch.iter().map(|&id| id as i64).collect();
            let rows = stmt.query_map(rusqlite::params_from_iter(params), |row| {
                Ok((row.get::<_, i64>(0)? as u64, row.get::<_, String>(1)?))
            })?;
            for r in rows {
                let (id, ann) = r?;
                out.insert(id, ann);
            }
        }
        Ok(out)
    }

    /// Begin a transaction for batch operations
    pub fn begin_transaction(&self) -> Result<()> {
        self.conn.execute_batch("BEGIN TRANSACTION")?;
        Ok(())
    }

    /// Commit a transaction
    pub fn commit_transaction(&self) -> Result<()> {
        self.conn.execute_batch("COMMIT")?;
        Ok(())
    }

    /// Store file role classification for a given relative path.
    pub fn set_file_role(&self, file_path: &Path, role: FileRole) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO file_metadata (path, role) VALUES (?1, ?2)",
            params![file_path.to_string_lossy().as_ref(), role.as_str()],
        )?;
        Ok(())
    }

    /// Get file role for a given path.
    pub fn get_file_role(&self, file_path: &Path) -> Result<FileRole> {
        let role_str: Option<String> = self
            .conn
            .query_row(
                "SELECT role FROM file_metadata WHERE path = ?1",
                params![file_path.to_string_lossy().as_ref()],
                |row| row.get(0),
            )
            .optional()?;
        Ok(role_str
            .as_deref()
            .map_or(FileRole::Unknown, FileRole::from_str))
    }

    /// Batch-get file roles for multiple chunk file paths.
    /// Uses a single WHERE IN query instead of N individual lookups.
    pub fn get_file_roles(&self, paths: &[&Path]) -> Result<HashMap<PathBuf, FileRole>> {
        if paths.is_empty() {
            return Ok(HashMap::new());
        }

        // Deduplicate paths to minimize query size
        let unique_paths: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            paths
                .iter()
                .filter_map(|p| {
                    let s = p.to_string_lossy().into_owned();
                    seen.insert(s.clone()).then_some(s)
                })
                .collect()
        };

        // SQLite has a default SQLITE_MAX_VARIABLE_NUMBER of 999.
        // Chunk the query to stay under the limit.
        let mut result = HashMap::with_capacity(unique_paths.len());

        for chunk in unique_paths.chunks(500) {
            let placeholders: String = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql =
                format!("SELECT path, role FROM file_metadata WHERE path IN ({placeholders})");
            let mut stmt = self.conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|s| s as &dyn rusqlite::types::ToSql)
                .collect();

            let rows = stmt.query_map(params.as_slice(), |row| {
                let path: String = row.get(0)?;
                let role_str: String = row.get(1)?;
                Ok((path, role_str))
            })?;

            for row in rows {
                let (path, role_str) = row?;
                result.insert(PathBuf::from(&path), FileRole::from_str(&role_str));
            }
        }

        // Fill in Unknown for paths not found in file_metadata
        for path in paths {
            let pb = (*path).to_path_buf();
            result.entry(pb).or_insert(FileRole::Unknown);
        }

        Ok(result)
    }

    /// Store a call graph edge (caller chunk -> callee function name).
    #[allow(clippy::similar_names)]
    pub fn store_call_graph_edge(
        &self,
        caller_chunk_id: u64,
        callee_name: &str,
        callee_chunk_id: Option<u64>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO call_graph (caller_chunk_id, callee_name, callee_chunk_id) VALUES (?1, ?2, ?3)",
            params![caller_chunk_id as i64, callee_name, callee_chunk_id.map(|id| id as i64)],
        )?;
        Ok(())
    }

    // ── v7 graph insert methods ──────────────────────────────────────

    /// Insert a symbol definition (function, class, trait, etc.) for a chunk.
    pub fn insert_symbol_def(
        &self,
        chunk_id: u64,
        name: &str,
        kind: &str,
        file_path: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO symbol_defs (chunk_id, symbol_name, symbol_kind, file_path) VALUES (?1, ?2, ?3, ?4)",
            params![chunk_id as i64, name, kind, file_path],
        )?;
        Ok(())
    }

    /// Insert a type reference (usage of a type name in a chunk).
    pub fn insert_type_ref(&self, chunk_id: u64, type_name: &str, context: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO type_refs (chunk_id, type_name, ref_context) VALUES (?1, ?2, ?3)",
            params![chunk_id as i64, type_name, context],
        )?;
        Ok(())
    }

    /// Insert a module-level import edge.
    #[allow(clippy::similar_names)]
    pub fn insert_module_edge(&self, importer: &str, imported: &str, stmt: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO module_edges (importer_path, imported_path, import_stmt) VALUES (?1, ?2, ?3)",
            params![importer, imported, stmt],
        )?;
        Ok(())
    }

    /// Insert a type hierarchy relationship (extends, implements, etc.).
    pub fn insert_type_hierarchy(&self, child: &str, parent: &str, relation: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO type_hierarchy (child_name, parent_name, relation) VALUES (?1, ?2, ?3)",
            params![child, parent, relation],
        )?;
        Ok(())
    }

    // ── v7 graph query methods (global resolution) ───────────────────

    /// Get all symbol definitions: (chunk_id, name, kind, file_path).
    pub fn get_all_symbol_defs(&self) -> Result<Vec<(u64, String, String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT chunk_id, symbol_name, symbol_kind, file_path FROM symbol_defs")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)? as u64,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get symbol definitions for a specific file: (chunk_id, name, kind).
    pub fn get_symbol_defs_for_file(&self, path: &str) -> Result<Vec<(u64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT chunk_id, symbol_name, symbol_kind FROM symbol_defs WHERE file_path = ?1",
        )?;
        let rows = stmt
            .query_map(params![path], |row| {
                Ok((row.get::<_, i64>(0)? as u64, row.get(1)?, row.get(2)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get imported paths for a given importer file.
    pub fn get_module_edges_for_file(&self, file_path: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT imported_path FROM module_edges WHERE importer_path = ?1")?;
        let rows = stmt
            .query_map(params![file_path], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get unresolved call graph edges: (rowid, callee_name, caller_file_path).
    pub fn get_unresolved_call_edges(&self) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT cg.rowid, cg.callee_name, c.file_path
             FROM call_graph cg
             JOIN chunks c ON c.id = cg.caller_chunk_id
             WHERE cg.callee_chunk_id IS NULL",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Resolve a call edge by setting the callee chunk ID.
    pub fn update_callee_chunk_id(&self, rowid: i64, callee_chunk_id: u64) -> Result<()> {
        self.conn.execute(
            "UPDATE call_graph SET callee_chunk_id = ?1 WHERE rowid = ?2",
            params![callee_chunk_id as i64, rowid],
        )?;
        Ok(())
    }

    /// Get unresolved type references: (rowid, type_name, referencing_file_path).
    pub fn get_unresolved_type_refs(&self) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT tr.rowid, tr.type_name, c.file_path \
             FROM type_refs tr \
             JOIN chunks c ON c.id = tr.chunk_id \
             WHERE tr.defining_chunk IS NULL",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Resolve a type reference by setting the defining chunk ID.
    pub fn update_type_ref_defining_chunk(&self, rowid: i64, defining_chunk: u64) -> Result<()> {
        self.conn.execute(
            "UPDATE type_refs SET defining_chunk = ?1 WHERE rowid = ?2",
            params![defining_chunk as i64, rowid],
        )?;
        Ok(())
    }

    /// Get unresolved type hierarchy entries: (rowid, child_name, parent_name).
    pub fn get_unresolved_hierarchy(&self) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT rowid, child_name, parent_name FROM type_hierarchy
             WHERE child_chunk IS NULL OR parent_chunk IS NULL",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Resolve type hierarchy chunks for a given row.
    pub fn update_hierarchy_chunks(
        &self,
        rowid: i64,
        child_chunk: Option<u64>,
        parent_chunk: Option<u64>,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE type_hierarchy SET child_chunk = ?1, parent_chunk = ?2 WHERE rowid = ?3",
            params![
                child_chunk.map(|id| id as i64),
                parent_chunk.map(|id| id as i64),
                rowid
            ],
        )?;
        Ok(())
    }

    // ── v7 graph query methods (search-time propagation) ─────────────

    /// Get resolved call edges originating from the given caller chunk IDs:
    /// returns (caller_chunk_id, callee_chunk_id).
    pub fn get_call_edges_from(&self, caller_ids: &[u64]) -> Result<Vec<(u64, u64)>> {
        if caller_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders: String = caller_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT caller_chunk_id, callee_chunk_id FROM call_graph
             WHERE caller_chunk_id IN ({placeholders}) AND callee_chunk_id IS NOT NULL"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params: Vec<Box<dyn rusqlite::types::ToSql>> = caller_ids
            .iter()
            .map(|id| Box::new(*id as i64) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(std::convert::AsRef::as_ref).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get resolved call edges pointing to the given callee chunk IDs:
    /// returns (callee_chunk_id, caller_chunk_id).
    pub fn get_call_edges_to(&self, callee_ids: &[u64]) -> Result<Vec<(u64, u64)>> {
        if callee_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders: String = callee_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT callee_chunk_id, caller_chunk_id FROM call_graph
             WHERE callee_chunk_id IN ({placeholders})"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params: Vec<Box<dyn rusqlite::types::ToSql>> = callee_ids
            .iter()
            .map(|id| Box::new(*id as i64) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(std::convert::AsRef::as_ref).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get type reference edges from defining chunks to usage chunks:
    /// returns (defining_chunk, usage_chunk).
    pub fn get_type_ref_edges_to_defs(&self, def_chunk_ids: &[u64]) -> Result<Vec<(u64, u64)>> {
        if def_chunk_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders: String = def_chunk_ids
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT defining_chunk, chunk_id FROM type_refs
             WHERE defining_chunk IN ({placeholders})"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params: Vec<Box<dyn rusqlite::types::ToSql>> = def_chunk_ids
            .iter()
            .map(|id| Box::new(*id as i64) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(std::convert::AsRef::as_ref).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get type reference edges from usage chunks to their defining chunks:
    /// returns (usage_chunk, defining_chunk).
    pub fn get_type_ref_edges_from_usages(
        &self,
        usage_chunk_ids: &[u64],
    ) -> Result<Vec<(u64, u64)>> {
        if usage_chunk_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders: String = usage_chunk_ids
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT chunk_id, defining_chunk FROM type_refs
             WHERE chunk_id IN ({placeholders}) AND defining_chunk IS NOT NULL"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params: Vec<Box<dyn rusqlite::types::ToSql>> = usage_chunk_ids
            .iter()
            .map(|id| Box::new(*id as i64) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(std::convert::AsRef::as_ref).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get type hierarchy edges (both directions) for the given chunk IDs:
    /// returns (source_chunk, related_chunk).
    pub fn get_hierarchy_edges_for(&self, chunk_ids: &[u64]) -> Result<Vec<(u64, u64)>> {
        if chunk_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders: String = chunk_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        // Query both directions: chunk as child -> parent, and chunk as parent -> child
        let sql = format!(
            "SELECT child_chunk, parent_chunk FROM type_hierarchy
             WHERE child_chunk IN ({placeholders}) AND parent_chunk IS NOT NULL
             UNION ALL
             SELECT parent_chunk, child_chunk FROM type_hierarchy
             WHERE parent_chunk IN ({placeholders}) AND child_chunk IS NOT NULL",
        );
        let mut stmt = self.conn.prepare(&sql)?;
        // Parameters are used twice (for each half of UNION ALL)
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
            Vec::with_capacity(chunk_ids.len() * 2);
        for id in chunk_ids {
            param_values.push(Box::new(*id as i64));
        }
        for id in chunk_ids {
            param_values.push(Box::new(*id as i64));
        }
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = param_values
            .iter()
            .map(std::convert::AsRef::as_ref)
            .collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ── v7 file-level import neighbors (search-time) ──────────────────

    /// Get file paths that share an import relationship with any of the given paths.
    /// Returns files that import (or are imported by) the input files, excluding
    /// the input files themselves.
    pub fn get_import_neighbors(&self, file_paths: &[String]) -> Result<Vec<String>> {
        if file_paths.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_neighbors = Vec::new();
        let input_set: std::collections::HashSet<&str> =
            file_paths.iter().map(String::as_str).collect();

        for chunk in file_paths.chunks(500) {
            let placeholders: String = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT DISTINCT imported_path FROM module_edges WHERE importer_path IN ({placeholders}) \
                 UNION \
                 SELECT DISTINCT importer_path FROM module_edges WHERE imported_path IN ({placeholders})",
            );
            let mut stmt = self.conn.prepare(&sql)?;
            // Parameters used twice (one for each half of UNION)
            let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
                Vec::with_capacity(chunk.len() * 2);
            for path in chunk {
                param_values.push(Box::new(path.clone()));
            }
            for path in chunk {
                param_values.push(Box::new(path.clone()));
            }
            let param_refs: Vec<&dyn rusqlite::types::ToSql> = param_values
                .iter()
                .map(std::convert::AsRef::as_ref)
                .collect();

            let rows = stmt
                .query_map(param_refs.as_slice(), |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;

            for path in rows {
                if !input_set.contains(path.as_str()) {
                    all_neighbors.push(path);
                }
            }
        }

        Ok(all_neighbors)
    }

    // ── Phase 6: Exact symbol lookup (LSP-parity) ─────────────────────

    /// Look up chunks by exact symbol name. Returns (chunk_id, file_path, symbol_kind).
    /// Uses exact case-sensitive match only — prefix/LIKE matching caused false positives.
    pub fn lookup_symbol_exact(&self, symbol_name: &str) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT sd.chunk_id, c.file_path, sd.symbol_kind
             FROM symbol_defs sd
             JOIN chunks c ON c.id = sd.chunk_id
             WHERE sd.symbol_name = ?1
             ORDER BY sd.symbol_kind ASC
             LIMIT 10",
        )?;
        let rows = stmt
            .query_map(params![symbol_name], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Find chunk_ids that call or reference a given symbol name.
    pub fn find_references(&self, symbol_name: &str) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT caller_chunk_id FROM call_graph WHERE callee_name = ?1
             UNION
             SELECT DISTINCT chunk_id FROM type_refs WHERE type_name = ?1
             LIMIT 20",
        )?;
        let rows = stmt
            .query_map(params![symbol_name], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ── v7 cleanup ───────────────────────────────────────────────────

    /// Delete all graph data (symbol_defs, type_refs, module_edges, type_hierarchy)
    /// and v0.3 auxiliary rows (chunk_annotations, pattern_matches,
    /// chunk_centrality) associated with a given file path.
    ///
    /// The v0.3 tables are created by `init_auxiliary_schema` in builder.rs,
    /// which runs only after the indexing transaction (so on the very first
    /// build the tables may not yet exist when this is called). Each v0.3
    /// DELETE is guarded by a `sqlite_master` probe — a single indexed lookup
    /// (~tens of µs) that lets the function stay safe across all call sites
    /// (builder, validator, future search-time updaters).
    pub fn delete_graph_data_for_file(&self, file_path: &str) -> Result<()> {
        // symbol_defs has its own file_path column
        self.conn.execute(
            "DELETE FROM symbol_defs WHERE file_path = ?1",
            params![file_path],
        )?;

        // type_refs: delete rows where chunk_id belongs to chunks from this file
        self.conn.execute(
            "DELETE FROM type_refs WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?1)",
            params![file_path],
        )?;

        // module_edges: delete rows where this file is the importer
        self.conn.execute(
            "DELETE FROM module_edges WHERE importer_path = ?1",
            params![file_path],
        )?;

        // type_hierarchy: delete rows where child_chunk or parent_chunk belongs to this file.
        // NOTE: `?1` is a numbered placeholder reused twice — rusqlite counts unique
        // placeholders, so we must pass exactly one value (passing two raised
        // InvalidParameterCount, which was being silently swallowed by `let _ =`
        // at the call sites in builder.rs).
        self.conn.execute(
            "DELETE FROM type_hierarchy WHERE child_chunk IN (SELECT id FROM chunks WHERE file_path = ?1)
             OR parent_chunk IN (SELECT id FROM chunks WHERE file_path = ?1)",
            params![file_path],
        )?;

        // ── v0.3 auxiliary tables ─────────────────────────────────────
        // Created by `init_auxiliary_schema` in builder.rs after the indexing
        // pass. Guard each DELETE with a `sqlite_master` probe so we never
        // raise "no such table" on a fresh / partially-built index.

        if self.table_exists("chunk_annotations")? {
            self.conn.execute(
                "DELETE FROM chunk_annotations WHERE chunk_id IN \
                 (SELECT id FROM chunks WHERE file_path = ?1)",
                params![file_path],
            )?;
        }

        if self.table_exists("pattern_matches")? {
            self.conn.execute(
                "DELETE FROM pattern_matches WHERE chunk_id IN \
                 (SELECT id FROM chunks WHERE file_path = ?1)",
                params![file_path],
            )?;
        }

        if self.table_exists("chunk_centrality")? {
            self.conn.execute(
                "DELETE FROM chunk_centrality WHERE chunk_id IN \
                 (SELECT id FROM chunks WHERE file_path = ?1)",
                params![file_path],
            )?;
        }

        Ok(())
    }

    /// Returns true when a table with `name` is present in this database.
    /// Uses an indexed lookup on `sqlite_master` — sub-100µs in practice.
    fn table_exists(&self, name: &str) -> Result<bool> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1",
                params![name],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        Ok(exists)
    }

    /// Create the `chunk_centrality` aux table if missing and upsert one
    /// per-chunk PageRank / structural-centrality score. Symmetric counterpart
    /// to [`Self::get_centrality_scores`].
    pub fn insert_centrality_score(&self, chunk_id: u64, centrality: f64) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunk_centrality (
                chunk_id INTEGER PRIMARY KEY,
                structural_centrality REAL NOT NULL
            );",
        )?;
        self.conn.execute(
            "INSERT OR REPLACE INTO chunk_centrality (chunk_id, structural_centrality) \
             VALUES (?1, ?2)",
            params![chunk_id as i64, centrality],
        )?;
        Ok(())
    }

    /// Map file paths to their chunk ids. Generic accessor used by the
    /// search-time import-cohesion expansion. Unknown paths contribute nothing;
    /// empty input returns an empty vec.
    pub fn get_chunk_ids_for_files(&self, file_paths: &[String]) -> Result<Vec<u64>> {
        if file_paths.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for batch in file_paths.chunks(500) {
            let placeholders: String = batch.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!("SELECT id FROM chunks WHERE file_path IN ({placeholders})");
            let mut stmt = self.conn.prepare(&sql)?;
            let params: Vec<&dyn rusqlite::types::ToSql> = batch
                .iter()
                .map(|p| p as &dyn rusqlite::types::ToSql)
                .collect();
            let rows = stmt.query_map(params.as_slice(), |r| Ok(r.get::<_, i64>(0)? as u64))?;
            for r in rows {
                out.push(r?);
            }
        }
        Ok(out)
    }

    /// Read stored PageRank / structural centrality for the given chunk ids.
    /// Returns an empty map when the aux table is absent (older index) so the
    /// centrality prior is a clean no-op on un-enriched indexes.
    pub fn get_centrality_scores(&self, chunk_ids: &[u64]) -> Result<HashMap<u64, f32>> {
        if chunk_ids.is_empty() || !self.table_exists("chunk_centrality")? {
            return Ok(HashMap::new());
        }
        let placeholders: String = chunk_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT chunk_id, structural_centrality FROM chunk_centrality \
             WHERE chunk_id IN ({placeholders})"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let params: Vec<Box<dyn rusqlite::types::ToSql>> = chunk_ids
            .iter()
            .map(|id| Box::new(*id as i64) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(std::convert::AsRef::as_ref).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |r| {
            Ok((r.get::<_, i64>(0)? as u64, r.get::<_, f64>(1)? as f32))
        })?;
        Ok(rows.collect::<rusqlite::Result<HashMap<u64, f32>>>()?)
    }

    // ── v7 stats ─────────────────────────────────────────────────────

    /// Return statistics about the code graph tables.
    pub fn graph_stats(&self) -> Result<GraphStats> {
        let calls_resolved: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM call_graph WHERE callee_chunk_id IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        let types_resolved: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM type_refs WHERE defining_chunk IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        let hierarchy_resolved: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM type_hierarchy WHERE child_chunk IS NOT NULL AND parent_chunk IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        let module_edges_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM module_edges", [], |row| row.get(0))?;
        let symbol_defs_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM symbol_defs", [], |row| row.get(0))?;

        Ok(GraphStats {
            calls_resolved: calls_resolved as usize,
            types_resolved: types_resolved as usize,
            hierarchy_resolved: hierarchy_resolved as usize,
            module_edges_count: module_edges_count as usize,
            symbol_defs_count: symbol_defs_count as usize,
        })
    }
}

// ════════════════════════════════════════════════════════════════════════
// E8(a) + E8(b): mmap'd file helper + parallel index prefetch
// ════════════════════════════════════════════════════════════════════════
//
// These helpers are called by the daemon at startup to (1) memory-map a single
// index file (Unix; falls back to read on Windows) and (2) warm the OS page
// cache for the heaviest index files in parallel, so that
// `HybridSearcher::open` does cache-warm sequential reads rather than cold disk
// I/O.
//
// We deliberately do NOT re-implement `HybridSearcher::open` — that lives in
// `search/hybrid.rs` (owned by W1). The page-cache prefetch achieves the
// same wall-clock latency win as true parallel opening since the bulk of
// "open" cost is I/O, and parallelising disk reads is what `rayon::join3`
// here actually delivers.

/// Memory-mapped bytes from a file (Unix). On Windows, holds an owned `Vec<u8>`.
///
/// This abstraction lets callers treat an index file as a `&[u8]` regardless of
/// platform, without forcing every consumer to handle two distinct types.
pub struct MappedBytes {
    #[cfg(not(target_os = "windows"))]
    inner: Mmap,
    #[cfg(target_os = "windows")]
    inner: Vec<u8>,
}

impl MappedBytes {
    /// Open a file via mmap (Unix) or `read` (Windows fallback per spec risk T5).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or mapped.
    pub fn open(path: &Path) -> Result<Self> {
        #[cfg(not(target_os = "windows"))]
        {
            let file = File::open(path)
                .with_context(|| format!("Failed to open {} for mmap", path.display()))?;
            // SAFETY: We hold the File open for the lifetime of the Mmap. If
            // another process truncates the file the mapping becomes UB, but
            // index files are written once at build and never mutated in place
            // (rebuild deletes and recreates).
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("Failed to mmap {}", path.display()))?;
            Ok(Self { inner: mmap })
        }
        #[cfg(target_os = "windows")]
        {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            Ok(Self { inner: bytes })
        }
    }

    /// Borrow the underlying bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.inner
    }

    /// Returns true if backed by a real memory mapping (vs. fallback read).
    /// Test/diagnostics helper only.
    #[doc(hidden)]
    pub fn is_mmapped(&self) -> bool {
        cfg!(not(target_os = "windows"))
    }
}

/// Outcome of a parallel index prefetch — what we touched, in nanoseconds.
///
/// This is informational only; the returned struct is used by tests and by
/// the cold-start benchmark. Production paths discard the result.
#[derive(Debug, Default)]
pub struct PrefetchOutcome {
    pub sqlite_ns: u128,
    pub sparse_ns: u128,
    pub dense_ns: u128,
    pub sqlite_ok: bool,
    pub sparse_ok: bool,
    pub dense_ok: bool,
}

/// Prefetch the three heavy index components in parallel, warming the OS page
/// cache so that `HybridSearcher::open` runs entirely against cached pages.
///
/// Uses nested `rayon::join` to fan out one task per component:
/// - SQLite chunk store (`chunks.db`)
/// - Tantivy sparse index (`sparse/`)
/// - Dense index — the coderank-hnsw vector store
///   (`dense/coderank-hnsw/vectors.bin`); the HNSW graph itself is rebuilt in
///   RAM on open from this file, so warming it warms the whole dense channel.
///
/// Each task touches its files via `std::fs::read` (or a directory walk for
/// Tantivy); failures are recorded in [`PrefetchOutcome`] but never propagated
/// — prefetch is a best-effort latency optimization, and any actual I/O error
/// will surface again when `HybridSearcher::open` runs.
///
/// Returns immediately if `index_dir` does not exist.
pub fn prefetch_index_files(index_dir: &Path) -> PrefetchOutcome {
    if !index_dir.exists() {
        return PrefetchOutcome::default();
    }

    let sqlite_path = index_dir.join("chunks.db");
    let sparse_dir = index_dir.join("sparse");
    // The dense vector store the daemon actually opens. Backend-resolution lives
    // in `hybrid.rs`; here we warm the sole built-in backend's store file
    // directly (a best-effort page-cache touch — a missing file just yields
    // dense_ok=false, exactly as the old plaid path did when absent).
    let dense_store = crate::search::dense_backend::dense_subdir(
        index_dir,
        crate::search::dense_backend::DenseBackendKind::CoderankHnsw,
    )
    .join("vectors.bin");

    // E8(b): fan out the three reads to the rayon thread pool. Rayon only
    // provides binary `join`, so we nest one call inside the other —
    // equivalent semantics to a hypothetical `join3` and what rayon's own
    // documentation recommends for three-way parallelism. All three tasks
    // run concurrently; this function blocks until all complete. Errors are
    // swallowed into the per-task `ok` flag — the goal is warming the OS
    // page cache, not strict correctness.
    let ((sqlite_ns, sqlite_ok), ((sparse_ns, sparse_ok), (dense_ns, dense_ok))) = rayon::join(
        || prefetch_file_timed(&sqlite_path),
        || {
            rayon::join(
                || prefetch_dir_timed(&sparse_dir),
                || {
                    if dense_store.exists() {
                        prefetch_file_timed(&dense_store)
                    } else {
                        (0, false)
                    }
                },
            )
        },
    );

    PrefetchOutcome {
        sqlite_ns,
        sparse_ns,
        dense_ns,
        sqlite_ok,
        sparse_ok,
        dense_ok,
    }
}

/// Read the entire file into a discarded buffer, timing the operation.
/// Used to warm the OS page cache. Returns `(nanos, ok)`.
fn prefetch_file_timed(path: &Path) -> (u128, bool) {
    let t = std::time::Instant::now();
    let ok = std::fs::read(path).is_ok();
    (t.elapsed().as_nanos(), ok)
}

/// Recursively read every file under `dir`, timing the operation.
/// Used to warm the OS page cache for Tantivy's segment files. Returns
/// `(nanos, ok)` where `ok` is true if at least one file was read.
fn prefetch_dir_timed(dir: &Path) -> (u128, bool) {
    let t = std::time::Instant::now();
    if !dir.is_dir() {
        return (t.elapsed().as_nanos(), false);
    }
    let mut any_ok = false;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && std::fs::read(&path).is_ok() {
                any_ok = true;
            }
        }
    }
    (t.elapsed().as_nanos(), any_ok)
}

#[cfg(test)]
mod e8_storage_tests {
    use super::*;
    use tempfile::TempDir;

    /// `MappedBytes::open` should succeed on a valid file and expose the raw bytes,
    /// regardless of whether it's backed by mmap (Unix) or `read` (Windows fallback).
    #[test]
    fn mapped_bytes_open_round_trips() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let mapped = MappedBytes::open(&path).expect("open should succeed");
        assert_eq!(mapped.as_bytes(), b"hello world");
    }

    /// On Unix targets we expect the mapping to be a real mmap; on Windows
    /// (or any cfg(target_os = "windows") build) we expect the read fallback.
    #[test]
    fn mapped_bytes_uses_mmap_on_unix() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.bin");
        std::fs::write(&path, b"x").unwrap();
        let mapped = MappedBytes::open(&path).unwrap();
        #[cfg(not(target_os = "windows"))]
        assert!(mapped.is_mmapped(), "expected mmap on non-Windows target");
        #[cfg(target_os = "windows")]
        assert!(!mapped.is_mmapped(), "expected read fallback on Windows");
    }

    /// `MappedBytes::open` must error on a missing path (so callers can fall
    /// back rather than silently hold an empty buffer).
    #[test]
    fn mapped_bytes_open_fails_on_missing_file() {
        let res = MappedBytes::open(Path::new("/nonexistent/path/that/does/not/exist.bin"));
        assert!(res.is_err());
    }

    /// `prefetch_index_files` on an empty directory should return all-zero
    /// outcome without panicking — the daemon may call this before any
    /// indexing has happened.
    #[test]
    fn prefetch_empty_dir_returns_default() {
        let tmp = TempDir::new().unwrap();
        let outcome = prefetch_index_files(tmp.path());
        assert!(!outcome.sqlite_ok);
        assert!(!outcome.sparse_ok);
        assert!(!outcome.dense_ok);
    }

    /// `prefetch_index_files` on a missing directory should return all-zero
    /// outcome (no panic, no error).
    #[test]
    fn prefetch_missing_dir_returns_default() {
        let outcome = prefetch_index_files(Path::new("/nonexistent/path/that/does/not/exist"));
        assert!(!outcome.sqlite_ok);
        assert!(!outcome.sparse_ok);
        assert!(!outcome.dense_ok);
    }

    /// When the SQLite file and the coderank-hnsw vector store exist, the
    /// prefetch should report `sqlite_ok` and `dense_ok` true.
    #[test]
    fn prefetch_reports_ok_for_present_files() {
        use crate::search::dense_backend::{DenseBackendKind, dense_subdir};
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("chunks.db"), b"fake").unwrap();
        let dense_dir = dense_subdir(tmp.path(), DenseBackendKind::CoderankHnsw);
        std::fs::create_dir_all(&dense_dir).unwrap();
        std::fs::write(dense_dir.join("vectors.bin"), b"y").unwrap();
        let outcome = prefetch_index_files(tmp.path());
        assert!(outcome.sqlite_ok);
        assert!(outcome.dense_ok);
    }
}

#[cfg(test)]
mod delete_graph_data_tests {
    use super::*;
    use crate::types::{Chunk, ChunkType};
    use tempfile::TempDir;

    /// Helper: open a temp ChunkStore plus create the v0.3 auxiliary tables
    /// (mirror of `init_auxiliary_schema` in builder.rs — kept here so the
    /// storage-layer test does not depend on the builder module).
    fn create_v0_3_aux_tables(store: &ChunkStore) {
        store
            .conn
            .execute_batch(
                "
                CREATE TABLE IF NOT EXISTS chunk_annotations (
                    chunk_id INTEGER PRIMARY KEY,
                    nl_annotation TEXT NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
                );

                CREATE TABLE IF NOT EXISTS pattern_matches (
                    chunk_id     INTEGER NOT NULL,
                    pattern_name TEXT NOT NULL,
                    language     TEXT NOT NULL,
                    description  TEXT NOT NULL,
                    file_path    TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, pattern_name),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
                );

                CREATE TABLE IF NOT EXISTS chunk_centrality (
                    chunk_id INTEGER PRIMARY KEY,
                    structural_centrality REAL NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
                );
                ",
            )
            .unwrap();
    }

    fn insert_test_chunk(store: &ChunkStore, file_path: &str) -> u64 {
        let chunk = Chunk {
            id: 0,
            file_path: PathBuf::from(file_path),
            start_line: 1,
            end_line: 10,
            content: "fn main() {}".to_string(),
            chunk_type: ChunkType::TextWindow { window_index: 0 },
        };
        store.insert_chunk(&chunk, 0xdead_beef, 0).unwrap()
    }

    /// Regression for Finding 9: `delete_graph_data_for_file` must clear
    /// rows in the v0.3 auxiliary tables (chunk_annotations,
    /// pattern_matches, chunk_centrality) so incremental reindex does not
    /// leave orphan chunk_id references.
    #[test]
    fn delete_clears_v0_3_aux_rows() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        create_v0_3_aux_tables(&store);

        let file = "src/lib.rs";
        let cid = insert_test_chunk(&store, file);

        // Populate one row in each v0.3 table for the chunk
        store
            .conn
            .execute(
                "INSERT INTO chunk_annotations (chunk_id, nl_annotation) VALUES (?1, ?2)",
                params![cid as i64, "an annotation"],
            )
            .unwrap();
        store
            .conn
            .execute(
                "INSERT INTO pattern_matches (chunk_id, pattern_name, language, description, file_path) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![cid as i64, "test_pattern", "rust", "a test pattern", file],
            )
            .unwrap();
        store
            .conn
            .execute(
                "INSERT INTO chunk_centrality (chunk_id, structural_centrality) VALUES (?1, ?2)",
                params![cid as i64, 0.42_f64],
            )
            .unwrap();

        // Sanity: rows are present
        let count_annotations = |s: &ChunkStore| -> i64 {
            s.conn
                .query_row("SELECT COUNT(*) FROM chunk_annotations", [], |r| r.get(0))
                .unwrap()
        };
        let count_patterns = |s: &ChunkStore| -> i64 {
            s.conn
                .query_row("SELECT COUNT(*) FROM pattern_matches", [], |r| r.get(0))
                .unwrap()
        };
        let count_centrality = |s: &ChunkStore| -> i64 {
            s.conn
                .query_row("SELECT COUNT(*) FROM chunk_centrality", [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(count_annotations(&store), 1);
        assert_eq!(count_patterns(&store), 1);
        assert_eq!(count_centrality(&store), 1);

        // The function under test
        store.delete_graph_data_for_file(file).unwrap();

        // All v0.3 aux rows for this file's chunks must be gone
        assert_eq!(
            count_annotations(&store),
            0,
            "chunk_annotations not cleared"
        );
        assert_eq!(count_patterns(&store), 0, "pattern_matches not cleared");
        assert_eq!(count_centrality(&store), 0, "chunk_centrality not cleared");
    }

    /// `delete_graph_data_for_file` on a *fresh* DB where the v0.3 tables
    /// have not yet been created (e.g. very first build, before
    /// `init_auxiliary_schema` runs) must NOT raise "no such table".
    #[test]
    fn delete_is_safe_when_v0_3_tables_missing() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();

        let file = "src/lib.rs";
        let _cid = insert_test_chunk(&store, file);

        // Do NOT create the v0.3 aux tables. The deletion must still succeed.
        store
            .delete_graph_data_for_file(file)
            .expect("delete must be safe when v0.3 tables are absent");
    }

    /// `delete_graph_data_for_file` should leave aux rows for *other* files
    /// untouched (regression guard against an overly broad DELETE).
    #[test]
    fn delete_only_clears_target_file_rows() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        create_v0_3_aux_tables(&store);

        let cid_a = insert_test_chunk(&store, "a.rs");
        let cid_b = insert_test_chunk(&store, "b.rs");

        for cid in [cid_a, cid_b] {
            store
                .conn
                .execute(
                    "INSERT INTO chunk_annotations (chunk_id, nl_annotation) VALUES (?1, ?2)",
                    params![cid as i64, "x"],
                )
                .unwrap();
        }

        store.delete_graph_data_for_file("a.rs").unwrap();

        let remaining: Vec<i64> = store
            .conn
            .prepare("SELECT chunk_id FROM chunk_annotations")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(remaining, vec![cid_b as i64]);
    }

    /// Regression: prior to Finding 9 the `type_hierarchy` DELETE used
    /// `?1` twice but passed `params![file_path, file_path]`, which made
    /// rusqlite return `InvalidParameterCount(2, 1)`. The error was masked
    /// by `let _ = ...` at the call sites. After the fix, the call must
    /// succeed against a populated type_hierarchy row.
    #[test]
    fn delete_clears_type_hierarchy_without_param_count_error() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();

        let file = "src/lib.rs";
        let cid = insert_test_chunk(&store, file);
        // Insert a hierarchy row with this chunk as child
        store
            .conn
            .execute(
                "INSERT INTO type_hierarchy (child_name, parent_name, relation, child_chunk, parent_chunk) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params!["Child", "Parent", "extends", cid as i64, cid as i64],
            )
            .unwrap();

        let count_before: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM type_hierarchy", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count_before, 1);

        store.delete_graph_data_for_file(file).unwrap();

        let count_after: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM type_hierarchy", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            count_after, 0,
            "type_hierarchy row should have been deleted"
        );
    }

    /// `table_exists` correctly identifies present vs absent tables.
    #[test]
    fn table_exists_probe_works() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        // `chunks` is created by `init_schema`
        assert!(store.table_exists("chunks").unwrap());
        // v0.3 aux tables are NOT created by `init_schema`
        assert!(!store.table_exists("chunk_annotations").unwrap());
        assert!(!store.table_exists("pattern_matches").unwrap());
        assert!(!store.table_exists("chunk_centrality").unwrap());

        create_v0_3_aux_tables(&store);
        assert!(store.table_exists("chunk_annotations").unwrap());
        assert!(store.table_exists("pattern_matches").unwrap());
        assert!(store.table_exists("chunk_centrality").unwrap());
    }

    /// `get_centrality_scores` returns stored PageRank for present chunks and
    /// omits chunks without a row (caller treats missing as 0). On an index
    /// lacking the aux table it returns an empty map (clean no-op).
    #[test]
    fn get_centrality_scores_reads_stored_pagerank() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();

        // No aux table yet → empty map.
        let id_probe = insert_test_chunk(&store, "a.rs");
        assert!(
            store.get_centrality_scores(&[id_probe]).unwrap().is_empty(),
            "missing aux table must yield empty map"
        );

        create_v0_3_aux_tables(&store);
        let id1 = insert_test_chunk(&store, "b.rs");
        let id2 = insert_test_chunk(&store, "c.rs");
        let id3 = insert_test_chunk(&store, "d.rs"); // no centrality row

        for (cid, val) in [(id1, 0.80_f64), (id2, 0.20_f64)] {
            store
                .conn
                .execute(
                    "INSERT INTO chunk_centrality (chunk_id, structural_centrality) VALUES (?1, ?2)",
                    params![cid as i64, val],
                )
                .unwrap();
        }

        let scores = store.get_centrality_scores(&[id1, id2, id3]).unwrap();
        assert_eq!(scores.len(), 2, "only chunks with rows are returned");
        assert!((scores[&id1] - 0.80).abs() < 1e-6);
        assert!((scores[&id2] - 0.20).abs() < 1e-6);
        assert!(!scores.contains_key(&id3), "chunk without row is absent");

        // Empty input → empty map.
        assert!(store.get_centrality_scores(&[]).unwrap().is_empty());
    }

    /// `get_chunk_ids_for_files` maps file paths to their chunk ids and ignores
    /// unknown paths and empty input.
    #[test]
    fn get_chunk_ids_for_files_maps_paths_to_chunks() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();

        let a = insert_test_chunk(&store, "src/a.rs");
        let b = insert_test_chunk(&store, "src/b.rs");

        let mut got = store
            .get_chunk_ids_for_files(&["src/a.rs".to_string(), "src/b.rs".to_string()])
            .unwrap();
        got.sort_unstable();
        let mut want = vec![a, b];
        want.sort_unstable();
        assert_eq!(got, want);

        // Unknown path → no ids; empty input → empty.
        assert!(
            store
                .get_chunk_ids_for_files(&["nope.rs".to_string()])
                .unwrap()
                .is_empty()
        );
        assert!(store.get_chunk_ids_for_files(&[]).unwrap().is_empty());
    }
}
