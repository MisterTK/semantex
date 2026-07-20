//! Project memory — durable, project-scoped notes an agent saves across
//! sessions (v13 Wave 2, MEMORY workstream).
//!
//! Not to be confused with [`crate::memory`] (process RSS monitoring /
//! host-memory caps) — that name was already taken at the crate root, so
//! this lives under `index::memory` instead, next to [`crate::index::layout`]
//! which owns the `memory.db` schema/path.
//!
//! # Schema conformance
//!
//! [`layout::open_memory_db`] (spine-owned) creates the baseline `notes`
//! table per contract §A:
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS notes (
//!     id         INTEGER PRIMARY KEY AUTOINCREMENT,
//!     created_ts INTEGER NOT NULL,
//!     updated_ts INTEGER NOT NULL,
//!     scope      TEXT NOT NULL,
//!     key        TEXT NOT NULL,
//!     content    TEXT NOT NULL,
//!     source     TEXT NOT NULL,
//!     UNIQUE(scope, key)
//! );
//! ```
//!
//! That schema is insufficient for the Wave 2 tool surface in two ways: (1)
//! there's no column for freeform tags, and (2) `save(content, scope, tags)`
//! has no natural value for `key` — the contract's `UNIQUE(scope, key)`
//! reads as "callers dedupe/upsert by an explicit key", but the Wave 2 tool
//! surface is append-only (`save` always creates a new note; there is no
//! `update`). This module extends the schema **additively** (own file, not
//! `layout.rs` — see the module ownership note below) via a migration guard
//! that runs on every [`MemoryStore::open`]:
//!
//! 1. `ALTER TABLE notes ADD COLUMN tags TEXT NOT NULL DEFAULT ''` if the
//!    column isn't already present (checked via `pragma_table_info`).
//! 2. An external-content FTS5 index (`notes_fts`) over `content`/`tags`,
//!    plus insert/update/delete triggers to keep it in sync, if FTS5 is
//!    available (verified empirically against the bundled rusqlite/libsqlite3
//!    in this workspace — `libsqlite3-sys`'s bundled build.rs always passes
//!    `-DSQLITE_ENABLE_FTS5`; still, the fallback below is defensive against
//!    a future bundled-build change or an unbundled system libsqlite3).
//!    A freshly-created `notes_fts` is populated via the FTS5 `'rebuild'`
//!    command so pre-existing rows (if any) are indexed too.
//!
//! Both migrations are idempotent (guarded by an existence check) and run
//! every time `open()` is called, so a project's `memory.db` upgrades itself
//! automatically regardless of which schema version created it.
//!
//! `key` is treated as an internal per-row uniqueness token (see
//! [`generate_key`]) rather than a caller-supplied field — nothing in the
//! Wave 2 tool surface needs scope-scoped upsert semantics, so we satisfy the
//! `UNIQUE(scope, key)` constraint without changing what callers pass. If a
//! future workstream wants explicit upsert-by-key, `key` is already sitting
//! there for it.
//!
//! `source` is not part of the Wave 2 `save(content, scope, tags)` surface
//! either; it's set to a fixed value (see [`NOTE_SOURCE`]) for every note
//! this module writes, again leaving the column free for a future caller
//! that wants to distinguish writers (CLI vs MCP vs some other agent).
//!
//! # Branch independence
//!
//! `memory.db` (like `history.db`) already lives directly under
//! `<project>/.semantex/` (see [`layout::memory_db_path`]) and is **not**
//! part of either per-branch mirror list: [`layout::mirror_root_as`] only
//! copies/hard-links `chunks.db`, `meta.json`, `models.toml`, `dense/**`,
//! `sparse/`, and `branch.json`; [`layout::restore_branch_dir_into_root`]
//! only ever touches that same set in reverse. Neither function's file list
//! mentions `memory.db`. So project memory is branch-independent *by
//! omission*, with no layout.rs edit required: a `git switch` mirrors/
//! restores the *code index* (chunks/dense/sparse/meta) but leaves
//! `memory.db` untouched at the container root throughout — notes survive
//! branch switches unchanged, and are shared across every branch of the same
//! project. See `branch_switch_leaves_memory_db_untouched` below for a test
//! that exercises this against the real mirror/restore functions.
//!
//! # Growth bound
//!
//! [`save`](MemoryStore::save) enforces a cap on the number of notes per
//! project (default [`DEFAULT_MAX_NOTES`], overridable via
//! [`MAX_NOTES_ENV`]). Oldest notes (by `created_ts`, ties broken by `id`)
//! are evicted once the cap is exceeded, with a `tracing::warn!`.

use crate::index::layout;
use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default cap on notes retained per project. Overridable via
/// [`MAX_NOTES_ENV`]. Deliberately generous — a note is a short freeform
/// string, and 1000 of them is a few hundred KB at most — while still
/// bounding unlimited growth over a long-lived project.
pub const DEFAULT_MAX_NOTES: usize = 1000;

/// Env var overriding [`DEFAULT_MAX_NOTES`]. `0` means "no cap" (not
/// recommended — memory.db would grow without bound).
pub const MAX_NOTES_ENV: &str = "SEMANTEX_MEMORY_MAX_NOTES";

/// Fixed `source` value this module writes for every note (see module doc).
const NOTE_SOURCE: &str = "agent";

/// Retry budget for the (astronomically unlikely) `UNIQUE(scope, key)`
/// collision on insert — see [`generate_key`]'s doc.
const MAX_SAVE_ATTEMPTS: u32 = 5;

/// Suggested `scope` conventions (freeform — not enforced). Documented here
/// so both the core API docs and the MCP tool descriptions can point at a
/// single source of truth.
pub const SCOPE_CONVENTIONS: &[&str] =
    &["global", "file:<rel_path>", "module:<dir>", "task:<slug>"];

/// A single project-memory note.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Note {
    pub id: i64,
    pub created_ts: i64,
    pub updated_ts: i64,
    pub scope: String,
    pub content: String,
    pub tags: Vec<String>,
    pub source: String,
}

fn parse_tags(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn format_tags(tags: &[String]) -> String {
    let mut cleaned: Vec<String> = tags
        .iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    cleaned.sort();
    cleaned.dedup();
    cleaned.join(",")
}

fn now_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

/// Per-process counter mixed into [`generate_key`] so two saves in the same
/// nanosecond within one process still get distinct keys.
static KEY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate an internal per-row uniqueness token for the `UNIQUE(scope, key)`
/// constraint (see module doc for why `key` isn't a caller-supplied field).
/// Nanosecond timestamp + a per-process monotonic counter + pid is
/// vanishingly unlikely to collide even under concurrent writers; `save`
/// still retries on the rare `UNIQUE` constraint violation defensively.
fn generate_key() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let counter = KEY_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{counter:x}-{:x}", std::process::id())
}

/// Read `SEMANTEX_MEMORY_MAX_NOTES`, falling back to [`DEFAULT_MAX_NOTES`] if
/// unset or unparseable.
fn max_notes() -> usize {
    std::env::var(MAX_NOTES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_NOTES)
}

/// Escape a single token for safe use inside an FTS5 MATCH string: wrapping
/// in double quotes turns off FTS5's query-syntax operators (`-`, `:`, `*`,
/// parens, etc.) for that token, so arbitrary user text can never produce an
/// FTS5 syntax error. Internal double quotes are doubled per FTS5's escaping
/// rule for quoted strings.
fn escape_fts5_token(token: &str) -> String {
    format!("\"{}\"", token.replace('"', "\"\""))
}

/// Build an FTS5 MATCH expression that matches any of `query`'s whitespace-
/// separated tokens (OR'd), each individually escaped. Returns `None` if
/// `query` has no non-empty tokens (nothing to match on).
fn build_match_query(query: &str) -> Option<String> {
    let tokens: Vec<String> = query
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .map(escape_fts5_token)
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" OR "))
    }
}

/// Handle to a project's `memory.db`. Open once per call site (cheap — a
/// short-lived SQLite connection with a couple of PRAGMAs); notes volume is
/// small (capped, see [`DEFAULT_MAX_NOTES`]) so there's no need for the
/// longer-lived-connection ceremony `ChunkStore` uses for the much larger
/// chunk index.
pub struct MemoryStore {
    conn: Connection,
    fts5_enabled: bool,
}

impl MemoryStore {
    /// Open (creating if absent) `<project_root>/.semantex/memory.db`,
    /// ensure the spine's baseline schema exists, and apply this module's
    /// additive migrations (tags column, FTS5 index) if not already applied.
    pub fn open(project_root: &Path) -> Result<Self> {
        let conn = layout::open_memory_db(&layout::memory_db_path(project_root))
            .context("Failed to open memory.db")?;
        conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;")?;
        let fts5_enabled = ensure_migrations(&conn)?;
        Ok(Self { conn, fts5_enabled })
    }

    /// Open an in-memory / arbitrary-path store directly (test helper — real
    /// callers always go through [`Self::open`] with a project root).
    #[cfg(test)]
    fn open_at(db_path: &Path) -> Result<Self> {
        let conn = layout::open_memory_db(db_path)?;
        let fts5_enabled = ensure_migrations(&conn)?;
        Ok(Self { conn, fts5_enabled })
    }

    /// Whether the FTS5 match-quality path is active for this connection
    /// (vs. the tokenized-LIKE fallback). Exposed for tests/diagnostics.
    #[must_use]
    pub fn fts5_enabled(&self) -> bool {
        self.fts5_enabled
    }

    /// Save a new note. Returns the new note's id.
    ///
    /// `scope` is a freeform string (see [`SCOPE_CONVENTIONS`] for suggested
    /// values) and must not be empty. `content` must not be empty. `tags`
    /// are normalized (trimmed, deduped, sorted) before storage.
    ///
    /// After inserting, enforces the per-project note cap (see
    /// [`DEFAULT_MAX_NOTES`] / [`MAX_NOTES_ENV`]): if the cap is exceeded,
    /// the oldest notes beyond it are evicted with a `tracing::warn!`.
    pub fn save(&self, content: &str, scope: &str, tags: &[String]) -> Result<i64> {
        let content = content.trim();
        if content.is_empty() {
            bail!("memory note content must not be empty");
        }
        let scope = scope.trim();
        if scope.is_empty() {
            bail!("memory note scope must not be empty");
        }
        let tags_str = format_tags(tags);
        let ts = now_ts();

        // Retry loop covers the astronomically unlikely `UNIQUE(scope, key)`
        // collision (see `generate_key`'s doc) — never expected to loop more
        // than once in practice.
        let mut last_err = None;
        for _ in 0..MAX_SAVE_ATTEMPTS {
            let key = generate_key();
            let result = self.conn.execute(
                "INSERT INTO notes (created_ts, updated_ts, scope, key, content, source, tags)
                 VALUES (?1, ?1, ?2, ?3, ?4, ?5, ?6)",
                params![ts, scope, key, content, NOTE_SOURCE, tags_str],
            );
            match result {
                Ok(_) => {
                    let id = self.conn.last_insert_rowid();
                    self.enforce_cap()?;
                    return Ok(id);
                }
                Err(e) if is_unique_violation(&e) => {
                    last_err = Some(e);
                }
                Err(e) => return Err(e).context("Failed to insert memory note"),
            }
        }
        Err(anyhow::anyhow!(
            "Failed to insert memory note after {MAX_SAVE_ATTEMPTS} attempts (key collisions): {last_err:?}"
        ))
    }

    /// Recall notes matching `query` (best matches first), optionally
    /// filtered by `scope`. `limit` is clamped to `[1, 200]`.
    ///
    /// If `query` is `None` or has no matchable tokens, behaves like
    /// [`Self::list`] (most-recent-first, no relevance scoring).
    ///
    /// Uses FTS5 `bm25()` ranking when available (see the module doc for
    /// when it isn't); otherwise falls back to a tokenized-LIKE match count
    /// (notes containing more of the query's distinct tokens rank higher),
    /// with recency as the tiebreaker either way.
    pub fn recall(
        &self,
        query: Option<&str>,
        scope: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Note>> {
        let limit = limit.clamp(1, 200);
        let query = query.map(str::trim).filter(|q| !q.is_empty());

        let Some(query) = query else {
            return self.list(scope, limit);
        };

        if self.fts5_enabled
            && let Some(match_expr) = build_match_query(query)
        {
            return self.recall_fts5(&match_expr, scope, limit);
        }
        self.recall_like(query, scope, limit)
    }

    fn recall_fts5(
        &self,
        match_expr: &str,
        scope: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Note>> {
        let mut stmt = self.conn.prepare(
            "SELECT n.id, n.created_ts, n.updated_ts, n.scope, n.content, n.tags, n.source
             FROM notes_fts
             JOIN notes n ON n.id = notes_fts.rowid
             WHERE notes_fts MATCH ?1
               AND (?2 IS NULL OR n.scope = ?2)
             ORDER BY bm25(notes_fts, 3.0, 1.0) ASC, n.created_ts DESC
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![match_expr, scope, limit as i64], row_to_note)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("Failed to recall memory notes (fts5 path)")
    }

    fn recall_like(&self, query: &str, scope: Option<&str>, limit: usize) -> Result<Vec<Note>> {
        let tokens: Vec<String> = query
            .split_whitespace()
            .filter(|t| !t.is_empty())
            .map(str::to_lowercase)
            .collect();
        if tokens.is_empty() {
            return self.list(scope, limit);
        }

        // Score = number of distinct query tokens found in content or tags
        // (case-insensitive substring match). Built as a single SQL
        // expression so scoring/ordering happens in SQLite, not Rust.
        let mut score_expr = String::new();
        let mut sql_params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        for token in &tokens {
            if !score_expr.is_empty() {
                score_expr.push_str(" + ");
            }
            score_expr.push_str(
                "(CASE WHEN instr(lower(content), ?) > 0 OR instr(lower(tags), ?) > 0 THEN 1 ELSE 0 END)",
            );
            let like_token = format!("%{token}%");
            sql_params.push(Box::new(like_token.clone()));
            sql_params.push(Box::new(like_token));
        }

        let sql = format!(
            "SELECT id, created_ts, updated_ts, scope, content, tags, source, ({score_expr}) AS score
             FROM notes
             WHERE (?{scope_idx} IS NULL OR scope = ?{scope_idx})
               AND ({score_expr}) > 0
             ORDER BY score DESC, created_ts DESC
             LIMIT ?{limit_idx}",
            scope_idx = tokens.len() * 2 + 1,
            limit_idx = tokens.len() * 2 + 2,
        );
        sql_params.push(Box::new(scope.map(str::to_string)));
        sql_params.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = sql_params.iter().map(AsRef::as_ref).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok(Note {
                id: row.get(0)?,
                created_ts: row.get(1)?,
                updated_ts: row.get(2)?,
                scope: row.get(3)?,
                content: row.get(4)?,
                tags: parse_tags(&row.get::<_, String>(5)?),
                source: row.get(6)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("Failed to recall memory notes (LIKE fallback path)")
    }

    /// List the most recent notes, optionally filtered by `scope`. No
    /// relevance scoring — pure recency order. `limit` is clamped to
    /// `[1, 200]`.
    pub fn list(&self, scope: Option<&str>, limit: usize) -> Result<Vec<Note>> {
        let limit = limit.clamp(1, 200);
        let mut stmt = self.conn.prepare(
            "SELECT id, created_ts, updated_ts, scope, content, tags, source
             FROM notes
             WHERE (?1 IS NULL OR scope = ?1)
             ORDER BY created_ts DESC, id DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![scope, limit as i64], row_to_note)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("Failed to list memory notes")
    }

    /// Fetch a single note by id, if it exists.
    pub fn get(&self, id: i64) -> Result<Option<Note>> {
        self.conn
            .query_row(
                "SELECT id, created_ts, updated_ts, scope, content, tags, source
                 FROM notes WHERE id = ?1",
                params![id],
                row_to_note,
            )
            .optional()
            .context("Failed to get memory note")
    }

    /// Delete a note by id. Returns `true` if a row was deleted, `false` if
    /// no note had that id.
    pub fn delete(&self, id: i64) -> Result<bool> {
        let changed = self
            .conn
            .execute("DELETE FROM notes WHERE id = ?1", params![id])
            .context("Failed to delete memory note")?;
        Ok(changed > 0)
    }

    /// Total note count (all scopes). Exposed for tests/diagnostics.
    #[must_use]
    pub fn count(&self) -> i64 {
        self.conn
            .query_row("SELECT COUNT(*) FROM notes", [], |r| r.get(0))
            .unwrap_or(0)
    }

    /// Evict oldest notes beyond [`max_notes`] (by `created_ts`, ties broken
    /// by `id`), warning once per call if eviction happened.
    fn enforce_cap(&self) -> Result<()> {
        let cap = max_notes();
        if cap == 0 {
            return Ok(()); // explicitly disabled
        }
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM notes", [], |r| r.get(0))?;
        let count = count.max(0) as usize;
        if count <= cap {
            return Ok(());
        }
        let overflow = count - cap;
        let deleted = self.conn.execute(
            "DELETE FROM notes WHERE id IN (
                 SELECT id FROM notes ORDER BY created_ts ASC, id ASC LIMIT ?1
             )",
            params![overflow as i64],
        )?;
        tracing::warn!(
            deleted,
            cap,
            "memory.db notes exceeded the per-project cap ({MAX_NOTES_ENV}={cap}); \
             evicted the oldest {deleted} note(s)"
        );
        Ok(())
    }
}

fn row_to_note(row: &rusqlite::Row) -> rusqlite::Result<Note> {
    Ok(Note {
        id: row.get(0)?,
        created_ts: row.get(1)?,
        updated_ts: row.get(2)?,
        scope: row.get(3)?,
        content: row.get(4)?,
        tags: parse_tags(&row.get::<_, String>(5)?),
        source: row.get(6)?,
    })
}

fn is_unique_violation(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(err, _)
            if err.code == rusqlite::ErrorCode::ConstraintViolation
    )
}

/// Apply this module's additive migrations to an already-open `memory.db`
/// connection (baseline `notes` schema assumed already created by
/// [`layout::open_memory_db`]). Returns whether the FTS5 match-quality path
/// is available. Idempotent — safe to call on every open.
fn ensure_migrations(conn: &Connection) -> Result<bool> {
    let has_tags: i64 = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('notes') WHERE name = 'tags'",
        [],
        |r| r.get(0),
    )?;
    if has_tags == 0 {
        conn.execute(
            "ALTER TABLE notes ADD COLUMN tags TEXT NOT NULL DEFAULT ''",
            [],
        )
        .context("Failed to add `tags` column to memory.db notes table")?;
    }

    let fts_existed: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'notes_fts'",
        [],
        |r| r.get(0),
    )?;
    if fts_existed > 0 {
        return Ok(true);
    }

    match conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
             content, tags, content='notes', content_rowid='id'
         );
         CREATE TRIGGER IF NOT EXISTS notes_memory_ai AFTER INSERT ON notes BEGIN
           INSERT INTO notes_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
         END;
         CREATE TRIGGER IF NOT EXISTS notes_memory_ad AFTER DELETE ON notes BEGIN
           INSERT INTO notes_fts(notes_fts, rowid, content, tags) VALUES('delete', old.id, old.content, old.tags);
         END;
         CREATE TRIGGER IF NOT EXISTS notes_memory_au AFTER UPDATE ON notes BEGIN
           INSERT INTO notes_fts(notes_fts, rowid, content, tags) VALUES('delete', old.id, old.content, old.tags);
           INSERT INTO notes_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
         END;",
    ) {
        Ok(()) => {
            // Backfill any pre-existing rows (schema upgrade on a non-empty
            // db) via FTS5's official external-content rebuild command.
            conn.execute("INSERT INTO notes_fts(notes_fts) VALUES('rebuild')", [])
                .context("Failed to rebuild notes_fts after creation")?;
            Ok(true)
        }
        Err(e) => {
            tracing::warn!(
                "FTS5 unavailable for memory.db notes ({e}); falling back to tokenized \
                 LIKE scoring for semantex_memory_recall"
            );
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn store() -> (TempDir, MemoryStore) {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("memory.db");
        let store = MemoryStore::open_at(&db_path).unwrap();
        (dir, store)
    }

    #[test]
    fn save_and_get_roundtrip() {
        let (_dir, store) = store();
        let id = store
            .save(
                "remember to use WAL mode",
                "global",
                &["sqlite".to_string()],
            )
            .unwrap();
        let note = store.get(id).unwrap().expect("note exists");
        assert_eq!(note.content, "remember to use WAL mode");
        assert_eq!(note.scope, "global");
        assert_eq!(note.tags, vec!["sqlite".to_string()]);
        assert_eq!(note.source, NOTE_SOURCE);
        assert!(note.created_ts > 0);
    }

    #[test]
    fn save_rejects_empty_content_or_scope() {
        let (_dir, store) = store();
        assert!(store.save("", "global", &[]).is_err());
        assert!(store.save("  ", "global", &[]).is_err());
        assert!(store.save("hello", "", &[]).is_err());
        assert!(store.save("hello", "   ", &[]).is_err());
    }

    #[test]
    fn tags_are_normalized_sorted_and_deduped() {
        let (_dir, store) = store();
        let id = store
            .save(
                "note",
                "global",
                &[
                    "zeta".to_string(),
                    "alpha".to_string(),
                    "alpha".to_string(),
                    " beta ".to_string(),
                ],
            )
            .unwrap();
        let note = store.get(id).unwrap().unwrap();
        assert_eq!(note.tags, vec!["alpha", "beta", "zeta"]);
    }

    #[test]
    fn save_generates_distinct_ids_for_repeated_calls() {
        let (_dir, store) = store();
        let a = store.save("first", "global", &[]).unwrap();
        let b = store.save("second", "global", &[]).unwrap();
        let c = store.save("first", "global", &[]).unwrap(); // duplicate content, must not collide
        assert!(a != b && b != c && a != c);
    }

    #[test]
    fn list_orders_most_recent_first_and_filters_scope() {
        let (_dir, store) = store();
        store.save("one", "global", &[]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        store.save("two", "file:src/lib.rs", &[]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        store.save("three", "global", &[]).unwrap();

        let all = store.list(None, 10).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].content, "three", "most recent first");

        let scoped = store.list(Some("global"), 10).unwrap();
        assert_eq!(scoped.len(), 2);
        assert!(scoped.iter().all(|n| n.scope == "global"));
    }

    #[test]
    fn delete_removes_note_and_reports_whether_one_existed() {
        let (_dir, store) = store();
        let id = store.save("gone soon", "global", &[]).unwrap();
        assert!(store.delete(id).unwrap());
        assert!(store.get(id).unwrap().is_none());
        assert!(!store.delete(id).unwrap(), "second delete finds nothing");
        assert!(!store.delete(999_999).unwrap());
    }

    #[test]
    fn recall_none_query_behaves_like_list() {
        let (_dir, store) = store();
        store.save("alpha note", "global", &[]).unwrap();
        store.save("beta note", "global", &[]).unwrap();
        let recalled = store.recall(None, None, 10).unwrap();
        let listed = store.list(None, 10).unwrap();
        assert_eq!(recalled, listed);
    }

    #[test]
    fn recall_ranks_by_match_quality() {
        let (_dir, store) = store();
        store
            .save("unrelated content about parsing json files", "global", &[])
            .unwrap();
        store
            .save("sqlite schema migration guard notes", "global", &[])
            .unwrap();
        store
            .save(
                "migration guard migration guard: always add a guard before altering the schema migration",
                "global",
                &[],
            )
            .unwrap();

        let results = store.recall(Some("migration guard"), None, 10).unwrap();
        // The unrelated note shares zero tokens with the query, so an OR
        // match must exclude it entirely.
        assert_eq!(
            results.len(),
            2,
            "unrelated note must not match `migration guard`: {results:?}"
        );
        // The note repeating "migration"/"guard" many times must outrank the
        // one mentioning them only once.
        assert!(results[0].content.starts_with("migration guard migration"));
        assert!(
            !results.iter().any(|n| n.content.starts_with("unrelated")),
            "unrelated note should never appear: {results:?}"
        );
    }

    #[test]
    fn recall_filters_by_scope() {
        let (_dir, store) = store();
        store
            .save("shared note about caching", "global", &[])
            .unwrap();
        store
            .save(
                "caching detail specific to this file",
                "file:src/cache.rs",
                &[],
            )
            .unwrap();

        let scoped = store
            .recall(Some("caching"), Some("file:src/cache.rs"), 10)
            .unwrap();
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].scope, "file:src/cache.rs");
    }

    #[test]
    fn recall_handles_fts5_special_characters_without_erroring() {
        let (_dir, store) = store();
        store.save("normal note", "global", &[]).unwrap();
        for weird in [
            "\"quoted\"",
            "a-b:c*(d)",
            "NOT AND OR",
            "'; DROP TABLE notes; --",
        ] {
            let result = store.recall(Some(weird), None, 10);
            assert!(
                result.is_ok(),
                "query {weird:?} should not error: {result:?}"
            );
        }
    }

    /// `SEMANTEX_MEMORY_MAX_NOTES`-mutating cases run inside a single test
    /// function (rather than as separate `#[test]`s) because cargo runs
    /// different test functions in parallel by default — two tests each
    /// mutating the same process-global env var would race on each other's
    /// values. Same pattern as `crate::memory`'s `env_cap_behaviour_serial`.
    #[test]
    fn cap_behaviour_serial() {
        let orig = std::env::var(MAX_NOTES_ENV).ok();

        // (1) Cap of 3 evicts down to the 3 most recent notes.
        unsafe { std::env::set_var(MAX_NOTES_ENV, "3") };
        {
            let (_dir, store) = store();
            for i in 0..5 {
                store.save(&format!("note {i}"), "global", &[]).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            assert_eq!(store.count(), 3, "cap must evict down to 3");
            let remaining = store.list(None, 10).unwrap();
            let contents: Vec<&str> = remaining.iter().map(|n| n.content.as_str()).collect();
            assert!(contents.contains(&"note 4"));
            assert!(contents.contains(&"note 3"));
            assert!(contents.contains(&"note 2"));
            assert!(!contents.contains(&"note 0"));
            assert!(!contents.contains(&"note 1"));
        }

        // (2) Cap of 0 disables eviction entirely.
        unsafe { std::env::set_var(MAX_NOTES_ENV, "0") };
        {
            let (_dir, store) = store();
            for i in 0..10 {
                store.save(&format!("note {i}"), "global", &[]).unwrap();
            }
            assert_eq!(store.count(), 10);
        }

        match orig {
            Some(v) => unsafe { std::env::set_var(MAX_NOTES_ENV, v) },
            None => unsafe { std::env::remove_var(MAX_NOTES_ENV) },
        }
    }

    #[test]
    fn open_via_layout_creates_migrated_schema_and_fts5_is_enabled_here() {
        let dir = TempDir::new().unwrap();
        let store = MemoryStore::open(dir.path()).unwrap();
        // The bundled rusqlite in this workspace is verified to ship FTS5
        // (libsqlite3-sys always passes -DSQLITE_ENABLE_FTS5 in its bundled
        // build.rs); this pins that expectation so a future dependency bump
        // that silently drops FTS5 is caught here rather than discovered as
        // a quiet ranking-quality regression in `recall`.
        assert!(store.fts5_enabled());
        assert!(layout::memory_db_path(dir.path()).exists());
    }

    #[test]
    fn reopening_an_existing_memory_db_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let id = {
            let store = MemoryStore::open(dir.path()).unwrap();
            store.save("persisted note", "global", &[]).unwrap()
        };
        // Re-open (simulates a second process/session): migrations must not
        // error on an already-migrated db, and the note must still be there.
        let store = MemoryStore::open(dir.path()).unwrap();
        let note = store.get(id).unwrap().expect("note survives reopen");
        assert_eq!(note.content, "persisted note");
    }

    // ── Branch independence ──────────────────────────────────────────────
    //
    // memory.db is never in either mirror-list (`layout::mirror_root_as` /
    // `layout::restore_branch_dir_into_root` only ever name chunks.db,
    // meta.json, models.toml, dense/**, sparse/, branch.json), so it should
    // never move or get overwritten by a branch switch. This test exercises
    // the real mirror + restore functions against a project that also has a
    // memory.db, simulating: index on branch A, save a note, switch to
    // branch B (mirror A away, nothing to restore for B), save another note,
    // switch back to A (mirror B away, restore A's snapshot) — the memory
    // notes from both branches must still all be present and unaffected by
    // either snapshot/restore, because memory.db was never part of either
    // operation.
    #[test]
    fn branch_switch_leaves_memory_db_untouched() {
        use crate::index::storage::ChunkStore;

        let dir = TempDir::new().unwrap();
        let project = dir.path();

        // Minimal "an index exists" precondition for mirror_root_as (it
        // no-ops without a chunks.db).
        let container = layout::container_dir(project);
        std::fs::create_dir_all(&container).unwrap();
        let chunk_store = ChunkStore::open(&container.join("chunks.db")).unwrap();
        drop(chunk_store);
        std::fs::write(
            container.join("meta.json"),
            serde_json::to_string(&serde_json::json!({"chunk_count": 0})).unwrap(),
        )
        .unwrap();

        // Save a note before any mirroring happens.
        let first_note_id = {
            let store = MemoryStore::open(project).unwrap();
            store.save("branch A note", "global", &[]).unwrap()
        };

        // Mirror branch A away, then simulate landing on branch B (no
        // snapshot yet) by just leaving the root as-is per the real
        // `detect_and_handle_branch_switch` contract, and save a second note
        // representing work on branch B.
        layout::mirror_into_branch_dir(project, "branch-a-00000000").unwrap();
        let second_note_id = {
            let store = MemoryStore::open(project).unwrap();
            store.save("branch B note", "global", &[]).unwrap()
        };
        layout::mirror_into_branch_dir(project, "branch-b-11111111").unwrap();

        // Switch back to branch A: restore its snapshot into the root. This
        // overwrites chunks.db/meta.json/dense/sparse — but must NOT touch
        // memory.db.
        layout::restore_branch_dir_into_root(project, "branch-a-00000000").unwrap();

        let store = MemoryStore::open(project).unwrap();
        let note_a = store.get(first_note_id).unwrap();
        let note_b = store.get(second_note_id).unwrap();
        assert!(
            note_a.is_some(),
            "branch A's note must survive mirroring away and restoring back"
        );
        assert!(
            note_b.is_some(),
            "branch B's note must survive even though branch A's snapshot was restored \
             over the code index — memory.db is shared, not per-branch"
        );
        assert_eq!(store.count(), 2);
    }
}
