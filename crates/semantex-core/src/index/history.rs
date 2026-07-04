//! Git history as a searchable dimension of the index (v13 Wave 2).
//!
//! Populates and reads the `history.db` schema (`commits`, `file_commits`,
//! `chunk_blame`) that [`crate::index::layout`] creates but never writes to
//! — that module owns the schema, this one owns every read/write against it
//! (see `layout::open_history_db`'s doc comment).
//!
//! # Population
//!
//! [`populate_history_best_effort`] is called once per successful index
//! build (see `index::builder::IndexBuilder::build`). It shells out to the
//! system `git` CLI (no libgit2 dependency, matching `layout.rs`'s existing
//! no-git2 discipline) to walk commit history and record it:
//!
//! - **Commits**: `sha`, `author`, `author_time`, and `message` (the commit
//!   subject on its own line, followed by a blank line and the — sanely
//!   truncated — body, mirroring `git log`'s own default format). The
//!   `commits` table's `message` column (spine schema) carries this
//!   combined text; [`CommitInfo::subject`] extracts just the first line
//!   for display. A `body` column is added on top via an idempotent
//!   `ALTER TABLE ... ADD COLUMN` (same idiom `builder.rs` uses for its own
//!   auxiliary columns) so the two are also queryable separately.
//! - **file_commits**: one row per file touched by a commit (`--name-only`).
//! - **chunk_blame** (OPT-IN, `SEMANTEX_HISTORY_BLAME=1`): best-effort
//!   `git blame --porcelain` mapped onto the chunk line ranges of files
//!   that changed in *this* build. Blame is expensive, so it only ever runs
//!   over files this build actually (re)indexed, never the whole tree.
//!
//! Population is **incremental**: a `history_state` table (this module's
//! own, not the spine's) records the last-processed commit sha, so each
//! build only fetches commits newer than that — except the very first run,
//! or a run so far behind that catching up would exceed
//! [`history_commit_limit`], which instead re-anchors to the most recent N
//! commits from `HEAD` (same as a cold start). This can leave a gap of
//! very old commits ungathered in that rare case, which is an acceptable
//! trade for a bounded, always-fast build — the alternative (an unbounded
//! `git log` on a long-neglected index) is the actual footgun.
//!
//! Every entry point here is **best-effort and non-fatal**: a non-git
//! directory is a silent no-op, and any `git` failure (missing binary,
//! shallow clone, corrupted repo, permission error) is logged at `warn`/
//! `debug` and swallowed — indexing must never fail because history
//! couldn't be gathered.
//!
//! # Query surface
//!
//! [`commits_touching_file`], [`search_commit_messages`], and
//! [`recent_commits`] are plain read helpers over an already-open
//! [`Connection`]. `search_commit_messages` uses a mirrored FTS5 virtual
//! table (`commits_fts`) when the bundled SQLite supports it, falling back
//! to a `LIKE` scan over `commits.message` otherwise (checked once via
//! `CREATE VIRTUAL TABLE`'s success/failure — no separate feature probe).

use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::index::layout;

/// Default number of commits captured on a cold start (or a catch-up re-anchor),
/// overridable via `SEMANTEX_HISTORY_COMMITS`.
pub const DEFAULT_HISTORY_COMMITS: usize = 500;

/// Max bytes kept of a commit body before truncation (sane cap — commit
/// bodies are occasionally enormous auto-generated changelogs).
const MAX_BODY_BYTES: usize = 4_000;

/// Record separator between commits in the `git log --pretty=format:` output
/// below. `\x1e` (ASCII Record Separator) is vanishingly unlikely to appear
/// in real commit text.
const RECORD_SEP: char = '\u{1e}';
/// Field separator within a single commit's record.
const FIELD_SEP: char = '\u{1f}';
/// Marker preceding each commit's hash in the `--name-only` fetch, so a
/// commit body containing a bare 40-hex-char line can't be confused for a
/// record boundary.
const NAME_ONLY_MARKER: char = '\u{2}';

// Age buckets for `humanize_age`, unix seconds.
const AGE_MINUTE: i64 = 60;
const AGE_HOUR: i64 = 60 * AGE_MINUTE;
const AGE_DAY: i64 = 24 * AGE_HOUR;
const AGE_MONTH: i64 = 30 * AGE_DAY;
const AGE_YEAR: i64 = 365 * AGE_DAY;

/// `SEMANTEX_HISTORY_COMMITS` override, falling back to
/// [`DEFAULT_HISTORY_COMMITS`]. `0` or unparseable values are ignored (the
/// default wins) rather than treated as "unbounded" — an explicit unbounded
/// mode isn't offered, to keep every build bounded.
pub fn history_commit_limit() -> usize {
    std::env::var("SEMANTEX_HISTORY_COMMITS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_HISTORY_COMMITS)
}

/// `SEMANTEX_HISTORY_BLAME=1` opts into the expensive per-chunk blame pass.
/// Off by default.
pub fn blame_enabled() -> bool {
    std::env::var("SEMANTEX_HISTORY_BLAME")
        .map(|v| v == "1")
        .unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────────────
// Public data types
// ─────────────────────────────────────────────────────────────────────────

/// One commit, as read back from `history.db`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitInfo {
    pub hash: String,
    pub author: String,
    /// Author time, unix seconds.
    pub ts: i64,
    /// First line of the stored message (the commit subject).
    pub subject: String,
}

/// Human-friendly relative age (`"3d ago"`, `"2mo ago"`, ...) for `ts`
/// (unix seconds). Shared by the CLI table, the docs scaffold, and the
/// `semantex_agent` "recent changes" addendum so all three read the same.
pub fn humanize_age(ts: i64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let delta = (now - ts).max(0);
    if delta < AGE_MINUTE {
        format!("{delta}s ago")
    } else if delta < AGE_HOUR {
        format!("{}m ago", delta / AGE_MINUTE)
    } else if delta < AGE_DAY {
        format!("{}h ago", delta / AGE_HOUR)
    } else if delta < AGE_MONTH {
        format!("{}d ago", delta / AGE_DAY)
    } else if delta < AGE_YEAR {
        format!("{}mo ago", delta / AGE_MONTH)
    } else {
        format!("{}y ago", delta / AGE_YEAR)
    }
}

fn row_to_commit(row: &rusqlite::Row) -> rusqlite::Result<CommitInfo> {
    let message: String = row.get(3)?;
    let subject = message.lines().next().unwrap_or("").to_string();
    Ok(CommitInfo {
        hash: row.get(0)?,
        author: row.get(1)?,
        ts: row.get(2)?,
        subject,
    })
}

/// Open (creating the schema if absent) `history.db` for `project_root`.
/// Thin wrapper over `layout::open_history_db` so callers in other crates
/// don't need to import `layout` too.
pub fn open(project_root: &Path) -> Result<Connection> {
    layout::open_history_db(&layout::history_db_path(project_root))
}

/// Whether `history.db` exists AND has at least one commit recorded.
/// Callers use this to decide whether to surface history-derived fields at
/// all (absent, not an empty array — see `docs_scaffold`'s history fields).
pub fn has_history(project_root: &Path) -> bool {
    let path = layout::history_db_path(project_root);
    if !path.exists() {
        return false;
    }
    let Ok(conn) = Connection::open(&path) else {
        return false;
    };
    conn.query_row("SELECT 1 FROM commits LIMIT 1", [], |_| Ok(()))
        .is_ok()
}

/// Commits that touched `path` (repo-relative), most recent first.
pub fn commits_touching_file(
    conn: &Connection,
    path: &str,
    limit: usize,
) -> Result<Vec<CommitInfo>> {
    let mut stmt = conn.prepare(
        "SELECT c.hash, c.author, c.ts, c.message \
         FROM commits c JOIN file_commits f ON f.hash = c.hash \
         WHERE f.path = ?1 ORDER BY c.ts DESC, c.rowid DESC LIMIT ?2",
    )?;
    let rows = stmt
        .query_map(params![path, limit as i64], row_to_commit)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Most recent commits repo-wide.
pub fn recent_commits(conn: &Connection, limit: usize) -> Result<Vec<CommitInfo>> {
    let mut stmt = conn.prepare(
        "SELECT hash, author, ts, message FROM commits ORDER BY ts DESC, rowid DESC LIMIT ?1",
    )?;
    let rows = stmt
        .query_map(params![limit as i64], row_to_commit)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Full-text-ish search over commit messages. Uses the mirrored `commits_fts`
/// FTS5 virtual table when the bundled SQLite supports it (probed once via
/// `ensure_commits_fts` at population time — see its doc comment); falls
/// back to a `LIKE` scan otherwise, and also falls back on any FTS query
/// error (e.g. a MATCH query string with unbalanced quoting).
pub fn search_commit_messages(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<CommitInfo>> {
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    if commits_fts_exists(conn) {
        let fts_result = (|| -> rusqlite::Result<Vec<CommitInfo>> {
            let mut stmt = conn.prepare(
                "SELECT c.hash, c.author, c.ts, c.message \
                 FROM commits_fts f JOIN commits c ON c.hash = f.hash \
                 WHERE commits_fts MATCH ?1 ORDER BY c.ts DESC, c.rowid DESC LIMIT ?2",
            )?;
            stmt.query_map(params![q, limit as i64], row_to_commit)?
                .collect::<rusqlite::Result<Vec<_>>>()
        })();
        if let Ok(rows) = fts_result {
            return Ok(rows);
        }
        // Malformed MATCH syntax or any other FTS-path error: fall through.
    }
    let like = format!("%{}%", q.replace(['%', '_'], ""));
    let mut stmt = conn.prepare(
        "SELECT hash, author, ts, message FROM commits WHERE message LIKE ?1 ORDER BY ts DESC, rowid DESC LIMIT ?2",
    )?;
    let rows = stmt
        .query_map(params![like, limit as i64], row_to_commit)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

// ─────────────────────────────────────────────────────────────────────────
// Schema extensions owned by this module (spine owns commits/file_commits/
// chunk_blame's own columns; everything below is additive).
// ─────────────────────────────────────────────────────────────────────────

fn column_exists(conn: &Connection, table: &str, column: &str) -> bool {
    let sql = format!("PRAGMA table_info({table})");
    let Ok(mut stmt) = conn.prepare(&sql) else {
        return false;
    };
    let Ok(mut rows) = stmt.query([]) else {
        return false;
    };
    while let Ok(Some(row)) = rows.next() {
        if let Ok(name) = row.get::<_, String>(1)
            && name == column
        {
            return true;
        }
    }
    false
}

/// `history_state`: this module's own key/value bookkeeping table (the
/// incremental-population watermark). Not part of the spine's contract §A
/// schema, so it lives here rather than in `layout::open_history_db`.
fn ensure_history_extensions(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS history_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );",
    )?;
    if !column_exists(conn, "commits", "body") {
        conn.execute_batch("ALTER TABLE commits ADD COLUMN body TEXT NOT NULL DEFAULT ''")?;
    }
    Ok(())
}

/// Best-effort: create the `commits_fts` FTS5 mirror if the bundled SQLite
/// supports it. Returns whether it exists (pre-existing or just created).
/// A single failed `CREATE VIRTUAL TABLE` (FTS5 not compiled in) is exactly
/// the "check" the contract asks for — no separate capability probe needed.
fn ensure_commits_fts(conn: &Connection) -> bool {
    if commits_fts_exists(conn) {
        return true;
    }
    conn.execute_batch("CREATE VIRTUAL TABLE commits_fts USING fts5(hash UNINDEXED, message)")
        .is_ok()
}

fn commits_fts_exists(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='commits_fts'",
        [],
        |_| Ok(()),
    )
    .is_ok()
}

fn read_state(conn: &Connection, key: &str) -> Option<String> {
    conn.query_row(
        "SELECT value FROM history_state WHERE key = ?1",
        params![key],
        |row| row.get(0),
    )
    .ok()
}

fn write_state(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO history_state (key, value) VALUES (?1, ?2) \
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// git CLI plumbing
// ─────────────────────────────────────────────────────────────────────────

fn run_git(project_root: &Path, args: &[&str]) -> Result<std::process::Output> {
    Command::new("git")
        .arg("-C")
        .arg(project_root)
        .args(args)
        .output()
        .context("failed to spawn `git`")
}

/// `HEAD`'s commit sha, or `None` for a non-git directory, a repo with zero
/// commits, or a missing `git` binary — all of which are "nothing to do"
/// for history population, not errors.
fn resolve_head(project_root: &Path) -> Option<String> {
    let output = run_git(project_root, &["rev-parse", "HEAD"]).ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!sha.is_empty()).then_some(sha)
}

fn commit_exists(project_root: &Path, sha: &str) -> bool {
    run_git(
        project_root,
        &["cat-file", "-e", &format!("{sha}^{{commit}}")],
    )
    .map(|o| o.status.success())
    .unwrap_or(false)
}

fn rev_list_count(project_root: &Path, range: &str) -> Option<usize> {
    let output = run_git(project_root, &["rev-list", "--count", range]).ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

struct RawCommit {
    hash: String,
    author: String,
    ts: i64,
    subject: String,
    body: String,
}

/// Fetch commit metadata for `range` (either `"HEAD"` for the whole reachable
/// history, or `"<sha>..HEAD"` for an incremental slice), oldest-first.
/// `max_count`, when set, caps the walk to that many of the most recent
/// commits in `range` (git's own `-n` semantics: select-then-reverse, not
/// reverse-then-select — see the module-level test asserting this).
fn fetch_commit_metadata(
    project_root: &Path,
    range: &str,
    max_count: Option<usize>,
) -> Result<Vec<RawCommit>> {
    let mut args: Vec<String> = vec![
        "log".into(),
        "--no-color".into(),
        "--date=unix".into(),
        "--reverse".into(),
    ];
    if let Some(n) = max_count {
        args.push("-n".into());
        args.push(n.to_string());
    }
    args.push(format!(
        "--pretty=format:%H{FIELD_SEP}%an{FIELD_SEP}%at{FIELD_SEP}%s{FIELD_SEP}%b{RECORD_SEP}"
    ));
    args.push(range.to_string());

    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let output = run_git(project_root, &arg_refs)?;
    if !output.status.success() {
        anyhow::bail!(
            "git log failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let text = String::from_utf8_lossy(&output.stdout);

    let mut commits = Vec::new();
    for record in text.split(RECORD_SEP) {
        let record = record.trim_start_matches('\n');
        if record.is_empty() {
            continue;
        }
        let mut fields = record.splitn(5, FIELD_SEP);
        let (Some(hash), Some(author), Some(ts_str), Some(subject)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let body = fields.next().unwrap_or("").trim().to_string();
        let Ok(ts) = ts_str.parse::<i64>() else {
            continue;
        };
        let body = truncate_body(&body);
        commits.push(RawCommit {
            hash: hash.to_string(),
            author: author.to_string(),
            ts,
            subject: subject.to_string(),
            body,
        });
    }
    Ok(commits)
}

fn truncate_body(body: &str) -> String {
    if body.len() <= MAX_BODY_BYTES {
        return body.to_string();
    }
    let mut end = MAX_BODY_BYTES;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &body[..end])
}

/// Fetch `hash -> files touched` for the same `range`/`max_count` as
/// [`fetch_commit_metadata`] (a second `git log` invocation rather than
/// trying to interleave `--name-only` output with the `--pretty=format:`
/// metadata above, which would require disambiguating file names that
/// happen to look like format fields).
fn fetch_changed_files(
    project_root: &Path,
    range: &str,
    max_count: Option<usize>,
) -> Result<HashMap<String, Vec<String>>> {
    let mut args: Vec<String> = vec![
        "log".into(),
        "--no-color".into(),
        "--reverse".into(),
        "--name-only".into(),
    ];
    if let Some(n) = max_count {
        args.push("-n".into());
        args.push(n.to_string());
    }
    args.push(format!("--pretty=format:{NAME_ONLY_MARKER}%H"));
    args.push(range.to_string());

    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let output = run_git(project_root, &arg_refs)?;
    if !output.status.success() {
        anyhow::bail!(
            "git log --name-only failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let text = String::from_utf8_lossy(&output.stdout);

    let mut map = HashMap::new();
    for block in text.split(NAME_ONLY_MARKER) {
        let mut lines = block.lines();
        let Some(hash) = lines.next() else { continue };
        let hash = hash.trim();
        if hash.is_empty() {
            continue;
        }
        let files: Vec<String> = lines
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
        map.insert(hash.to_string(), files);
    }
    Ok(map)
}

/// Insert `commits` (+ mirrored `commits_fts` row, best-effort) and
/// `file_commits` rows for a freshly-fetched batch. `INSERT OR IGNORE` on
/// `commits`/`file_commits` (their existing PKs) makes re-processing an
/// already-known commit — the re-anchor path can do this — harmless.
fn store_commit_batch(
    conn: &mut Connection,
    commits: &[RawCommit],
    files_by_hash: &HashMap<String, Vec<String>>,
) -> Result<()> {
    let has_fts = ensure_commits_fts(conn);
    let tx = conn.transaction()?;
    {
        let mut insert_commit = tx.prepare(
            "INSERT OR IGNORE INTO commits (hash, author, ts, message, body) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        let mut insert_file =
            tx.prepare("INSERT OR IGNORE INTO file_commits (path, hash) VALUES (?1, ?2)")?;
        for c in commits {
            let message = if c.body.is_empty() {
                c.subject.clone()
            } else {
                format!("{}\n\n{}", c.subject, c.body)
            };
            insert_commit.execute(params![c.hash, c.author, c.ts, message, c.body])?;
            if has_fts {
                // FTS5 has no natural uniqueness constraint; guard against
                // duplicate rows on the (rare) re-anchor-reprocesses-a-known-
                // commit path with a delete-then-insert.
                tx.execute("DELETE FROM commits_fts WHERE hash = ?1", params![c.hash])?;
                tx.execute(
                    "INSERT INTO commits_fts (hash, message) VALUES (?1, ?2)",
                    params![c.hash, message],
                )?;
            }
            if let Some(files) = files_by_hash.get(&c.hash) {
                for f in files {
                    insert_file.execute(params![f, c.hash])?;
                }
            }
        }
    }
    tx.commit()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Population entry points
// ─────────────────────────────────────────────────────────────────────────

/// Populate `history.db`'s commit tables for `project_root`, incrementally.
/// No-op (Ok) for a non-git directory or a repo with zero commits.
pub fn populate_history(project_root: &Path) -> Result<()> {
    populate_history_with_bound(project_root, history_commit_limit())
}

/// Core of [`populate_history`], with the cold-start/re-anchor bound passed
/// explicitly rather than read from `SEMANTEX_HISTORY_COMMITS` — lets tests
/// exercise the bounding behavior deterministically without mutating global
/// process env (which `cargo test`'s parallel unit-test threads would race
/// on otherwise).
fn populate_history_with_bound(project_root: &Path, bound: usize) -> Result<()> {
    let Some(head) = resolve_head(project_root) else {
        return Ok(());
    };
    let mut conn = open(project_root)?;
    ensure_history_extensions(&conn)?;

    let last_sha = read_state(&conn, "last_sha");

    if last_sha.as_deref() == Some(head.as_str()) {
        return Ok(()); // already caught up
    }

    let (range, max_count): (String, Option<usize>) = match last_sha.as_deref() {
        Some(sha) if commit_exists(project_root, sha) => {
            let incremental_range = format!("{sha}..HEAD");
            let count = rev_list_count(project_root, &incremental_range).unwrap_or(usize::MAX);
            if count == 0 {
                write_state(&conn, "last_sha", &head)?;
                return Ok(());
            }
            if count > bound {
                // Too far behind to catch up gap-free within the bound —
                // re-anchor to the most recent `bound` commits, same as a
                // cold start. Leaves older commits between the stale
                // watermark and this new window unfetched (documented
                // trade-off — see module doc).
                ("HEAD".to_string(), Some(bound))
            } else {
                (incremental_range, None)
            }
        }
        _ => ("HEAD".to_string(), Some(bound)),
    };

    let commits = fetch_commit_metadata(project_root, &range, max_count)?;
    if commits.is_empty() {
        write_state(&conn, "last_sha", &head)?;
        return Ok(());
    }
    let files_by_hash = fetch_changed_files(project_root, &range, max_count).unwrap_or_default();

    store_commit_batch(&mut conn, &commits, &files_by_hash)?;

    // The newest commit actually processed (last in the --reverse-ordered
    // batch) becomes the new watermark — equal to `head` whenever the fetch
    // reached HEAD (always true here: both the plain incremental range and
    // the bounded re-anchor/cold-start range end at HEAD).
    let newest = &commits.last().expect("checked non-empty above").hash;
    write_state(&conn, "last_sha", newest)?;
    Ok(())
}

/// Best-effort wrapper: logs and swallows any error rather than failing the
/// index build. Called from `index::builder::IndexBuilder::build` after a
/// successful build, under the same `.semantex.lock` the builder already
/// holds.
pub fn populate_history_best_effort(project_root: &Path, changed_files: &[PathBuf]) {
    if let Err(e) = populate_history(project_root) {
        tracing::warn!("history population failed (index still valid): {e}");
    }
    if blame_enabled() {
        populate_chunk_blame_best_effort(project_root, changed_files);
    }
}

/// OPT-IN (`SEMANTEX_HISTORY_BLAME=1`): map `git blame --porcelain` onto the
/// chunk line ranges of each file in `changed_files`, best-effort per file
/// (a blame failure for one file — binary content, file since deleted,
/// whatever — never blocks the rest).
fn populate_chunk_blame_best_effort(project_root: &Path, changed_files: &[PathBuf]) {
    if changed_files.is_empty() {
        return;
    }
    if resolve_head(project_root).is_none() {
        return; // non-git / no commits
    }
    let chunks_db = layout::container_dir(project_root).join("chunks.db");
    if !chunks_db.exists() {
        return;
    }
    let Ok(history_conn) = open(project_root) else {
        return;
    };
    let Ok(chunks_conn) = Connection::open(&chunks_db) else {
        return;
    };
    for rel in changed_files {
        if let Err(e) =
            populate_chunk_blame_for_file(project_root, &chunks_conn, &history_conn, rel)
        {
            tracing::debug!(file = %rel.display(), "chunk blame skipped: {e}");
        }
    }
}

/// Parse `git blame --porcelain` output into `final_line -> commit sha`.
/// Every attributed line, whether it's the first time a commit's full
/// metadata block appears or a terse repeat, starts with a header line
/// `<40-hex-sha> <orig-line> <final-line>[ <num-lines>]` and ends with the
/// single `\t<content>` line that closes that record — so tracking "most
/// recent header seen" and flushing it on the next tab-prefixed line
/// correctly handles both forms without needing to distinguish them.
fn parse_blame_porcelain(output: &str) -> Vec<(u32, String)> {
    let mut result = Vec::new();
    let mut current: Option<(u32, String)> = None;
    for line in output.lines() {
        if line.starts_with('\t') {
            if let Some(pair) = current.take() {
                result.push(pair);
            }
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(sha) = it.next() else { continue };
        if sha.len() != 40 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
            continue; // metadata line (author/committer/summary/...)
        }
        let Some(_orig_line) = it.next() else {
            continue;
        };
        let Some(final_line) = it.next().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        current = Some((final_line, sha.to_string()));
    }
    result
}

fn populate_chunk_blame_for_file(
    project_root: &Path,
    chunks_conn: &Connection,
    history_conn: &Connection,
    rel_path: &Path,
) -> Result<()> {
    let path_str = rel_path.to_string_lossy();
    let mut stmt =
        chunks_conn.prepare("SELECT id, start_line, end_line FROM chunks WHERE file_path = ?1")?;
    let chunk_rows: Vec<(i64, u32, u32)> = stmt
        .query_map(params![path_str.as_ref()], |row| {
            Ok((
                row.get(0)?,
                row.get::<_, i64>(1)? as u32,
                row.get::<_, i64>(2)? as u32,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if chunk_rows.is_empty() {
        return Ok(());
    }

    let output = Command::new("git")
        .arg("-C")
        .arg(project_root)
        .args(["blame", "--porcelain", "--"])
        .arg(rel_path)
        .output()
        .context("failed to spawn `git blame`")?;
    if !output.status.success() {
        return Ok(()); // e.g. file not tracked / deleted — skip silently
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut line_hash: HashMap<u32, String> = HashMap::new();
    for (line_no, sha) in parse_blame_porcelain(&text) {
        line_hash.insert(line_no, sha);
    }

    let mut insert = history_conn
        .prepare("INSERT OR IGNORE INTO chunk_blame (chunk_id, hash) VALUES (?1, ?2)")?;
    for (chunk_id, start, end) in chunk_rows {
        let mut seen: HashSet<&str> = HashSet::new();
        for line_no in start..=end {
            if let Some(sha) = line_hash.get(&line_no)
                && seen.insert(sha.as_str())
            {
                insert.execute(params![chunk_id, sha])?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command as StdCommand;
    use tempfile::TempDir;

    fn git(dir: &Path, args: &[&str]) {
        let status = StdCommand::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .status()
            .expect("git must be on PATH for these tests");
        assert!(status.success(), "git {args:?} failed");
    }

    /// Scripted tempdir git repo: N commits, each touching one file.
    fn init_repo(dir: &Path, commit_count: usize) {
        git(dir, &["init", "-q"]);
        git(dir, &["config", "user.email", "test@example.com"]);
        git(dir, &["config", "user.name", "Test User"]);
        for i in 0..commit_count {
            std::fs::write(dir.join(format!("file{i}.txt")), format!("content {i}")).unwrap();
            git(dir, &["add", "."]);
            git(dir, &["commit", "-q", "-m", &format!("commit number {i}")]);
        }
    }

    #[test]
    fn non_git_dir_populate_is_a_silent_no_op() {
        let tmp = TempDir::new().unwrap();
        populate_history(tmp.path()).unwrap();
        assert!(!has_history(tmp.path()));
        // history.db must not even be created for a non-git dir.
        assert!(!layout::history_db_path(tmp.path()).exists());
    }

    #[test]
    fn populate_records_all_commits_within_bound() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 5);

        populate_history(tmp.path()).unwrap();
        assert!(has_history(tmp.path()));

        let conn = open(tmp.path()).unwrap();
        let commits = recent_commits(&conn, 100).unwrap();
        assert_eq!(commits.len(), 5);
        // Most recent first.
        assert_eq!(commits[0].subject, "commit number 4");
        assert_eq!(commits[4].subject, "commit number 0");
        assert_eq!(commits[0].author, "Test User");
    }

    #[test]
    fn populate_records_file_commits() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 3);
        populate_history(tmp.path()).unwrap();

        let conn = open(tmp.path()).unwrap();
        let hits = commits_touching_file(&conn, "file1.txt", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "commit number 1");
    }

    #[test]
    fn incremental_repopulation_only_adds_new_commits() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 3);
        populate_history(tmp.path()).unwrap();

        // A second call with nothing new must be a cheap no-op (same count).
        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        assert_eq!(recent_commits(&conn, 100).unwrap().len(), 3);
        drop(conn);

        // New commits land incrementally.
        std::fs::write(tmp.path().join("file3.txt"), "content 3").unwrap();
        git(tmp.path(), &["add", "."]);
        git(tmp.path(), &["commit", "-q", "-m", "commit number 3"]);
        populate_history(tmp.path()).unwrap();

        let conn = open(tmp.path()).unwrap();
        let commits = recent_commits(&conn, 100).unwrap();
        assert_eq!(commits.len(), 4);
        assert_eq!(commits[0].subject, "commit number 3");
    }

    #[test]
    fn history_commit_limit_bounds_a_cold_start() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 10);

        // Bound injected directly (not via env — see `populate_history_with_bound`'s
        // doc comment) so this test can't race other tests' env reads.
        populate_history_with_bound(tmp.path(), 3).unwrap();

        let conn = open(tmp.path()).unwrap();
        let commits = recent_commits(&conn, 100).unwrap();
        assert_eq!(
            commits.len(),
            3,
            "bounded cold start must cap at the override"
        );
        // Must be the 3 MOST RECENT commits (7, 8, 9), not the oldest 3.
        assert_eq!(commits[0].subject, "commit number 9");
        assert_eq!(commits[2].subject, "commit number 7");
    }

    #[test]
    fn default_history_commits_constant_is_500() {
        // The documented default — `history_commit_limit()` itself isn't
        // asserted against here to avoid coupling this test to whatever
        // SEMANTEX_HISTORY_COMMITS happens to be set to in the ambient
        // environment (see `populate_history_with_bound`'s doc comment for
        // why tests inject the bound directly instead of mutating env).
        assert_eq!(DEFAULT_HISTORY_COMMITS, 500);
    }

    #[test]
    fn search_commit_messages_finds_substring() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 3);
        populate_history(tmp.path()).unwrap();

        let conn = open(tmp.path()).unwrap();
        let hits = search_commit_messages(&conn, "number 2", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "commit number 2");

        let none = search_commit_messages(&conn, "nonexistent-xyz", 10).unwrap();
        assert!(none.is_empty());

        let empty_query = search_commit_messages(&conn, "   ", 10).unwrap();
        assert!(empty_query.is_empty());
    }

    #[test]
    fn body_is_stored_and_truncated() {
        let tmp = TempDir::new().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Test User"]);
        std::fs::write(tmp.path().join("f.txt"), "x").unwrap();
        git(tmp.path(), &["add", "."]);
        let long_body = "y".repeat(10_000);
        git(
            tmp.path(),
            &[
                "commit",
                "-q",
                "-m",
                &format!("subject line\n\n{long_body}"),
            ],
        );

        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        let body: String = conn
            .query_row("SELECT body FROM commits LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert!(body.len() <= MAX_BODY_BYTES + 4); // small slack for the ellipsis char
        assert!(body.starts_with('y'));
    }

    #[test]
    fn humanize_age_buckets() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        assert!(humanize_age(now).ends_with("s ago"));
        assert!(humanize_age(now - 3600).ends_with("h ago"));
        assert!(humanize_age(now - 86400 * 2).ends_with("d ago"));
        assert!(humanize_age(now - 86400 * 400).ends_with("y ago"));
    }

    #[test]
    fn git_log_reverse_selects_most_recent_then_reverses_order() {
        // Documents the exact semantics `fetch_commit_metadata` relies on:
        // `-n N --reverse` picks the N MOST RECENT commits (same set as
        // plain `-n N`), just printed oldest-first — it does NOT pick the
        // oldest N commits in the whole history.
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 10);
        let newest_first = fetch_commit_metadata(tmp.path(), "HEAD", Some(3)).unwrap();
        assert_eq!(newest_first.len(), 3);
        assert_eq!(newest_first[0].subject, "commit number 7");
        assert_eq!(newest_first[2].subject, "commit number 9");
    }

    #[test]
    fn chunk_blame_maps_lines_to_commits_when_opted_in() {
        let tmp = TempDir::new().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Test User"]);
        std::fs::write(
            tmp.path().join("lib.rs"),
            "line one\nline two\nline three\n",
        )
        .unwrap();
        git(tmp.path(), &["add", "."]);
        git(tmp.path(), &["commit", "-q", "-m", "add lib.rs"]);

        populate_history(tmp.path()).unwrap();

        // Seed a chunks.db with one chunk spanning lines 1-3.
        let chunks_db = layout::container_dir(tmp.path()).join("chunks.db");
        std::fs::create_dir_all(chunks_db.parent().unwrap()).unwrap();
        let chunks_conn = Connection::open(&chunks_db).unwrap();
        chunks_conn
            .execute_batch(
                "CREATE TABLE chunks (id INTEGER PRIMARY KEY, file_path TEXT, \
                 start_line INTEGER, end_line INTEGER);
                 INSERT INTO chunks (id, file_path, start_line, end_line) \
                 VALUES (1, 'lib.rs', 1, 3);",
            )
            .unwrap();
        drop(chunks_conn);

        populate_chunk_blame_best_effort(tmp.path(), &[PathBuf::from("lib.rs")]);

        let history_conn = open(tmp.path()).unwrap();
        let hash: String = history_conn
            .query_row("SELECT hash FROM chunk_blame WHERE chunk_id = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(hash.len(), 40);
    }

    #[test]
    fn blame_disabled_by_default_leaves_chunk_blame_empty() {
        assert!(!blame_enabled());
    }
}
