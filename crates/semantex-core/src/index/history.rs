//! Git history as a searchable dimension of the index (v13 Wave 2).
//!
//! Populates and reads the `history.db` schema (`commits`, `file_commits`,
//! `chunk_blame`) that [`crate::index::layout`] creates but never writes to
//! — that module owns the schema, this one owns every read/write against it
//! (see `layout::open_history_db`'s doc comment).
//!
//! `history.db` is **repo-GLOBAL**, like `memory.db`: it lives at the
//! container root and is deliberately EXCLUDED from the per-branch snapshot
//! mirror/restore machinery (`layout::mirror_root_as` /
//! `layout::restore_branch_dir_into_root`). Branches share one underlying
//! commit DAG, git itself is the ever-present source of truth, and
//! population is incremental + idempotent — so the file self-heals from git
//! on the next build no matter what, and branch switches leave it fully
//! untouched.
//!
//! # Population
//!
//! [`populate_history_best_effort`] is called once per successful index
//! build (see `index::builder::IndexBuilder::build`). It shells out to the
//! system `git` CLI (no libgit2 dependency, matching `layout.rs`'s existing
//! no-git2 discipline) to walk commit history and record it:
//!
//! - **Commits**: `sha`, `author`, committer time, and `message` (the commit
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

// Both `git log` fetches below use `%x00` (NUL) as their in-band separator.
// Git guarantees NUL can never appear in commit metadata or messages (and
// POSIX filenames can't contain it either), so parsing is EXACT: a crafted
// commit body cannot inject fake record/field boundaries and forge commit
// rows (adversarial-review fix #3 — the previous \x1e/\x1f separators were
// injectable via a hostile commit message).

/// Max commits expanded per `semantex_history` detail call — bounds the
/// per-call `git show` subprocess fan-out (same discipline as
/// [`MAX_BLAME_FILES`]). Excess shas are reported as skipped, not dropped.
pub const MAX_DETAIL_COMMITS: usize = 10;

/// Max files blamed per build when `SEMANTEX_HISTORY_BLAME=1` — caps the
/// per-build `git blame` process fan-out (a 10k-file cold build must not
/// spawn 10k subprocesses). Files beyond the cap are skipped with a warn;
/// they get blamed on whichever later build re-touches them.
const MAX_BLAME_FILES: usize = 500;

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
    /// Committer time, unix seconds (`%ct` — see `fetch_commit_metadata`
    /// for why committer beats author time for recency ordering).
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
/// don't need to import `layout` too. Sets a busy timeout so a reader
/// racing the builder's population write waits briefly instead of failing
/// with SQLITE_BUSY.
pub fn open(project_root: &Path) -> Result<Connection> {
    let conn = layout::open_history_db(&layout::history_db_path(project_root))?;
    let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
    Ok(conn)
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
    let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
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

/// Composable commit filters for the `semantex_history` surface. Every
/// present field ANDs into the query; `Default` (all `None`, `limit: 0`)
/// means "most recent commits" with the default limit of 20.
#[derive(Debug, Clone, Default)]
pub struct HistoryFilter {
    /// Strictly-greater-than committer-timestamp bound (`ts > since_ts`) —
    /// "since tag X" for release notes must exclude the tagged commit itself.
    pub since_ts: Option<i64>,
    /// Case-insensitive author substring (SQLite LIKE).
    pub author: Option<String>,
    /// Exact repo-relative path, matched against `file_commits.path`.
    pub file: Option<String>,
    /// Free-text match over the stored commit message.
    pub message_query: Option<String>,
    /// Max rows returned; `0` means the default (20).
    pub limit: usize,
}

/// Query commits with all of `filter`'s present fields ANDed together,
/// most recent first.
///
/// A message-only filter delegates to [`search_commit_messages`] so it
/// benefits from FTS5 when available; any composed filter uses a plain
/// LIKE for the message term (recency ordering dominates anyway, and
/// composing FTS MATCH with joins buys nothing here). LIKE wildcards in
/// user input are stripped, mirroring `search_commit_messages`'s fallback.
pub fn filter_commits(conn: &Connection, filter: &HistoryFilter) -> Result<Vec<CommitInfo>> {
    let limit = if filter.limit == 0 { 20 } else { filter.limit };
    if let Some(q) = &filter.message_query
        && filter.since_ts.is_none()
        && filter.author.is_none()
        && filter.file.is_none()
    {
        return search_commit_messages(conn, q, limit);
    }

    let mut sql = String::from("SELECT DISTINCT c.hash, c.author, c.ts, c.message FROM commits c");
    let mut bound: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    if filter.file.is_some() {
        sql.push_str(" JOIN file_commits f ON f.hash = c.hash");
    }
    sql.push_str(" WHERE 1=1");
    if let Some(ts) = filter.since_ts {
        sql.push_str(" AND c.ts > ?");
        bound.push(Box::new(ts));
    }
    if let Some(author) = &filter.author {
        sql.push_str(" AND c.author LIKE ?");
        bound.push(Box::new(format!("%{}%", author.replace(['%', '_'], ""))));
    }
    if let Some(file) = &filter.file {
        sql.push_str(" AND f.path = ?");
        bound.push(Box::new(file.clone()));
    }
    if let Some(q) = &filter.message_query {
        sql.push_str(" AND c.message LIKE ?");
        bound.push(Box::new(format!("%{}%", q.replace(['%', '_'], ""))));
    }
    sql.push_str(" ORDER BY c.ts DESC, c.rowid DESC LIMIT ?");
    bound.push(Box::new(limit as i64));

    let mut stmt = conn.prepare(&sql)?;
    let refs: Vec<&dyn rusqlite::ToSql> = bound.iter().map(|b| b.as_ref()).collect();
    let rows = stmt
        .query_map(rusqlite::params_from_iter(refs), row_to_commit)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Repo-relative paths touched by `hash` (from `file_commits`), sorted.
pub fn files_for_commit(conn: &Connection, hash: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT path FROM file_commits WHERE hash = ?1 ORDER BY path")?;
    let rows = stmt
        .query_map(params![hash], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Committer timestamp of the OLDEST captured commit, or `None` when the
/// table is empty. Used to warn when a `since` selector predates the
/// bounded capture window (see `history_commit_limit`).
pub fn oldest_commit_ts(conn: &Connection) -> Option<i64> {
    conn.query_row("SELECT MIN(ts) FROM commits", [], |row| {
        row.get::<_, Option<i64>>(0)
    })
    .ok()
    .flatten()
}

/// Resolve a user-supplied `since` selector to a unix committer timestamp.
/// Accepted forms, tried in order: a strict UTC calendar date `YYYY-MM-DD`
/// (midnight UTC), then any git rev (tag, sha, branch, `HEAD~3`, ...).
/// Returns `None` when neither resolves. Selectors starting with `-` are
/// rejected outright so user input can never be parsed as a git flag.
pub fn resolve_since_to_ts(project_root: &Path, since: &str) -> Option<i64> {
    let s = since.trim();
    if s.is_empty() || s.starts_with('-') {
        return None;
    }
    if let Some(ts) = parse_utc_date_to_ts(s) {
        return Some(ts);
    }
    let verify = run_git(
        project_root,
        &["rev-parse", "--verify", &format!("{s}^{{commit}}")],
    )
    .ok()?;
    if !verify.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&verify.stdout).trim().to_string();
    let show = run_git(project_root, &["show", "-s", "--format=%ct", &sha, "--"]).ok()?;
    if !show.status.success() {
        return None;
    }
    String::from_utf8_lossy(&show.stdout).trim().parse().ok()
}

/// Strict `YYYY-MM-DD` → unix seconds at 00:00:00 UTC. Hand-rolled
/// days-from-civil (Howard Hinnant's algorithm) — deliberately no chrono/
/// time dependency for one conversion. Rejects out-of-range month/day
/// components but does not validate month lengths (2026-02-31 resolves a
/// few days into March — harmless for a lower-bound filter).
fn parse_utc_date_to_ts(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let y: i64 = s[0..4].parse().ok()?;
    let m: i64 = s[5..7].parse().ok()?;
    let d: i64 = s[8..10].parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400)
}

/// Full detail for one commit: metadata + files + live `git show` output.
#[derive(Debug, Clone, Serialize)]
pub struct CommitDetail {
    pub hash: String,
    pub author: String,
    pub ts: i64,
    pub subject: String,
    pub body: String,
    /// Paths from `file_commits` — empty when the commit predates the
    /// captured window (the `stat` text still names every file).
    pub files: Vec<String>,
    /// `git show --stat` summary (per-file +/- counts), fetched live.
    pub stat: String,
    /// Unified diff, truncated to the caller's byte budget.
    pub patch: String,
    pub patch_truncated: bool,
}

/// Live per-commit detail. `sha` must be 7-40 hex chars — anything else
/// (including flag-shaped strings) is rejected before git ever sees it.
/// Metadata comes from `git show` itself (not the DB) so commits outside
/// the captured history window still resolve; `files` comes from the DB
/// and may be empty for such commits.
pub fn commit_detail(
    project_root: &Path,
    conn: &Connection,
    sha: &str,
    patch_budget: usize,
) -> Result<CommitDetail> {
    let sha = sha.trim();
    if sha.len() < 7 || sha.len() > 40 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        anyhow::bail!("invalid commit sha '{sha}' (need 7-40 hex chars)");
    }

    // Same NUL-separated 5-field format as `fetch_commit_metadata` (see the
    // module-level separator note for why NUL and only NUL is injection-proof).
    let meta = run_git(
        project_root,
        &[
            "show",
            "-s",
            "--no-color",
            "--pretty=format:%H%x00%an%x00%ct%x00%s%x00%b%x00",
            sha,
            "--",
        ],
    )?;
    if !meta.status.success() {
        anyhow::bail!(
            "commit '{sha}' not found: {}",
            String::from_utf8_lossy(&meta.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&meta.stdout);
    let fields: Vec<&str> = text.split('\0').collect();
    if fields.len() < 5 {
        anyhow::bail!("unexpected `git show` output for '{sha}'");
    }
    let full_hash = fields[0].trim().to_string();
    let ts: i64 = fields[2].trim().parse().unwrap_or(0);

    let files = files_for_commit(conn, &full_hash).unwrap_or_default();

    // `--format=` suppresses the commit header on both calls — we already
    // have the metadata, these fetch only the stat table / diff body.
    let stat_out = run_git(
        project_root,
        &[
            "show",
            "--stat",
            "--no-color",
            "--format=",
            &full_hash,
            "--",
        ],
    )?;
    let stat = if stat_out.status.success() {
        String::from_utf8_lossy(&stat_out.stdout).trim().to_string()
    } else {
        String::new()
    };

    let patch_out = run_git(
        project_root,
        &[
            "show",
            "--patch",
            "--no-color",
            "--format=",
            &full_hash,
            "--",
        ],
    )?;
    let raw_patch = if patch_out.status.success() {
        String::from_utf8_lossy(&patch_out.stdout).into_owned()
    } else {
        String::new()
    };
    let (patch, patch_truncated) = truncate_at_boundary(&raw_patch, patch_budget);

    Ok(CommitDetail {
        hash: full_hash,
        author: fields[1].to_string(),
        ts,
        subject: fields[3].to_string(),
        body: truncate_body(fields[4].trim()),
        files,
        stat,
        patch,
        patch_truncated,
    })
}

/// Byte-bounded, char-boundary-safe truncation. Returns the (possibly
/// shortened) string and whether truncation happened.
fn truncate_at_boundary(s: &str, max: usize) -> (String, bool) {
    if s.len() <= max {
        return (s.trim_end().to_string(), false);
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    (s[..end].to_string(), true)
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
    // Trailing `--` for the same tracked-file-named-`HEAD` disambiguation as
    // the two `git log` fetches (adversarial-review fix #1).
    let output = run_git(project_root, &["rev-list", "--count", range, "--"]).ok()?;
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
///
/// The format string emits exactly 5 NUL-terminated fields per commit
/// (`%x00` — see the module-level separator note for why NUL and only NUL
/// is injection-proof). `%ct` (committer time) rather than `%at`: recency
/// ordering should reflect when a commit landed on this branch — rebased/
/// cherry-picked work keeps its original author date and would otherwise
/// look ancient in `recent_commits`.
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
    args.push("--pretty=format:%H%x00%an%x00%ct%x00%s%x00%b%x00".into());
    args.push(range.to_string());
    // Disambiguate the range from paths: without this, a tracked file or
    // directory literally named `HEAD` makes git error out with "ambiguous
    // argument" and history never populates (adversarial-review fix #1).
    args.push("--".into());

    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let output = run_git(project_root, &arg_refs)?;
    if !output.status.success() {
        anyhow::bail!(
            "git log failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let text = String::from_utf8_lossy(&output.stdout);

    // Every field (including the last) is NUL-terminated, so the stream is
    // exactly 5 fields per commit; `chunks_exact` drops the trailing empty
    // remainder after the final NUL. `--pretty=format:` inserts a plain
    // newline BETWEEN entries, which lands at the start of the next
    // record's hash field — trimmed below.
    let fields: Vec<&str> = text.split('\0').collect();
    let mut commits = Vec::new();
    for chunk in fields.chunks_exact(5) {
        let hash = chunk[0].trim_start_matches('\n').trim();
        if hash.is_empty() {
            continue;
        }
        let Ok(ts) = chunk[2].parse::<i64>() else {
            continue;
        };
        let body = truncate_body(chunk[4].trim());
        commits.push(RawCommit {
            hash: hash.to_string(),
            author: chunk[1].to_string(),
            ts,
            subject: chunk[3].to_string(),
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
///
/// - `%x00%H` marker: NUL cannot appear in a POSIX filename or a commit
///   message, so record boundaries are exact (see the module separator note).
/// - `-c core.quotepath=off`: default git C-quotes non-ASCII paths in
///   `--name-only` output (`"h\303\251llo.txt"`), which would be stored
///   verbatim and never match real index paths (adversarial-review fix #2).
/// - `--relative`: paths come back relative to `project_root` (the `-C`
///   cwd), matching how the indexer stores them, INCLUDING when
///   `project_root` is a subdirectory of the git repo — default `--name-only`
///   paths are toplevel-relative and would silently never match
///   (adversarial-review fix #5). Also filters out-of-tree files for free.
fn fetch_changed_files(
    project_root: &Path,
    range: &str,
    max_count: Option<usize>,
) -> Result<HashMap<String, Vec<String>>> {
    let mut args: Vec<String> = vec![
        "-c".into(),
        "core.quotepath=off".into(),
        "log".into(),
        "--no-color".into(),
        "--reverse".into(),
        "--name-only".into(),
        "--relative".into(),
    ];
    if let Some(n) = max_count {
        args.push("-n".into());
        args.push(n.to_string());
    }
    args.push("--pretty=format:%x00%H".into());
    args.push(range.to_string());
    // Same `HEAD`-named-file disambiguation as `fetch_commit_metadata`
    // (adversarial-review fix #1).
    args.push("--".into());

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
    for block in text.split('\0') {
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
/// whatever — never blocks the rest). Capped at [`MAX_BLAME_FILES`] per
/// build so a cold build over a huge tree can't fan out into thousands of
/// `git blame` subprocesses.
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
    let files = if changed_files.len() > MAX_BLAME_FILES {
        tracing::warn!(
            total = changed_files.len(),
            cap = MAX_BLAME_FILES,
            "SEMANTEX_HISTORY_BLAME: capping per-build blame fan-out; \
             skipped files get blamed when a later build re-touches them"
        );
        &changed_files[..MAX_BLAME_FILES]
    } else {
        changed_files
    };
    for rel in files {
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
        if sha.bytes().all(|b| b == b'0') {
            // The all-zero sha is git's "not committed yet" sentinel for
            // uncommitted working-tree lines — not a real commit to attribute.
            current = None;
            continue;
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

    /// Adversarial-review fix #1: a tracked file (or directory) literally
    /// named `HEAD` must not break population — without the trailing `--`
    /// on the `git log`/`rev-list` invocations, git errors with "ambiguous
    /// argument 'HEAD'" and history never populates for that repo, forever.
    #[test]
    fn tracked_file_named_head_does_not_break_population() {
        let tmp = TempDir::new().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Test User"]);
        std::fs::write(tmp.path().join("HEAD"), "not a ref, a real file").unwrap();
        git(tmp.path(), &["add", "."]);
        git(tmp.path(), &["commit", "-q", "-m", "add a file named HEAD"]);

        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        let commits = recent_commits(&conn, 10).unwrap();
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].subject, "add a file named HEAD");
        // file_commits must record the HEAD-named file too.
        let hits = commits_touching_file(&conn, "HEAD", 10).unwrap();
        assert_eq!(hits.len(), 1);

        // Incremental catch-up (rev-list --count path) must also survive.
        std::fs::write(tmp.path().join("other.txt"), "x").unwrap();
        git(tmp.path(), &["add", "."]);
        git(tmp.path(), &["commit", "-q", "-m", "second"]);
        populate_history(tmp.path()).unwrap();
        assert_eq!(recent_commits(&conn, 10).unwrap().len(), 2);
    }

    /// Adversarial-review fix #2: default git C-quotes non-ASCII paths in
    /// `--name-only` output (`"h\303\251llo.txt"`), which — stored verbatim
    /// — would never match real index paths. `-c core.quotepath=off` keeps
    /// them raw.
    #[test]
    fn non_ascii_filenames_are_stored_unquoted() {
        let tmp = TempDir::new().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Test User"]);
        std::fs::write(tmp.path().join("héllo.txt"), "bonjour").unwrap();
        git(tmp.path(), &["add", "."]);
        git(tmp.path(), &["commit", "-q", "-m", "add héllo"]);

        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        let hits = commits_touching_file(&conn, "héllo.txt", 10).unwrap();
        assert_eq!(hits.len(), 1, "non-ASCII path must match verbatim");
        // And no C-quoted phantom row.
        let quoted: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM file_commits WHERE path LIKE '\"%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(quoted, 0, "no C-quoted paths may be stored");
    }

    /// Adversarial-review fix #3: a commit whose BODY contains crafted
    /// separator sequences must not parse as additional fake commit rows
    /// (fake authors/subjects would surface to agents via recent_changes /
    /// include_history). With NUL separators this is structurally
    /// impossible — git strips NUL from messages — but this test pins the
    /// property against regression to injectable separators.
    #[test]
    fn hostile_commit_body_cannot_forge_commit_rows() {
        let tmp = TempDir::new().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Real Author"]);
        std::fs::write(tmp.path().join("f.txt"), "x").unwrap();
        git(tmp.path(), &["add", "."]);
        // Body embedding a fully-formed fake record in the OLD \x1e/\x1f
        // separator scheme, plus stray newlines for good measure.
        let hostile_body = format!(
            "innocent first line\n\
             {rs}\ndeadbeefdeadbeefdeadbeefdeadbeefdeadbeef{fs}Fake Author{fs}1{fs}fake subject{fs}fake body{rs}\n\
             trailing text",
            rs = '\u{1e}',
            fs = '\u{1f}',
        );
        git(
            tmp.path(),
            &[
                "commit",
                "-q",
                "-m",
                &format!("real subject\n\n{hostile_body}"),
            ],
        );

        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        let commits = recent_commits(&conn, 10).unwrap();
        assert_eq!(commits.len(), 1, "exactly one real commit, no forged rows");
        assert_eq!(commits[0].author, "Real Author");
        assert_eq!(commits[0].subject, "real subject");
        assert_ne!(commits[0].hash, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef");
        let fake: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM commits WHERE author = 'Fake Author'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(fake, 0);
    }

    /// Adversarial-review fix #5: when the indexed project is a SUBDIRECTORY
    /// of the git repo, `--name-only` paths are toplevel-relative by default
    /// while the index stores project-relative paths — `--relative` makes
    /// them match (and filters out-of-tree files).
    #[test]
    fn subdir_project_stores_project_relative_paths() {
        let tmp = TempDir::new().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Test User"]);
        let sub = tmp.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("inner.txt"), "in subdir").unwrap();
        std::fs::write(tmp.path().join("outer.txt"), "outside project").unwrap();
        git(tmp.path(), &["add", "."]);
        git(tmp.path(), &["commit", "-q", "-m", "touch inner and outer"]);

        // The PROJECT is the subdirectory, not the repo toplevel.
        populate_history(&sub).unwrap();
        let conn = open(&sub).unwrap();
        let hits = commits_touching_file(&conn, "inner.txt", 10).unwrap();
        assert_eq!(
            hits.len(),
            1,
            "paths must be project-relative, not toplevel-relative"
        );
        // The toplevel-relative form must NOT be stored.
        assert!(
            commits_touching_file(&conn, "sub/inner.txt", 10)
                .unwrap()
                .is_empty()
        );
        // Out-of-tree files are filtered by --relative.
        assert!(
            commits_touching_file(&conn, "outer.txt", 10)
                .unwrap()
                .is_empty()
        );
        assert!(
            commits_touching_file(&conn, "../outer.txt", 10)
                .unwrap()
                .is_empty()
        );
    }

    /// Blame parsing must skip git's all-zero "not committed yet" sentinel
    /// sha — uncommitted working-tree lines are not attributable commits.
    #[test]
    fn parse_blame_porcelain_skips_uncommitted_zero_sha() {
        let out = "\
0000000000000000000000000000000000000000 1 1 1\n\
author Not Committed Yet\n\
\tuncommitted line\n\
deadbeefdeadbeefdeadbeefdeadbeefdeadbeef 2 2 1\n\
author Real\n\
\tcommitted line\n";
        let parsed = parse_blame_porcelain(out);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].0, 2);
        assert_eq!(parsed[0].1, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef");
    }

    /// Commit `file` with `msg` at a controlled committer timestamp so
    /// since-filter tests are deterministic (ts resolution is seconds).
    fn commit_with_date(dir: &Path, file: &str, msg: &str, unix_ts: i64) {
        std::fs::write(dir.join(file), msg).unwrap();
        git(dir, &["add", "."]);
        let date = format!("{unix_ts} +0000");
        let status = StdCommand::new("git")
            .arg("-C")
            .arg(dir)
            .env("GIT_COMMITTER_DATE", &date)
            .env("GIT_AUTHOR_DATE", &date)
            .args(["commit", "-q", "-m", msg])
            .status()
            .expect("git must be on PATH for these tests");
        assert!(status.success(), "dated commit failed");
    }

    #[test]
    fn filter_commits_since_ts_is_strictly_greater() {
        let tmp = TempDir::new().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Test User"]);
        commit_with_date(tmp.path(), "a.txt", "first", 1_000_000_000);
        commit_with_date(tmp.path(), "b.txt", "second", 1_000_000_100);
        commit_with_date(tmp.path(), "c.txt", "third", 1_000_000_200);
        populate_history(tmp.path()).unwrap();

        let conn = open(tmp.path()).unwrap();
        let filter = HistoryFilter {
            since_ts: Some(1_000_000_100),
            limit: 10,
            ..Default::default()
        };
        let hits = filter_commits(&conn, &filter).unwrap();
        // Strictly greater: the ts==since_ts commit ("second") is excluded.
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "third");
    }

    #[test]
    fn filter_commits_composes_file_and_message() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 3); // messages "commit number 0..2", files file0..2.txt
        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();

        let hits = filter_commits(
            &conn,
            &HistoryFilter {
                file: Some("file1.txt".into()),
                message_query: Some("number".into()),
                limit: 10,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "commit number 1");

        let none = filter_commits(
            &conn,
            &HistoryFilter {
                file: Some("file1.txt".into()),
                message_query: Some("nonexistent-xyz".into()),
                limit: 10,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn filter_commits_author_substring() {
        let tmp = TempDir::new().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Test User"]);
        std::fs::write(tmp.path().join("a.txt"), "x").unwrap();
        git(tmp.path(), &["add", "."]);
        git(
            tmp.path(),
            &[
                "-c",
                "user.name=Alice Wonder",
                "commit",
                "-q",
                "-m",
                "by alice",
            ],
        );
        std::fs::write(tmp.path().join("b.txt"), "y").unwrap();
        git(tmp.path(), &["add", "."]);
        git(
            tmp.path(),
            &[
                "-c",
                "user.name=Bob Builder",
                "commit",
                "-q",
                "-m",
                "by bob",
            ],
        );
        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();

        let hits = filter_commits(
            &conn,
            &HistoryFilter {
                author: Some("Alice".into()),
                limit: 10,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "by alice");
    }

    #[test]
    fn filter_commits_message_only_delegates_to_fts_search() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 3);
        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();

        let via_filter = filter_commits(
            &conn,
            &HistoryFilter {
                message_query: Some("number 2".into()),
                limit: 10,
                ..Default::default()
            },
        )
        .unwrap();
        let via_search = search_commit_messages(&conn, "number 2", 10).unwrap();
        assert_eq!(via_filter, via_search);
        assert_eq!(via_filter.len(), 1);
    }

    #[test]
    fn filter_commits_zero_limit_defaults_to_twenty() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 25);
        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        let hits = filter_commits(&conn, &HistoryFilter::default()).unwrap();
        assert_eq!(hits.len(), 20);
    }

    #[test]
    fn files_for_commit_lists_touched_paths() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 2);
        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        let commits = recent_commits(&conn, 10).unwrap();
        let files = files_for_commit(&conn, &commits[0].hash).unwrap();
        assert_eq!(files, vec!["file1.txt".to_string()]);
    }

    #[test]
    fn oldest_commit_ts_returns_min() {
        let tmp = TempDir::new().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Test User"]);
        commit_with_date(tmp.path(), "a.txt", "first", 1_000_000_000);
        commit_with_date(tmp.path(), "b.txt", "second", 1_000_000_100);
        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        assert_eq!(oldest_commit_ts(&conn), Some(1_000_000_000));
    }

    #[test]
    fn parse_utc_date_handles_known_values() {
        assert_eq!(parse_utc_date_to_ts("1970-01-01"), Some(0));
        assert_eq!(parse_utc_date_to_ts("2020-01-01"), Some(1_577_836_800));
        assert_eq!(parse_utc_date_to_ts("2026-07-10"), Some(1_783_641_600));
        assert_eq!(parse_utc_date_to_ts("2020-13-01"), None);
        assert_eq!(parse_utc_date_to_ts("2020-00-10"), None);
        assert_eq!(parse_utc_date_to_ts("not-a-date"), None);
        assert_eq!(parse_utc_date_to_ts("2020-1-1"), None); // strict 10-char form
    }

    #[test]
    fn resolve_since_rejects_flag_like_and_empty_input() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(resolve_since_to_ts(tmp.path(), "-n"), None);
        assert_eq!(resolve_since_to_ts(tmp.path(), "--all"), None);
        assert_eq!(resolve_since_to_ts(tmp.path(), "   "), None);
    }

    #[test]
    fn resolve_since_tag_sha_and_garbage() {
        let tmp = TempDir::new().unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Test User"]);
        commit_with_date(tmp.path(), "a.txt", "first", 1_000_000_000);
        git(tmp.path(), &["tag", "v1"]);
        commit_with_date(tmp.path(), "b.txt", "second", 1_000_000_100);

        assert_eq!(resolve_since_to_ts(tmp.path(), "v1"), Some(1_000_000_000));
        // A date string resolves without touching git.
        assert_eq!(
            resolve_since_to_ts(tmp.path(), "2001-09-09"),
            parse_utc_date_to_ts("2001-09-09")
        );
        // HEAD-relative revs work.
        assert_eq!(
            resolve_since_to_ts(tmp.path(), "HEAD~1"),
            Some(1_000_000_000)
        );
        assert_eq!(resolve_since_to_ts(tmp.path(), "no-such-ref-xyz"), None);
    }

    #[test]
    fn commit_detail_returns_stat_files_and_patch() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 2);
        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        let head = recent_commits(&conn, 1).unwrap().remove(0);

        let d = commit_detail(tmp.path(), &conn, &head.hash, 64 * 1024).unwrap();
        assert_eq!(d.hash, head.hash);
        assert_eq!(d.subject, "commit number 1");
        assert_eq!(d.files, vec!["file1.txt".to_string()]);
        assert!(d.stat.contains("file1.txt"), "stat: {}", d.stat);
        assert!(d.patch.contains("+content 1"), "patch: {}", d.patch);
        assert!(!d.patch_truncated);
    }

    #[test]
    fn commit_detail_truncates_patch_to_budget() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 1);
        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        let head = recent_commits(&conn, 1).unwrap().remove(0);

        let d = commit_detail(tmp.path(), &conn, &head.hash, 10).unwrap();
        assert!(d.patch_truncated);
        assert!(d.patch.len() <= 10);
    }

    #[test]
    fn commit_detail_short_prefix_resolves() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 1);
        populate_history(tmp.path()).unwrap();
        let conn = open(tmp.path()).unwrap();
        let head = recent_commits(&conn, 1).unwrap().remove(0);

        let d = commit_detail(tmp.path(), &conn, &head.hash[..8], 4096).unwrap();
        assert_eq!(
            d.hash, head.hash,
            "short prefix must resolve to the full hash"
        );
    }

    #[test]
    fn commit_detail_rejects_non_hex_or_flag_shas() {
        let tmp = TempDir::new().unwrap();
        init_repo(tmp.path(), 1);
        let conn = open(tmp.path()).unwrap();
        assert!(commit_detail(tmp.path(), &conn, "HEAD", 4096).is_err());
        assert!(commit_detail(tmp.path(), &conn, "--stat", 4096).is_err());
        assert!(commit_detail(tmp.path(), &conn, "abc", 4096).is_err()); // < 7 chars
        assert!(
            commit_detail(
                tmp.path(),
                &conn,
                "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
                4096
            )
            .is_err()
        ); // valid form, nonexistent commit
    }
}
