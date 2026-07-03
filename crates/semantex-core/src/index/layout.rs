//! Storage layout v13 — per-branch index directories under
//! `<project>/.semantex/indexes/<branch_key>/`, plus the top-level project
//! meta, `history.db` / `memory.db` schema creation, and legacy-layout
//! migration.
//!
//! # Design notes (read before touching this file)
//!
//! Pre-v13, `<project>/.semantex/` WAS the index directory: `chunks.db`,
//! `meta.json`, `sparse/`, `dense/<backend>/` all lived directly at that root,
//! and dozens of call sites across three crates (`semantex-cli`,
//! `semantex-mcp`, and non-owned modules within this crate) resolve "the
//! index dir" as `SemantexConfig::project_index_dir(project)` and then read
//! those files straight from it. Wave 1 (this file) owns the storage schema
//! but does NOT own those call sites, and the quality gate requires the
//! entire existing test suite to keep passing unmodified.
//!
//! So: **the container root (`<project>/.semantex/`) stays the live,
//! authoritative location for the currently-open branch** — exactly as
//! today, byte-for-byte, so every existing reader keeps working with zero
//! changes. On top of that, this module builds the v13 `indexes/<branch_key>/`
//! structure the contract specifies, mirroring the root's current content
//! into the branch directory after every successful build/update.
//!
//! The mirror is a **snapshot, not an alias**. Root artifacts that mutate
//! in place after a later build (SQLite writes pages inside `chunks.db`;
//! `fs::write` truncates-and-rewrites `dense/**` and `meta.json` through the
//! same inode) are **copied** — `chunks.db` via SQLite `VACUUM INTO` (which
//! also folds in any not-yet-checkpointed WAL content), the rest byte-copies.
//! Hard-linking those would silently morph an old branch's mirror into the
//! new branch's data on the next incremental build. Only `sparse/` (tantivy)
//! is hard-linked: tantivy segment files are write-once and its own
//! `meta.json` is replaced by atomic rename, both of which are hard-link-safe
//! — and they're the bulk of the index, so the snapshot stays near-free on
//! disk. A `hard_link` failure (e.g. cross-device) falls back to a real copy.
//!
//! The top-level `meta.json` gains the v13 fields (`layout_version`,
//! `project_id`, `default_branch`) via [`ProjectMeta`], which
//! `#[serde(flatten)]`s the existing [`IndexMeta`] so legacy readers that
//! `serde_json::from_str::<IndexMeta>()` the same file (unknown fields are
//! ignored by serde) keep working unmodified — this is how
//! `semantex-mcp`'s warm-state fast path and every other non-owned consumer
//! stays green without being touched.
//!
//! `indexes/<branch_key>/meta.json` is a snapshot of whatever the root
//! `meta.json` was at mirror time — plain [`IndexMeta`] on the very first
//! migration of a legacy index, the flattened [`ProjectMeta`] superset on
//! every sync after that. Either way it parses as `IndexMeta` (flattening
//! exists precisely for that compatibility). The contract also asks for
//! "branch name, head_commit" on the per-index meta; rather than adding
//! fields to the shared `IndexMeta` struct (which nine files across the
//! workspace construct via struct literals with no `..Default::default()`,
//! several outside spine's ownership), that data is written to a small
//! sidecar, `indexes/<branch_key>/branch.json` ([`BranchMeta`]).

use crate::types::IndexMeta;
use anyhow::Result;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Current storage layout version (contract §A).
pub const LAYOUT_VERSION: u32 = 13;

/// Branch key used when the branch cannot be resolved (detached HEAD, or the
/// project directory is not a git repository at all).
pub const DEFAULT_BRANCH_KEY: &str = "default";

/// Top-level `<project>/.semantex/meta.json` under v13.
///
/// Deliberately a superset of [`IndexMeta`] (via `#[serde(flatten)]`) so that
/// any code still doing `serde_json::from_str::<IndexMeta>(&meta_json)` on
/// this exact file keeps parsing successfully — see the module doc for why
/// that compatibility matters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMeta {
    pub layout_version: u32,
    pub project_id: String,
    pub default_branch: String,
    #[serde(flatten)]
    pub active_index_meta: IndexMeta,
}

/// Sidecar next to `indexes/<branch_key>/meta.json` carrying the branch
/// identity that the contract asks for on the per-index meta, without
/// widening the shared [`IndexMeta`] struct (see module doc).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BranchMeta {
    pub branch: String,
    pub branch_key: String,
    #[serde(default)]
    pub head_commit: Option<String>,
}

// ── Path helpers ───────────────────────────────────────────────────────────

/// `<project>/.semantex/` — the top-level container. Unchanged from
/// pre-v13; still the live/authoritative dir for the active branch.
pub fn container_dir(project_root: &Path) -> PathBuf {
    project_root.join(".semantex")
}

/// `<project>/.semantex/indexes/`
pub fn indexes_root(project_root: &Path) -> PathBuf {
    container_dir(project_root).join("indexes")
}

/// `<project>/.semantex/indexes/<branch_key>/`
pub fn branch_index_dir(project_root: &Path, branch_key: &str) -> PathBuf {
    indexes_root(project_root).join(branch_key)
}

/// `<project>/.semantex/history.db`
pub fn history_db_path(project_root: &Path) -> PathBuf {
    container_dir(project_root).join("history.db")
}

/// `<project>/.semantex/memory.db`
pub fn memory_db_path(project_root: &Path) -> PathBuf {
    container_dir(project_root).join("memory.db")
}

/// `<project>/.semantex/meta.json` (top-level, v13 [`ProjectMeta`] shape).
pub fn project_meta_path(project_root: &Path) -> PathBuf {
    container_dir(project_root).join("meta.json")
}

// ── branch_key derivation ───────────────────────────────────────────────────

/// Longest sanitized-name prefix kept in a branch_key. The 8-hex ref-name
/// hash carries the identity; the name prefix is only for human readability,
/// so capping it keeps `indexes/<branch_key>/...` paths clear of filesystem
/// name-length limits (ENAMETOOLONG) for pathological branch names.
const BRANCH_KEY_NAME_MAX: usize = 64;

/// Replace every non-alphanumeric (ASCII) byte with `-`, capped at
/// [`BRANCH_KEY_NAME_MAX`] chars. Applied to the branch name before
/// appending the ref-name hash. Output is pure ASCII (non-ASCII chars are
/// not alphanumeric-ASCII, so they map to `-`).
fn sanitize_branch_name(name: &str) -> String {
    name.chars()
        .take(BRANCH_KEY_NAME_MAX)
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

/// First 8 hex chars of SHA256(ref_name).
fn ref_name_hash8(ref_name: &str) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    let mut hasher = Sha256::new();
    hasher.update(ref_name.as_bytes());
    let digest = hasher.finalize();
    digest[..4].iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// branch_key = sanitized branch name (non-alnum → `-`) + `-` + first 8 hex
/// of SHA256(ref name). Exposed standalone (rather than folded into
/// `current_branch_key`) so callers that already have a branch name (e.g.
/// the registry, or a federated target from another project) can compute the
/// same key without touching the filesystem.
pub fn branch_key_for_branch(branch_name: &str) -> String {
    format!(
        "{}-{}",
        sanitize_branch_name(branch_name),
        ref_name_hash8(branch_name)
    )
}

/// Resolve the `.git` directory for `project_root`, following the `gitdir:`
/// pointer file used by worktrees and submodules. Returns `None` if
/// `project_root` is not a git repository.
fn resolve_git_dir(project_root: &Path) -> Option<PathBuf> {
    let dot_git = project_root.join(".git");
    if dot_git.is_dir() {
        return Some(dot_git);
    }
    if dot_git.is_file() {
        let content = std::fs::read_to_string(&dot_git).ok()?;
        let pointer = content
            .lines()
            .find_map(|l| l.trim().strip_prefix("gitdir:"))?;
        let p = PathBuf::from(pointer.trim());
        return Some(if p.is_absolute() {
            p
        } else {
            project_root.join(p)
        });
    }
    None
}

/// Resolve the branch name HEAD currently points to. Returns `None` for a
/// detached HEAD (raw commit hash), an unreadable HEAD, or a non-git
/// directory — all of which map to the `"default"` branch_key.
pub fn resolve_git_head_branch(project_root: &Path) -> Option<String> {
    let git_dir = resolve_git_dir(project_root)?;
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    head.trim()
        .strip_prefix("ref: refs/heads/")
        .map(str::to_string)
}

/// Look up `ref_name` (e.g. `refs/heads/main`) in `git_dir`: loose ref file
/// first, then `packed-refs`.
fn resolve_ref_in(git_dir: &Path, ref_name: &str) -> Option<String> {
    if let Ok(hash) = std::fs::read_to_string(git_dir.join(ref_name)) {
        return Some(hash.trim().to_string());
    }
    let packed = std::fs::read_to_string(git_dir.join("packed-refs")).ok()?;
    packed.lines().find_map(|line| {
        let line = line.trim();
        if line.starts_with('#') {
            return None;
        }
        let mut parts = line.split_whitespace();
        let hash = parts.next()?;
        let name = parts.next()?;
        (name == ref_name).then(|| hash.to_string())
    })
}

/// Resolve HEAD's commit hash on a best-effort basis (loose ref file, with
/// `packed-refs` and worktree-`commondir` fallbacks) rather than shelling out
/// to `git`, matching the rest of this module's no-git2-dependency approach.
pub fn resolve_git_head_commit(project_root: &Path) -> Option<String> {
    let git_dir = resolve_git_dir(project_root)?;
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    if let Some(rest) = head.strip_prefix("ref: ") {
        if let Some(hash) = resolve_ref_in(&git_dir, rest) {
            return Some(hash);
        }
        // Linked worktree: its private gitdir (`.git/worktrees/<name>/`) holds
        // HEAD but NOT the shared refs/packed-refs — those live in the main
        // repo's git dir, pointed to by the `commondir` file (a path relative
        // to the worktree gitdir, usually `../..`).
        if let Ok(common) = std::fs::read_to_string(git_dir.join("commondir")) {
            let common = PathBuf::from(common.trim());
            let common_dir = if common.is_absolute() {
                common
            } else {
                git_dir.join(common)
            };
            return resolve_ref_in(&common_dir, rest);
        }
        return None;
    }
    // Detached HEAD: the file content IS the commit hash.
    Some(head.to_string())
}

/// branch_key for the branch currently checked out at `project_root`.
/// `"default"` for detached HEAD or a non-git directory (contract §A).
pub fn current_branch_key(project_root: &Path) -> String {
    match resolve_git_head_branch(project_root) {
        Some(branch) => branch_key_for_branch(&branch),
        None => DEFAULT_BRANCH_KEY.to_string(),
    }
}

/// The branch name to record alongside `current_branch_key` — mirrors its
/// git-resolution fallback so the two always agree on what "default" means.
pub fn current_branch_name(project_root: &Path) -> String {
    resolve_git_head_branch(project_root).unwrap_or_else(|| DEFAULT_BRANCH_KEY.to_string())
}

// ── history.db / memory.db schema (Wave 2 populates; spine creates) ────────

/// Open (creating if absent) `history.db` at `path`, ensuring the schema
/// from contract §A exists. Returns the raw connection — Wave 2 owns all
/// read/write access beyond schema creation.
pub fn open_history_db(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS commits (
            hash    TEXT PRIMARY KEY,
            author  TEXT NOT NULL,
            ts      INTEGER NOT NULL,
            message TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS file_commits (
            path TEXT NOT NULL,
            hash TEXT NOT NULL,
            PRIMARY KEY (path, hash)
        );
        CREATE TABLE IF NOT EXISTS chunk_blame (
            chunk_id INTEGER NOT NULL,
            hash     TEXT NOT NULL,
            PRIMARY KEY (chunk_id, hash)
        );
        ",
    )?;
    Ok(conn)
}

/// Open (creating if absent) `memory.db` at `path`, ensuring the schema from
/// contract §A exists. Returns the raw connection — Wave 2 owns all
/// read/write access beyond schema creation.
pub fn open_memory_db(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            created_ts INTEGER NOT NULL,
            updated_ts INTEGER NOT NULL,
            scope      TEXT NOT NULL,
            key        TEXT NOT NULL,
            content    TEXT NOT NULL,
            source     TEXT NOT NULL,
            UNIQUE(scope, key)
        );
        ",
    )?;
    Ok(conn)
}

// ── legacy → v13 migration / mirror ─────────────────────────────────────────

/// Recursively hard-link every file under `src` into the same relative path
/// under `dst`, creating directories as needed. Falls back to a real copy
/// for any file where `hard_link` fails (e.g. a cross-device mount) so the
/// mirror is always complete, just potentially non-free on odd filesystems.
///
/// ONLY safe for write-once/rename-replaced artifacts (tantivy `sparse/`):
/// hard links share the inode, so anything the builder later mutates
/// *in place* (SQLite page writes, `fs::write` truncation) would mutate the
/// mirror too. Mutable artifacts go through [`copy_tree`] instead.
fn hardlink_tree(src: &Path, dst: &Path) -> Result<()> {
    if src.is_dir() {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let name = entry.file_name();
            hardlink_tree(&src.join(&name), &dst.join(&name))?;
        }
    } else if src.is_file() {
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if std::fs::hard_link(src, dst).is_err() {
            std::fs::copy(src, dst)?;
        }
    }
    Ok(())
}

/// Recursively byte-copy every file under `src` into `dst` — a true
/// snapshot with its own inodes, immune to later in-place mutation of the
/// source. Used for the artifacts the builder rewrites in place
/// (`dense/**`, `meta.json`, `models.toml`).
fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
    if src.is_dir() {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let name = entry.file_name();
            copy_tree(&src.join(&name), &dst.join(&name))?;
        }
    } else if src.is_file() {
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(src, dst)?;
    }
    Ok(())
}

/// Snapshot a live SQLite database into `dst`.
///
/// Primary path is `VACUUM INTO`, which produces a complete, consistent,
/// compacted copy **including any content still sitting in the `-wal` file**
/// — critical because the builder's own WAL-mode connection is typically
/// still open (un-checkpointed) when the mirror runs, so a plain file copy
/// of `chunks.db` alone would miss the newest rows. If `VACUUM INTO` is
/// unavailable/fails, fall back to `PRAGMA wal_checkpoint(TRUNCATE)` (fold
/// the WAL back into the main file) followed by a byte copy.
fn snapshot_sqlite_db(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(src)?;
    // VACUUM INTO refuses to overwrite; dst lives in a fresh tmp dir so it
    // never exists, but be defensive for the fallback/retry case.
    if dst.exists() {
        std::fs::remove_file(dst)?;
    }
    let dst_str = dst.to_string_lossy().into_owned();
    match conn.execute("VACUUM INTO ?1", [&dst_str]) {
        Ok(_) => Ok(()),
        Err(e) => {
            tracing::debug!("VACUUM INTO failed ({e}); falling back to checkpoint + copy");
            // Best-effort checkpoint; TRUNCATE also resets the -wal file so
            // the main db file is complete on its own. (query_row: this
            // PRAGMA returns a result row, so `execute` would error.)
            let _ = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()));
            std::fs::copy(src, dst)?;
            Ok(())
        }
    }
}

/// Mirror the container root's current index content into
/// `indexes/<branch_key>/`, migrating a legacy (pre-v13) flat layout on
/// first encounter and refreshing an existing mirror on every subsequent
/// call. Rename-based and crash-safe: the mirror is built in a temporary
/// sibling directory and only swapped into place once complete, so a crash
/// mid-mirror leaves either the previous mirror or nothing — never a
/// half-written one — and the root (which this function never modifies)
/// stays fully functional throughout. No-op (returns `Ok(())`) if the
/// container root has no `chunks.db` yet (nothing built).
///
/// This is deliberately NOT a destructive move: the root stays the live,
/// authoritative copy that every existing (non-owned) reader depends on —
/// see the module doc for why.
pub fn mirror_into_branch_dir(project_root: &Path, branch_key: &str) -> Result<()> {
    let container = container_dir(project_root);
    if !container.join("chunks.db").exists() {
        return Ok(());
    }

    let branch_dir = branch_index_dir(project_root, branch_key);
    let tmp_dir = indexes_root(project_root).join(format!("{branch_key}.tmp-mirror"));
    if tmp_dir.exists() {
        std::fs::remove_dir_all(&tmp_dir)?;
    }
    std::fs::create_dir_all(&tmp_dir)?;

    // chunks.db: SQLite snapshot — includes un-checkpointed WAL content and
    // detaches the mirror from future in-place page writes (see
    // `snapshot_sqlite_db`). NEVER hard-link a live SQLite db.
    snapshot_sqlite_db(&container.join("chunks.db"), &tmp_dir.join("chunks.db"))?;

    // meta.json / models.toml: rewritten in place by `fs::write` on later
    // builds (same inode, truncate + rewrite) — must be copies.
    for name in ["meta.json", "models.toml"] {
        let src = container.join(name);
        if src.is_file() {
            copy_tree(&src, &tmp_dir.join(name))?;
        }
    }
    // dense/**: vectors/pointer files are rewritten via `fs::write` on
    // incremental builds — must be copies.
    let dense_src = container.join("dense");
    if dense_src.is_dir() {
        copy_tree(&dense_src, &tmp_dir.join("dense"))?;
    }
    // sparse/ (tantivy): segment files are write-once and tantivy's own
    // meta.json is replaced by atomic rename — hard-link-safe, and it's the
    // bulk of the index, so links keep the snapshot near-free on disk.
    let sparse_src = container.join("sparse");
    if sparse_src.is_dir() {
        hardlink_tree(&sparse_src, &tmp_dir.join("sparse"))?;
    }

    if branch_dir.exists() {
        std::fs::remove_dir_all(&branch_dir)?;
    }
    std::fs::rename(&tmp_dir, &branch_dir)?;

    // Branch identity sidecar (contract: per-index meta carries branch name +
    // head_commit; kept out of the shared IndexMeta struct — see module doc).
    let branch_meta = BranchMeta {
        branch: current_branch_name(project_root),
        branch_key: branch_key.to_string(),
        head_commit: resolve_git_head_commit(project_root),
    };
    let json = serde_json::to_string_pretty(&branch_meta)?;
    std::fs::write(branch_dir.join("branch.json"), json)?;

    Ok(())
}

/// Ensure the full v13 layout exists for `project_root`: migrate/refresh the
/// per-branch mirror (if any index content exists yet), create the
/// `history.db` / `memory.db` schemas, and upgrade the top-level
/// `meta.json` to the v13 [`ProjectMeta`] shape (superset-compatible with
/// legacy `IndexMeta` readers). `project_id` is a stable identifier for this
/// project (see [`crate::index::registry`] for how it's minted/looked up).
///
/// Mirror-step failures are logged and do not fail the overall call — the
/// root index (built by the existing, unmodified `IndexBuilder` pipeline)
/// remains valid and searchable regardless of whether the v13 mirror could
/// be written.
pub fn sync_v13_layout(project_root: &Path, project_id: &str) -> Result<PathBuf> {
    let container = container_dir(project_root);
    std::fs::create_dir_all(&container)?;
    let branch_key = current_branch_key(project_root);

    if let Err(e) = mirror_into_branch_dir(project_root, &branch_key) {
        tracing::warn!("v13 layout mirror failed (root index still valid): {e}");
    }

    let _ = open_history_db(&history_db_path(project_root))?;
    let _ = open_memory_db(&memory_db_path(project_root))?;

    let meta_path = project_meta_path(project_root);
    if let Ok(existing) = std::fs::read_to_string(&meta_path)
        && let Ok(active_index_meta) = serde_json::from_str::<IndexMeta>(&existing)
    {
        let project_meta = ProjectMeta {
            layout_version: LAYOUT_VERSION,
            project_id: project_id.to_string(),
            default_branch: current_branch_name(project_root),
            active_index_meta,
        };
        let json = serde_json::to_string_pretty(&project_meta)?;
        std::fs::write(&meta_path, json)?;
    }

    Ok(branch_index_dir(project_root, &branch_key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn sanitize_replaces_non_alnum() {
        assert_eq!(sanitize_branch_name("feature/foo-bar"), "feature-foo-bar");
        assert_eq!(sanitize_branch_name("main"), "main");
        assert_eq!(sanitize_branch_name("a b.c"), "a-b-c");
    }

    #[test]
    fn branch_key_is_deterministic_and_unique_per_name() {
        let k1 = branch_key_for_branch("main");
        let k2 = branch_key_for_branch("main");
        let k3 = branch_key_for_branch("develop");
        assert_eq!(k1, k2, "same branch name must yield the same key");
        assert_ne!(k1, k3, "different branch names must yield different keys");
        assert!(k1.starts_with("main-"));
        // sanitized-name + '-' + 8 hex chars
        let hash_part = k1.rsplit('-').next().unwrap();
        assert_eq!(hash_part.len(), 8);
        assert!(hash_part.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn non_git_dir_resolves_to_default_branch_key() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(current_branch_key(tmp.path()), DEFAULT_BRANCH_KEY);
        assert_eq!(current_branch_name(tmp.path()), DEFAULT_BRANCH_KEY);
    }

    #[test]
    fn resolves_current_branch_from_real_git_dir() {
        let tmp = TempDir::new().unwrap();
        let git_dir = tmp.path().join(".git");
        std::fs::create_dir_all(git_dir.join("refs").join("heads")).unwrap();
        std::fs::write(git_dir.join("HEAD"), "ref: refs/heads/feature/x\n").unwrap();
        std::fs::write(git_dir.join("refs").join("heads").join("x"), "deadbeef\n").unwrap();
        assert_eq!(
            resolve_git_head_branch(tmp.path()),
            Some("feature/x".to_string())
        );
        assert_eq!(
            current_branch_key(tmp.path()),
            branch_key_for_branch("feature/x")
        );
    }

    #[test]
    fn detached_head_resolves_to_default() {
        let tmp = TempDir::new().unwrap();
        let git_dir = tmp.path().join(".git");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::write(git_dir.join("HEAD"), "deadbeefdeadbeefdeadbeef\n").unwrap();
        assert_eq!(resolve_git_head_branch(tmp.path()), None);
        assert_eq!(current_branch_key(tmp.path()), DEFAULT_BRANCH_KEY);
    }

    #[test]
    fn worktree_gitdir_pointer_is_followed() {
        let tmp = TempDir::new().unwrap();
        let real_git = tmp.path().join("real.git").join("worktrees").join("wt1");
        std::fs::create_dir_all(&real_git).unwrap();
        std::fs::write(real_git.join("HEAD"), "ref: refs/heads/main\n").unwrap();

        let worktree = tmp.path().join("worktree-dir");
        std::fs::create_dir_all(&worktree).unwrap();
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", real_git.display()),
        )
        .unwrap();

        assert_eq!(resolve_git_head_branch(&worktree), Some("main".to_string()));
    }

    fn sample_index_meta() -> IndexMeta {
        IndexMeta {
            schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: PathBuf::from("/x"),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 3,
            chunk_count: 9,
            embedding_model: "CodeRankEmbed".to_string(),
            embedding_dim: 768,
            use_bm25_stemmer: true,
            dense_backend: "coderank-hnsw".to_string(),
            embedder_fingerprint: "fp".to_string(),
        }
    }

    /// The whole point of flattening: existing readers that parse
    /// `meta.json` as plain `IndexMeta` must keep working against the v13
    /// top-level `ProjectMeta` shape.
    #[test]
    fn project_meta_is_superset_compatible_with_index_meta() {
        let project_meta = ProjectMeta {
            layout_version: LAYOUT_VERSION,
            project_id: "proj-abc".to_string(),
            default_branch: "main".to_string(),
            active_index_meta: sample_index_meta(),
        };
        let json = serde_json::to_string(&project_meta).unwrap();

        // A legacy reader doing exactly what state::is_stale / builder.rs do.
        let back: IndexMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.schema_version, IndexMeta::CURRENT_SCHEMA_VERSION);
        assert_eq!(back.embedder_fingerprint, "fp");

        // And it must also round-trip as ProjectMeta.
        let back2: ProjectMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back2.layout_version, LAYOUT_VERSION);
        assert_eq!(back2.project_id, "proj-abc");
        assert_eq!(back2.default_branch, "main");
        assert_eq!(back2.active_index_meta.chunk_count, 9);
    }

    #[test]
    fn history_and_memory_db_schemas_are_created() {
        let tmp = TempDir::new().unwrap();
        let history = open_history_db(&tmp.path().join("history.db")).unwrap();
        history
            .execute(
                "INSERT INTO commits (hash, author, ts, message) VALUES ('h1', 'a', 1, 'm')",
                [],
            )
            .unwrap();
        history
            .execute(
                "INSERT INTO file_commits (path, hash) VALUES ('f.rs', 'h1')",
                [],
            )
            .unwrap();
        history
            .execute(
                "INSERT INTO chunk_blame (chunk_id, hash) VALUES (1, 'h1')",
                [],
            )
            .unwrap();

        let memory = open_memory_db(&tmp.path().join("memory.db")).unwrap();
        memory
            .execute(
                "INSERT INTO notes (created_ts, updated_ts, scope, key, content, source) \
                 VALUES (1, 1, 's', 'k', 'c', 'src')",
                [],
            )
            .unwrap();
        // UNIQUE(scope, key) must reject a duplicate.
        let dup = memory.execute(
            "INSERT INTO notes (created_ts, updated_ts, scope, key, content, source) \
             VALUES (2, 2, 's', 'k', 'c2', 'src2')",
            [],
        );
        assert!(dup.is_err(), "duplicate (scope,key) must be rejected");

        // Re-opening (idempotent schema creation) must not error.
        open_history_db(&tmp.path().join("history.db")).unwrap();
        open_memory_db(&tmp.path().join("memory.db")).unwrap();
    }

    /// Round-trip: build an index with the LEGACY (pre-v13) flat layout,
    /// then run the v13 sync — the content must be migrated into
    /// `indexes/<branch_key>/` and stay searchable there.
    #[test]
    fn legacy_layout_round_trips_through_v13_migration() {
        use crate::index::storage::ChunkStore;
        use crate::types::{Chunk, ChunkType};

        let tmp = TempDir::new().unwrap();
        let project = tmp.path();
        let container = container_dir(project);
        std::fs::create_dir_all(&container).unwrap();

        // Simulate a legacy build: chunks.db + meta.json directly at the root.
        let chunk_id = {
            let store = ChunkStore::open(&container.join("chunks.db")).unwrap();
            let chunk = Chunk {
                id: 0,
                file_path: PathBuf::from("src/lib.rs"),
                start_line: 1,
                end_line: 5,
                content: "fn legacy() {}".to_string(),
                chunk_type: ChunkType::TextWindow { window_index: 0 },
            };
            store.insert_chunk(&chunk, 0xdead, 0).unwrap()
        };
        std::fs::write(
            container.join("meta.json"),
            serde_json::to_string(&sample_index_meta()).unwrap(),
        )
        .unwrap();
        std::fs::create_dir_all(container.join("sparse")).unwrap();
        std::fs::write(container.join("sparse").join("seg.dat"), b"tantivy-ish").unwrap();

        let branch_dir = sync_v13_layout(project, "proj-123").unwrap();

        // Migrated: the branch dir now has its own chunks.db with the same data.
        assert!(branch_dir.join("chunks.db").exists());
        assert!(branch_dir.join("meta.json").exists());
        assert!(branch_dir.join("branch.json").exists());
        assert!(branch_dir.join("sparse").join("seg.dat").exists());

        let migrated_store = ChunkStore::open(&branch_dir.join("chunks.db")).unwrap();
        let chunk = migrated_store.get_chunk(chunk_id).unwrap();
        assert_eq!(chunk.content, "fn legacy() {}");

        // The branch meta.json is a snapshot of the root meta.json as of
        // mirror time (plain IndexMeta on this first legacy migration; the
        // flattened ProjectMeta on later syncs) — either way it must parse
        // as IndexMeta, the shape every per-index reader expects.
        let branch_meta_json = std::fs::read_to_string(branch_dir.join("meta.json")).unwrap();
        let branch_meta: IndexMeta = serde_json::from_str(&branch_meta_json).unwrap();
        assert_eq!(branch_meta.chunk_count, 9);

        // The root is UNCHANGED and still fully functional (no readers broke).
        let root_store = ChunkStore::open(&container.join("chunks.db")).unwrap();
        let root_chunk = root_store.get_chunk(chunk_id).unwrap();
        assert_eq!(root_chunk.content, "fn legacy() {}");

        // Top-level meta.json is now the v13 ProjectMeta shape, but still
        // parses as plain IndexMeta for legacy readers.
        let top_json = std::fs::read_to_string(container.join("meta.json")).unwrap();
        let top_as_project: ProjectMeta = serde_json::from_str(&top_json).unwrap();
        assert_eq!(top_as_project.layout_version, LAYOUT_VERSION);
        assert_eq!(top_as_project.project_id, "proj-123");
        let top_as_legacy: IndexMeta = serde_json::from_str(&top_json).unwrap();
        assert_eq!(top_as_legacy.chunk_count, 9);

        // Idempotent: running the sync again must not error and must refresh cleanly.
        let branch_dir_2 = sync_v13_layout(project, "proj-123").unwrap();
        assert_eq!(branch_dir, branch_dir_2);
        assert!(branch_dir_2.join("chunks.db").exists());

        // history.db / memory.db created at the container root.
        assert!(history_db_path(project).exists());
        assert!(memory_db_path(project).exists());
    }

    /// A brand-new project (no `chunks.db` yet) must not error — sync is a
    /// safe no-op for the mirror step, though it still creates the aux DBs.
    #[test]
    fn sync_on_empty_project_is_a_safe_no_op_for_mirror() {
        let tmp = TempDir::new().unwrap();
        let branch_dir = sync_v13_layout(tmp.path(), "proj-empty").unwrap();
        assert!(!branch_dir.join("chunks.db").exists());
        assert!(history_db_path(tmp.path()).exists());
        assert!(memory_db_path(tmp.path()).exists());
    }

    #[test]
    fn branch_key_name_prefix_is_capped() {
        let long_name = "x".repeat(500);
        let key = branch_key_for_branch(&long_name);
        // capped prefix + '-' + 8 hex chars
        assert_eq!(key.len(), BRANCH_KEY_NAME_MAX + 1 + 8, "key was: {key}");
        // Identity still lives in the hash: two long names sharing the first
        // 64 chars must still get distinct keys.
        let key2 = branch_key_for_branch(&format!("{}{}", "x".repeat(64), "different-tail"));
        assert_ne!(key, key2);
        // And the same name is still deterministic.
        assert_eq!(key, branch_key_for_branch(&long_name));
    }

    /// Linked-worktree layout: the worktree's private gitdir holds HEAD but
    /// refs/packed-refs live in the main repo's git dir, reachable via the
    /// `commondir` pointer file. head_commit must resolve through it.
    #[test]
    fn worktree_commondir_fallback_resolves_head_commit() {
        let tmp = TempDir::new().unwrap();
        // Main repo git dir with the actual ref.
        let main_git = tmp.path().join("main.git");
        std::fs::create_dir_all(main_git.join("refs").join("heads")).unwrap();
        std::fs::write(
            main_git.join("refs").join("heads").join("main"),
            "cafebabe\n",
        )
        .unwrap();
        // Worktree private gitdir: HEAD + commondir, but NO refs of its own.
        let wt_git = main_git.join("worktrees").join("wt1");
        std::fs::create_dir_all(&wt_git).unwrap();
        std::fs::write(wt_git.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        std::fs::write(wt_git.join("commondir"), "../..\n").unwrap();

        let worktree = tmp.path().join("wt");
        std::fs::create_dir_all(&worktree).unwrap();
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", wt_git.display()),
        )
        .unwrap();

        assert_eq!(
            resolve_git_head_commit(&worktree),
            Some("cafebabe".to_string())
        );
        // packed-refs via commondir too.
        std::fs::remove_file(main_git.join("refs").join("heads").join("main")).unwrap();
        std::fs::write(main_git.join("packed-refs"), "deadf00d refs/heads/main\n").unwrap();
        assert_eq!(
            resolve_git_head_commit(&worktree),
            Some("deadf00d".to_string())
        );
    }

    /// THE snapshot guarantee (adversarial-review followup #3): after a
    /// mirror, mutating the ROOT index in place — new SQLite rows in
    /// chunks.db, `fs::write` truncation of dense/** and meta.json (exactly
    /// what an incremental re-index after a branch switch does) — must NOT
    /// change the mirrored branch dir. Hard-linked aliases fail this test;
    /// snapshots pass it.
    #[test]
    fn mirror_is_a_snapshot_not_an_alias_of_the_root() {
        use crate::index::storage::ChunkStore;
        use crate::types::{Chunk, ChunkType};

        let tmp = TempDir::new().unwrap();
        let project = tmp.path();
        let container = container_dir(project);
        std::fs::create_dir_all(container.join("dense").join("backend")).unwrap();

        let chunk = |id: u64, content: &str| Chunk {
            id,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 1,
            end_line: 2,
            content: content.to_string(),
            chunk_type: ChunkType::TextWindow { window_index: 0 },
        };

        let store = ChunkStore::open(&container.join("chunks.db")).unwrap();
        let old_id = store
            .insert_chunk(&chunk(0, "fn old_branch() {}"), 1, 0)
            .unwrap();
        std::fs::write(
            container.join("meta.json"),
            serde_json::to_string(&sample_index_meta()).unwrap(),
        )
        .unwrap();
        std::fs::write(
            container.join("dense").join("backend").join("vectors.bin"),
            b"old-branch-vectors",
        )
        .unwrap();

        mirror_into_branch_dir(project, "oldbranch-aaaa1111").unwrap();
        let mirror = branch_index_dir(project, "oldbranch-aaaa1111");

        // Simulate `git switch` + incremental re-index mutating the ROOT
        // in place, through the same inodes the old code hard-linked:
        // SQLite in-place writes...
        let new_id = store
            .insert_chunk(&chunk(0, "fn new_branch() {}"), 2, 0)
            .unwrap();
        drop(store);
        // ...and fs::write truncation of dense + meta.
        std::fs::write(
            container.join("dense").join("backend").join("vectors.bin"),
            b"NEW-branch-vectors-completely-different",
        )
        .unwrap();
        let mut new_meta = sample_index_meta();
        new_meta.chunk_count = 999;
        std::fs::write(
            container.join("meta.json"),
            serde_json::to_string(&new_meta).unwrap(),
        )
        .unwrap();

        // The mirror must still be the OLD branch's data, untouched.
        let mirror_store = ChunkStore::open(&mirror.join("chunks.db")).unwrap();
        assert_eq!(
            mirror_store.get_chunk(old_id).unwrap().content,
            "fn old_branch() {}"
        );
        assert!(
            mirror_store.get_chunk(new_id).is_err(),
            "post-mirror root writes must NOT leak into the mirror"
        );
        assert_eq!(
            std::fs::read(mirror.join("dense").join("backend").join("vectors.bin")).unwrap(),
            b"old-branch-vectors"
        );
        let mirror_meta: IndexMeta =
            serde_json::from_str(&std::fs::read_to_string(mirror.join("meta.json")).unwrap())
                .unwrap();
        assert_eq!(
            mirror_meta.chunk_count, 9,
            "meta snapshot must not follow the root"
        );
    }

    /// WAL correctness (followup #3a): the builder's WAL-mode connection is
    /// still OPEN (rows un-checkpointed, living only in `chunks.db-wal`)
    /// when the mirror runs. The snapshot must still contain those rows —
    /// `VACUUM INTO` (or checkpoint-then-copy) guarantees it; a naive file
    /// copy/hard-link of `chunks.db` alone would silently drop them.
    #[test]
    fn mirror_captures_unchekpointed_wal_content() {
        use crate::index::storage::ChunkStore;
        use crate::types::{Chunk, ChunkType};

        let tmp = TempDir::new().unwrap();
        let project = tmp.path();
        let container = container_dir(project);
        std::fs::create_dir_all(&container).unwrap();

        // ChunkStore opens in WAL mode; KEEP the connection open across the
        // mirror so nothing gets checkpointed by a connection close.
        let store = ChunkStore::open(&container.join("chunks.db")).unwrap();
        let id = store
            .insert_chunk(
                &Chunk {
                    id: 0,
                    file_path: PathBuf::from("src/wal.rs"),
                    start_line: 1,
                    end_line: 1,
                    content: "fn only_in_wal() {}".to_string(),
                    chunk_type: ChunkType::TextWindow { window_index: 0 },
                },
                7,
                0,
            )
            .unwrap();

        mirror_into_branch_dir(project, "walbranch-bbbb2222").unwrap();

        let mirror = branch_index_dir(project, "walbranch-bbbb2222");
        let mirror_store = ChunkStore::open(&mirror.join("chunks.db")).unwrap();
        assert_eq!(
            mirror_store.get_chunk(id).unwrap().content,
            "fn only_in_wal() {}",
            "rows still in the root's WAL must be present in the snapshot"
        );
        drop(store);
    }

    /// Branch-switch simulation (followup #4 semantics): after switching git
    /// HEAD to a new branch, a sync — even one triggered by a build that
    /// found zero content changes — must create the NEW branch's index dir
    /// while leaving the old branch's mirror intact.
    #[test]
    fn sync_after_branch_switch_creates_new_branch_dir_and_keeps_old() {
        use crate::index::storage::ChunkStore;
        use crate::types::{Chunk, ChunkType};

        let tmp = TempDir::new().unwrap();
        let project = tmp.path();
        let container = container_dir(project);
        std::fs::create_dir_all(&container).unwrap();

        // Fake git repo on branch `main`.
        let git = project.join(".git");
        std::fs::create_dir_all(git.join("refs").join("heads")).unwrap();
        std::fs::write(git.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        std::fs::write(git.join("refs").join("heads").join("main"), "abc123\n").unwrap();

        {
            let store = ChunkStore::open(&container.join("chunks.db")).unwrap();
            store
                .insert_chunk(
                    &Chunk {
                        id: 0,
                        file_path: PathBuf::from("src/a.rs"),
                        start_line: 1,
                        end_line: 1,
                        content: "fn a() {}".to_string(),
                        chunk_type: ChunkType::TextWindow { window_index: 0 },
                    },
                    1,
                    0,
                )
                .unwrap();
        }
        std::fs::write(
            container.join("meta.json"),
            serde_json::to_string(&sample_index_meta()).unwrap(),
        )
        .unwrap();

        sync_v13_layout(project, "proj-sw").unwrap();
        let main_dir = branch_index_dir(project, &branch_key_for_branch("main"));
        assert!(main_dir.join("chunks.db").exists());

        // `git switch -c feature/new` (same tree → a re-index would find
        // zero content changes and take the builder's early return — which
        // now also syncs).
        std::fs::write(git.join("HEAD"), "ref: refs/heads/feature/new\n").unwrap();
        std::fs::write(git.join("refs").join("heads").join("feature"), "").unwrap(); // noise
        sync_v13_layout(project, "proj-sw").unwrap();

        let feature_dir = branch_index_dir(project, &branch_key_for_branch("feature/new"));
        assert!(
            feature_dir.join("chunks.db").exists(),
            "new branch dir must be created by a no-change sync"
        );
        assert!(
            main_dir.join("chunks.db").exists(),
            "old branch mirror must survive the switch"
        );
        // Each carries its own branch identity sidecar.
        let feature_meta: BranchMeta = serde_json::from_str(
            &std::fs::read_to_string(feature_dir.join("branch.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(feature_meta.branch, "feature/new");
        let main_meta: BranchMeta =
            serde_json::from_str(&std::fs::read_to_string(main_dir.join("branch.json")).unwrap())
                .unwrap();
        assert_eq!(main_meta.branch, "main");
    }
}
