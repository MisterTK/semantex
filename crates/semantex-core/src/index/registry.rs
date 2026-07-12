//! Global project registry — tracks all repos that have been indexed.
//!
//! Stored at `<semantex_home>/projects.json` (i.e. `~/.semantex/projects.json`
//! by default; [`crate::config::SemantexConfig::semantex_home`] honors the
//! `SEMANTEX_HOME` env var, so the whole registry relocates with it — which is
//! also how tests and sandboxed environments keep it away from real user
//! state). **v2** (contract §A) is a versioned object: `{ version: 2,
//! projects: [{ path, project_id, display_name, branches: [...],
//! embedder_fingerprint }] }`. **v1** (pre-v13) was a bare JSON array of
//! canonical absolute path strings; [`load`] transparently upgrades a v1 file
//! to v2 in place (the next [`register`]/[`upsert_branch`] call persists the
//! upgraded shape).
//!
//! Writes are atomic (tmp file + rename in the same directory), so a crash
//! mid-save can never leave a torn/corrupt `projects.json` behind. Concurrent
//! writers (two `semantex index` runs racing) are **last-write-wins** at
//! whole-file granularity: each writer read-modify-writes the full file, and
//! the final rename decides. That can drop the other writer's single upsert,
//! but never corrupts the file — acceptable for a best-effort discovery aid.
//!
//! Both the CLI session hook and the MCP server read this to discover repos
//! that may have drifted (index age > threshold) without waiting for a user
//! to open them — via [`read_all`], which keeps its pre-v13 signature
//! (`Vec<PathBuf>`) so neither caller needed to change.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Resolve the registry file location. Routed through
/// [`SemantexConfig::semantex_home`](crate::config::SemantexConfig::semantex_home)
/// so it honors `SEMANTEX_HOME` (and never resolves to a developer's real
/// home when that override is set).
fn registry_path() -> PathBuf {
    crate::config::SemantexConfig::semantex_home().join("projects.json")
}

/// One tracked branch of a registered project (contract §A).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BranchEntry {
    pub branch: String,
    pub branch_key: String,
    #[serde(default)]
    pub last_indexed_ts: i64,
    #[serde(default)]
    pub head_commit: Option<String>,
}

/// One registered project (contract §A).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProjectEntry {
    pub path: PathBuf,
    #[serde(default)]
    pub project_id: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub branches: Vec<BranchEntry>,
    #[serde(default)]
    pub embedder_fingerprint: String,
    /// True if `path` is a linked git worktree rather than the repo's
    /// primary checkout. Plain JSON registry, so old entries missing this
    /// key deserialize as `false` — no schema version bump needed.
    #[serde(default)]
    pub is_worktree: bool,
}

/// The versioned registry file shape (contract §A: `"version": 2`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryV2 {
    pub version: u32,
    pub projects: Vec<ProjectEntry>,
}

impl Default for RegistryV2 {
    fn default() -> Self {
        Self {
            version: 2,
            projects: Vec::new(),
        }
    }
}

/// Derive a stable, filesystem/JSON-safe project id from a canonical path.
/// Purely a function of the path (not random) so re-registering the same
/// project always yields the same id, and so `IndexBuilder` can compute the
/// same id independently when stamping `ProjectMeta::project_id`
/// (`index/layout.rs`) without a registry round-trip.
pub fn project_id_for_path(canonical: &Path) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    let mut hasher = Sha256::new();
    hasher.update(canonical.to_string_lossy().as_bytes());
    let digest = hasher.finalize();
    digest[..8].iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

fn display_name_for_path(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.to_string_lossy().to_string(),
        |n| n.to_string_lossy().to_string(),
    )
}

/// Path-parameterized core of [`load`]. Tests point this at a tempdir file so
/// they never read (let alone delete) a developer's real registry.
fn load_from(path: &Path) -> RegistryV2 {
    let Ok(content) = std::fs::read_to_string(path) else {
        return RegistryV2::default();
    };
    if let Ok(v2) = serde_json::from_str::<RegistryV2>(&content) {
        return v2;
    }
    // v1: a bare array of canonical path strings.
    if let Ok(v1_paths) = serde_json::from_str::<Vec<String>>(&content) {
        let projects = v1_paths
            .into_iter()
            .map(|p| {
                let path = PathBuf::from(&p);
                ProjectEntry {
                    project_id: project_id_for_path(&path),
                    display_name: display_name_for_path(&path),
                    path,
                    branches: Vec::new(),
                    embedder_fingerprint: String::new(),
                    is_worktree: false,
                }
            })
            .collect();
        return RegistryV2 {
            version: 2,
            projects,
        };
    }
    RegistryV2::default()
}

/// Load the registry, transparently upgrading a v1 (bare path array) file to
/// the v2 shape in memory. Returns an empty v2 registry if the file is
/// absent, unreadable, or unparseable as either shape — the registry is a
/// best-effort discovery aid, never a hard dependency.
pub fn load() -> RegistryV2 {
    load_from(&registry_path())
}

/// Atomically persist the registry to `path`: serialize to a tmp file in the
/// SAME directory, then `rename` over the destination. A crash at any point
/// leaves either the old complete file or the new complete file — never a
/// truncated/torn one (which `load_from` would silently read as an empty
/// registry, losing every registered project). Concurrent writers are
/// last-write-wins at whole-file granularity (see module doc).
fn save_to(path: &Path, registry: &RegistryV2) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return false;
    }
    let Ok(json) = serde_json::to_string_pretty(registry) else {
        return false;
    };
    // Pid-suffixed so two racing processes never stomp each other's tmp file;
    // same dir as the destination so the rename is atomic (no cross-device).
    let tmp = parent.join(format!(
        ".{}.tmp.{}",
        path.file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default(),
        std::process::id()
    ));
    if std::fs::write(&tmp, json).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    let ok = std::fs::rename(&tmp, path).is_ok();
    if !ok {
        let _ = std::fs::remove_file(&tmp);
    }
    ok
}

/// Read all registered project paths from the registry.
///
/// Signature preserved from v1 (`Vec<PathBuf>`) so existing callers
/// (`semantex-mcp`, `semantex-cli`) keep compiling and behaving unchanged
/// across the v1 → v2 upgrade.
pub fn read_all() -> Vec<PathBuf> {
    load().projects.into_iter().map(|p| p.path).collect()
}

/// Read the full v2 registry (path, project_id, branches, embedder
/// fingerprint, etc.) — used by [`crate::search::federation`] to resolve
/// cross-project search targets. `read_all` stays the lightweight
/// path-only accessor for existing callers.
pub fn read_all_v2() -> RegistryV2 {
    load()
}

/// True if `path` is the OS temp root itself (`std::env::temp_dir()`, or the
/// common Unix aliases `/tmp` / `/private/tmp`) — never a subdirectory under
/// it. Every tool on the machine writes here continuously, so treating the
/// bare root as a tracked "project" produces an index that's permanently
/// stale: it fails every staleness check forever, which is exactly the
/// unbounded-refresh loop [`retain`] and the session-hook spawn cap exist to
/// stop. A genuine scratch repo *under* the temp root (a test fixture, a
/// smoke-test clone) is unaffected — only the bare root is refused.
pub fn is_system_temp_root(path: &Path) -> bool {
    let mut candidates = vec![
        std::env::temp_dir(),
        PathBuf::from("/tmp"),
        PathBuf::from("/private/tmp"),
    ];
    candidates.dedup();
    candidates
        .iter()
        .any(|c| path == c || c.canonicalize().is_ok_and(|canon| canon == path))
}

/// True if `path` is the OS temp root OR anything underneath it — a scratch
/// test fixture, a smoke-test clone, any depth.
///
/// This is the broader counterpart to [`is_system_temp_root`], deliberately
/// scoped to *unattended* auto-indexing (the session-start hook's
/// index-whatever-cwd-this-session-opened-in trigger, and the MCP server's
/// index-on-first-tool-use trigger) — nobody asked for a throwaway scratch
/// directory to become a permanently tracked project. It is intentionally
/// NOT applied to the explicit `semantex index <path>` CLI command or to
/// [`register`]/[`upsert_branch`] themselves: this project's own test and
/// benchmark harnesses deliberately build indexes against corpora under
/// `/tmp` (a scratch fixture, a cloned sample repo), and that's a real,
/// intentional action a human or a script asked for — refusing it would
/// break working infrastructure, not protect anyone. Any registry entry
/// that *does* end up under a temp root through such a deliberate call
/// self-heals via [`retain`] once the directory is gone.
pub fn is_under_system_temp_root(path: &Path) -> bool {
    let mut candidates = vec![
        std::env::temp_dir(),
        PathBuf::from("/tmp"),
        PathBuf::from("/private/tmp"),
    ];
    candidates.dedup();
    candidates
        .iter()
        .any(|c| path.starts_with(c) || c.canonicalize().is_ok_and(|canon| path.starts_with(canon)))
}

/// Above this many immediate subdirectories with their own `.git`, a
/// `.git`-less `path` looks like a multi-repo workspace container rather than
/// a single project — see [`is_likely_multi_repo_container`].
const MULTI_REPO_THRESHOLD: usize = 3;

/// True if `path` looks like a container of multiple independent repos (a
/// `~/dev`-style workspace folder holding dozens of separate checkouts)
/// rather than a single project to index.
///
/// Criteria: `path` itself has no `.git`, AND at least
/// [`MULTI_REPO_THRESHOLD`] of its *immediate* subdirectories do. The walker
/// has no concept of a repo boundary — it would flatten every nested repo's
/// files into one undifferentiated project, with no per-repo size cap, no
/// way to search them individually, and (for a real multi-repo workspace) a
/// single build spanning dozens of independent codebases at once.
///
/// A single repo containing git submodules is unaffected: submodules live
/// several directories deep, not as `path`'s own immediate children, and
/// `path` itself has a `.git` (submodule-containing repos return `false`
/// immediately). A plain non-git project directory (no `.git` anywhere) also
/// returns `false` — only having *several sibling repos* trips this, not
/// merely lacking version control.
pub fn is_likely_multi_repo_container(path: &Path) -> bool {
    if path.join(".git").exists() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    let nested_repo_count = entries
        .filter_map(Result::ok)
        .filter(|e| e.path().join(".git").exists())
        .count();
    nested_repo_count >= MULTI_REPO_THRESHOLD
}

/// True if `path` is a linked git worktree checkout rather than the primary
/// clone that owns its `.git` directory. Compares `git rev-parse --git-dir`
/// against `--git-common-dir`: identical for a normal checkout; different
/// for a linked worktree (`--git-dir` resolves under
/// `<main-repo>/.git/worktrees/<name>`, `--git-common-dir` points at the
/// shared `.git`). Git-native, so this is correct regardless of which tool
/// created the worktree (`git worktree add`, an IDE, an agent harness, ...)
/// — not a match on a path convention like `.claude/worktrees/`. Any git
/// failure (missing binary, non-git directory) returns `false`: fails
/// closed, so a detection error just means "index and register normally",
/// never "silently skip a real project".
pub fn is_worktree_checkout(path: &Path) -> bool {
    let (Some(git_dir), Some(common_dir)) = (
        git_rev_parse(path, "--git-dir"),
        git_rev_parse(path, "--git-common-dir"),
    ) else {
        return false;
    };
    match (
        resolve_under(path, &git_dir),
        resolve_under(path, &common_dir),
    ) {
        (Some(g), Some(c)) => g != c,
        _ => false,
    }
}

fn git_rev_parse(path: &Path, flag: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(path)
        .args(["rev-parse", flag])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// `git rev-parse --git-dir`/`--git-common-dir` return a path relative to
/// `path` when possible — resolve against `path` and canonicalize so a
/// relative and an absolute answer for the same directory compare equal.
fn resolve_under(path: &Path, git_path: &str) -> Option<PathBuf> {
    let p = Path::new(git_path);
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        path.join(p)
    };
    joined.canonicalize().ok()
}

/// Path-parameterized core of [`retain`].
fn retain_at(path: &Path, mut keep: impl FnMut(&ProjectEntry) -> bool) -> usize {
    let mut registry = load_from(path);
    let before = registry.projects.len();
    registry.projects.retain(|p| keep(p));
    let removed = before - registry.projects.len();
    if removed > 0 {
        save_to(path, &registry);
    }
    removed
}

/// Drop registry entries for which `keep` returns `false`, persisting the
/// result. Returns the number of entries removed (0 skips the write).
///
/// The registry only ever grows via [`register`]/[`upsert_branch`] — nothing
/// previously removed a project once its directory was deleted or it turned
/// out to be a redundant nested path within another registered repo. This is
/// the self-healing counterpart, used by callers that periodically sweep the
/// registry (e.g. the session-start hook) rather than any repo-specific logic.
pub fn retain(keep: impl FnMut(&ProjectEntry) -> bool) -> usize {
    retain_at(&registry_path(), keep)
}

/// Path-parameterized core of [`register`].
fn register_at(path: &Path, canonical: &Path) {
    if is_system_temp_root(canonical) {
        return;
    }
    let mut registry = load_from(path);
    if registry.projects.iter().any(|p| p.path == canonical) {
        return; // already registered
    }
    registry.projects.push(ProjectEntry {
        path: canonical.to_path_buf(),
        project_id: project_id_for_path(canonical),
        display_name: display_name_for_path(canonical),
        branches: Vec::new(),
        embedder_fingerprint: String::new(),
        is_worktree: is_worktree_checkout(canonical),
    });
    save_to(path, &registry);
}

/// Register a project in the global registry (upsert — no duplicates).
///
/// Signature preserved from v1. Internally upgrades/creates a v2
/// [`ProjectEntry`] for `canonical` if one doesn't already exist.
pub fn register(canonical: &Path) {
    register_at(&registry_path(), canonical);
}

/// Path-parameterized core of [`upsert_branch`].
#[allow(clippy::too_many_arguments)]
fn upsert_branch_at(
    path: &Path,
    canonical: &Path,
    branch: &str,
    branch_key: &str,
    last_indexed_ts: i64,
    head_commit: Option<String>,
    embedder_fingerprint: &str,
) {
    if is_system_temp_root(canonical) {
        return;
    }
    let mut registry = load_from(path);
    let entry = if let Some(existing) = registry.projects.iter_mut().find(|p| p.path == canonical) {
        existing
    } else {
        registry.projects.push(ProjectEntry {
            path: canonical.to_path_buf(),
            project_id: project_id_for_path(canonical),
            display_name: display_name_for_path(canonical),
            branches: Vec::new(),
            embedder_fingerprint: String::new(),
            is_worktree: is_worktree_checkout(canonical),
        });
        registry.projects.last_mut().expect("just pushed")
    };
    entry.embedder_fingerprint = embedder_fingerprint.to_string();
    if let Some(existing_branch) = entry
        .branches
        .iter_mut()
        .find(|b| b.branch_key == branch_key)
    {
        existing_branch.branch = branch.to_string();
        existing_branch.last_indexed_ts = last_indexed_ts;
        existing_branch.head_commit = head_commit;
    } else {
        entry.branches.push(BranchEntry {
            branch: branch.to_string(),
            branch_key: branch_key.to_string(),
            last_indexed_ts,
            head_commit,
        });
    }
    save_to(path, &registry);
}

/// Record (upsert) that `branch` of the project at `canonical` was just
/// indexed, stamping `last_indexed_ts` (Unix seconds) and the resolved
/// `head_commit`. Creates the project entry first via [`register`]-equivalent
/// logic if it doesn't exist yet. Available for Wave 2's multi-branch daemon
/// to keep the registry's branch list in sync with what's actually been built.
pub fn upsert_branch(
    canonical: &Path,
    branch: &str,
    branch_key: &str,
    last_indexed_ts: i64,
    head_commit: Option<String>,
    embedder_fingerprint: &str,
) {
    upsert_branch_at(
        &registry_path(),
        canonical,
        branch,
        branch_key,
        last_indexed_ts,
        head_commit,
        embedder_fingerprint,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // Every test operates on its own tempdir registry file via the `_at`/
    // `_from`/`_to` internals — the public wrappers only add path resolution
    // (`registry_path()`), so this covers all the logic while guaranteeing no
    // test can ever read, overwrite, or delete a developer's real
    // `~/.semantex/projects.json`. (The old delete-and-restore guard was
    // unsafe: a SIGKILL/panic=abort inside the window destroyed the real
    // file permanently.) No env-var mutation either, so tests stay
    // parallel-safe.
    fn tmp_registry() -> (TempDir, PathBuf) {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("projects.json");
        (tmp, path)
    }

    fn git_test(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .status()
            .expect("git must be on PATH for these tests");
        assert!(status.success(), "git {args:?} failed");
    }

    /// A minimal one-commit repo with a deterministic `main` branch name.
    fn init_git_repo(dir: &Path) {
        git_test(dir, &["init", "-q"]);
        git_test(dir, &["config", "user.email", "test@example.com"]);
        git_test(dir, &["config", "user.name", "Test User"]);
        std::fs::write(dir.join("a.txt"), "a").unwrap();
        git_test(dir, &["add", "."]);
        git_test(dir, &["commit", "-q", "-m", "init"]);
        git_test(dir, &["branch", "-M", "main"]);
    }

    #[test]
    fn is_worktree_checkout_false_for_non_git_dir() {
        let tmp = TempDir::new().unwrap();
        assert!(!is_worktree_checkout(tmp.path()));
    }

    #[test]
    fn is_worktree_checkout_distinguishes_main_from_linked_worktree() {
        let main_repo = TempDir::new().unwrap();
        init_git_repo(main_repo.path());

        let worktree_dir = TempDir::new().unwrap();
        // `git worktree add` requires the target directory to not exist yet.
        std::fs::remove_dir(worktree_dir.path()).unwrap();
        git_test(
            main_repo.path(),
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "feature-x",
                worktree_dir.path().to_str().unwrap(),
            ],
        );

        assert!(!is_worktree_checkout(main_repo.path()));
        assert!(is_worktree_checkout(worktree_dir.path()));
    }

    #[test]
    fn register_at_stamps_is_worktree_correctly() {
        let main_repo = TempDir::new().unwrap();
        init_git_repo(main_repo.path());

        let worktree_dir = TempDir::new().unwrap();
        std::fs::remove_dir(worktree_dir.path()).unwrap();
        git_test(
            main_repo.path(),
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "feature-y",
                worktree_dir.path().to_str().unwrap(),
            ],
        );

        let (_tmp, reg) = tmp_registry();
        let main_canonical = main_repo.path().canonicalize().unwrap();
        let worktree_canonical = worktree_dir.path().canonicalize().unwrap();
        register_at(&reg, &main_canonical);
        register_at(&reg, &worktree_canonical);

        let registry = load_from(&reg);
        let main_entry = registry
            .projects
            .iter()
            .find(|p| p.path == main_canonical)
            .unwrap();
        let worktree_entry = registry
            .projects
            .iter()
            .find(|p| p.path == worktree_canonical)
            .unwrap();
        assert!(!main_entry.is_worktree);
        assert!(worktree_entry.is_worktree);
    }

    #[test]
    fn register_and_read_all_round_trip() {
        let (_tmp, reg) = tmp_registry();

        let p1 = PathBuf::from("/tmp/project-one");
        let p2 = PathBuf::from("/tmp/project-two");
        register_at(&reg, &p1);
        register_at(&reg, &p2);
        register_at(&reg, &p1); // duplicate — must not double-register

        let all: Vec<PathBuf> = load_from(&reg)
            .projects
            .into_iter()
            .map(|p| p.path)
            .collect();
        assert_eq!(all.len(), 2);
        assert!(all.contains(&p1));
        assert!(all.contains(&p2));

        let v2 = load_from(&reg);
        assert_eq!(v2.version, 2);
        let entry = v2.projects.iter().find(|p| p.path == p1).unwrap();
        assert_eq!(entry.project_id, project_id_for_path(&p1));
        assert_eq!(entry.display_name, "project-one");
    }

    #[test]
    fn multi_repo_container_true_above_threshold_when_root_has_no_git() {
        let tmp = TempDir::new().unwrap();
        for name in ["repo-a", "repo-b", "repo-c"] {
            std::fs::create_dir_all(tmp.path().join(name).join(".git")).unwrap();
        }
        assert!(is_likely_multi_repo_container(tmp.path()));
    }

    #[test]
    fn multi_repo_container_false_below_threshold() {
        let tmp = TempDir::new().unwrap();
        for name in ["repo-a", "repo-b"] {
            std::fs::create_dir_all(tmp.path().join(name).join(".git")).unwrap();
        }
        assert!(!is_likely_multi_repo_container(tmp.path()));
    }

    #[test]
    fn multi_repo_container_false_when_root_itself_is_a_repo() {
        // A repo with several git submodules must not trip the guard.
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join(".git")).unwrap();
        for name in ["sub-a", "sub-b", "sub-c"] {
            std::fs::create_dir_all(tmp.path().join(name).join(".git")).unwrap();
        }
        assert!(!is_likely_multi_repo_container(tmp.path()));
    }

    #[test]
    fn multi_repo_container_false_for_plain_non_git_directory() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        assert!(!is_likely_multi_repo_container(tmp.path()));
    }

    #[test]
    fn is_system_temp_root_true_for_env_temp_dir_and_unix_aliases() {
        assert!(is_system_temp_root(&std::env::temp_dir()));
        assert!(is_system_temp_root(Path::new("/tmp")));
        assert!(is_system_temp_root(Path::new("/private/tmp")));
    }

    #[test]
    fn is_system_temp_root_false_for_a_subdirectory_or_unrelated_path() {
        assert!(!is_system_temp_root(Path::new("/tmp/some-scratch-repo")));
        assert!(!is_system_temp_root(Path::new("/Users/dev/my-project")));
    }

    #[test]
    fn is_under_system_temp_root_true_for_root_and_any_depth_subdirectory() {
        assert!(is_under_system_temp_root(Path::new("/tmp")));
        assert!(is_under_system_temp_root(Path::new(
            "/tmp/some-scratch-repo"
        )));
        assert!(is_under_system_temp_root(Path::new(
            "/private/tmp/pytest-of-tk/pytest-1/tiny_corpus"
        )));
    }

    #[test]
    fn is_under_system_temp_root_false_for_unrelated_path() {
        assert!(!is_under_system_temp_root(Path::new(
            "/Users/dev/my-project"
        )));
    }

    #[test]
    fn register_at_refuses_the_system_temp_root() {
        let (_tmp, reg) = tmp_registry();
        register_at(&reg, &std::env::temp_dir());
        assert!(load_from(&reg).projects.is_empty());
    }

    #[test]
    fn upsert_branch_at_refuses_the_system_temp_root() {
        let (_tmp, reg) = tmp_registry();
        upsert_branch_at(
            &reg,
            &std::env::temp_dir(),
            "main",
            "main-abc",
            1,
            None,
            "fp",
        );
        assert!(load_from(&reg).projects.is_empty());
    }

    #[test]
    fn retain_removes_only_entries_the_predicate_rejects() {
        let (_tmp, reg) = tmp_registry();

        let p1 = PathBuf::from("/tmp/keep-me");
        let p2 = PathBuf::from("/tmp/drop-me");
        register_at(&reg, &p1);
        register_at(&reg, &p2);

        let removed = retain_at(&reg, |p| p.path != p2);
        assert_eq!(removed, 1);

        let all: Vec<PathBuf> = load_from(&reg)
            .projects
            .into_iter()
            .map(|p| p.path)
            .collect();
        assert_eq!(all, vec![p1]);
    }

    #[test]
    fn retain_is_a_no_op_write_when_nothing_is_removed() {
        let (_tmp, reg) = tmp_registry();

        let p1 = PathBuf::from("/tmp/keep-me");
        register_at(&reg, &p1);

        let removed = retain_at(&reg, |_| true);
        assert_eq!(removed, 0);

        let all: Vec<PathBuf> = load_from(&reg)
            .projects
            .into_iter()
            .map(|p| p.path)
            .collect();
        assert_eq!(all, vec![p1]);
    }

    #[test]
    fn v1_array_is_upgraded_to_v2_on_load() {
        let (_tmp, reg) = tmp_registry();

        let v1_json =
            serde_json::to_string(&vec!["/repo/a".to_string(), "/repo/b".to_string()]).unwrap();
        std::fs::write(&reg, v1_json).unwrap();

        let v2 = load_from(&reg);
        assert_eq!(v2.version, 2);
        assert_eq!(v2.projects.len(), 2);
        assert!(v2.projects.iter().any(|p| p.path == Path::new("/repo/a")));

        // Registering upgrades the persisted file to v2 (versioned object,
        // v1 content preserved).
        register_at(&reg, Path::new("/repo/c"));
        let raw = std::fs::read_to_string(&reg).unwrap();
        let persisted: RegistryV2 = serde_json::from_str(&raw).unwrap();
        assert_eq!(persisted.version, 2);
        assert_eq!(persisted.projects.len(), 3);
    }

    #[test]
    fn upsert_branch_creates_project_and_updates_existing_branch() {
        let (_tmp, reg) = tmp_registry();

        let proj = PathBuf::from("/tmp/branchy-project");
        upsert_branch_at(
            &reg,
            &proj,
            "main",
            "main-abc12345",
            100,
            Some("c1".into()),
            "fp1",
        );
        let v2 = load_from(&reg);
        let entry = v2.projects.iter().find(|p| p.path == proj).unwrap();
        assert_eq!(entry.branches.len(), 1);
        assert_eq!(entry.branches[0].last_indexed_ts, 100);
        assert_eq!(entry.embedder_fingerprint, "fp1");

        // Re-indexing the SAME branch_key updates in place, not append.
        upsert_branch_at(
            &reg,
            &proj,
            "main",
            "main-abc12345",
            200,
            Some("c2".into()),
            "fp2",
        );
        let v2 = load_from(&reg);
        let entry = v2.projects.iter().find(|p| p.path == proj).unwrap();
        assert_eq!(entry.branches.len(), 1);
        assert_eq!(entry.branches[0].last_indexed_ts, 200);
        assert_eq!(entry.branches[0].head_commit, Some("c2".to_string()));
        assert_eq!(entry.embedder_fingerprint, "fp2");

        // A second branch appends rather than replacing.
        upsert_branch_at(&reg, &proj, "develop", "develop-def67890", 50, None, "fp2");
        let v2 = load_from(&reg);
        let entry = v2.projects.iter().find(|p| p.path == proj).unwrap();
        assert_eq!(entry.branches.len(), 2);
    }

    #[test]
    fn save_is_atomic_no_tmp_file_left_behind() {
        let (_tmp, reg) = tmp_registry();
        register_at(&reg, Path::new("/tmp/atomic-project"));

        // The registry file exists and parses; no tmp artifact lingers in
        // the directory (write went through tmp-then-rename).
        assert!(reg.exists());
        let entries: Vec<_> = std::fs::read_dir(reg.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["projects.json".to_string()], "{entries:?}");
    }

    #[test]
    fn corrupt_registry_loads_as_empty_and_is_replaced_on_next_save() {
        let (_tmp, reg) = tmp_registry();
        std::fs::write(&reg, "{ this is not json").unwrap();
        assert!(load_from(&reg).projects.is_empty());
        register_at(&reg, Path::new("/tmp/recovered"));
        assert_eq!(load_from(&reg).projects.len(), 1);
    }

    #[test]
    fn registry_path_honors_semantex_home_layout() {
        // registry_path() must be `<semantex_home>/projects.json` — the same
        // resolution the rest of the crate uses (gate.rs, models dir), which
        // honors the SEMANTEX_HOME env override. We assert the relationship
        // rather than mutating the env (env mutation is process-global and
        // racy under the parallel test runner).
        assert_eq!(
            registry_path(),
            crate::config::SemantexConfig::semantex_home().join("projects.json")
        );
    }

    #[test]
    fn project_id_is_stable_for_same_path() {
        let p = PathBuf::from("/tmp/stable-id-project");
        assert_eq!(project_id_for_path(&p), project_id_for_path(&p));
        assert_ne!(
            project_id_for_path(&p),
            project_id_for_path(&PathBuf::from("/tmp/other-project"))
        );
    }
}
