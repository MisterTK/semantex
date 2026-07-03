//! Global project registry — tracks all repos that have been indexed.
//!
//! Stored at `~/.semantex/projects.json`. **v2** (contract §A) is a versioned
//! object: `{ version: 2, projects: [{ path, project_id, display_name,
//! branches: [...], embedder_fingerprint }] }`. **v1** (pre-v13) was a bare
//! JSON array of canonical absolute path strings; [`load`] transparently
//! upgrades a v1 file to v2 in place (the next [`register`]/[`upsert_branch`]
//! call persists the upgraded shape).
//!
//! Both the CLI session hook and the MCP server read this to discover repos
//! that may have drifted (index age > threshold) without waiting for a user
//! to open them — via [`read_all`], which keeps its pre-v13 signature
//! (`Vec<PathBuf>`) so neither caller needed to change.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

fn registry_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".semantex").join("projects.json"))
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

/// Load the registry, transparently upgrading a v1 (bare path array) file to
/// the v2 shape in memory. Returns an empty v2 registry if the file is
/// absent, unreadable, or unparseable as either shape — the registry is a
/// best-effort discovery aid, never a hard dependency.
pub fn load() -> RegistryV2 {
    let Some(path) = registry_path() else {
        return RegistryV2::default();
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
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

fn save(registry: &RegistryV2) -> bool {
    let Some(path) = registry_path() else {
        return false;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Ok(json) = serde_json::to_string_pretty(registry) else {
        return false;
    };
    std::fs::write(&path, json).is_ok()
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

/// Register a project in the global registry (upsert — no duplicates).
///
/// Signature preserved from v1. Internally upgrades/creates a v2
/// [`ProjectEntry`] for `canonical` if one doesn't already exist.
pub fn register(canonical: &Path) {
    let mut registry = load();
    if registry.projects.iter().any(|p| p.path == canonical) {
        return; // already registered
    }
    registry.projects.push(ProjectEntry {
        path: canonical.to_path_buf(),
        project_id: project_id_for_path(canonical),
        display_name: display_name_for_path(canonical),
        branches: Vec::new(),
        embedder_fingerprint: String::new(),
    });
    save(&registry);
}

/// Record (upsert) that `branch` of the project at `canonical` was just
/// indexed, stamping `last_indexed_ts` (Unix seconds) and the resolved
/// `head_commit`. Creates the project entry first via [`register`]-equivalent
/// logic if it doesn't exist yet. Used by `IndexBuilder::build` to keep the
/// registry's branch list in sync with what's actually been built.
pub fn upsert_branch(
    canonical: &Path,
    branch: &str,
    branch_key: &str,
    last_indexed_ts: i64,
    head_commit: Option<String>,
    embedder_fingerprint: &str,
) {
    let mut registry = load();
    let entry = if let Some(existing) = registry.projects.iter_mut().find(|p| p.path == canonical) {
        existing
    } else {
        registry.projects.push(ProjectEntry {
            path: canonical.to_path_buf(),
            project_id: project_id_for_path(canonical),
            display_name: display_name_for_path(canonical),
            branches: Vec::new(),
            embedder_fingerprint: String::new(),
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
    save(&registry);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // `registry_path()` is a fixed `~/.semantex/projects.json` — every test
    // that touches the real file must be serialized against every other, and
    // must restore the original content afterward so this suite doesn't
    // clobber a developer's real registry.
    static REGISTRY_FILE_LOCK: Mutex<()> = Mutex::new(());

    struct RegistryFileGuard {
        path: PathBuf,
        original: Option<String>,
    }

    impl RegistryFileGuard {
        fn acquire() -> Self {
            let path = registry_path().expect("home dir resolvable in test env");
            let original = std::fs::read_to_string(&path).ok();
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::remove_file(&path);
            Self { path, original }
        }
    }

    impl Drop for RegistryFileGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(content) => {
                    let _ = std::fs::write(&self.path, content);
                }
                None => {
                    let _ = std::fs::remove_file(&self.path);
                }
            }
        }
    }

    #[test]
    fn register_and_read_all_round_trip() {
        let _lock = REGISTRY_FILE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _guard = RegistryFileGuard::acquire();

        let p1 = PathBuf::from("/tmp/project-one");
        let p2 = PathBuf::from("/tmp/project-two");
        register(&p1);
        register(&p2);
        register(&p1); // duplicate — must not double-register

        let all = read_all();
        assert_eq!(all.len(), 2);
        assert!(all.contains(&p1));
        assert!(all.contains(&p2));

        let v2 = read_all_v2();
        assert_eq!(v2.version, 2);
        let entry = v2.projects.iter().find(|p| p.path == p1).unwrap();
        assert_eq!(entry.project_id, project_id_for_path(&p1));
        assert_eq!(entry.display_name, "project-one");
    }

    #[test]
    fn v1_array_is_upgraded_to_v2_on_load() {
        let _lock = REGISTRY_FILE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _guard = RegistryFileGuard::acquire();

        let path = registry_path().unwrap();
        let v1_json =
            serde_json::to_string(&vec!["/repo/a".to_string(), "/repo/b".to_string()]).unwrap();
        std::fs::write(&path, v1_json).unwrap();

        let v2 = load();
        assert_eq!(v2.version, 2);
        assert_eq!(v2.projects.len(), 2);
        assert!(v2.projects.iter().any(|p| p.path == Path::new("/repo/a")));

        // read_all() (the pre-existing API surface) must also see the
        // upgraded content transparently.
        let all = read_all();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn upsert_branch_creates_project_and_updates_existing_branch() {
        let _lock = REGISTRY_FILE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _guard = RegistryFileGuard::acquire();

        let proj = PathBuf::from("/tmp/branchy-project");
        upsert_branch(
            &proj,
            "main",
            "main-abc12345",
            100,
            Some("c1".into()),
            "fp1",
        );
        let v2 = read_all_v2();
        let entry = v2.projects.iter().find(|p| p.path == proj).unwrap();
        assert_eq!(entry.branches.len(), 1);
        assert_eq!(entry.branches[0].last_indexed_ts, 100);
        assert_eq!(entry.embedder_fingerprint, "fp1");

        // Re-indexing the SAME branch_key updates in place, not append.
        upsert_branch(
            &proj,
            "main",
            "main-abc12345",
            200,
            Some("c2".into()),
            "fp2",
        );
        let v2 = read_all_v2();
        let entry = v2.projects.iter().find(|p| p.path == proj).unwrap();
        assert_eq!(entry.branches.len(), 1);
        assert_eq!(entry.branches[0].last_indexed_ts, 200);
        assert_eq!(entry.branches[0].head_commit, Some("c2".to_string()));
        assert_eq!(entry.embedder_fingerprint, "fp2");

        // A second branch appends rather than replacing.
        upsert_branch(&proj, "develop", "develop-def67890", 50, None, "fp2");
        let v2 = read_all_v2();
        let entry = v2.projects.iter().find(|p| p.path == proj).unwrap();
        assert_eq!(entry.branches.len(), 2);
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
