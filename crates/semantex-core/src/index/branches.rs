//! Branch-switch orchestration (Wave 2, contract §F "multi-branch
//! watch/daemon").
//!
//! Storage layout v13 (`index/layout.rs`) keeps `<project>/.semantex/` as the
//! single LIVE index and mirrors it into `indexes/<branch_key>/` snapshots
//! after every build. That alone makes switching branches *safe* (nothing is
//! lost), but not *cheap*: without this module, a `git switch` back to a
//! branch that was already indexed still pays a full hash-diff re-index of
//! the whole tree, because the root's `chunks.db` still holds the OTHER
//! branch's file hashes.
//!
//! This module makes it cheap: every entry point that opens or updates an
//! index (`semantex index`, `watch`, `serve`, the MCP auto-index hook) calls
//! [`detect_and_handle_branch_switch`] first. It compares
//! [`layout::current_branch_key`] against the branch recorded in the root's
//! [`layout::BranchMeta`] sidecar and, on a mismatch:
//!
//! - **the new branch already has a snapshot** (`indexes/<new_key>/` exists)
//!   → restore it into the root ([`layout::restore_branch_dir_into_root`]).
//!   The caller's normal incremental build then only has to catch drift
//!   since that snapshot was taken (files changed after the last time this
//!   branch was indexed) — not re-embed the world.
//! - **it doesn't** → snapshot the OUTGOING branch's current root content
//!   under its OLD key ([`layout::mirror_into_branch_dir`]) so it isn't lost,
//!   then leave the root as-is; the caller's normal incremental build
//!   re-indexes it in place for the new branch (today's behavior — file-hash
//!   comparison naturally picks up every file that differs between the two
//!   branches).
//!
//! Also owns the two other pieces of Wave 2 bookkeeping that hang off the
//! same "a build just happened" moment: retention (capping how many
//! `indexes/<branch_key>/` snapshots a project accumulates) and the registry
//! `branches[]` upsert (`index::registry::upsert_branch`, shipped in Wave 1
//! but zero production callers until this module wires it up).

use crate::index::layout;
use crate::index::registry;
use anyhow::Result;
use std::path::Path;

/// What [`detect_and_handle_branch_switch`] did, for logging/tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchSwitchAction {
    /// No branch recorded for the root yet (brand-new project, or a
    /// `.semantex/` that predates the Wave 2 root sidecar) — nothing to
    /// reconcile. The caller's normal build path establishes the sidecar.
    FirstBuild,
    /// The root's recorded branch already matches HEAD — nothing to do.
    Unchanged { branch_key: String },
    /// HEAD moved to a branch with a saved snapshot; it was restored into
    /// the root. The caller should still run its normal incremental build
    /// to catch any drift since the snapshot was taken.
    Restored {
        from_branch_key: String,
        to_branch_key: String,
    },
    /// HEAD moved to a branch with no saved snapshot. The outgoing branch's
    /// root content was snapshotted under its old key; the root itself is
    /// untouched, so the caller's normal incremental build re-indexes it in
    /// place for the new branch.
    SnapshottedOutgoing {
        from_branch_key: String,
        to_branch_key: String,
    },
}

impl BranchSwitchAction {
    /// Whether this action mutated the root's index content (restore or
    /// snapshot-outgoing) — i.e. whether a caller that only builds "if
    /// something changed" should force a build regardless of its own
    /// change-detection.
    pub fn switched(&self) -> bool {
        matches!(
            self,
            BranchSwitchAction::Restored { .. } | BranchSwitchAction::SnapshottedOutgoing { .. }
        )
    }
}

/// Cheap, read-only check: has HEAD moved to a different branch than the one
/// the root's live index currently belongs to? Safe to call on a hot path
/// (e.g. MCP `initialize`) since it does no filesystem mutation beyond the
/// two small reads `current_branch_key`/`read_root_branch_meta` already do.
///
/// Returns `false` for a brand-new project (nothing recorded yet) — that
/// case is "first build", not "switch", and the normal
/// not-indexed-yet trigger already covers it.
pub fn branch_switch_pending(project_root: &Path) -> bool {
    let Some(root_meta) = layout::read_root_branch_meta(project_root) else {
        return false;
    };
    root_meta.branch_key != layout::current_branch_key(project_root)
}

/// Detect a branch switch since the root was last synced and, if one
/// happened, reconcile the root's index content (see module doc for the two
/// cases). Idempotent and cheap when nothing has changed (`Unchanged`).
pub fn detect_and_handle_branch_switch(project_root: &Path) -> Result<BranchSwitchAction> {
    let new_key = layout::current_branch_key(project_root);
    let Some(root_meta) = layout::read_root_branch_meta(project_root) else {
        return Ok(BranchSwitchAction::FirstBuild);
    };
    if root_meta.branch_key == new_key {
        return Ok(BranchSwitchAction::Unchanged {
            branch_key: new_key,
        });
    }

    let new_branch_dir = layout::branch_index_dir(project_root, &new_key);
    if new_branch_dir.join("chunks.db").exists() {
        layout::restore_branch_dir_into_root(project_root, &new_key)?;
        tracing::info!(
            from = %root_meta.branch_key,
            to = %new_key,
            "Branch switch detected: restored existing snapshot into root"
        );
        Ok(BranchSwitchAction::Restored {
            from_branch_key: root_meta.branch_key,
            to_branch_key: new_key,
        })
    } else {
        // Snapshot the OUTGOING branch's still-live root content under its
        // OLD identity (`root_meta`, recorded at the last sync) — NOT
        // `mirror_into_branch_dir`, which would re-derive the sidecar's
        // branch name from the CURRENT git HEAD (already the new branch by
        // the time this runs) and mislabel the outgoing snapshot.
        layout::mirror_root_as(project_root, &root_meta)?;
        tracing::info!(
            from = %root_meta.branch_key,
            to = %new_key,
            "Branch switch detected: snapshotted outgoing branch, root will be re-indexed in place"
        );
        Ok(BranchSwitchAction::SnapshottedOutgoing {
            from_branch_key: root_meta.branch_key,
            to_branch_key: new_key,
        })
    }
}

/// Cap on stored `indexes/<branch_key>/` snapshots per project, overridable
/// via `SEMANTEX_MAX_BRANCH_INDEXES` (any non-positive/unparseable value
/// falls back to the default).
fn max_branch_indexes() -> usize {
    std::env::var("SEMANTEX_MAX_BRANCH_INDEXES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(5)
}

/// Evict the oldest branch snapshots beyond the retention cap
/// ([`max_branch_indexes`]), always keeping `current_branch_key` regardless
/// of recency. Recency is the `branch.json` sidecar's mtime, refreshed by
/// every [`layout::mirror_into_branch_dir`] call. Returns the evicted
/// branch_keys (empty if nothing needed evicting).
pub fn enforce_retention(project_root: &Path, current_branch_key: &str) -> Result<Vec<String>> {
    enforce_retention_with_cap(project_root, current_branch_key, max_branch_indexes())
}

/// Cap-parameterized core of [`enforce_retention`] — tests call this
/// directly instead of mutating the process-global `SEMANTEX_MAX_BRANCH_INDEXES`
/// env var, which would race against other tests in the same binary (see
/// `index::registry`'s `_at`/`_from` split for the same pattern/rationale).
pub fn enforce_retention_with_cap(
    project_root: &Path,
    current_branch_key: &str,
    cap: usize,
) -> Result<Vec<String>> {
    let indexes_root = layout::indexes_root(project_root);
    if !indexes_root.is_dir() {
        return Ok(Vec::new());
    }

    let mut candidates: Vec<(String, std::time::SystemTime)> = Vec::new();
    for entry in std::fs::read_dir(&indexes_root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        // Skip in-flight tmp-swap dirs from mirror_into_branch_dir /
        // restore_branch_dir_into_root — never candidates for eviction.
        if name.contains(".tmp-") {
            continue;
        }
        let branch_json = indexes_root.join(&name).join("branch.json");
        let mtime = std::fs::metadata(&branch_json)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        candidates.push((name, mtime));
    }

    if candidates.len() <= cap {
        return Ok(Vec::new());
    }

    // Newest first, so the first `cap` distinct keys (current always
    // included) are the ones we keep.
    candidates.sort_by(|a, b| b.1.cmp(&a.1));

    let mut keep: std::collections::HashSet<String> = std::collections::HashSet::new();
    keep.insert(current_branch_key.to_string());
    for (name, _) in &candidates {
        if keep.len() >= cap {
            break;
        }
        keep.insert(name.clone());
    }

    let mut evicted = Vec::new();
    for (name, _) in &candidates {
        if keep.contains(name) {
            continue;
        }
        std::fs::remove_dir_all(indexes_root.join(name))?;
        evicted.push(name.clone());
    }
    if !evicted.is_empty() {
        tracing::info!(?evicted, cap, "Evicted stale branch index snapshots");
    }
    Ok(evicted)
}

/// Read the embedder fingerprint just stamped into the root's `meta.json` by
/// a build (best-effort — empty string if unreadable, which just means the
/// registry entry records an empty fingerprint rather than failing).
fn root_embedder_fingerprint(project_root: &Path) -> String {
    let meta_path = layout::container_dir(project_root).join("meta.json");
    std::fs::read_to_string(meta_path)
        .ok()
        .and_then(|s| serde_json::from_str::<crate::types::IndexMeta>(&s).ok())
        .map(|m| m.embedder_fingerprint)
        .unwrap_or_default()
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Record (upsert into the global registry) that `project_root`'s current
/// branch was just indexed, and enforce the branch-snapshot retention cap.
/// Call this right after a successful build/update from a real entry point
/// (CLI `index`/`watch`/`serve`, MCP auto-index) — never from
/// `IndexBuilder::build` itself (see that function's neighboring
/// `sync_v13_layout_best_effort` doc for why: it runs inside a huge number of
/// tests against tempdir projects, and making every one of those a hidden
/// writer of the real global registry would be a flakiness regression).
///
/// Best-effort: registry/retention failures are logged, never propagated —
/// the index build they follow already succeeded and is fully usable.
pub fn record_branch_indexed(project_root: &Path) {
    let branch = layout::current_branch_name(project_root);
    let branch_key = layout::current_branch_key(project_root);
    let head_commit = layout::resolve_git_head_commit(project_root);
    let fingerprint = root_embedder_fingerprint(project_root);

    registry::upsert_branch(
        project_root,
        &branch,
        &branch_key,
        unix_now(),
        head_commit,
        &fingerprint,
    );

    if let Err(e) = enforce_retention(project_root, &branch_key) {
        tracing::warn!("Branch snapshot retention pass failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::storage::ChunkStore;
    use crate::types::{Chunk, ChunkType, IndexMeta};
    use tempfile::TempDir;

    fn sample_index_meta() -> IndexMeta {
        IndexMeta {
            schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: std::path::PathBuf::from("/x"),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 1,
            chunk_count: 1,
            embedding_model: "CodeRankEmbed".to_string(),
            embedding_dim: 768,
            use_bm25_stemmer: true,
            dense_backend: "coderank-hnsw".to_string(),
            embedder_fingerprint: "fp".to_string(),
        }
    }

    fn write_fake_git_head(project: &Path, branch: &str) {
        let git = project.join(".git");
        std::fs::create_dir_all(git.join("refs").join("heads")).unwrap();
        std::fs::write(git.join("HEAD"), format!("ref: refs/heads/{branch}\n")).unwrap();
        std::fs::write(git.join("refs").join("heads").join(branch), "deadbeef\n").unwrap();
    }

    fn build_root_index(project: &Path, content: &str) -> u64 {
        let container = layout::container_dir(project);
        std::fs::create_dir_all(&container).unwrap();
        let store = ChunkStore::open(&container.join("chunks.db")).unwrap();
        let id = store
            .insert_chunk(
                &Chunk {
                    id: 0,
                    file_path: std::path::PathBuf::from("src/a.rs"),
                    start_line: 1,
                    end_line: 1,
                    content: content.to_string(),
                    chunk_type: ChunkType::TextWindow { window_index: 0 },
                },
                1,
                0,
            )
            .unwrap();
        std::fs::write(
            container.join("meta.json"),
            serde_json::to_string(&sample_index_meta()).unwrap(),
        )
        .unwrap();
        id
    }

    /// The core round trip: index branch A, switch to B (no snapshot yet —
    /// snapshots A, leaves root for in-place reindex), "index" B, switch back
    /// to A — must RESTORE A's original snapshot, not re-embed from scratch.
    #[test]
    fn round_trip_switch_a_to_b_and_back_restores_a() {
        let tmp = TempDir::new().unwrap();
        let project = tmp.path();

        write_fake_git_head(project, "a");
        let a_id = build_root_index(project, "fn on_a() {}");
        layout::sync_v13_layout(project, "proj").unwrap();

        // First sync ever: FirstBuild (no root sidecar existed before it).
        // Now the root sidecar records "a". Switch HEAD to "b".
        write_fake_git_head(project, "b");
        let action = detect_and_handle_branch_switch(project).unwrap();
        assert!(
            matches!(action, BranchSwitchAction::SnapshottedOutgoing { .. }),
            "{action:?}"
        );

        // Root content is untouched (still "a"'s chunk) — the caller's
        // incremental build would now re-index it in place for "b".
        let root_store =
            ChunkStore::open(&layout::container_dir(project).join("chunks.db")).unwrap();
        assert_eq!(root_store.get_chunk(a_id).unwrap().content, "fn on_a() {}");
        drop(root_store);

        // Simulate the caller's incremental build for "b": overwrite root
        // content with B's, then sync (this is what IndexBuilder::build +
        // sync_v13_layout_best_effort do together).
        std::fs::remove_file(layout::container_dir(project).join("chunks.db")).unwrap();
        build_root_index(project, "fn on_b() {}");
        layout::sync_v13_layout(project, "proj").unwrap();

        let b_key = layout::current_branch_key(project);
        assert!(
            layout::branch_index_dir(project, &b_key)
                .join("chunks.db")
                .exists()
        );

        // Switch back to "a" — a should have a snapshot from the very first
        // sync, so this must be a RESTORE, not another snapshot-outgoing.
        write_fake_git_head(project, "a");
        let action_back = detect_and_handle_branch_switch(project).unwrap();
        assert!(
            matches!(action_back, BranchSwitchAction::Restored { .. }),
            "{action_back:?}"
        );

        // Root now holds A's original content again, without ever having
        // re-run the (expensive, simulated-away-here) embed step for A.
        let restored_store =
            ChunkStore::open(&layout::container_dir(project).join("chunks.db")).unwrap();
        assert_eq!(
            restored_store.get_chunk(a_id).unwrap().content,
            "fn on_a() {}",
            "switching back to A must restore A's snapshot, not keep B's content"
        );
    }

    #[test]
    fn unchanged_branch_is_a_no_op() {
        let tmp = TempDir::new().unwrap();
        let project = tmp.path();
        write_fake_git_head(project, "main");
        build_root_index(project, "fn f() {}");
        layout::sync_v13_layout(project, "proj").unwrap();

        let action = detect_and_handle_branch_switch(project).unwrap();
        assert!(matches!(action, BranchSwitchAction::Unchanged { .. }));
    }

    #[test]
    fn brand_new_project_is_first_build() {
        let tmp = TempDir::new().unwrap();
        let action = detect_and_handle_branch_switch(tmp.path()).unwrap();
        assert_eq!(action, BranchSwitchAction::FirstBuild);
    }

    #[test]
    fn branch_switch_pending_is_cheap_and_read_only() {
        let tmp = TempDir::new().unwrap();
        let project = tmp.path();
        write_fake_git_head(project, "main");
        build_root_index(project, "fn f() {}");
        layout::sync_v13_layout(project, "proj").unwrap();

        assert!(!branch_switch_pending(project));
        write_fake_git_head(project, "feature");
        assert!(branch_switch_pending(project));
        // Read-only: root content must be untouched by the check itself.
        assert!(
            layout::read_root_branch_meta(project)
                .unwrap()
                .branch
                .eq("main"),
            "branch_switch_pending must not mutate the root sidecar"
        );
    }

    /// Retention: cap stored snapshots, always keep the current branch.
    #[test]
    fn retention_evicts_oldest_but_never_the_current_branch() {
        let tmp = TempDir::new().unwrap();
        let project = tmp.path();

        for (i, name) in ["one", "two", "three"].iter().enumerate() {
            write_fake_git_head(project, name);
            build_root_index(project, &format!("fn on_{name}() {{}}"));
            layout::sync_v13_layout(project, "proj").unwrap();
            // Ensure distinct mtimes across iterations for deterministic
            // recency ordering (filesystem mtime resolution can be coarse).
            let key = layout::branch_key_for_branch(name);
            let bump =
                std::time::SystemTime::now() + std::time::Duration::from_secs((i as u64 + 1) * 2);
            let branch_json = layout::branch_index_dir(project, &key).join("branch.json");
            let f = std::fs::File::open(&branch_json).unwrap();
            let _ = f.set_modified(bump);
        }

        let current_key = layout::branch_key_for_branch("three");
        let evicted = enforce_retention_with_cap(project, &current_key, 2).unwrap();

        // Cap is 2: current ("three") + the most recent other one ("two")
        // survive; "one" is evicted.
        assert_eq!(evicted, vec![layout::branch_key_for_branch("one")]);
        assert!(
            layout::branch_index_dir(project, &current_key)
                .join("chunks.db")
                .exists()
        );
        assert!(
            layout::branch_index_dir(project, &layout::branch_key_for_branch("two"))
                .join("chunks.db")
                .exists()
        );
        assert!(!layout::branch_index_dir(project, &layout::branch_key_for_branch("one")).exists());
    }

    #[test]
    fn retention_default_cap_is_five_and_keeps_all_when_under_cap() {
        let tmp = TempDir::new().unwrap();
        let project = tmp.path();

        for name in ["a", "b", "c"] {
            write_fake_git_head(project, name);
            build_root_index(project, &format!("fn on_{name}() {{}}"));
            layout::sync_v13_layout(project, "proj").unwrap();
        }
        let current_key = layout::branch_key_for_branch("c");
        let evicted = enforce_retention_with_cap(project, &current_key, 5).unwrap();
        assert!(evicted.is_empty(), "3 snapshots is under a cap of 5");
    }

    /// `max_branch_indexes()` itself (the env-var-reading wrapper) — the one
    /// place we DO touch the env var, in a single test, asserting only the
    /// pure parsing function's behavior rather than running it through any
    /// filesystem/registry side effects that a concurrent test could observe.
    #[test]
    fn max_branch_indexes_parses_env_override_and_rejects_garbage() {
        // Run serially within this one test — no other test reads
        // SEMANTEX_MAX_BRANCH_INDEXES, so this can't race.
        unsafe {
            std::env::set_var("SEMANTEX_MAX_BRANCH_INDEXES", "3");
        }
        assert_eq!(max_branch_indexes(), 3);
        unsafe {
            std::env::set_var("SEMANTEX_MAX_BRANCH_INDEXES", "not-a-number");
        }
        assert_eq!(max_branch_indexes(), 5);
        unsafe {
            std::env::set_var("SEMANTEX_MAX_BRANCH_INDEXES", "0");
        }
        assert_eq!(max_branch_indexes(), 5);
        unsafe {
            std::env::remove_var("SEMANTEX_MAX_BRANCH_INDEXES");
        }
        assert_eq!(max_branch_indexes(), 5);
    }
}
