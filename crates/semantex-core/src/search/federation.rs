//! Query federation (contract §B) — the interface Wave 2 wires actual
//! cross-repo/cross-branch callers through. Spine ships:
//!
//! 1. The addressing types ([`IndexTarget`], [`SearchScope`]) and
//!    [`resolve_targets`], which turns a scope into a concrete list of
//!    targets using the v2 [`Registry`](crate::index::registry::RegistryV2).
//! 2. The [`IndexSearcher`] trait — the seam a per-target search
//!    implementation plugs into (single-repo today; a multi-client daemon
//!    or cross-repo runner in Wave 2).
//! 3. Cross-target fusion ([`fuse_targets`]): per-target min-max score
//!    normalization, then Reciprocal Rank Fusion (k=60 by default) across
//!    targets, with `(project, branch)` provenance preserved on every
//!    [`FederatedHit`].
//!
//! `SearchScope::CurrentRepo` always resolves to exactly one [`IndexTarget`]
//! for the caller's own repo/branch, and [`search_single_target`] performs no
//! normalization or fusion at all — together these guarantee existing
//! single-repo search behavior is completely unchanged for callers that
//! don't opt into federation. No current search path routes through this
//! module yet; Wave 2 owns wiring real callers to it (contract §B, §F).

use crate::index::layout;
use crate::index::registry::RegistryV2;
use crate::types::SearchResult;
use anyhow::Result;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

/// The registry type `resolve_targets` reads from. Alias rather than a
/// re-export so federation.rs's public surface matches the contract's
/// `resolve_targets(&SearchScope, &Registry, &Path)` signature verbatim.
pub type Registry = RegistryV2;

/// One addressable index: a project root + the branch_key of the index to
/// query within it (contract §A branch_key derivation —
/// [`crate::index::layout::current_branch_key`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexTarget {
    pub project_root: PathBuf,
    pub branch_key: String,
}

/// A search hit with its originating target preserved, so a caller merging
/// hits from many repos/branches can still tell the user where each one
/// came from.
#[derive(Debug, Clone)]
pub struct FederatedHit {
    pub target: IndexTarget,
    pub result: SearchResult,
}

/// Which index(es) a federated query should run against. Append-only per
/// the contract — new variants may be added in later waves, but existing
/// ones never change meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchScope {
    /// Just the caller's own repo, on whatever branch is currently checked
    /// out. The only scope in use today — resolves to a single target and
    /// is what every existing (non-federated) search path is equivalent to.
    CurrentRepo,
    /// Every project + branch the registry knows about.
    All,
    /// A caller-specified subset, matched against each registered project's
    /// `display_name` or path.
    Named(Vec<String>),
}

/// Default Reciprocal Rank Fusion constant for cross-target fusion
/// (contract §B: "RRF k=60 across targets"). Mirrors
/// `SemantexConfig::rrf_k`'s existing default for the intra-target
/// dense/sparse/exact fusion (`config.rs`), so both fusion stages agree.
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Resolve `scope` into the concrete list of targets to query, using
/// `registry` for cross-project lookups and `cwd` to identify "the caller's
/// own repo" for [`SearchScope::CurrentRepo`].
///
/// `CurrentRepo` never consults the registry at all — it always resolves to
/// exactly one target for `cwd`'s current branch, so a caller that never
/// touches federation continues to behave exactly as before v13.
pub fn resolve_targets(scope: &SearchScope, registry: &Registry, cwd: &Path) -> Vec<IndexTarget> {
    match scope {
        SearchScope::CurrentRepo => vec![IndexTarget {
            project_root: cwd.to_path_buf(),
            branch_key: layout::current_branch_key(cwd),
        }],
        SearchScope::All => registry.projects.iter().flat_map(project_targets).collect(),
        SearchScope::Named(names) => registry
            .projects
            .iter()
            .filter(|p| {
                names.iter().any(|n| {
                    n == &p.display_name
                        || n == &p.path.to_string_lossy()
                        || p.path
                            .file_name()
                            .is_some_and(|f| f.to_string_lossy() == *n)
                })
            })
            .flat_map(project_targets)
            .collect(),
    }
}

/// One target per branch the registry has recorded for this project; if
/// none are recorded yet (registered but never fully indexed under v13),
/// fall back to resolving the branch currently checked out on disk.
fn project_targets(project: &crate::index::registry::ProjectEntry) -> Vec<IndexTarget> {
    if project.branches.is_empty() {
        return vec![IndexTarget {
            branch_key: layout::current_branch_key(&project.path),
            project_root: project.path.clone(),
        }];
    }
    project
        .branches
        .iter()
        .map(|b| IndexTarget {
            project_root: project.path.clone(),
            branch_key: b.branch_key.clone(),
        })
        .collect()
}

/// The seam a per-target search execution plugs into. A single-repo runner
/// implements this trivially (query its one local index); a Wave 2
/// multi-client daemon or cross-repo runner implements it by dispatching to
/// whichever process/searcher owns that `(project_root, branch_key)`.
pub trait IndexSearcher {
    /// Run `query` against `target`, returning up to `limit` hits. Errors
    /// (e.g. the target has no index yet) are the caller's to handle —
    /// federation callers typically skip a failing target rather than fail
    /// the whole federated query.
    fn search_target(
        &self,
        target: &IndexTarget,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>>;
}

/// Wrap a single target's hits as [`FederatedHit`]s with NO normalization or
/// re-scoring — the single-target case current (pre-federation) search
/// paths are equivalent to. Scores and ordering are passed through exactly
/// as the underlying search produced them.
pub fn search_single_target(target: &IndexTarget, hits: Vec<SearchResult>) -> Vec<FederatedHit> {
    hits.into_iter()
        .map(|result| FederatedHit {
            target: target.clone(),
            result,
        })
        .collect()
}

/// Fuse hits from multiple targets into one globally-ranked list, using the
/// default [`DEFAULT_RRF_K`]. See [`fuse_targets_with_k`] for the algorithm.
pub fn fuse_targets(per_target_hits: Vec<(IndexTarget, Vec<SearchResult>)>) -> Vec<FederatedHit> {
    fuse_targets_with_k(per_target_hits, DEFAULT_RRF_K)
}

/// Cross-target fusion (contract §B): within each target, hits are
/// min-max-normalized to `[0.0, 1.0]` and re-ranked by that normalized
/// score (this is what makes the subsequent RRF pass compare *ranks*
/// established on a common scale, rather than raw scores that different
/// backends may put on very different numeric ranges). Then every hit's
/// final score is `1 / (k + rank_within_its_target)` (1-indexed), i.e.
/// standard Reciprocal Rank Fusion applied across targets — each hit
/// appears in exactly one target's list, so this reduces to re-scoring by
/// rank rather than merging duplicate keys, but it is the same formula used
/// for the existing intra-target dense/sparse/exact fusion
/// (`search/triple_fusion.rs`), just applied one level up. Provenance
/// (`target.project_root`, `target.branch_key`) is preserved on every
/// returned hit. The final list is sorted by fused score, descending.
pub fn fuse_targets_with_k(
    per_target_hits: Vec<(IndexTarget, Vec<SearchResult>)>,
    k: f32,
) -> Vec<FederatedHit> {
    let mut fused = Vec::new();

    for (target, hits) in per_target_hits {
        if hits.is_empty() {
            continue;
        }
        let min = hits.iter().map(|h| h.score).fold(f32::INFINITY, f32::min);
        let max = hits
            .iter()
            .map(|h| h.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(f32::EPSILON);

        let mut ranked: Vec<(usize, f32)> = hits
            .iter()
            .enumerate()
            .map(|(i, h)| (i, (h.score - min) / range))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        for (rank, (orig_idx, _normalized)) in ranked.into_iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank as f32 + 1.0));
            let mut result = hits[orig_idx].clone();
            result.score = rrf_score;
            fused.push(FederatedHit {
                target: target.clone(),
                result,
            });
        }
    }

    fused.sort_by(|a, b| {
        b.result
            .score
            .partial_cmp(&a.result.score)
            .unwrap_or(Ordering::Equal)
    });
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::registry::{BranchEntry, ProjectEntry};
    use crate::types::{Chunk, ChunkType, Confidence, SearchSource};

    fn make_result(id: u64, score: f32) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                id,
                file_path: PathBuf::from(format!("f{id}.rs")),
                start_line: 1,
                end_line: 2,
                content: format!("chunk {id}"),
                chunk_type: ChunkType::TextWindow { window_index: 0 },
            },
            score,
            source: SearchSource::Dense,
            score_dense: score,
            score_sparse: 0.0,
            score_exact: 0.0,
            confidence: Confidence::Inferred,
            confidence_score: 0.0,
        }
    }

    #[test]
    fn current_repo_scope_resolves_to_single_target_ignoring_registry() {
        let registry = Registry::default();
        let cwd = PathBuf::from("/some/project");
        let targets = resolve_targets(&SearchScope::CurrentRepo, &registry, &cwd);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].project_root, cwd);
    }

    #[test]
    fn all_scope_resolves_one_target_per_registered_branch() {
        let mut registry = Registry::default();
        registry.projects.push(ProjectEntry {
            path: PathBuf::from("/repo/a"),
            project_id: "a".into(),
            display_name: "a".into(),
            branches: vec![
                BranchEntry {
                    branch: "main".into(),
                    branch_key: "main-aaaa1111".into(),
                    last_indexed_ts: 1,
                    head_commit: None,
                },
                BranchEntry {
                    branch: "dev".into(),
                    branch_key: "dev-bbbb2222".into(),
                    last_indexed_ts: 2,
                    head_commit: None,
                },
            ],
            embedder_fingerprint: "fp".into(),
        });

        let targets = resolve_targets(&SearchScope::All, &registry, Path::new("/anywhere"));
        assert_eq!(targets.len(), 2);
        assert!(
            targets
                .iter()
                .all(|t| t.project_root == Path::new("/repo/a"))
        );
    }

    #[test]
    fn all_scope_falls_back_to_current_branch_when_none_recorded() {
        let mut registry = Registry::default();
        registry.projects.push(ProjectEntry {
            path: PathBuf::from("/repo/no-branches"),
            project_id: "x".into(),
            display_name: "x".into(),
            branches: vec![],
            embedder_fingerprint: String::new(),
        });
        let targets = resolve_targets(&SearchScope::All, &registry, Path::new("/anywhere"));
        assert_eq!(targets.len(), 1);
        // Non-git dir → "default" branch_key.
        assert_eq!(targets[0].branch_key, "default");
    }

    #[test]
    fn named_scope_matches_display_name_and_path_suffix() {
        let mut registry = Registry::default();
        registry.projects.push(ProjectEntry {
            path: PathBuf::from("/home/dev/my-project"),
            project_id: "p".into(),
            display_name: "my-project".into(),
            branches: vec![],
            embedder_fingerprint: String::new(),
        });
        registry.projects.push(ProjectEntry {
            path: PathBuf::from("/home/dev/other-project"),
            project_id: "q".into(),
            display_name: "other-project".into(),
            branches: vec![],
            embedder_fingerprint: String::new(),
        });

        let targets = resolve_targets(
            &SearchScope::Named(vec!["my-project".to_string()]),
            &registry,
            Path::new("/anywhere"),
        );
        assert_eq!(targets.len(), 1);
        assert_eq!(
            targets[0].project_root,
            PathBuf::from("/home/dev/my-project")
        );

        // No match → empty, not an error.
        let none = resolve_targets(
            &SearchScope::Named(vec!["nonexistent".to_string()]),
            &registry,
            Path::new("/anywhere"),
        );
        assert!(none.is_empty());
    }

    #[test]
    fn single_target_search_passes_scores_through_unchanged() {
        let target = IndexTarget {
            project_root: PathBuf::from("/repo"),
            branch_key: "main-abc12345".to_string(),
        };
        let hits = vec![make_result(1, 0.9), make_result(2, 0.4)];
        let federated = search_single_target(&target, hits.clone());
        assert_eq!(federated.len(), 2);
        assert_eq!(federated[0].result.score, 0.9);
        assert_eq!(federated[1].result.score, 0.4);
        assert!(federated.iter().all(|h| h.target == target));
    }

    #[test]
    fn fuse_targets_preserves_provenance_and_ranks_by_rrf() {
        let target_a = IndexTarget {
            project_root: PathBuf::from("/repo/a"),
            branch_key: "main-aaaa1111".to_string(),
        };
        let target_b = IndexTarget {
            project_root: PathBuf::from("/repo/b"),
            branch_key: "main-bbbb2222".to_string(),
        };

        // target_a's top hit has a much larger raw score than target_b's,
        // but RRF is rank-based: target_b's #1 must still beat target_a's #2.
        let hits_a = vec![make_result(10, 100.0), make_result(11, 1.0)];
        let hits_b = vec![make_result(20, 0.9)];

        let fused = fuse_targets(vec![(target_a.clone(), hits_a), (target_b.clone(), hits_b)]);

        assert_eq!(fused.len(), 3);
        // Rank-1 hits from both targets should score 1/(60+1); rank-2 lower.
        let rank1_score = 1.0 / (DEFAULT_RRF_K + 1.0);
        let rank2_score = 1.0 / (DEFAULT_RRF_K + 2.0);
        assert!((fused[0].result.score - rank1_score).abs() < 1e-6);
        assert!((fused[1].result.score - rank1_score).abs() < 1e-6);
        assert!((fused[2].result.score - rank2_score).abs() < 1e-6);

        // Provenance: chunk 20 (target_b's only hit) must carry target_b.
        let hit20 = fused.iter().find(|h| h.result.chunk.id == 20).unwrap();
        assert_eq!(hit20.target, target_b);
        let hit10 = fused.iter().find(|h| h.result.chunk.id == 10).unwrap();
        assert_eq!(hit10.target, target_a);

        // Output must be sorted descending by fused score.
        for w in fused.windows(2) {
            assert!(w[0].result.score >= w[1].result.score);
        }
    }

    #[test]
    fn fuse_targets_skips_empty_target_lists() {
        let target = IndexTarget {
            project_root: PathBuf::from("/repo"),
            branch_key: "main-abc12345".to_string(),
        };
        let fused = fuse_targets(vec![(target, vec![])]);
        assert!(fused.is_empty());
    }
}
