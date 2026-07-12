//! Query federation (contract §B) — the interface Wave 2 wires actual
//! cross-repo/cross-branch callers through. Spine ships:
//!
//! 1. The addressing types ([`IndexTarget`], [`SearchScope`]) and
//!    [`resolve_targets`], which turns a scope into a concrete list of
//!    targets using the v2 [`Registry`](crate::index::registry::RegistryV2).
//! 2. The [`IndexSearcher`] trait — the seam a per-target search
//!    implementation plugs into (single-repo today; a multi-client daemon
//!    or cross-repo runner in Wave 2).
//! 3. Cross-target fusion ([`fuse_targets`]): rank-based Reciprocal Rank
//!    Fusion (k=60 by default) across targets — rank-based fusion is
//!    inherently immune to targets whose backends put raw scores on
//!    different numeric ranges (see [`fuse_targets_with_k`] for why no
//!    score normalization step is needed) — with `(project, branch)`
//!    provenance preserved on every [`FederatedHit`].
//!
//! `SearchScope::CurrentRepo` always resolves to exactly one [`IndexTarget`]
//! for the caller's own repo/branch, and [`search_single_target`] performs no
//! normalization or fusion at all — together these guarantee existing
//! single-repo search behavior is completely unchanged for callers that
//! don't opt into federation. No current search path routes through this
//! module yet; Wave 2 owns wiring real callers to it (contract §B, §F).

use crate::index::layout;
use crate::index::registry::RegistryV2;
use crate::index::state::{self, IndexState};
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

/// Parse a caller-supplied scope *string* into a [`SearchScope`] — the ONE
/// grammar the CLI's `--scope` flag and the MCP tools' string-typed `scope`
/// argument share, so `scope: "frontend"` means the same thing at every
/// entry point: `"repo"`/empty → `CurrentRepo`; `"all"` → `All`; anything
/// else is split on commas into project display names/paths (`Named`).
///
/// Unknown strings are deliberately treated as `Named` rather than silently
/// mapped to `CurrentRepo`: a typo ("al", "All", "frontnd") then resolves to
/// zero targets and is surfaced per-name by [`run_federated_search`] in
/// `skipped` — diagnosable — instead of quietly searching the wrong scope.
pub fn parse_scope_str(raw: &str) -> SearchScope {
    match raw {
        "repo" | "" => SearchScope::CurrentRepo,
        "all" => SearchScope::All,
        other => {
            let names: Vec<String> = other
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            if names.is_empty() {
                SearchScope::CurrentRepo
            } else {
                SearchScope::Named(names)
            }
        }
    }
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
            .filter(|p| names.iter().any(|n| project_matches_name(p, n)))
            .flat_map(project_targets)
            .collect(),
    }
}

/// The `Named`-scope matching rule: `name` matches a registered project when
/// it equals the project's `display_name`, its full path, or the path's
/// final component. Shared by [`resolve_targets`] and
/// [`run_federated_search`]'s unmatched-name reporting so the two can never
/// disagree about what "matched" means.
fn project_matches_name(project: &crate::index::registry::ProjectEntry, name: &str) -> bool {
    name == project.display_name
        || name == project.path.to_string_lossy()
        || project
            .path
            .file_name()
            .is_some_and(|f| f.to_string_lossy() == name)
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

/// Cross-target fusion (contract §B): purely **rank-based** Reciprocal Rank
/// Fusion. Within each target, hits are ordered by their raw score
/// (descending), then every hit's final score is
/// `1 / (k + rank_within_its_target)` (1-indexed) — the same formula used
/// for the existing intra-target dense/sparse/exact fusion
/// (`search/triple_fusion.rs`), just applied one level up. Each hit appears
/// in exactly one target's list, so this reduces to re-scoring by rank
/// rather than merging duplicate keys.
///
/// Because RRF consumes only *ranks*, per-target score normalization (e.g.
/// min-max) is deliberately omitted: any monotonic rescaling of one
/// target's scores is a rank-order no-op, so it could never change the
/// output. This is exactly what makes RRF the right cross-target combiner —
/// different targets' backends may put raw scores on wildly different
/// numeric ranges, and rank-based fusion is immune to that by construction.
///
/// Provenance (`target.project_root`, `target.branch_key`) is preserved on
/// every returned hit. The final list is sorted by fused score, descending
/// (ties between same-rank hits of different targets are input-order
/// stable).
pub fn fuse_targets_with_k(
    per_target_hits: Vec<(IndexTarget, Vec<SearchResult>)>,
    k: f32,
) -> Vec<FederatedHit> {
    let mut fused = Vec::new();

    for (target, hits) in per_target_hits {
        let mut ranked: Vec<usize> = (0..hits.len()).collect();
        ranked.sort_by(|&a, &b| {
            hits[b]
                .score
                .partial_cmp(&hits[a].score)
                .unwrap_or(Ordering::Equal)
        });

        for (rank, orig_idx) in ranked.into_iter().enumerate() {
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

// ── Wave 2: cross-repo callers ──────────────────────────────────────────────
//
// Everything below this point is the Wave 2 wiring the module doc promised:
// a concrete on-disk `IndexSearcher`, target readiness/skip handling, and the
// orchestration function (`run_federated_search`) that `semantex-mcp` and
// `semantex-cli` call for any `SearchScope` other than `CurrentRepo`.
//
// `SearchScope::CurrentRepo` callers MUST keep calling their existing
// single-repo code path directly and never route through
// `run_federated_search` — that is what guarantees byte-identical output for
// every pre-v13 consumer (contract Wave 2 §3). Federation is opt-in.

/// Resolve the on-disk index directory to open for `target`.
///
/// Wave 2 federates ACROSS REPOS, not across a repo's branch snapshots: this
/// always resolves to the container root (`<project_root>/.semantex/`) — the
/// live, currently-checked-out-branch index (see `index/layout.rs`'s module
/// doc: the root stays authoritative for "the currently-open branch" after
/// every build). `target.branch_key` is carried on `IndexTarget`/
/// `FederatedHit` for provenance/display only; it is NOT used to select a
/// `indexes/<branch_key>/` snapshot here. Multi-branch federation (reading a
/// branch OTHER than the one currently checked out on disk) is a distinct
/// Wave 2 preview item (contract §F) this batch does not implement.
pub fn target_index_dir(target: &IndexTarget) -> PathBuf {
    layout::container_dir(&target.project_root)
}

/// Check whether `target`'s index is ready to be queried, without ever
/// triggering a build. Federation degrades gracefully: a target whose index
/// is missing, mid-build, or stale is skipped (with a reason) rather than
/// failing the whole federated query.
pub fn target_readiness(target: &IndexTarget) -> Result<(), String> {
    match state::detect(&target.project_root) {
        IndexState::Ready => Ok(()),
        IndexState::NotIndexed => Err("not indexed".to_string()),
        IndexState::Building => Err("index build in progress".to_string()),
        IndexState::Stale => Err("index schema is stale — needs rebuild".to_string()),
    }
}

/// A target that a federated query skipped rather than erroring on — either
/// its index wasn't ready ([`target_readiness`]) or [`IndexSearcher::search_target`]
/// itself failed (e.g. a corrupt on-disk index). Surfaced in response
/// metadata so callers can tell the user which repos were left out.
#[derive(Debug, Clone)]
pub struct SkippedTarget {
    pub target: IndexTarget,
    pub reason: String,
}

/// Result of [`run_federated_search`]: the fused hits plus any targets that
/// were skipped along the way, and an optional caller-facing `note` for the
/// cases where scope resolution itself produced nothing queryable.
#[derive(Debug, Clone, Default)]
pub struct FederatedSearchOutcome {
    pub hits: Vec<FederatedHit>,
    pub skipped: Vec<SkippedTarget>,
    /// Set when the scope resolved to zero targets for a reason the user
    /// should see (today: `All` against an empty registry). Callers surface
    /// this instead of returning a bare, indistinguishable empty success.
    pub note: Option<String>,
}

/// Resolve `scope` to targets, query every ready one through `searcher`, and
/// fuse the results with [`fuse_targets`] (RRF k=60). Targets that aren't
/// ready ([`target_readiness`]) or whose search call errors are recorded in
/// `skipped` instead of aborting the whole query — one bad repo in `All`
/// scope must never prevent results from the rest.
///
/// `limit` is applied twice: each target is queried for up to `limit` hits,
/// and the final FUSED list is also truncated to `limit` — so callers get at
/// most `limit` results total, not `limit × N_targets`.
///
/// Diagnosability of "nothing happened": a `Named` name that matched no
/// registered project is reported in `skipped` (reason `"no registered
/// project matched '<name>'"`, with the name itself as the synthetic
/// target's `project_root`), and `All` against an empty registry sets
/// `note` — an empty result is never silently ambiguous with "searched
/// everything, found nothing".
pub fn run_federated_search(
    scope: &SearchScope,
    registry: &Registry,
    cwd: &Path,
    query: &str,
    limit: usize,
    searcher: &dyn IndexSearcher,
) -> FederatedSearchOutcome {
    let mut targets = resolve_targets(scope, registry, cwd);

    // A project with N recorded branches yields N targets, but every
    // branch_key resolves to the same live container root today (see
    // target_index_dir) — dedupe by resolved index dir (keeping the first)
    // so such a project is searched once, not N times, which would both
    // duplicate its hits and give them N× the RRF weight.
    {
        let mut seen = std::collections::HashSet::new();
        targets.retain(|t| seen.insert(target_index_dir(t)));
    }

    let mut skipped = Vec::new();

    if let SearchScope::Named(names) = scope {
        for name in names {
            if !registry
                .projects
                .iter()
                .any(|p| project_matches_name(p, name))
            {
                skipped.push(SkippedTarget {
                    target: IndexTarget {
                        project_root: PathBuf::from(name),
                        branch_key: String::new(),
                    },
                    reason: format!("no registered project matched '{name}'"),
                });
            }
        }
    }

    let note = if matches!(scope, SearchScope::All) && registry.projects.is_empty() {
        Some(
            "registry has no projects — nothing to federate (build an index in a repo to register it)"
                .to_string(),
        )
    } else {
        None
    };

    let mut per_target_hits: Vec<(IndexTarget, Vec<SearchResult>)> =
        Vec::with_capacity(targets.len());

    for target in targets {
        if let Err(reason) = target_readiness(&target) {
            skipped.push(SkippedTarget { target, reason });
            continue;
        }
        match searcher.search_target(&target, query, limit) {
            Ok(hits) => per_target_hits.push((target, hits)),
            Err(e) => skipped.push(SkippedTarget {
                target,
                reason: e.to_string(),
            }),
        }
    }

    let mut hits = fuse_targets(per_target_hits);
    hits.truncate(limit);
    FederatedSearchOutcome {
        hits,
        skipped,
        note,
    }
}

/// Look up the `display_name` the registry has recorded for the project at
/// `project_root`, falling back to the root's own directory name when the
/// project isn't registered (e.g. `SearchScope::CurrentRepo`'s target, which
/// never touches the registry — see [`resolve_targets`]).
pub fn project_display_name(registry: &Registry, project_root: &Path) -> String {
    registry
        .projects
        .iter()
        .find(|p| p.path == project_root)
        .map(|p| p.display_name.clone())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| {
            project_root.file_name().map_or_else(
                || project_root.to_string_lossy().to_string(),
                |n| n.to_string_lossy().to_string(),
            )
        })
}

/// Builds the per-target `SearchQuery` from the raw query text and the
/// requested per-target limit — lets [`SequentialIndexSearcher`] callers opt
/// into grep-mode/no-rerank/etc. the same way the single-repo path does,
/// without widening `IndexSearcher::search_target`'s signature. A boxed
/// closure (rather than a bare `fn`) so callers can capture CLI flags
/// (`--rerank`, `--grep-mode`, ...) instead of needing one free function per
/// flag combination. Named as a `type` alias (rather than inlined) per
/// clippy::type_complexity.
pub type QueryBuilder<'a> = Box<dyn Fn(&str, usize) -> crate::search::SearchQuery + 'a>;

/// A simple, no-cache [`IndexSearcher`]: opens `target`'s index, runs one
/// hybrid search, and drops the searcher before returning. Peak memory is
/// bounded to one open index at a time — the right default for a short-lived
/// CLI process with no long-lived cache to amortize the open cost across
/// calls. Long-lived hosts (the MCP server, a future multi-client daemon)
/// should instead implement `IndexSearcher` on top of their own searcher
/// cache (see `semantex-mcp`'s LRU `McpServer::get_searcher`) so warm repos
/// stay warm across federated calls instead of reopening every time.
pub struct SequentialIndexSearcher<'a> {
    pub config: &'a crate::config::SemantexConfig,
    pub build_query: QueryBuilder<'a>,
}

impl IndexSearcher for SequentialIndexSearcher<'_> {
    fn search_target(
        &self,
        target: &IndexTarget,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let index_dir = target_index_dir(target);
        let searcher = crate::search::hybrid::HybridSearcher::open(&index_dir, self.config)?;
        let sq = (self.build_query)(query, limit);
        Ok(searcher.search(&sq)?.results)
    }
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
            is_worktree: false,
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
            is_worktree: false,
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
            is_worktree: false,
        });
        registry.projects.push(ProjectEntry {
            path: PathBuf::from("/home/dev/other-project"),
            project_id: "q".into(),
            display_name: "other-project".into(),
            branches: vec![],
            embedder_fingerprint: String::new(),
            is_worktree: false,
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

    // ── Wave 2 additions ────────────────────────────────────────────────────

    fn write_ready_index(project_root: &Path) {
        let semantex_dir = project_root.join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        let meta = serde_json::json!({
            "schema_version": crate::types::IndexMeta::CURRENT_SCHEMA_VERSION,
            "project_path": project_root,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "file_count": 1,
            "chunk_count": 1,
            "embedding_model": "test",
            "embedding_dim": 8,
            "use_bm25_stemmer": true,
            "dense_backend": "coderank-hnsw",
            "embedder_fingerprint": "test",
        });
        std::fs::write(semantex_dir.join("meta.json"), meta.to_string()).unwrap();
    }

    #[test]
    fn target_index_dir_is_the_project_container_root() {
        let target = IndexTarget {
            project_root: PathBuf::from("/repo/a"),
            branch_key: "some-other-branch-11112222".to_string(),
        };
        // Deliberately NOT indexes/<branch_key>/ — see target_index_dir's doc.
        assert_eq!(
            target_index_dir(&target),
            PathBuf::from("/repo/a/.semantex")
        );
    }

    #[test]
    fn target_readiness_reports_not_indexed_for_a_fresh_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let target = IndexTarget {
            project_root: tmp.path().to_path_buf(),
            branch_key: "default".to_string(),
        };
        assert_eq!(target_readiness(&target), Err("not indexed".to_string()));
    }

    #[test]
    fn target_readiness_is_ok_for_a_ready_index() {
        let tmp = tempfile::TempDir::new().unwrap();
        write_ready_index(tmp.path());
        let target = IndexTarget {
            project_root: tmp.path().to_path_buf(),
            branch_key: "default".to_string(),
        };
        assert_eq!(target_readiness(&target), Ok(()));
    }

    /// Stub [`IndexSearcher`] for orchestration tests: returns canned hits per
    /// project_root, or errors for paths not in its map — lets tests exercise
    /// `run_federated_search`'s skip/fuse behaviour without touching disk.
    struct StubSearcher {
        by_project: std::collections::HashMap<PathBuf, Vec<SearchResult>>,
    }

    impl IndexSearcher for StubSearcher {
        fn search_target(
            &self,
            target: &IndexTarget,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SearchResult>> {
            self.by_project
                .get(&target.project_root)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("stub: no hits configured for this target"))
        }
    }

    #[test]
    fn run_federated_search_skips_not_ready_targets_and_reports_them() {
        let tmp_ready = tempfile::TempDir::new().unwrap();
        write_ready_index(tmp_ready.path());
        let tmp_not_indexed = tempfile::TempDir::new().unwrap();

        let mut registry = Registry::default();
        registry.projects.push(ProjectEntry {
            path: tmp_ready.path().to_path_buf(),
            project_id: "ready".into(),
            display_name: "ready-project".into(),
            branches: vec![],
            embedder_fingerprint: String::new(),
            is_worktree: false,
        });
        registry.projects.push(ProjectEntry {
            path: tmp_not_indexed.path().to_path_buf(),
            project_id: "notready".into(),
            display_name: "not-ready-project".into(),
            branches: vec![],
            embedder_fingerprint: String::new(),
            is_worktree: false,
        });

        let mut by_project = std::collections::HashMap::new();
        by_project.insert(tmp_ready.path().to_path_buf(), vec![make_result(1, 0.5)]);
        let searcher = StubSearcher { by_project };

        let outcome = run_federated_search(
            &SearchScope::All,
            &registry,
            Path::new("/anywhere"),
            "query",
            10,
            &searcher,
        );

        assert_eq!(outcome.hits.len(), 1);
        assert_eq!(outcome.hits[0].result.chunk.id, 1);
        assert_eq!(outcome.skipped.len(), 1);
        assert_eq!(
            outcome.skipped[0].target.project_root,
            tmp_not_indexed.path()
        );
        assert_eq!(outcome.skipped[0].reason, "not indexed");
    }

    #[test]
    fn run_federated_search_reports_searcher_errors_as_skipped_not_a_hard_failure() {
        let tmp = tempfile::TempDir::new().unwrap();
        write_ready_index(tmp.path());

        let mut registry = Registry::default();
        registry.projects.push(ProjectEntry {
            path: tmp.path().to_path_buf(),
            project_id: "p".into(),
            display_name: "p".into(),
            branches: vec![],
            embedder_fingerprint: String::new(),
            is_worktree: false,
        });

        // No hits configured for this project → StubSearcher errors.
        let searcher = StubSearcher {
            by_project: std::collections::HashMap::new(),
        };

        let outcome = run_federated_search(
            &SearchScope::All,
            &registry,
            Path::new("/anywhere"),
            "query",
            10,
            &searcher,
        );

        assert!(outcome.hits.is_empty());
        assert_eq!(outcome.skipped.len(), 1);
        assert_eq!(outcome.skipped[0].target.project_root, tmp.path());
    }

    /// Fix 1 regression guard: a project with N recorded branches must be
    /// searched ONCE per federated query, because every branch_key resolves
    /// to the same live container root — N searches would return N duplicate
    /// copies of each hit and give that repo N× the RRF weight.
    #[test]
    fn run_federated_search_dedupes_multi_branch_projects_by_index_dir() {
        use std::sync::Mutex;

        struct CountingSearcher {
            calls: Mutex<usize>,
            hits: Vec<SearchResult>,
        }
        impl IndexSearcher for CountingSearcher {
            fn search_target(
                &self,
                _target: &IndexTarget,
                _query: &str,
                _limit: usize,
            ) -> Result<Vec<SearchResult>> {
                *self.calls.lock().unwrap() += 1;
                Ok(self.hits.clone())
            }
        }

        let tmp = tempfile::TempDir::new().unwrap();
        write_ready_index(tmp.path());

        let mut registry = Registry::default();
        registry.projects.push(ProjectEntry {
            path: tmp.path().to_path_buf(),
            project_id: "p".into(),
            display_name: "p".into(),
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
            is_worktree: false,
        });
        // Sanity: the registry really does resolve to 2 targets pre-dedupe.
        assert_eq!(
            resolve_targets(&SearchScope::All, &registry, Path::new("/anywhere")).len(),
            2
        );

        let searcher = CountingSearcher {
            calls: Mutex::new(0),
            hits: vec![make_result(1, 0.9), make_result(2, 0.5)],
        };

        let outcome = run_federated_search(
            &SearchScope::All,
            &registry,
            Path::new("/anywhere"),
            "query",
            10,
            &searcher,
        );

        assert_eq!(
            *searcher.calls.lock().unwrap(),
            1,
            "a multi-branch project must be searched exactly once"
        );
        assert_eq!(outcome.hits.len(), 2, "no duplicated hits");
        let ids: Vec<u64> = outcome.hits.iter().map(|h| h.result.chunk.id).collect();
        assert_eq!(ids, vec![1, 2]);
        assert!(outcome.skipped.is_empty());
    }

    /// Fix 2 regression guard: `limit` bounds the FUSED list, not just each
    /// target — N targets × limit hits each must still return `limit` total.
    #[test]
    fn run_federated_search_truncates_fused_list_to_limit() {
        let tmp_a = tempfile::TempDir::new().unwrap();
        let tmp_b = tempfile::TempDir::new().unwrap();
        write_ready_index(tmp_a.path());
        write_ready_index(tmp_b.path());

        let mut registry = Registry::default();
        for (id, tmp) in [("a", &tmp_a), ("b", &tmp_b)] {
            registry.projects.push(ProjectEntry {
                path: tmp.path().to_path_buf(),
                project_id: id.into(),
                display_name: id.into(),
                branches: vec![],
                embedder_fingerprint: String::new(),
            is_worktree: false,
            });
        }

        let mut by_project = std::collections::HashMap::new();
        by_project.insert(
            tmp_a.path().to_path_buf(),
            (0..5)
                .map(|i| make_result(i, 1.0 - i as f32 * 0.1))
                .collect(),
        );
        by_project.insert(
            tmp_b.path().to_path_buf(),
            (10..15)
                .map(|i| make_result(i, 1.0 - i as f32 * 0.01))
                .collect(),
        );
        let searcher = StubSearcher { by_project };

        let outcome = run_federated_search(
            &SearchScope::All,
            &registry,
            Path::new("/anywhere"),
            "query",
            3,
            &searcher,
        );

        assert_eq!(
            outcome.hits.len(),
            3,
            "fused list must be truncated to `limit`, not limit × N_targets"
        );
        // And it keeps the TOP fused hits: both targets' rank-1 hits survive.
        let ids: std::collections::HashSet<u64> =
            outcome.hits.iter().map(|h| h.result.chunk.id).collect();
        assert!(ids.contains(&0), "target a's rank-1 hit kept");
        assert!(ids.contains(&10), "target b's rank-1 hit kept");
    }

    /// Fix 4: the shared string grammar both the CLI flag and the MCP
    /// string-typed `scope` argument parse with.
    #[test]
    fn parse_scope_str_grammar() {
        assert_eq!(parse_scope_str("repo"), SearchScope::CurrentRepo);
        assert_eq!(parse_scope_str(""), SearchScope::CurrentRepo);
        assert_eq!(parse_scope_str("all"), SearchScope::All);
        assert_eq!(
            parse_scope_str("frontend"),
            SearchScope::Named(vec!["frontend".to_string()])
        );
        assert_eq!(
            parse_scope_str("frontend, backend"),
            SearchScope::Named(vec!["frontend".to_string(), "backend".to_string()])
        );
        // Near-misses of the enum keywords are Named (→ diagnosable via
        // skipped), NOT silently CurrentRepo and NOT silently All.
        assert_eq!(
            parse_scope_str("All"),
            SearchScope::Named(vec!["All".to_string()])
        );
        assert_eq!(
            parse_scope_str("al"),
            SearchScope::Named(vec!["al".to_string()])
        );
        // Degenerate comma-only input has no names → safe default.
        assert_eq!(parse_scope_str(",,"), SearchScope::CurrentRepo);
    }

    /// Fix 5: empty resolutions must be diagnosable, not empty successes.
    #[test]
    fn run_federated_search_reports_unmatched_named_scope_names() {
        let tmp = tempfile::TempDir::new().unwrap();
        write_ready_index(tmp.path());
        let mut registry = Registry::default();
        registry.projects.push(ProjectEntry {
            path: tmp.path().to_path_buf(),
            project_id: "real".into(),
            display_name: "real-project".into(),
            branches: vec![],
            embedder_fingerprint: String::new(),
            is_worktree: false,
        });

        let mut by_project = std::collections::HashMap::new();
        by_project.insert(tmp.path().to_path_buf(), vec![make_result(1, 0.5)]);
        let searcher = StubSearcher { by_project };

        let outcome = run_federated_search(
            &SearchScope::Named(vec!["real-project".to_string(), "typo-projct".to_string()]),
            &registry,
            Path::new("/anywhere"),
            "query",
            10,
            &searcher,
        );

        // The matched project still returns hits...
        assert_eq!(outcome.hits.len(), 1);
        // ...and the unmatched name is called out by name in skipped.
        assert_eq!(outcome.skipped.len(), 1);
        assert_eq!(
            outcome.skipped[0].reason,
            "no registered project matched 'typo-projct'"
        );
        assert_eq!(
            outcome.skipped[0].target.project_root,
            PathBuf::from("typo-projct")
        );
        assert!(outcome.note.is_none());
    }

    #[test]
    fn run_federated_search_all_scope_empty_registry_sets_note() {
        let registry = Registry::default();
        let searcher = StubSearcher {
            by_project: std::collections::HashMap::new(),
        };
        let outcome = run_federated_search(
            &SearchScope::All,
            &registry,
            Path::new("/anywhere"),
            "query",
            10,
            &searcher,
        );
        assert!(outcome.hits.is_empty());
        assert!(outcome.skipped.is_empty());
        let note = outcome.note.expect("All + empty registry must set note");
        assert!(note.contains("no projects"), "actionable note: {note}");
    }

    #[test]
    fn project_display_name_prefers_registry_entry_then_falls_back_to_dir_name() {
        let mut registry = Registry::default();
        registry.projects.push(ProjectEntry {
            path: PathBuf::from("/home/dev/my-project"),
            project_id: "p".into(),
            display_name: "My Project".into(),
            branches: vec![],
            embedder_fingerprint: String::new(),
            is_worktree: false,
        });

        assert_eq!(
            project_display_name(&registry, Path::new("/home/dev/my-project")),
            "My Project"
        );
        // Unregistered path (e.g. CurrentRepo's own target) → dir name fallback.
        assert_eq!(
            project_display_name(&registry, Path::new("/home/dev/unregistered")),
            "unregistered"
        );
    }
}
