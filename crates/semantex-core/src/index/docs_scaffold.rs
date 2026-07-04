//! Deterministic documentation-scaffold generator (v13 Wave 2).
//!
//! Maintainer decision (Wave 0 contract §F): semantex ships **zero LLM
//! wiring**. This module never writes prose — it produces structurally
//! complete JSON scaffolds (symbol inventories, call-graph edges, import
//! edges, existing doc-comment text, file:line provenance) that a *host*
//! agent (the user's own LLM — Claude Code, etc.) turns into maintained
//! markdown docs under `semantex_docs/` in the user's repo. See the
//! `semantex-docs` Skill (`plugin/skills/semantex-docs/SKILL.md`) for the
//! writing workflow this data feeds.
//!
//! Two scopes, mirroring the `semantex_docs_context` MCP tool's `scope` param:
//! - [`build_overview_scaffold`]: reuses [`crate::index::architecture`] (god
//!   nodes / communities / boundaries — PageRank + BFS community detection +
//!   cross-directory coupling) plus a repo-wide module inventory and
//!   language/file stats.
//! - [`build_module_scaffold`]: per-file symbol inventory sourced from
//!   `symbol_defs` + `structured_meta`, call-graph edges in/out, import edges
//!   in/out, all carrying file:line provenance so the writing agent can
//!   verify every claim against the actual source before committing it.
//!
//! Every query here is read-only; this module never mutates the index. New
//! read paths that aren't already on `ChunkStore` are added there as
//! clearly-named new methods (`get_importers_of_file`,
//! `get_call_edges_from_file`, `get_call_edges_into_file`,
//! `get_chunk_counts_by_file`, `get_symbol_counts_by_file`) rather than
//! changing existing ones.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::chunking::structured_meta::DocTag;
use crate::file::detector::FileType;
use crate::index::architecture::{self, ArchBudget, budget_for_chunk_count};
use crate::index::history;
use crate::index::storage::ChunkStore;
use crate::types::Chunk;

/// Max recent repo-wide commits surfaced in [`OverviewScaffold::recent_changes`].
const RECENT_CHANGES_LIMIT: usize = 10;
/// Max commits per file surfaced in [`ModuleScaffold::commits_touching`].
const COMMITS_TOUCHING_LIMIT: usize = 10;

/// File:line provenance for a scaffold item, so the writing agent can verify
/// every claim against the actual source before committing prose to markdown.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Provenance {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
}

impl Provenance {
    fn from_chunk(chunk: &Chunk) -> Self {
        Self {
            file: chunk.file_path.display().to_string(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
        }
    }
}

/// One symbol (function/method/class/etc.) in a module scaffold. Fields come
/// straight from `symbol_defs` (name/kind/provenance) and `structured_meta`
/// (signature/params/docstring/calls) — see `chunking::structured_meta` for
/// the 6-layer extraction this is sourced from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolScaffold {
    pub name: String,
    pub kind: String,
    pub qualified_name: Option<String>,
    pub signature: Option<String>,
    pub params: Vec<String>,
    pub return_type: Option<String>,
    /// Existing doc-comment text already captured in the chunk (Layer 1 of
    /// `StructuredChunkMeta`). The writing agent should preserve/refresh this
    /// rather than inventing new prose when it's present and still accurate.
    pub docstring: Option<String>,
    pub doc_tags: Vec<DocTag>,
    pub parent_class: Option<String>,
    pub semantic_role: Option<String>,
    /// Outgoing call targets by name (may include unresolved names).
    pub calls: Vec<String>,
    /// Incoming callers by name, as captured at chunk-extraction time.
    pub called_by: Vec<String>,
    pub provenance: Provenance,
}

/// One outgoing call edge from a module's code to a named callee, resolved to
/// a chunk (provenance) where the global graph resolution pass succeeded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CallEdgeOut {
    pub callee_name: String,
    pub resolved: Option<Provenance>,
}

/// One incoming call edge — some other symbol calls into this module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallEdgeIn {
    pub caller_symbol: Option<String>,
    pub provenance: Provenance,
}

/// Per-module (file) documentation scaffold — the `{ module: <path> }` scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleScaffold {
    pub path: String,
    pub language: Option<String>,
    pub file_role: Option<String>,
    pub symbols: Vec<SymbolScaffold>,
    /// Files this module imports.
    pub imports: Vec<String>,
    /// Files that import this module.
    pub imported_by: Vec<String>,
    pub calls_out: Vec<CallEdgeOut>,
    pub calls_in: Vec<CallEdgeIn>,
    /// Commits that touched this file, most recent first (v13 Wave 2 —
    /// history) — provenance candidates for the writing agent ("last
    /// touched by <author> <age> ago: <subject>"). `None` (absent, not `[]`)
    /// when `history.db` hasn't been populated yet — see
    /// [`OverviewScaffold::recent_changes`] for why that distinction matters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commits_touching: Option<Vec<CommitScaffold>>,
}

/// God node in the overview scaffold — a serde-able mirror of
/// `architecture::GodNode`. `GodNode` intentionally has no serde derive (its
/// only other consumer, `tool_architecture`, hand-builds `serde_json::Value`);
/// this type carries the same fields for the docs-scaffold JSON surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GodNodeScaffold {
    pub symbol: String,
    pub centrality: f64,
    pub semantic_role: Option<String>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryPointScaffold {
    pub symbol: String,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityScaffold {
    pub label: String,
    pub size: usize,
    pub member_files: Vec<String>,
    pub entry_points: Vec<EntryPointScaffold>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryScaffold {
    pub from: String,
    pub to: String,
    pub edge_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LanguageStat {
    pub language: String,
    pub file_count: usize,
    pub chunk_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInventoryEntry {
    pub path: String,
    pub language: Option<String>,
    pub file_role: Option<String>,
    pub symbol_count: usize,
}

/// One commit, as surfaced in a docs scaffold (v13 Wave 2 — history).
/// A serde-friendly compact mirror of `history::CommitInfo`, adding the
/// human-friendly relative `age` string the writing agent (and every other
/// history-derived rendering — the CLI table, the `semantex_agent` "recent
/// changes" addendum) shows instead of a raw unix timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitScaffold {
    pub hash: String,
    pub author: String,
    pub age: String,
    pub subject: String,
}

impl From<history::CommitInfo> for CommitScaffold {
    fn from(c: history::CommitInfo) -> Self {
        Self {
            age: history::humanize_age(c.ts),
            hash: c.hash,
            author: c.author,
            subject: c.subject,
        }
    }
}

/// Repo-wide documentation scaffold — the `"overview"` scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverviewScaffold {
    pub total_files: u64,
    pub total_chunks: u64,
    pub god_nodes: Vec<GodNodeScaffold>,
    pub communities: Vec<CommunityScaffold>,
    pub boundaries: Vec<BoundaryScaffold>,
    /// Every indexed file with its symbol count, language and inferred role.
    /// Sorted by symbol_count desc so the writing agent sees the highest
    /// signal modules first when a budget truncates the list.
    pub module_inventory: Vec<ModuleInventoryEntry>,
    /// Aggregated per-language file/chunk counts.
    pub language_stats: Vec<LanguageStat>,
    /// Most recent commits repo-wide (v13 Wave 2 — history). `None` (field
    /// absent from the JSON, not an empty array) when `history.db` hasn't
    /// been populated yet — an empty array would read as "this repo has no
    /// history", which isn't the same claim as "history wasn't gathered".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recent_changes: Option<Vec<CommitScaffold>>,
}

/// Map a file path to a language label via extension detection. Returns
/// `None` for `FileType::Unknown` (no recognized extension).
fn language_label(path: &Path) -> Option<String> {
    match FileType::detect(path) {
        FileType::Unknown => None,
        ft => Some(format!("{ft:?}").to_lowercase()),
    }
}

/// Open `history.db` read-only, if it exists, alongside `db_path`
/// (`<index_dir>/chunks.db` → `<index_dir>/history.db` — the two always
/// live side by side; see `layout::history_db_path`/`project_index_dir`).
/// `None` for a missing file OR a file that fails to open — either way, the
/// caller's history fields stay absent rather than erroring the whole
/// scaffold build over an auxiliary, best-effort signal.
fn open_history_readonly(db_path: &Path) -> Option<rusqlite::Connection> {
    let history_path = db_path.parent()?.join("history.db");
    if !history_path.exists() {
        return None;
    }
    let conn = rusqlite::Connection::open(&history_path).ok()?;
    // Wait briefly instead of failing with SQLITE_BUSY if a concurrent
    // build is populating history.db right now.
    let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
    Some(conn)
}

/// Hydrate a batch of chunk ids into a `chunk_id -> Chunk` map in one query.
fn chunk_map(store: &ChunkStore, ids: &[u64]) -> HashMap<u64, Chunk> {
    store
        .get_chunks(ids)
        .unwrap_or_default()
        .into_iter()
        .map(|c| (c.id, c))
        .collect()
}

/// Build the repo-wide `overview` scaffold: architectural overview (god
/// nodes / communities / boundaries, reused verbatim from
/// `index::architecture`) plus a module inventory and language/file stats
/// derived from `symbol_defs` and `chunks`.
pub fn build_overview_scaffold(store: &ChunkStore, db_path: &Path) -> Result<OverviewScaffold> {
    let total_files = store.file_count()?;
    let total_chunks = store.chunk_count()?;

    // Mirrors the MCP `tool_architecture` handler's adaptive-budget /
    // tiny-repo override so the docs scaffold's arch section matches what
    // `semantex_architecture` would report for the same index.
    let chunk_count_usize = total_chunks as usize;
    let budget = if total_chunks < architecture::ARCH_TINY_REPO_THRESHOLD {
        ArchBudget {
            god_nodes: 5,
            communities: 0,
            boundaries: 0,
            deep_examples_max: 0,
            exhaustive_max: 0,
        }
    } else {
        budget_for_chunk_count(chunk_count_usize)
    };

    let overview = architecture::build_arch_overview(store, db_path, Some(budget))
        .context("Failed to build architectural overview for docs scaffold")?;

    // Defect #15 pattern (see architecture.rs): symbol names must come from
    // `Chunk::symbol_name()`, not `structured_meta.name`. Batch-hydrate.
    let god_ids: Vec<u64> = overview.god_nodes.iter().map(|g| g.chunk_id).collect();
    let god_chunks = chunk_map(store, &god_ids);

    let god_nodes: Vec<GodNodeScaffold> = overview
        .god_nodes
        .iter()
        .map(|g| {
            let symbol = god_chunks
                .get(&g.chunk_id)
                .and_then(|c| c.symbol_name().map(str::to_string))
                .unwrap_or_else(|| g.symbol.clone());
            GodNodeScaffold {
                symbol,
                centrality: g.centrality,
                semantic_role: g.semantic_role.clone(),
                provenance: Provenance {
                    file: g.file.clone(),
                    start_line: g.start_line,
                    end_line: g.end_line,
                },
            }
        })
        .collect();

    let communities: Vec<CommunityScaffold> = overview
        .communities
        .iter()
        .map(|c| CommunityScaffold {
            label: c.label.clone(),
            size: c.size,
            member_files: c.member_files.clone(),
            entry_points: c
                .entry_points
                .iter()
                .map(|ep| EntryPointScaffold {
                    symbol: ep.symbol.clone(),
                    provenance: Provenance {
                        file: ep.file.clone(),
                        start_line: ep.start_line,
                        end_line: ep.end_line,
                    },
                })
                .collect(),
        })
        .collect();

    let boundaries: Vec<BoundaryScaffold> = overview
        .boundaries
        .iter()
        .map(|b| BoundaryScaffold {
            from: b.from.clone(),
            to: b.to.clone(),
            edge_count: b.edge_count,
        })
        .collect();

    // Module inventory + language stats: two grouped queries instead of
    // N per-file queries.
    let chunk_counts = store.get_chunk_counts_by_file().unwrap_or_default();
    let symbol_counts = store.get_symbol_counts_by_file().unwrap_or_default();
    let all_paths = store.get_all_file_paths()?;
    let path_refs: Vec<&Path> = all_paths.iter().map(std::path::PathBuf::as_path).collect();
    let roles = store.get_file_roles(&path_refs).unwrap_or_default();

    let mut module_inventory = Vec::with_capacity(all_paths.len());
    let mut language_stats_map: HashMap<String, LanguageStat> = HashMap::new();
    for path in &all_paths {
        let path_str = path.display().to_string();
        let symbol_count = symbol_counts.get(&path_str).copied().unwrap_or(0);
        let chunk_count = chunk_counts.get(&path_str).copied().unwrap_or(0);
        let language = language_label(path);
        let file_role = roles.get(path).map(|r| r.as_str().to_string());

        module_inventory.push(ModuleInventoryEntry {
            path: path_str,
            language: language.clone(),
            file_role,
            symbol_count,
        });

        let key = language.unwrap_or_else(|| "unrecognized".to_string());
        let stat = language_stats_map
            .entry(key.clone())
            .or_insert_with(|| LanguageStat {
                language: key,
                file_count: 0,
                chunk_count: 0,
            });
        stat.file_count += 1;
        stat.chunk_count += chunk_count;
    }

    module_inventory.sort_by(|a, b| {
        b.symbol_count
            .cmp(&a.symbol_count)
            .then_with(|| a.path.cmp(&b.path))
    });
    let mut language_stats: Vec<LanguageStat> = language_stats_map.into_values().collect();
    language_stats.sort_by(|a, b| {
        b.chunk_count
            .cmp(&a.chunk_count)
            .then_with(|| a.language.cmp(&b.language))
    });

    // v13 Wave 2 — history: absent (not `[]`) unless history.db actually
    // has commits to report (see `OverviewScaffold::recent_changes`'s doc).
    let recent_changes = open_history_readonly(db_path)
        .and_then(|conn| history::recent_commits(&conn, RECENT_CHANGES_LIMIT).ok())
        .filter(|v| !v.is_empty())
        .map(|v| v.into_iter().map(CommitScaffold::from).collect());

    Ok(OverviewScaffold {
        total_files,
        total_chunks,
        god_nodes,
        communities,
        boundaries,
        module_inventory,
        language_stats,
        recent_changes,
    })
}

/// Build the per-file `{ module: <path> }` scaffold: symbol inventory, import
/// edges in/out, and call-graph edges in/out, all with file:line provenance.
///
/// `module_path` must match the `file_path` stored in the index (the path
/// relative to the project root, as recorded by the indexer). `db_path` is
/// the `chunks.db` path (same one every other scaffold/tool caller already
/// has); it's used only to locate the sibling `history.db` for the
/// `commits_touching` field (v13 Wave 2 — history) — absent, not a probe
/// error, if `history.db` doesn't exist yet.
pub fn build_module_scaffold(
    store: &ChunkStore,
    db_path: &Path,
    module_path: &str,
) -> Result<ModuleScaffold> {
    let symbol_rows = store
        .get_symbol_defs_for_file(module_path)
        .with_context(|| format!("Failed to read symbol_defs for `{module_path}`"))?;
    let chunk_ids: Vec<u64> = symbol_rows.iter().map(|(id, _, _)| *id).collect();
    let chunks_by_id = chunk_map(store, &chunk_ids);

    let mut symbols = Vec::with_capacity(symbol_rows.len());
    for (chunk_id, name, kind) in &symbol_rows {
        let Some(chunk) = chunks_by_id.get(chunk_id) else {
            continue;
        };
        let meta = store.get_structured_meta(*chunk_id).ok().flatten();
        symbols.push(SymbolScaffold {
            name: name.clone(),
            kind: kind.clone(),
            qualified_name: meta.as_ref().and_then(|m| m.qualified_name.clone()),
            signature: meta.as_ref().and_then(|m| m.signature.clone()),
            params: meta.as_ref().map(|m| m.params.clone()).unwrap_or_default(),
            return_type: meta.as_ref().and_then(|m| m.return_type.clone()),
            docstring: meta.as_ref().and_then(|m| m.docstring.clone()),
            doc_tags: meta
                .as_ref()
                .map(|m| m.doc_tags.clone())
                .unwrap_or_default(),
            parent_class: meta.as_ref().and_then(|m| m.parent_class.clone()),
            semantic_role: meta
                .as_ref()
                .and_then(|m| m.semantic_role.map(|r| r.as_label().to_string())),
            calls: meta.as_ref().map(|m| m.calls.clone()).unwrap_or_default(),
            called_by: meta
                .as_ref()
                .map(|m| m.called_by.clone())
                .unwrap_or_default(),
            provenance: Provenance::from_chunk(chunk),
        });
    }
    symbols.sort_by_key(|s| s.provenance.start_line);

    let imports = store
        .get_module_edges_for_file(module_path)
        .unwrap_or_default();
    let imported_by = store.get_importers_of_file(module_path).unwrap_or_default();

    let out_edges = store
        .get_call_edges_from_file(module_path)
        .unwrap_or_default();
    let resolved_ids: Vec<u64> = out_edges.iter().filter_map(|(_, _, cid)| *cid).collect();
    let resolved_chunks = chunk_map(store, &resolved_ids);
    let mut calls_out: Vec<CallEdgeOut> = out_edges
        .iter()
        .map(|(_, callee_name, cid)| CallEdgeOut {
            callee_name: callee_name.clone(),
            resolved: cid
                .and_then(|id| resolved_chunks.get(&id))
                .map(Provenance::from_chunk),
        })
        .collect();
    calls_out.sort_by(|a, b| {
        a.callee_name
            .cmp(&b.callee_name)
            .then_with(|| a.resolved.cmp(&b.resolved))
    });
    calls_out.dedup();

    let in_edges = store
        .get_call_edges_into_file(module_path)
        .unwrap_or_default();
    let caller_ids: Vec<u64> = in_edges.iter().map(|(caller_id, _)| *caller_id).collect();
    let caller_chunks = chunk_map(store, &caller_ids);
    let mut calls_in: Vec<CallEdgeIn> = in_edges
        .iter()
        .filter_map(|(caller_id, _)| {
            let chunk = caller_chunks.get(caller_id)?;
            Some(CallEdgeIn {
                caller_symbol: chunk.symbol_name().map(str::to_string),
                provenance: Provenance::from_chunk(chunk),
            })
        })
        .collect();
    calls_in.sort_by(|a, b| {
        a.provenance
            .file
            .cmp(&b.provenance.file)
            .then_with(|| a.provenance.start_line.cmp(&b.provenance.start_line))
    });
    calls_in.dedup_by(|a, b| a.provenance == b.provenance);

    let path = Path::new(module_path);
    let language = language_label(path);
    let file_role = store
        .get_file_role(path)
        .ok()
        .map(|r| r.as_str().to_string());

    // v13 Wave 2 — history: absent (not `[]`) unless history.db actually has
    // commits for this file — see `ModuleScaffold::commits_touching`'s doc.
    let commits_touching = open_history_readonly(db_path)
        .and_then(|conn| {
            history::commits_touching_file(&conn, module_path, COMMITS_TOUCHING_LIMIT).ok()
        })
        .filter(|v| !v.is_empty())
        .map(|v| v.into_iter().map(CommitScaffold::from).collect());

    Ok(ModuleScaffold {
        path: module_path.to_string(),
        language,
        file_role,
        symbols,
        imports,
        imported_by,
        calls_out,
        calls_in,
        commits_touching,
    })
}

/// Approximate token budget as bytes (4 chars/token — the same rough
/// heuristic `search/agent.rs`'s byte-budget formatting uses) and trim the
/// module scaffold's lists until the serialized JSON roughly fits, or until
/// there's nothing left worth trimming. Deterministic and bounded: each
/// iteration strictly shrinks a list, so the loop always terminates.
pub fn apply_module_budget(scaffold: &mut ModuleScaffold, budget_tokens: usize) {
    if budget_tokens == 0 {
        return;
    }
    let approx_bytes = budget_tokens.saturating_mul(4);
    loop {
        let size = serde_json::to_vec(&*scaffold).map(|v| v.len()).unwrap_or(0);
        if size <= approx_bytes {
            return;
        }
        // History provenance is a nice-to-have addition, not a claim in
        // itself — trim it before anything the writing agent would actually
        // cite in prose.
        if let Some(commits) = scaffold.commits_touching.as_mut()
            && commits.len() > 3
        {
            let keep = commits.len() / 2;
            commits.truncate(keep.max(3));
            continue;
        }
        if scaffold.calls_in.len() > 5 {
            let keep = scaffold.calls_in.len() / 2;
            scaffold.calls_in.truncate(keep.max(5));
            continue;
        }
        if scaffold.calls_out.len() > 5 {
            let keep = scaffold.calls_out.len() / 2;
            scaffold.calls_out.truncate(keep.max(5));
            continue;
        }
        if scaffold.symbols.len() > 3 {
            let keep = scaffold.symbols.len() - (scaffold.symbols.len() / 4).max(1);
            scaffold.symbols.truncate(keep.max(3));
            continue;
        }
        // Nothing left that can shrink further without going empty.
        return;
    }
}

/// Same idea as [`apply_module_budget`] but for the overview scaffold: trims
/// `module_inventory` (already sorted by symbol_count desc, so the highest
/// signal entries survive) first, since it's typically the largest list on
/// big repos.
pub fn apply_overview_budget(scaffold: &mut OverviewScaffold, budget_tokens: usize) {
    if budget_tokens == 0 {
        return;
    }
    let approx_bytes = budget_tokens.saturating_mul(4);
    loop {
        let size = serde_json::to_vec(&*scaffold).map(|v| v.len()).unwrap_or(0);
        if size <= approx_bytes {
            return;
        }
        // History provenance trims first — same reasoning as `apply_module_budget`.
        if let Some(changes) = scaffold.recent_changes.as_mut()
            && changes.len() > 3
        {
            let keep = changes.len() / 2;
            changes.truncate(keep.max(3));
            continue;
        }
        if scaffold.module_inventory.len() > 20 {
            let keep = scaffold.module_inventory.len() / 2;
            scaffold.module_inventory.truncate(keep.max(20));
            continue;
        }
        if scaffold.boundaries.len() > 5 {
            let keep = scaffold.boundaries.len() / 2;
            scaffold.boundaries.truncate(keep.max(5));
            continue;
        }
        if scaffold.communities.len() > 1 {
            let keep = scaffold.communities.len() - 1;
            scaffold.communities.truncate(keep);
            continue;
        }
        return;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::structured_meta::StructuredChunkMeta;
    use crate::types::{AstNodeKind, ChunkType};

    fn seed_fixture(store: &ChunkStore) -> (u64, u64) {
        // lib.rs: pub fn run() calls helper()
        let mut meta_run = StructuredChunkMeta {
            name: Some("run".to_string()),
            signature: Some("pub fn run()".to_string()),
            docstring: Some("Runs the thing.".to_string()),
            calls: vec!["helper".to_string()],
            ..StructuredChunkMeta::default()
        };
        meta_run.generate_nl_summary();
        let run_chunk = Chunk {
            id: 0,
            file_path: "src/lib.rs".into(),
            start_line: 1,
            end_line: 5,
            content: "pub fn run() { helper(); }".to_string(),
            chunk_type: ChunkType::AstNode {
                name: "run".to_string(),
                kind: AstNodeKind::Function,
                language: "rust".to_string(),
                structured_meta: Some(Box::new(meta_run)),
            },
        };
        let run_id = store.insert_chunk(&run_chunk, 1, 0).unwrap();
        store
            .insert_symbol_def(run_id, "run", "function", "src/lib.rs")
            .unwrap();

        // helper.rs: fn helper() — called by run()
        let mut meta_helper = StructuredChunkMeta {
            name: Some("helper".to_string()),
            signature: Some("fn helper()".to_string()),
            docstring: Some("Helper routine.".to_string()),
            ..StructuredChunkMeta::default()
        };
        meta_helper.generate_nl_summary();
        let helper_chunk = Chunk {
            id: 0,
            file_path: "src/helper.rs".into(),
            start_line: 10,
            end_line: 12,
            content: "fn helper() {}".to_string(),
            chunk_type: ChunkType::AstNode {
                name: "helper".to_string(),
                kind: AstNodeKind::Function,
                language: "rust".to_string(),
                structured_meta: Some(Box::new(meta_helper)),
            },
        };
        let helper_id = store.insert_chunk(&helper_chunk, 2, 0).unwrap();
        store
            .insert_symbol_def(helper_id, "helper", "function", "src/helper.rs")
            .unwrap();

        store
            .store_call_graph_edge(run_id, "helper", Some(helper_id))
            .unwrap();
        store
            .insert_module_edge("src/lib.rs", "src/helper.rs", "mod helper;")
            .unwrap();

        (run_id, helper_id)
    }

    #[test]
    fn module_scaffold_captures_symbol_and_call_edges() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        seed_fixture(&store);

        let scaffold = build_module_scaffold(&store, &db_path, "src/lib.rs").unwrap();
        assert_eq!(scaffold.path, "src/lib.rs");
        assert_eq!(scaffold.symbols.len(), 1);
        assert_eq!(scaffold.symbols[0].name, "run");
        assert_eq!(
            scaffold.symbols[0].docstring.as_deref(),
            Some("Runs the thing.")
        );
        assert_eq!(scaffold.symbols[0].provenance.file, "src/lib.rs");
        assert_eq!(scaffold.symbols[0].provenance.start_line, 1);
        assert_eq!(scaffold.imports, vec!["src/helper.rs".to_string()]);
        assert_eq!(scaffold.calls_out.len(), 1);
        assert_eq!(scaffold.calls_out[0].callee_name, "helper");
        assert_eq!(
            scaffold.calls_out[0]
                .resolved
                .as_ref()
                .map(|p| p.file.as_str()),
            Some("src/helper.rs")
        );
    }

    #[test]
    fn module_scaffold_reverse_edges_on_callee_side() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        seed_fixture(&store);

        let scaffold = build_module_scaffold(&store, &db_path, "src/helper.rs").unwrap();
        assert_eq!(scaffold.symbols.len(), 1);
        assert_eq!(scaffold.symbols[0].name, "helper");
        assert_eq!(scaffold.imported_by, vec!["src/lib.rs".to_string()]);
        assert_eq!(scaffold.calls_in.len(), 1);
        assert_eq!(scaffold.calls_in[0].caller_symbol.as_deref(), Some("run"));
        assert_eq!(scaffold.calls_in[0].provenance.file, "src/lib.rs");
    }

    #[test]
    fn module_scaffold_unknown_file_returns_empty_not_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        seed_fixture(&store);

        let scaffold = build_module_scaffold(&store, &db_path, "src/does_not_exist.rs").unwrap();
        assert!(scaffold.symbols.is_empty());
        assert!(scaffold.imports.is_empty());
        assert!(scaffold.imported_by.is_empty());
    }

    /// v13 Wave 2 (history): with no `history.db` sitting next to
    /// `chunks.db`, both scaffolds' history fields must be `None` — absent
    /// from the JSON, not `[]` (see each field's doc comment for why that
    /// distinction is load-bearing).
    #[test]
    fn history_fields_absent_when_history_db_does_not_exist() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        seed_fixture(&store);

        let overview = build_overview_scaffold(&store, &db_path).unwrap();
        assert!(overview.recent_changes.is_none());
        assert!(
            !serde_json::to_string(&overview)
                .unwrap()
                .contains("recent_changes"),
            "the field itself must be omitted, not serialized as null/[]"
        );

        let module = build_module_scaffold(&store, &db_path, "src/lib.rs").unwrap();
        assert!(module.commits_touching.is_none());
        assert!(
            !serde_json::to_string(&module)
                .unwrap()
                .contains("commits_touching")
        );
    }

    /// v13 Wave 2 (history): once `history.db` has commits, both scaffolds
    /// surface them — repo-wide `recent_changes` on the overview, and
    /// per-file `commits_touching` on the module scaffold (only for files
    /// `file_commits` actually names).
    #[test]
    fn history_fields_present_when_history_db_populated() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        seed_fixture(&store);

        let history_path = tmp.path().join("history.db");
        let history_conn = crate::index::layout::open_history_db(&history_path).unwrap();
        history_conn
            .execute(
                "INSERT INTO commits (hash, author, ts, message) \
                 VALUES ('deadbeef', 'Ada', 1700000000, 'Add helper()')",
                [],
            )
            .unwrap();
        history_conn
            .execute(
                "INSERT INTO file_commits (path, hash) VALUES ('src/lib.rs', 'deadbeef')",
                [],
            )
            .unwrap();
        drop(history_conn);

        let overview = build_overview_scaffold(&store, &db_path).unwrap();
        let recent = overview
            .recent_changes
            .expect("recent_changes must be present once history.db has commits");
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].subject, "Add helper()");
        assert_eq!(recent[0].author, "Ada");

        let module = build_module_scaffold(&store, &db_path, "src/lib.rs").unwrap();
        let touching = module
            .commits_touching
            .expect("commits_touching must be present for a file file_commits names");
        assert_eq!(touching.len(), 1);
        assert_eq!(touching[0].hash, "deadbeef");

        // A file with no file_commits rows still gets `None`, not `[]`.
        let untouched = build_module_scaffold(&store, &db_path, "src/helper.rs").unwrap();
        assert!(untouched.commits_touching.is_none());
    }

    #[test]
    fn overview_scaffold_module_inventory_and_language_stats() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        seed_fixture(&store);
        store
            .set_file_entry(&crate::types::FileEntry {
                path: "src/lib.rs".into(),
                hash: 1,
                size: 10,
                mtime: 0,
            })
            .unwrap();
        store
            .set_file_entry(&crate::types::FileEntry {
                path: "src/helper.rs".into(),
                hash: 2,
                size: 10,
                mtime: 0,
            })
            .unwrap();

        let overview = build_overview_scaffold(&store, &db_path).unwrap();
        assert_eq!(overview.total_files, 2);
        assert_eq!(overview.total_chunks, 2);
        assert_eq!(overview.module_inventory.len(), 2);
        // Both files have exactly one symbol_def, so ordering falls back to
        // path ascending — helper.rs before lib.rs.
        assert_eq!(overview.module_inventory[0].path, "src/helper.rs");
        assert_eq!(overview.module_inventory[0].symbol_count, 1);
        assert_eq!(
            overview.module_inventory[0].language.as_deref(),
            Some("rust")
        );

        let rust_stats = overview
            .language_stats
            .iter()
            .find(|s| s.language == "rust")
            .expect("rust language stat present");
        assert_eq!(rust_stats.file_count, 2);
        assert_eq!(rust_stats.chunk_count, 2);
    }

    #[test]
    fn overview_scaffold_tiny_repo_still_succeeds_with_empty_arch() {
        // Empty index: no centrality table, no module_edges — the
        // architecture overview collapses to empty sections, but the
        // scaffold itself must still build without error.
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();

        let overview = build_overview_scaffold(&store, &db_path).unwrap();
        assert_eq!(overview.total_files, 0);
        assert_eq!(overview.total_chunks, 0);
        assert!(overview.god_nodes.is_empty());
        assert!(overview.module_inventory.is_empty());
    }

    #[test]
    fn apply_module_budget_shrinks_large_scaffold() {
        let mut scaffold = ModuleScaffold {
            path: "src/big.rs".to_string(),
            language: Some("rust".to_string()),
            file_role: None,
            symbols: (0..50)
                .map(|i| SymbolScaffold {
                    name: format!("sym_{i}"),
                    kind: "function".to_string(),
                    qualified_name: None,
                    signature: Some(format!("fn sym_{i}()")),
                    params: vec![],
                    return_type: None,
                    docstring: Some("x".repeat(200)),
                    doc_tags: vec![],
                    parent_class: None,
                    semantic_role: None,
                    calls: vec![],
                    called_by: vec![],
                    provenance: Provenance {
                        file: "src/big.rs".to_string(),
                        start_line: i,
                        end_line: i + 1,
                    },
                })
                .collect(),
            imports: vec![],
            imported_by: vec![],
            calls_out: vec![],
            calls_in: vec![],
            commits_touching: None,
        };
        let before = serde_json::to_vec(&scaffold).unwrap().len();
        apply_module_budget(&mut scaffold, 200); // ~800 bytes — much smaller than `before`
        let after = serde_json::to_vec(&scaffold).unwrap().len();
        assert!(after < before, "budget trim must shrink the scaffold");
        assert!(!scaffold.symbols.is_empty(), "must not trim to nothing");
    }

    #[test]
    fn apply_module_budget_zero_is_noop() {
        let mut scaffold = ModuleScaffold {
            path: "src/x.rs".to_string(),
            language: None,
            file_role: None,
            symbols: vec![],
            imports: vec![],
            imported_by: vec![],
            calls_out: vec![],
            calls_in: vec![],
            commits_touching: None,
        };
        apply_module_budget(&mut scaffold, 0);
        assert!(scaffold.symbols.is_empty());
    }

    #[test]
    fn language_label_maps_known_extension_and_none_for_unknown() {
        assert_eq!(
            language_label(Path::new("src/lib.rs")),
            Some("rust".to_string())
        );
        assert_eq!(language_label(Path::new("README.weirdext")), None);
    }
}
