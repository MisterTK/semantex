//! Architectural overview helpers: god nodes (by PageRank centrality),
//! communities (BFS over the call graph from top-centrality seeds), and
//! cross-directory boundaries.
//!
//! These functions were originally implemented in the MCP server's M6
//! `semantex_architecture` handler. They've been extracted here so both the
//! MCP tool (when hidden but still dispatchable for back-compat) and the
//! agent pipeline (`AgentRoute::Architecture`) can call the same logic.
//!
//! All helpers open the chunks SQLite database in read-only mode; they do
//! not require a `ChunkStore` instance for the parts that only read
//! aggregated centrality / boundary data. `build_communities` needs a
//! `ChunkStore` to walk the call-graph and resolve chunk file paths.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};

use crate::index::storage::ChunkStore;

/// A "god node" — a top-centrality chunk by PageRank score over the
/// call+import+hierarchy graph (E5 from the v0.3 SOTA spec).
#[derive(Debug, Clone)]
pub struct GodNode {
    pub chunk_id: u64,
    pub centrality: f64,
    pub symbol: String,
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub semantic_role: Option<String>,
}

/// A community — a connected component over the call graph rooted in the
/// top-centrality seeds. Members are file paths; entry points are the
/// highest-centrality symbols inside the component.
#[derive(Debug, Clone)]
pub struct Community {
    pub label: String,
    pub size: usize,
    pub member_files: Vec<String>,
    pub entry_points: Vec<EntryPoint>,
}

#[derive(Debug, Clone)]
pub struct EntryPoint {
    pub symbol: String,
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
}

/// A directory-level coupling edge: how many import edges cross from one
/// top-level directory to another.
#[derive(Debug, Clone)]
pub struct Boundary {
    pub from: String,
    pub to: String,
    pub edge_count: u64,
}

/// Aggregated overview returned by `build_arch_overview`.
#[derive(Debug, Clone, Default)]
pub struct ArchOverview {
    pub god_nodes: Vec<GodNode>,
    pub communities: Vec<Community>,
    pub boundaries: Vec<Boundary>,
}

/// Top-N chunks ordered by `structural_centrality` desc. Returns
/// `(chunk_id, centrality_score)` pairs.
pub fn query_top_centrality(db_path: &Path, n: usize) -> Result<Vec<(u64, f64)>> {
    let conn =
        rusqlite::Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("Failed to open {} read-only", db_path.display()))?;
    // chunk_centrality may not exist on pre-v0.3 indexes — guard with sqlite_master.
    let has_table: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_centrality'",
            [],
            |row| row.get::<_, i64>(0).map(|n| n != 0),
        )
        .unwrap_or(false);
    if !has_table {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(
        "SELECT chunk_id, structural_centrality FROM chunk_centrality \
         ORDER BY structural_centrality DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map([n as i64], |row| {
        Ok((row.get::<_, i64>(0)? as u64, row.get::<_, f64>(1)?))
    })?;
    let mut out = Vec::with_capacity(n);
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Hydrate top-N god nodes from the centrality table into the richer
/// `GodNode` struct (with symbol name, file path, role).
pub fn build_god_nodes(store: &ChunkStore, db_path: &Path, n: usize) -> Result<Vec<GodNode>> {
    let raw = query_top_centrality(db_path, n)?;
    let mut out = Vec::with_capacity(raw.len());
    for (cid, centrality) in raw {
        let Ok(chunk) = store.get_chunk(cid) else {
            continue;
        };
        let meta = store.get_structured_meta(cid).ok().flatten();
        let symbol = meta
            .as_ref()
            .and_then(|m| m.name.clone())
            .unwrap_or_else(|| chunk.file_path.display().to_string());
        let semantic_role = meta
            .and_then(|m| m.semantic_role)
            .map(|r| r.as_label().to_string());
        out.push(GodNode {
            chunk_id: cid,
            centrality,
            symbol,
            file: chunk.file_path.display().to_string(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            semantic_role,
        });
    }
    Ok(out)
}

/// Detect simple communities via BFS over call edges from the top-centrality
/// seed set. Returns up to 5 largest components, each as a `Community` with
/// member file paths and entry-point symbols.
pub fn build_communities(store: &ChunkStore, db_path: &Path) -> Result<Vec<Community>> {
    // Seeds: top 200 chunks by PageRank centrality. Bounded so we don't walk
    // the whole graph on giant repos.
    let seeds = match query_top_centrality(db_path, 200) {
        Ok(s) if !s.is_empty() => s,
        _ => return Ok(Vec::new()),
    };
    let seed_ids: Vec<u64> = seeds.iter().map(|(id, _)| *id).collect();

    let call_out = store.get_call_edges_from(&seed_ids)?;
    let call_in = store.get_call_edges_to(&seed_ids)?;
    let mut adj: HashMap<u64, HashSet<u64>> = HashMap::new();
    let add_edge = |a: u64, b: u64, adj: &mut HashMap<u64, HashSet<u64>>| {
        adj.entry(a).or_default().insert(b);
        adj.entry(b).or_default().insert(a);
    };
    for sid in &seed_ids {
        adj.entry(*sid).or_default();
    }
    for (a, b) in &call_out {
        add_edge(*a, *b, &mut adj);
    }
    for (a, b) in &call_in {
        add_edge(*a, *b, &mut adj);
    }

    // BFS to enumerate components.
    let mut visited: HashSet<u64> = HashSet::new();
    let mut components: Vec<Vec<u64>> = Vec::new();
    let mut nodes: Vec<u64> = adj.keys().copied().collect();
    nodes.sort_unstable(); // deterministic iteration (Finding from earlier review)
    for node in nodes {
        if visited.contains(&node) {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if !visited.insert(n) {
                continue;
            }
            component.push(n);
            if let Some(neighbors) = adj.get(&n) {
                let mut sorted: Vec<u64> = neighbors.iter().copied().collect();
                sorted.sort_unstable();
                for nb in sorted {
                    if !visited.contains(&nb) {
                        stack.push(nb);
                    }
                }
            }
        }
        components.push(component);
    }

    components.sort_by_key(|c| std::cmp::Reverse(c.len()));
    components.truncate(5);

    let score_by_id: HashMap<u64, f64> = seeds.iter().copied().collect();

    let mut out = Vec::with_capacity(components.len());
    for (idx, component) in components.iter().enumerate() {
        if component.is_empty() {
            continue;
        }
        let mut member_files: Vec<String> = Vec::new();
        let mut member_seen: HashSet<String> = HashSet::new();
        let mut entry_points: Vec<EntryPoint> = Vec::new();

        let mut sorted: Vec<u64> = component.clone();
        sorted.sort_by(|a, b| {
            score_by_id
                .get(b)
                .partial_cmp(&score_by_id.get(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &cid in sorted.iter().take(component.len().min(20)) {
            if let Ok(chunk) = store.get_chunk(cid) {
                let p = chunk.file_path.display().to_string();
                if member_seen.insert(p.clone()) {
                    member_files.push(p);
                }
                if entry_points.len() < 3 {
                    let meta = store.get_structured_meta(cid).ok().flatten();
                    let name = meta
                        .and_then(|m| m.name)
                        .unwrap_or_else(|| chunk.file_path.display().to_string());
                    entry_points.push(EntryPoint {
                        symbol: name,
                        file: chunk.file_path.display().to_string(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                    });
                }
            }
            if member_files.len() >= 8 {
                break;
            }
        }

        out.push(Community {
            label: format!("community-{}", idx + 1),
            size: component.len(),
            member_files,
            entry_points,
        });
    }

    Ok(out)
}

/// Count import edges that cross top-level directory boundaries. Returns up
/// to 25 entries sorted by edge_count desc.
pub fn build_boundaries(db_path: &Path) -> Result<Vec<Boundary>> {
    let conn =
        rusqlite::Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("Failed to open {} read-only", db_path.display()))?;
    let has_table: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='module_edges'",
            [],
            |row| row.get::<_, i64>(0).map(|n| n != 0),
        )
        .unwrap_or(false);
    if !has_table {
        return Ok(Vec::new());
    }

    let mut counts: HashMap<(String, String), u64> = HashMap::new();
    {
        let mut stmt = conn.prepare("SELECT importer_path, imported_path FROM module_edges")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for r in rows {
            let (a, b) = r?;
            let da = top_level_dir(&a);
            let db = top_level_dir(&b);
            if da == db {
                continue;
            }
            *counts.entry((da, db)).or_insert(0) += 1;
        }
    }

    let mut sorted: Vec<((String, String), u64)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    sorted.truncate(25);

    Ok(sorted
        .into_iter()
        .map(|((from, to), edge_count)| Boundary {
            from,
            to,
            edge_count,
        })
        .collect())
}

/// First path component (top-level directory). For `crates/foo/bar.rs` →
/// `"crates"`. Empty for files at the root.
pub fn top_level_dir(p: &str) -> String {
    Path::new(p)
        .components()
        .next()
        .and_then(|c| c.as_os_str().to_str())
        .unwrap_or("")
        .to_string()
}

/// One-call architectural overview: god nodes + communities + boundaries.
/// Default sizes (10/5/25) are spec-driven and small enough to fit in an
/// agent's first turn without bloating context.
pub fn build_arch_overview(store: &ChunkStore, db_path: &Path) -> Result<ArchOverview> {
    Ok(ArchOverview {
        god_nodes: build_god_nodes(store, db_path, 10)?,
        communities: build_communities(store, db_path)?,
        boundaries: build_boundaries(db_path)?,
    })
}

/// Pattern-catalog query: return `(chunk_id, pattern_name, language)` triples
/// matching a named pattern. Used by both M5 `tool_examples` and the agent
/// pipeline's deep-with-examples path.
pub fn query_pattern_matches(
    db_path: &Path,
    pattern: &str,
    language: Option<&str>,
    max: usize,
) -> Result<Vec<(u64, String, String)>> {
    let conn =
        rusqlite::Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("Failed to open {} read-only", db_path.display()))?;
    let has_table: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pattern_matches'",
            [],
            |row| row.get::<_, i64>(0).map(|n| n != 0),
        )
        .unwrap_or(false);
    if !has_table {
        return Ok(Vec::new());
    }
    let mut out: Vec<(u64, String, String)> = Vec::new();
    if let Some(lang) = language {
        let mut stmt = conn.prepare(
            "SELECT chunk_id, pattern_name, language FROM pattern_matches \
             WHERE pattern_name = ?1 AND language = ?2 LIMIT ?3",
        )?;
        let rows = stmt.query_map(rusqlite::params![pattern, lang, max as i64], |row| {
            Ok((
                row.get::<_, i64>(0)? as u64,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for r in rows {
            out.push(r?);
        }
    } else {
        let mut stmt = conn.prepare(
            "SELECT chunk_id, pattern_name, language FROM pattern_matches \
             WHERE pattern_name = ?1 LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![pattern, max as i64], |row| {
            Ok((
                row.get::<_, i64>(0)? as u64,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for r in rows {
            out.push(r?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_dir_extracts_first_component() {
        assert_eq!(top_level_dir("crates/semantex-mcp/src/server.rs"), "crates");
        assert_eq!(top_level_dir("src/main.rs"), "src");
        assert_eq!(top_level_dir("Cargo.toml"), "Cargo.toml");
        assert_eq!(top_level_dir(""), "");
    }

    #[test]
    fn query_top_centrality_handles_missing_table() {
        // A non-existent path triggers the guarded "no chunk_centrality" branch.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let result = query_top_centrality(tmp.path(), 10).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn build_boundaries_handles_missing_table() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let result = build_boundaries(tmp.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn query_pattern_matches_handles_missing_table() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let result = query_pattern_matches(tmp.path(), "rust.drop_impl", None, 3).unwrap();
        assert!(result.is_empty());
    }
}
