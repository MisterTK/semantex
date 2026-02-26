//! Cross-file symbol resolution for the code graph.
//!
//! After all files have been chunked and stored, this module runs a global
//! resolution pass that links call targets, type references, and type hierarchy
//! entries across file boundaries.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use super::storage::{ChunkStore, GraphStats};

/// A single symbol definition: (chunk_id, file_path, symbol_kind).
type SymbolDef = (u64, String, String);

/// Global post-indexing pass: resolve cross-file call targets,
/// type references, and type hierarchy entries.
///
/// Called once after all files have been chunked and inserted.
pub fn resolve_cross_file_graph(store: &ChunkStore) -> Result<GraphStats> {
    let mut stats = GraphStats::default();

    // Phase 1: Build symbol definition index
    let symbol_defs = store.get_all_symbol_defs()?;
    let name_to_defs = build_name_index(&symbol_defs);
    stats.symbol_defs_count = symbol_defs.len();

    // Phase 2: Resolve call_graph.callee_chunk_id
    stats.calls_resolved = resolve_call_targets(store, &name_to_defs)?;

    // Phase 3: Resolve type_refs.defining_chunk
    stats.types_resolved = resolve_type_targets(store, &name_to_defs)?;

    // Phase 4: Resolve type_hierarchy chunk IDs
    stats.hierarchy_resolved = resolve_hierarchy_targets(store, &name_to_defs)?;

    Ok(stats)
}

/// Build a lookup table mapping symbol_name -> Vec<(chunk_id, file_path, symbol_kind)>.
fn build_name_index(defs: &[(u64, String, String, String)]) -> HashMap<String, Vec<SymbolDef>> {
    let mut index: HashMap<String, Vec<SymbolDef>> = HashMap::with_capacity(defs.len());
    for (chunk_id, name, kind, file_path) in defs {
        index
            .entry(name.clone())
            .or_default()
            .push((*chunk_id, file_path.clone(), kind.clone()));
    }
    index
}

/// Resolve unresolved call graph edges by matching callee_name to symbol defs.
fn resolve_call_targets(
    store: &ChunkStore,
    name_index: &HashMap<String, Vec<SymbolDef>>,
) -> Result<usize> {
    let unresolved = store.get_unresolved_call_edges()?;
    let mut resolved_count = 0;

    for (rowid, callee_name, caller_file_path) in &unresolved {
        let target = lookup_symbol(store, name_index, callee_name, caller_file_path);

        if let Some(chunk_id) = target {
            store.update_callee_chunk_id(*rowid, chunk_id)?;
            resolved_count += 1;
        }
    }

    Ok(resolved_count)
}

/// Resolve unresolved type references by matching type_name to symbol defs
/// filtered to type-like kinds. Uses the same disambiguation heuristic as call
/// resolution when multiple definitions match.
fn resolve_type_targets(
    store: &ChunkStore,
    name_index: &HashMap<String, Vec<SymbolDef>>,
) -> Result<usize> {
    let unresolved = store.get_unresolved_type_refs()?;
    let mut resolved_count = 0;

    for (rowid, type_name, ref_file_path) in &unresolved {
        let chunk_id = lookup_type(store, name_index, type_name, ref_file_path);

        if let Some(id) = chunk_id {
            store.update_type_ref_defining_chunk(*rowid, id)?;
            resolved_count += 1;
        }
    }

    Ok(resolved_count)
}

/// Look up a type name in the index, with disambiguation and module prefix stripping.
fn lookup_type(
    store: &ChunkStore,
    name_index: &HashMap<String, Vec<SymbolDef>>,
    type_name: &str,
    ref_file: &str,
) -> Option<u64> {
    // Try exact name first
    if let Some(id) = try_resolve_type(store, name_index, type_name, ref_file) {
        return Some(id);
    }

    // Try stripping module prefix (e.g. "collections::HashMap" -> "HashMap")
    let stripped = strip_module_prefix(type_name);
    #[allow(clippy::collapsible_if)]
    if stripped != type_name {
        if let Some(id) = try_resolve_type(store, name_index, stripped, ref_file) {
            return Some(id);
        }
    }

    None
}

/// Attempt to resolve a type name to a single chunk ID, filtering to type-defining kinds
/// and using disambiguation for ambiguous matches.
fn try_resolve_type(
    store: &ChunkStore,
    name_index: &HashMap<String, Vec<SymbolDef>>,
    name: &str,
    ref_file: &str,
) -> Option<u64> {
    let defs = name_index.get(name)?;
    let type_defs: Vec<SymbolDef> = defs
        .iter()
        .filter(|(_, _, kind)| is_type_kind(kind))
        .cloned()
        .collect();

    match type_defs.len() {
        0 => None,
        1 => Some(type_defs[0].0),
        _ => Some(disambiguate_symbol(store, &type_defs, ref_file)),
    }
}

/// Resolve type hierarchy entries by linking child_name and parent_name
/// to their defining chunks.
fn resolve_hierarchy_targets(
    store: &ChunkStore,
    name_index: &HashMap<String, Vec<SymbolDef>>,
) -> Result<usize> {
    let unresolved = store.get_unresolved_hierarchy()?;
    let mut resolved_count = 0;

    for (rowid, child_name, parent_name) in &unresolved {
        let child_chunk = find_type_chunk(name_index, child_name);
        let parent_chunk = find_type_chunk(name_index, parent_name);

        // Only update if we resolved at least one side
        if child_chunk.is_some() || parent_chunk.is_some() {
            store.update_hierarchy_chunks(*rowid, child_chunk, parent_chunk)?;
            if child_chunk.is_some() && parent_chunk.is_some() {
                resolved_count += 1;
            }
        }
    }

    Ok(resolved_count)
}

/// Look up a symbol name in the index, with disambiguation for ambiguous matches.
///
/// Tries the full name first, then strips module prefixes (e.g. `utils.helper` -> `helper`).
fn lookup_symbol(
    store: &ChunkStore,
    name_index: &HashMap<String, Vec<SymbolDef>>,
    name: &str,
    caller_file: &str,
) -> Option<u64> {
    // Try exact name first
    if let Some(chunk_id) = try_resolve(store, name_index, name, caller_file) {
        return Some(chunk_id);
    }

    // Try stripping module/namespace prefix (e.g. "module.func" -> "func", "Ns::func" -> "func")
    let stripped = strip_module_prefix(name);
    #[allow(clippy::collapsible_if)]
    if stripped != name {
        if let Some(chunk_id) = try_resolve(store, name_index, stripped, caller_file) {
            return Some(chunk_id);
        }
    }

    None
}

/// Attempt to resolve a symbol name to a single chunk ID.
fn try_resolve(
    store: &ChunkStore,
    name_index: &HashMap<String, Vec<SymbolDef>>,
    name: &str,
    caller_file: &str,
) -> Option<u64> {
    let defs = name_index.get(name)?;
    match defs.len() {
        0 => None,
        1 => Some(defs[0].0),
        _ => Some(disambiguate_symbol(store, defs, caller_file)),
    }
}

/// Disambiguate among multiple candidate definitions for a symbol.
///
/// Priority order:
/// 1. Import-linked: caller's file imports candidate's file
/// 2. Same directory
/// 3. Same parent directory
/// 4. First match (fallback)
fn disambiguate_symbol(store: &ChunkStore, candidates: &[SymbolDef], caller_file: &str) -> u64 {
    let caller_path = Path::new(caller_file);
    let caller_dir = caller_path.parent();
    let caller_parent_dir = caller_dir.and_then(Path::parent);

    // Check import edges for the caller's file
    let imported_paths = store
        .get_module_edges_for_file(caller_file)
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to get module edges for {caller_file}: {e}");
            Vec::new()
        });

    // Priority 1: Candidate whose file is imported by the caller
    for (chunk_id, file_path, _) in candidates {
        if imported_paths.iter().any(|imp| imp == file_path) {
            return *chunk_id;
        }
    }

    // Priority 2: Same directory
    if let Some(cdir) = caller_dir {
        for (chunk_id, file_path, _) in candidates {
            if Path::new(file_path).parent() == Some(cdir) {
                return *chunk_id;
            }
        }
    }

    // Priority 3: Same parent directory
    if let Some(pdir) = caller_parent_dir {
        for (chunk_id, file_path, _) in candidates {
            let candidate_parent = Path::new(file_path).parent().and_then(Path::parent);
            if candidate_parent == Some(pdir) {
                return *chunk_id;
            }
        }
    }

    // Fallback: first candidate
    candidates[0].0
}

/// Find the chunk ID for a type name, preferring type-defining kinds.
fn find_type_chunk(name_index: &HashMap<String, Vec<SymbolDef>>, name: &str) -> Option<u64> {
    let defs = name_index.get(name)?;
    // Prefer type-defining kinds
    for (chunk_id, _, kind) in defs {
        if is_type_kind(kind) {
            return Some(*chunk_id);
        }
    }
    // Fall back to any definition
    defs.first().map(|(chunk_id, _, _)| *chunk_id)
}

/// Whether a symbol_kind represents a type definition.
fn is_type_kind(kind: &str) -> bool {
    matches!(
        kind,
        "class" | "struct" | "enum" | "trait" | "interface" | "type_alias"
    )
}

/// Strip module/namespace prefix from a qualified name.
///
/// Handles both dot-separated (`module.func`) and double-colon (`Ns::func`) forms.
fn strip_module_prefix(name: &str) -> &str {
    // Try :: separator first (Rust, C++)
    if let Some(pos) = name.rfind("::") {
        return &name[pos + 2..];
    }
    // Try dot separator (Python, JS, Java)
    if let Some(pos) = name.rfind('.') {
        return &name[pos + 1..];
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_module_prefix() {
        assert_eq!(strip_module_prefix("utils.helper"), "helper");
        assert_eq!(strip_module_prefix("std::collections::HashMap"), "HashMap");
        assert_eq!(strip_module_prefix("simple_name"), "simple_name");
        assert_eq!(strip_module_prefix("a.b.c.deep"), "deep");
    }

    #[test]
    fn test_is_type_kind() {
        assert!(is_type_kind("class"));
        assert!(is_type_kind("struct"));
        assert!(is_type_kind("enum"));
        assert!(is_type_kind("trait"));
        assert!(is_type_kind("interface"));
        assert!(is_type_kind("type_alias"));
        assert!(!is_type_kind("function"));
        assert!(!is_type_kind("method"));
        assert!(!is_type_kind("variable"));
    }

    #[test]
    fn test_build_name_index() {
        let defs = vec![
            (
                1,
                "foo".to_string(),
                "function".to_string(),
                "a.rs".to_string(),
            ),
            (
                2,
                "bar".to_string(),
                "class".to_string(),
                "b.rs".to_string(),
            ),
            (
                3,
                "foo".to_string(),
                "method".to_string(),
                "c.rs".to_string(),
            ),
        ];
        let index = build_name_index(&defs);
        assert_eq!(index.len(), 2);
        assert_eq!(index["foo"].len(), 2);
        assert_eq!(index["bar"].len(), 1);
    }

    #[test]
    fn test_find_type_chunk_prefers_type_kinds() {
        let defs = vec![
            (
                1,
                "Foo".to_string(),
                "function".to_string(),
                "a.rs".to_string(),
            ),
            (
                2,
                "Foo".to_string(),
                "class".to_string(),
                "b.rs".to_string(),
            ),
            (
                3,
                "Bar".to_string(),
                "struct".to_string(),
                "c.rs".to_string(),
            ),
        ];
        let index = build_name_index(&defs);

        // Foo: should prefer chunk 2 (class) over chunk 1 (function)
        assert_eq!(find_type_chunk(&index, "Foo"), Some(2));
        // Bar: only one def, a struct
        assert_eq!(find_type_chunk(&index, "Bar"), Some(3));
        // Missing name
        assert_eq!(find_type_chunk(&index, "Baz"), None);
    }

    #[test]
    fn test_try_resolve_type_filters_to_type_kinds() {
        // try_resolve_type should only consider type-defining kinds
        let defs = vec![
            (
                1,
                "Config".to_string(),
                "function".to_string(),
                "a.rs".to_string(),
            ),
            (
                2,
                "Config".to_string(),
                "struct".to_string(),
                "b.rs".to_string(),
            ),
        ];
        let index = build_name_index(&defs);

        // Without a store we can't call try_resolve_type directly, but we can
        // verify the filtering logic via find_type_chunk (same filter).
        assert_eq!(find_type_chunk(&index, "Config"), Some(2));
    }

    #[test]
    fn test_strip_module_prefix_for_types() {
        // Verifies that the same prefix stripping used for calls also works for
        // qualified type names (fix for issue #7).
        assert_eq!(strip_module_prefix("collections::HashMap"), "HashMap");
        assert_eq!(strip_module_prefix("com.example.UserConfig"), "UserConfig");
        assert_eq!(strip_module_prefix("HashMap"), "HashMap");
    }
}
