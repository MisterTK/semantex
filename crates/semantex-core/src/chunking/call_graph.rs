use std::collections::HashMap;

use crate::chunking::structured_meta::StructuredChunkMeta;

/// Build bidirectional call graph across all chunks in a file.
///
/// After individual chunk extraction, this post-pass fills in the `called_by` field
/// for each chunk by cross-referencing the `calls` fields.
///
/// Bidirectional caller/callee tracking.
/// Feeds `called_by` into the NL summary for BM25 enrichment.
pub fn build_call_graph(chunks_meta: &mut [(String, StructuredChunkMeta)]) {
    // Index: function name -> chunk indices that define it
    let mut name_to_indices: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, (_, meta)) in chunks_meta.iter().enumerate() {
        if let Some(ref name) = meta.name {
            name_to_indices.entry(name.clone()).or_default().push(i);
            // Also index qualified name for method resolution
            if let Some(ref qname) = meta.qualified_name {
                name_to_indices.entry(qname.clone()).or_default().push(i);
            }
        }
    }

    // Snapshot calls before mutating called_by, to avoid borrow conflicts
    let calls_snapshot: Vec<(usize, Vec<String>, Option<String>)> = chunks_meta
        .iter()
        .enumerate()
        .map(|(i, (_, meta))| (i, meta.calls.clone(), meta.name.clone()))
        .collect();

    // For each chunk, look at its calls and populate called_by on the targets
    for (caller_idx, calls, caller_name) in &calls_snapshot {
        let caller_label = caller_name.as_deref().unwrap_or("anonymous");
        for call_target in calls {
            // Try basename match (strip module prefix with rsplit('.'))
            let target_name = call_target
                .rsplit('.')
                .next()
                .unwrap_or(call_target.as_str());
            if let Some(target_indices) = name_to_indices.get(target_name) {
                for &target_idx in target_indices {
                    if target_idx != *caller_idx {
                        let called_by = &mut chunks_meta[target_idx].1.called_by;
                        let label = caller_label.to_string();
                        if !called_by.contains(&label) {
                            called_by.push(label);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_call_graph_basic() {
        let mut chunks = vec![
            (
                "fn foo() { bar(); }".to_string(),
                StructuredChunkMeta {
                    name: Some("foo".to_string()),
                    calls: vec!["bar".to_string()],
                    ..Default::default()
                },
            ),
            (
                "fn bar() { }".to_string(),
                StructuredChunkMeta {
                    name: Some("bar".to_string()),
                    ..Default::default()
                },
            ),
        ];

        build_call_graph(&mut chunks);

        assert!(chunks[1].1.called_by.contains(&"foo".to_string()));
        assert!(chunks[0].1.called_by.is_empty());
    }

    #[test]
    fn test_build_call_graph_module_prefix() {
        let mut chunks = vec![
            (
                "fn caller() { utils.helper(); }".to_string(),
                StructuredChunkMeta {
                    name: Some("caller".to_string()),
                    calls: vec!["utils.helper".to_string()],
                    ..Default::default()
                },
            ),
            (
                "fn helper() { }".to_string(),
                StructuredChunkMeta {
                    name: Some("helper".to_string()),
                    ..Default::default()
                },
            ),
        ];

        build_call_graph(&mut chunks);

        assert!(chunks[1].1.called_by.contains(&"caller".to_string()));
    }

    #[test]
    fn test_build_call_graph_no_self_reference() {
        let mut chunks = vec![(
            "fn recursive() { recursive(); }".to_string(),
            StructuredChunkMeta {
                name: Some("recursive".to_string()),
                calls: vec!["recursive".to_string()],
                ..Default::default()
            },
        )];

        build_call_graph(&mut chunks);

        // Should not add self as called_by (caller_idx == target_idx)
        assert!(chunks[0].1.called_by.is_empty());
    }

    #[test]
    fn test_build_call_graph_deduplication() {
        let mut chunks = vec![
            (
                "fn a() { b(); b(); }".to_string(),
                StructuredChunkMeta {
                    name: Some("a".to_string()),
                    calls: vec!["b".to_string(), "b".to_string()],
                    ..Default::default()
                },
            ),
            (
                "fn b() { }".to_string(),
                StructuredChunkMeta {
                    name: Some("b".to_string()),
                    ..Default::default()
                },
            ),
        ];

        build_call_graph(&mut chunks);

        // "a" should appear only once in b's called_by
        assert_eq!(chunks[1].1.called_by.len(), 1);
    }

    #[test]
    fn test_build_call_graph_empty() {
        let mut chunks: Vec<(String, StructuredChunkMeta)> = vec![];
        build_call_graph(&mut chunks);
        // Should not panic on empty input
    }

    #[test]
    fn test_build_call_graph_anonymous_caller() {
        let mut chunks = vec![
            (
                "{ bar(); }".to_string(),
                StructuredChunkMeta {
                    name: None,
                    calls: vec!["bar".to_string()],
                    ..Default::default()
                },
            ),
            (
                "fn bar() { }".to_string(),
                StructuredChunkMeta {
                    name: Some("bar".to_string()),
                    ..Default::default()
                },
            ),
        ];

        build_call_graph(&mut chunks);

        assert!(chunks[1].1.called_by.contains(&"anonymous".to_string()));
    }

    #[test]
    fn test_build_call_graph_qualified_name_match() {
        let mut chunks = vec![
            (
                "fn caller() { MyClass.process(); }".to_string(),
                StructuredChunkMeta {
                    name: Some("caller".to_string()),
                    calls: vec!["MyClass.process".to_string()],
                    ..Default::default()
                },
            ),
            (
                "fn process() { }".to_string(),
                StructuredChunkMeta {
                    name: Some("process".to_string()),
                    qualified_name: Some("MyClass.process".to_string()),
                    ..Default::default()
                },
            ),
        ];

        build_call_graph(&mut chunks);

        // Should match via basename "process"
        assert!(chunks[1].1.called_by.contains(&"caller".to_string()));
    }
}
