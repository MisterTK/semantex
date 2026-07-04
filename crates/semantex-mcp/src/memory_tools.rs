//! `semantex_memory_save` / `semantex_memory_recall` — project memory (v13
//! Wave 2, MEMORY workstream).
//!
//! Param parsing + result rendering for both tools lives here (same reason
//! `docs_context.rs` is split out of `server.rs`: keeps the already-large
//! `server.rs` from growing further). `McpServer::tool_memory_save` /
//! `tool_memory_recall` in `server.rs` are thin wrappers matching every
//! other tool handler's shape (resolve path → delegate) — memory notes
//! don't need the index-readiness gate or a `ChunkStore` open that most
//! other tools require, since they live in their own `memory.db`,
//! independent of the code index.
//!
//! See `semantex_core::index::memory` for the CRUD implementation, schema
//! migration, cap eviction, and FTS5-vs-LIKE match-quality ranking.

use anyhow::{Context, Result, bail};
use semantex_core::index::memory::{MemoryStore, Note};
use std::path::Path;

/// Default `scope` when the caller omits it. Matches
/// `semantex_core::index::memory::SCOPE_CONVENTIONS`'s first suggested
/// value.
const DEFAULT_SCOPE: &str = "global";

/// Default `limit` for `semantex_memory_recall` when the caller omits it —
/// deliberately small: memory recall is meant to surface a handful of the
/// most relevant notes, not dump the whole store.
const DEFAULT_RECALL_LIMIT: u64 = 5;
const MAX_RECALL_LIMIT: u64 = 50;

/// Result of a memory tool call: structured note data plus a compact
/// human-readable rendering.
#[derive(Debug)]
pub struct MemoryToolResult {
    pub text: String,
    pub structured: serde_json::Value,
}

fn parse_content(args: &serde_json::Value) -> Result<String> {
    let content = args
        .get("content")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("Missing required parameter: content"))?;
    if content.trim().is_empty() {
        bail!("`content` must not be empty");
    }
    Ok(content.to_string())
}

fn parse_scope(args: &serde_json::Value) -> String {
    args.get("scope")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_SCOPE)
        .to_string()
}

fn parse_tags(args: &serde_json::Value) -> Vec<String> {
    args.get("tags")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_query(args: &serde_json::Value) -> Option<String> {
    args.get("query")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn parse_limit(args: &serde_json::Value) -> u64 {
    args.get("limit")
        .and_then(serde_json::Value::as_u64)
        .map_or(DEFAULT_RECALL_LIMIT, |l| l.clamp(1, MAX_RECALL_LIMIT))
}

fn note_to_json(note: &Note) -> serde_json::Value {
    serde_json::json!({
        "id": note.id,
        "created_ts": note.created_ts,
        "updated_ts": note.updated_ts,
        "scope": note.scope,
        "content": note.content,
        "tags": note.tags,
        "source": note.source,
    })
}

fn short(s: &str, max: usize) -> String {
    let s = s.trim().replace('\n', " ");
    if s.chars().count() <= max {
        return s;
    }
    let truncated: String = s.chars().take(max).collect();
    format!("{truncated}…")
}

/// Run `semantex_memory_save`: parse `content`/`scope`/`tags`, save the note,
/// and return its id plus a one-line confirmation.
pub fn run_save(project_root: &Path, args: &serde_json::Value) -> Result<MemoryToolResult> {
    let content = parse_content(args)?;
    let scope = parse_scope(args);
    let tags = parse_tags(args);

    let store = MemoryStore::open(project_root).context("Failed to open project memory store")?;
    let id = store.save(&content, &scope, &tags)?;

    let text = format!(
        "Saved note #{id} (scope=\"{scope}\"{}): {}",
        if tags.is_empty() {
            String::new()
        } else {
            format!(", tags=[{}]", tags.join(", "))
        },
        short(&content, 120)
    );
    Ok(MemoryToolResult {
        text,
        structured: serde_json::json!({ "id": id, "scope": scope, "tags": tags }),
    })
}

/// Run `semantex_memory_recall`: parse `query`/`scope`/`limit`, recall
/// matching notes (best match first), and return them structured plus a
/// compact text listing.
pub fn run_recall(project_root: &Path, args: &serde_json::Value) -> Result<MemoryToolResult> {
    let query = parse_query(args);
    let scope = args
        .get("scope")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let limit = parse_limit(args);

    let store = MemoryStore::open(project_root).context("Failed to open project memory store")?;
    let notes = store.recall(query.as_deref(), scope, limit as usize)?;

    let mut text = String::new();
    {
        use std::fmt::Write as _;
        if notes.is_empty() {
            let _ = write!(
                text,
                "No memory notes found{}{}.",
                query
                    .as_ref()
                    .map_or(String::new(), |q| format!(" matching \"{q}\"")),
                scope.map_or(String::new(), |s| format!(" in scope \"{s}\""))
            );
        } else {
            let _ = writeln!(text, "{} memory note(s):", notes.len());
            for note in &notes {
                let _ = writeln!(
                    text,
                    "- #{} [{}]{}: {}",
                    note.id,
                    note.scope,
                    if note.tags.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", note.tags.join(", "))
                    },
                    short(&note.content, 160)
                );
            }
        }
    }

    let structured = serde_json::json!({
        "notes": notes.iter().map(note_to_json).collect::<Vec<_>>(),
        "count": notes.len(),
    });
    Ok(MemoryToolResult { text, structured })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_content_missing_errors() {
        let err = parse_content(&serde_json::json!({})).unwrap_err();
        assert!(format!("{err}").contains("Missing required parameter"));
    }

    #[test]
    fn parse_content_empty_errors() {
        let err = parse_content(&serde_json::json!({ "content": "   " })).unwrap_err();
        assert!(format!("{err}").contains("must not be empty"));
    }

    #[test]
    fn parse_scope_defaults_to_global() {
        assert_eq!(parse_scope(&serde_json::json!({})), "global");
        assert_eq!(parse_scope(&serde_json::json!({ "scope": "  " })), "global");
        assert_eq!(
            parse_scope(&serde_json::json!({ "scope": "file:src/lib.rs" })),
            "file:src/lib.rs"
        );
    }

    #[test]
    fn parse_tags_defaults_empty_and_filters_non_strings() {
        assert_eq!(parse_tags(&serde_json::json!({})), Vec::<String>::new());
        assert_eq!(
            parse_tags(&serde_json::json!({ "tags": ["a", "b", 1, "c"] })),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn parse_query_trims_and_treats_blank_as_none() {
        assert_eq!(parse_query(&serde_json::json!({})), None);
        assert_eq!(parse_query(&serde_json::json!({ "query": "   " })), None);
        assert_eq!(
            parse_query(&serde_json::json!({ "query": " hello " })),
            Some("hello".to_string())
        );
    }

    #[test]
    fn parse_limit_defaults_and_clamps() {
        assert_eq!(parse_limit(&serde_json::json!({})), DEFAULT_RECALL_LIMIT);
        assert_eq!(parse_limit(&serde_json::json!({ "limit": 0 })), 1);
        assert_eq!(
            parse_limit(&serde_json::json!({ "limit": 9999 })),
            MAX_RECALL_LIMIT
        );
        assert_eq!(parse_limit(&serde_json::json!({ "limit": 10 })), 10);
    }

    #[test]
    fn run_save_and_recall_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let save_out = run_save(
            dir.path(),
            &serde_json::json!({
                "content": "prefer WAL mode for memory.db",
                "scope": "module:index",
                "tags": ["sqlite", "convention"],
            }),
        )
        .unwrap();
        assert!(save_out.text.contains("Saved note"));
        let id = save_out.structured["id"].as_i64().unwrap();
        assert!(id > 0);

        let recall_out = run_recall(
            dir.path(),
            &serde_json::json!({ "query": "WAL mode", "scope": "module:index" }),
        )
        .unwrap();
        let notes = recall_out.structured["notes"].as_array().unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0]["id"].as_i64().unwrap(), id);
        assert_eq!(
            notes[0]["tags"].as_array().unwrap(),
            &[
                serde_json::Value::String("convention".into()),
                serde_json::Value::String("sqlite".into())
            ]
        );
        assert!(recall_out.text.contains("memory note"));
    }

    #[test]
    fn run_recall_empty_store_returns_helpful_text() {
        let dir = tempfile::TempDir::new().unwrap();
        let out = run_recall(dir.path(), &serde_json::json!({ "query": "anything" })).unwrap();
        assert_eq!(out.structured["count"].as_u64(), Some(0));
        assert!(out.text.contains("No memory notes found"));
    }

    #[test]
    fn run_save_missing_content_errors() {
        let dir = tempfile::TempDir::new().unwrap();
        let err = run_save(dir.path(), &serde_json::json!({})).unwrap_err();
        assert!(format!("{err}").contains("Missing required parameter"));
    }
}
