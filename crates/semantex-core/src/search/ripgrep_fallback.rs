//! Ripgrep fallback — shell out to system `rg` when the semantex index is unavailable.
//!
//! Provides keyword-level search results in a semantex-compatible format so users
//! get useful results immediately while the index builds in the background.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// A single result from ripgrep.
#[derive(Debug, Clone)]
pub struct RgResult {
    pub file: PathBuf,
    pub line_number: u32,
    pub content: String,
}

/// Check whether `rg` (ripgrep) is available on the system PATH.
/// The result is cached for the lifetime of the process.
pub fn is_rg_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::process::Command::new("rg")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

/// Run ripgrep on `path` with the given `query` and return up to `max_results` matches.
///
/// Uses `--fixed-strings` so the query is treated as a literal string, not a regex.
/// Returns file paths relative to `path`.
pub fn search(query: &str, path: &Path, max_results: usize) -> Result<Vec<RgResult>> {
    if !is_rg_available() {
        anyhow::bail!(
            "ripgrep (rg) is not installed. Install it for fallback search while the index builds."
        );
    }

    // --max-count is per-file, not global. We cap total results in parse_rg_json.
    let output = std::process::Command::new("rg")
        .arg("--json")
        .arg("--fixed-strings") // treat query as literal, not regex
        .arg("--max-count")
        .arg(max_results.to_string())
        .arg("-S") // smart-case: case-insensitive unless query has uppercase
        .arg("--")
        .arg(query)
        .arg(path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .context("Failed to execute ripgrep")?;

    // rg exits with 1 on "no matches" — that's not an error for us
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_rg_json(&stdout, path, max_results))
}

/// Parse ripgrep's JSON-lines output, filtering for `type: "match"` entries.
/// File paths are made relative to `base_path`.
fn parse_rg_json(output: &str, base_path: &Path, max_results: usize) -> Vec<RgResult> {
    let mut results = Vec::new();

    for line in output.lines() {
        if results.len() >= max_results {
            break;
        }

        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if value.get("type").and_then(|v| v.as_str()) != Some("match") {
            continue;
        }

        let Some(data) = value.get("data") else {
            continue;
        };

        let raw_file = data
            .get("path")
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("");

        let line_number = data
            .get("line_number")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32;

        let content = data
            .get("lines")
            .and_then(|l| l.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .trim_end()
            .to_string();

        if !raw_file.is_empty() {
            // Make path relative to the search directory
            let file_path = PathBuf::from(raw_file);
            let relative = file_path
                .strip_prefix(base_path)
                .unwrap_or(&file_path)
                .to_path_buf();
            results.push(RgResult {
                file: relative,
                line_number,
                content,
            });
        }
    }

    results
}

/// Format ripgrep results as a bare JSON array, matching semantex's normal `--json` output format.
pub fn format_as_json(results: &[RgResult]) -> String {
    let items: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "file": r.file.display().to_string(),
                "start_line": r.line_number,
                "end_line": r.line_number,
                "score": 0.0,
                "content": r.content,
                "source": "ripgrep",
            })
        })
        .collect();

    serde_json::to_string_pretty(&items).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_parse_rg_json_empty() {
        let results = parse_rg_json("", Path::new("."), 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_rg_json_match() {
        let json_line = r#"{"type":"match","data":{"path":{"text":"src/main.rs"},"lines":{"text":"fn main() {\n"},"line_number":1}}"#;
        let results = parse_rg_json(json_line, Path::new("."), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file, PathBuf::from("src/main.rs"));
        assert_eq!(results[0].line_number, 1);
        assert_eq!(results[0].content, "fn main() {");
    }

    #[test]
    fn test_parse_rg_json_strips_prefix() {
        let json_line = r#"{"type":"match","data":{"path":{"text":"/abs/project/src/main.rs"},"lines":{"text":"fn main()"},"line_number":1}}"#;
        let results = parse_rg_json(json_line, Path::new("/abs/project"), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file, PathBuf::from("src/main.rs"));
    }

    #[test]
    fn test_parse_rg_json_max_results() {
        let json_lines = (0..20)
            .map(|i| {
                format!(
                    r#"{{"type":"match","data":{{"path":{{"text":"file.rs"}},"lines":{{"text":"line {i}"}},"line_number":{i}}}}}"#,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let results = parse_rg_json(&json_lines, Path::new("."), 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_format_as_json() {
        let results = vec![RgResult {
            file: PathBuf::from("src/lib.rs"),
            line_number: 42,
            content: "pub fn search()".to_string(),
        }];
        let json = format_as_json(&results);
        assert!(json.contains("ripgrep"));
        assert!(json.contains("src/lib.rs"));
        // Verify it's a bare array, not wrapped in {"status", "results"}
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
    }
}
