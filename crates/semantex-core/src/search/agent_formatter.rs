use crate::server::protocol::{
    DeepSearchResponse, GraphWalkResponse, SearchResponse, SearchResultItem,
};
use std::fmt::Write as _;

/// Default response budget in bytes (~3K tokens).
pub const DEFAULT_BUDGET: usize = 12_000;

/// Max items per graph section.
const MAX_GRAPH_SECTION: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatStyle {
    Default,
    Grep,
}

/// Format a single result item as a compact reference string: `file:start-end[ name][ [kind]]`.
pub(crate) fn format_ref(item: &SearchResultItem) -> String {
    let mut s = format!("{}:{}-{}", item.file, item.start_line, item.end_line);
    if let Some(name) = &item.name {
        s.push(' ');
        s.push_str(name);
    }
    if let Some(kind) = &item.kind {
        let _ = write!(s, " [{kind}]");
    }
    s
}

/// Format search results as a human-readable string.
pub fn format_search_results(
    response: &SearchResponse,
    style: FormatStyle,
    budget: usize,
) -> String {
    if response.results.is_empty() {
        return "No results.".to_string();
    }

    match style {
        FormatStyle::Grep => format_grep(response),
        FormatStyle::Default => format_default(response, budget),
    }
}

fn format_grep(response: &SearchResponse) -> String {
    let mut lines = Vec::new();
    for item in &response.results {
        let first_line = item
            .content
            .as_deref()
            .unwrap_or("")
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        lines.push(format!("{}:{}: {}", item.file, item.start_line, first_line));
    }
    let count = lines.len();
    lines.push(String::new());
    lines.push(format!("[{} matches, {}ms]", count, response.duration_ms));
    lines.join("\n")
}

fn format_default(response: &SearchResponse, budget: usize) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut total_bytes = 0usize;
    let total_count = response.results.len();
    let mut written = 0usize;

    for item in &response.results {
        // Skip items with no name AND no content AND no summary
        if item.name.is_none() && item.content.is_none() && item.summary.is_none() {
            continue;
        }

        let mut block = String::new();

        let _ = write!(
            block,
            "{}:{}-{}",
            item.file, item.start_line, item.end_line
        );
        if let Some(name) = &item.name {
            block.push(' ');
            block.push_str(name);
        }
        if let Some(kind) = &item.kind {
            let _ = write!(block, " [{kind}]");
        }
        let _ = write!(block, " ({:.2})", item.score);
        block.push('\n');

        // Line 2: summary or content (first 200 chars, newlines replaced with spaces)
        let preview = item
            .summary
            .as_deref()
            .filter(|s| !s.is_empty())
            .or_else(|| item.content.as_deref().filter(|s| !s.is_empty()));
        if let Some(text) = preview {
            let text_normalized = text.replace('\n', " ");
            let truncated = if text_normalized.len() > 200 {
                // Walk back from byte 200 to find a valid char boundary
                let end = (0..=200.min(text_normalized.len()))
                    .rev()
                    .find(|&i| text_normalized.is_char_boundary(i))
                    .unwrap_or(0);
                &text_normalized[..end]
            } else {
                &text_normalized
            };
            block.push_str("  ");
            block.push_str(truncated);
            block.push('\n');
        }

        let block_len = block.len();

        // Budget check: stop if over budget and at least one result written
        if written > 0 && total_bytes + block_len > budget {
            let remaining = total_count - written;
            parts.push(format!("... and {remaining} more results"));
            break;
        }

        total_bytes += block_len;
        written += 1;
        parts.push(block);
    }

    let confidence = response.confidence.as_deref().unwrap_or("unknown");
    let footer = format!(
        "[{} results, {}ms, confidence: {}]",
        total_count, response.duration_ms, confidence
    );

    let mut output = parts.join("\n");
    output.push('\n');
    output.push('\n');
    output.push_str(&footer);
    output
}

/// Format deep search results.
pub fn format_deep_results(response: &DeepSearchResponse, budget: usize) -> String {
    if response.answer.is_empty() && response.sources.is_empty() {
        return "No results.".to_string();
    }

    let mut out = String::new();

    // Answer — truncate at sentence boundary if over budget
    if !response.answer.is_empty() {
        if response.answer.len() > budget {
            // Find last '.' before the limit
            let truncate_at = response.answer[..budget]
                .rfind('.')
                .map_or(budget, |i| i + 1);
            out.push_str(&response.answer[..truncate_at]);
            out.push_str("\n\n[answer truncated]");
        } else {
            out.push_str(&response.answer);
        }
    }

    // Sources section
    if !response.sources.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str("Sources:\n");
        for src in &response.sources {
            let _ = write!(
                out,
                "  {}:{}-{}",
                src.file, src.start_line, src.end_line
            );
            if let Some(name) = &src.name {
                out.push(' ');
                out.push_str(name);
            }
            if let Some(kind) = &src.kind {
                let _ = write!(out, " [{kind}]");
            }
            out.push('\n');
        }
    }

    // Metrics footer
    let m = &response.metrics;
    let footer = format!(
        "\n[searched: {} chunks, read: {}, {}ms]",
        m.chunks_searched, m.chunks_read, m.total_ms
    );
    out.push_str(&footer);

    out
}

/// Format graph walk results.
fn format_section(out: &mut String, title: &str, items: &[SearchResultItem]) {
    if items.is_empty() {
        return;
    }
    if !out.is_empty() {
        out.push('\n');
    }
    let _ = writeln!(out, "{} ({}):", title, items.len());
    let shown = items.len().min(MAX_GRAPH_SECTION);
    for item in &items[..shown] {
        let _ = writeln!(out, "  {}", format_ref(item));
    }
    if items.len() > MAX_GRAPH_SECTION {
        let _ = writeln!(
            out,
            "  ... and {} more",
            items.len() - MAX_GRAPH_SECTION
        );
    }
}

pub fn format_graph_results(response: &GraphWalkResponse) -> String {
    let all_empty = response.target.is_empty()
        && response.callers.is_empty()
        && response.callees.is_empty()
        && response.type_refs.is_empty()
        && response.hierarchy.is_empty();

    if all_empty {
        return "No graph data found.".to_string();
    }

    let mut out = String::new();

    // Target section (no count suffix if it's just the target)
    if !response.target.is_empty() {
        out.push_str("Target:\n");
        let shown = response.target.len().min(MAX_GRAPH_SECTION);
        for item in &response.target[..shown] {
            let _ = writeln!(out, "  {}", format_ref(item));
        }
    }

    format_section(&mut out, "Callers", &response.callers);
    format_section(&mut out, "Callees", &response.callees);
    format_section(&mut out, "Type References", &response.type_refs);
    format_section(&mut out, "Type Hierarchy", &response.hierarchy);

    out
}

/// Format code blocks with line numbers.
pub fn format_code_blocks(
    results: &[SearchResultItem],
    code_contents: &[String],
    budget: usize,
) -> String {
    let mut out = String::new();
    let mut total_chars = 0usize;
    let mut blocks_written = 0usize;

    for (item, code) in results.iter().zip(code_contents.iter()) {
        if code.is_empty() {
            continue;
        }

        let lang = item.language.as_deref().unwrap_or("");

        let mut block = String::new();
        // Header
        let _ = write!(
            block,
            "### {}:{}-{}",
            item.file, item.start_line, item.end_line
        );
        if let Some(name) = &item.name {
            let _ = write!(block, " — {name}");
        }
        if let Some(kind) = &item.kind {
            let _ = write!(block, " [{kind}]");
        }
        block.push('\n');

        // Fenced code block with line numbers
        let _ = writeln!(block, "```{lang}");
        let mut line_num = item.start_line;
        for line in code.lines() {
            let _ = writeln!(block, "{line_num:4} | {line}");
            line_num += 1;
        }
        block.push_str("```\n");

        let block_len = block.len();

        // Budget check: stop if adding this block would exceed budget
        // But always include at least the first block
        if blocks_written > 0 && total_chars + block_len > budget {
            break;
        }

        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&block);
        total_chars += block_len;
        blocks_written += 1;
    }

    if blocks_written == 0 {
        return "No code blocks to display.".to_string();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::protocol::*;

    fn make_result(
        file: &str,
        start: u32,
        end: u32,
        name: Option<&str>,
        kind: Option<&str>,
        score: f32,
    ) -> SearchResultItem {
        SearchResultItem {
            file: file.into(),
            start_line: start,
            end_line: end,
            score,
            source: "Dense".into(),
            chunk_type: "AstNode".into(),
            name: name.map(Into::into),
            language: Some("rust".into()),
            content: Some("fn example() {}".into()),
            kind: kind.map(Into::into),
            summary: None,
        }
    }

    #[test]
    fn test_format_default_single_result() {
        let resp = SearchResponse {
            results: vec![make_result(
                "src/auth.rs",
                10,
                20,
                Some("Auth"),
                Some("struct"),
                0.85,
            )],
            duration_ms: 17,
            dense_count: 5,
            sparse_count: 8,
            fused_count: 10,
            metrics: None,
            confidence: Some("high".into()),
        };
        let out = format_search_results(&resp, FormatStyle::Default, DEFAULT_BUDGET);
        assert!(out.contains("src/auth.rs:10-20"));
        assert!(out.contains("Auth"));
        assert!(out.contains("[struct]"));
        assert!(out.contains("(0.85)"));
        assert!(out.contains("[1 results, 17ms"));
    }

    #[test]
    fn test_format_grep_style() {
        let resp = SearchResponse {
            results: vec![make_result("src/auth.rs", 42, 42, None, None, 0.5)],
            duration_ms: 3,
            dense_count: 0,
            sparse_count: 1,
            fused_count: 1,
            metrics: None,
            confidence: Some("medium".into()),
        };
        let out = format_search_results(&resp, FormatStyle::Grep, DEFAULT_BUDGET);
        assert!(out.contains("src/auth.rs:42:"));
        assert!(out.contains("[1 matches, 3ms]"));
    }

    #[test]
    fn test_format_empty() {
        let resp = SearchResponse {
            results: vec![],
            duration_ms: 1,
            dense_count: 0,
            sparse_count: 0,
            fused_count: 0,
            metrics: None,
            confidence: Some("none".into()),
        };
        assert_eq!(
            format_search_results(&resp, FormatStyle::Default, DEFAULT_BUDGET),
            "No results."
        );
    }

    #[test]
    fn test_format_deep_with_sources() {
        let resp = DeepSearchResponse {
            answer: "Auth uses JWT tokens.".into(),
            sources: vec![DeepSearchSource {
                file: "src/auth.rs".into(),
                start_line: 10,
                end_line: 50,
                name: Some("Auth".into()),
                kind: Some("struct".into()),
            }],
            metrics: DeepResponseMetrics {
                search_ms: 10,
                triage_ms: 2,
                graph_ms: 3,
                read_ms: 5,
                summarize_ms: 8,
                total_ms: 28,
                chunks_searched: 20,
                chunks_read: 8,
                confidence_zone: String::new(),
            },
            confidence: 0.9,
        };
        let out = format_deep_results(&resp, DEFAULT_BUDGET);
        assert!(out.contains("Auth uses JWT tokens."));
        assert!(out.contains("Sources:"));
        assert!(out.contains("src/auth.rs:10-50 Auth [struct]"));
        assert!(out.contains("[searched: 20 chunks, read: 8, 28ms]"));
    }

    #[test]
    fn test_format_graph_omits_empty_sections() {
        let resp = GraphWalkResponse {
            target: vec![make_result(
                "src/auth.rs",
                10,
                50,
                Some("Auth"),
                Some("struct"),
                1.0,
            )],
            callers: vec![make_result(
                "src/mid.rs",
                5,
                20,
                Some("check"),
                Some("fn"),
                0.9,
            )],
            callees: vec![],
            type_refs: vec![],
            hierarchy: vec![],
        };
        let out = format_graph_results(&resp);
        assert!(out.contains("Target:"));
        assert!(out.contains("Callers (1):"));
        assert!(!out.contains("Callees"));
        assert!(!out.contains("Type References"));
    }

    #[test]
    fn test_format_search_budget_truncates() {
        let results: Vec<SearchResultItem> = (0..20)
            .map(|i| {
                make_result(
                    &format!("file{i}.rs"),
                    1,
                    50,
                    Some(&format!("Func{i}")),
                    Some("fn"),
                    0.9,
                )
            })
            .collect();
        let resp = SearchResponse {
            results,
            duration_ms: 10,
            dense_count: 10,
            sparse_count: 10,
            fused_count: 20,
            metrics: None,
            confidence: Some("high".into()),
        };
        let out = format_search_results(&resp, FormatStyle::Default, 500);
        assert!(out.contains("more results"));
        assert!(out.contains("[20 results,"));
    }

    #[test]
    fn test_format_deep_budget_truncates() {
        let long_answer = "x".repeat(15000);
        let resp = DeepSearchResponse {
            answer: long_answer,
            sources: vec![],
            metrics: DeepResponseMetrics {
                search_ms: 10,
                triage_ms: 2,
                graph_ms: 3,
                read_ms: 5,
                summarize_ms: 8,
                total_ms: 28,
                chunks_searched: 20,
                chunks_read: 8,
                confidence_zone: String::new(),
            },
            confidence: 0.5,
        };
        let out = format_deep_results(&resp, 5000);
        assert!(out.len() < 6000);
        assert!(out.contains("[answer truncated]"));
    }

    #[test]
    fn test_code_blocks_no_content() {
        // All items have empty content → should return the sentinel string
        let results: Vec<SearchResultItem> = (0..3)
            .map(|i| {
                let mut item = make_result(&format!("file{i}.rs"), 1, 10, None, None, 0.5);
                item.content = Some(String::new());
                item
            })
            .collect();
        let code: Vec<String> = vec![String::new(); 3];
        assert_eq!(
            format_code_blocks(&results, &code, DEFAULT_BUDGET),
            "No code blocks to display."
        );
    }

    #[test]
    fn test_format_graph_all_empty() {
        let resp = GraphWalkResponse {
            target: vec![],
            callers: vec![],
            callees: vec![],
            type_refs: vec![],
            hierarchy: vec![],
        };
        assert_eq!(format_graph_results(&resp), "No graph data found.");
    }

    #[test]
    fn test_code_blocks_budget() {
        let results: Vec<SearchResultItem> = (0..10)
            .map(|i| {
                make_result(
                    &format!("file{i}.rs"),
                    1,
                    100,
                    Some(&format!("Fn{i}")),
                    Some("fn"),
                    0.5,
                )
            })
            .collect();
        let code: Vec<String> = (0..10).map(|_| "x".repeat(2000)).collect();
        let out = format_code_blocks(&results, &code, 6000);
        let block_count = out.matches("###").count();
        assert!((1..=4).contains(&block_count));
    }
}
