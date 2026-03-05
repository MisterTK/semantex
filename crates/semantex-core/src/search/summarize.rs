// crates/semantex-core/src/search/summarize.rs
// Extractive summarizer: no LLM, fully algorithmic, <10ms target.
//
// Produces structured, chunk-level blocks (not line-level fragments).
// Each block: header + docstring excerpt + NL summary + key code lines.

use std::fmt::Write as _;

/// A chunk prepared for summarization, with metadata and content.
#[derive(Debug, Clone)]
pub struct ReadChunk {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub name: Option<String>,
    pub kind: Option<String>,
    pub content: String,
    /// NL summary from StructuredChunkMeta (pre-generated, rule-based).
    pub summary: Option<String>,
    pub docstring: Option<String>,
}

/// Stop words filtered out before query term matching.
static STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to", "of", "for", "and", "or",
    "with", "from", "that", "this", "it", "its", "by", "be", "as", "we", "can", "will", "have",
    "has", "had", "do", "does", "did", "not", "but", "if",
];

/// Tokenize a string into lowercase words, splitting on whitespace and punctuation.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Extract query terms after removing stop words.
pub fn extract_query_terms(query: &str) -> Vec<String> {
    tokenize(query)
        .into_iter()
        .filter(|t| !STOP_WORDS.contains(&t.as_str()) && t.len() > 1)
        .collect()
}

/// Count how many query terms appear in a line (case-insensitive).
fn count_query_terms(line: &str, terms: &[String]) -> usize {
    let lower = line.to_lowercase();
    terms.iter().filter(|t| lower.contains(t.as_str())).count()
}

/// Detect whether a line looks like a function/class/struct signature.
fn is_signature_line(line: &str) -> bool {
    let trimmed = line.trim();
    [
        "fn ",
        "pub fn ",
        "async fn ",
        "pub async fn ",
        "def ",
        "class ",
        "struct ",
        "impl ",
        "interface ",
        "func ",
        "function ",
        "pub struct ",
        "pub class ",
        "pub impl ",
        "export fn ",
        "export function ",
        "export class ",
        "export default ",
        "static ",
        "private ",
        "protected ",
        "public ",
        "void ",
        "int ",
        "string ",
        "bool ",
    ]
    .iter()
    .any(|prefix| trimmed.starts_with(prefix))
}

/// Extract the first N sentences from text.
fn first_n_sentences(text: &str, n: usize) -> String {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
                if sentences.len() >= n {
                    break;
                }
            }
            current.clear();
        }
    }
    // If no sentence-ending punctuation found, use the whole text
    if sentences.is_empty() && !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    sentences.join(" ")
}

/// Build an extractive summary from a set of `ReadChunk`s.
///
/// Produces structured, chunk-level blocks. Each block contains:
/// - Header: `file:start-end name [kind]`
/// - Docstring excerpt (2-space indent, first 2 sentences)
/// - NL summary (2-space indent)
/// - Top 3 query-matching content lines (4-space indent)
///
/// Chunks are presented in input order (search relevance from the deep pipeline).
/// Total output capped at ~3000 chars (~750 tokens).
pub fn extractive_summarize(query: &str, chunks: &[ReadChunk]) -> String {
    const MAX_LEN: usize = 3000;
    if chunks.is_empty() {
        return "No relevant code found for this query.".to_string();
    }

    let terms = extract_query_terms(query);
    if terms.is_empty() {
        return String::new();
    }

    let mut output = String::new();

    for chunk in chunks {
        let block = build_chunk_block(chunk, &terms);
        if block.is_empty() {
            continue;
        }

        // Stop if adding this block would exceed budget (but always include first block)
        if output.len() + block.len() > MAX_LEN && !output.is_empty() {
            break;
        }
        output.push_str(&block);
        output.push('\n');
    }

    let result = output.trim_end().to_string();
    if result.is_empty() {
        // Fallback: if no blocks were produced (all chunks had no matches),
        // produce a minimal header-only listing
        let mut fallback = String::new();
        for chunk in chunks.iter().take(5) {
            let header = chunk_header(chunk);
            if fallback.len() + header.len() + 1 > MAX_LEN {
                break;
            }
            fallback.push_str(&header);
            fallback.push('\n');
        }
        return fallback.trim_end().to_string();
    }
    result
}

/// Build the header line for a chunk: `file:start-end name [kind]`
fn chunk_header(chunk: &ReadChunk) -> String {
    let mut header = String::new();
    // Use short filename for readability
    let short_file = std::path::Path::new(&chunk.file)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(&chunk.file);
    let _ = write!(
        header,
        "{}:{}-{}",
        short_file, chunk.start_line, chunk.end_line
    );
    if let Some(ref name) = chunk.name {
        let _ = write!(header, " {name}");
    }
    if let Some(ref kind) = chunk.kind {
        let _ = write!(header, " [{kind}]");
    }
    header
}

/// Clean an NL summary for use in deep search answers.
///
/// NL summaries from StructuredChunkMeta have the format:
///   "function name; parameters: ...; returns ...; calls ...; called_by ..."
///
/// Most of this is noise — it just restates the function signature. We:
/// 1. Strip the calls/called_by/imports/uses noise lists
/// 2. Skip summaries that are pure signature restatements (start with "function ",
///    "module ", "struct ", "enum ", "impl " — these just wordify the header)
///
/// Returns None if the summary adds no value beyond the chunk header.
fn clean_summary(summary: &str) -> Option<String> {
    // Strip noise: calls, called_by, imports, uses
    let noise_markers = ["; calls ", "; called by ", "; imports ", "; uses types "];
    let mut end = summary.len();
    for marker in &noise_markers {
        if let Some(pos) = summary.find(marker) {
            end = end.min(pos);
        }
    }
    let truncated = &summary[..end];

    // Skip pure signature restatements — these start with a kind keyword
    // and contain only parameter/return info that the code already shows
    let lower = truncated.to_lowercase();
    let signature_prefixes = [
        "function ",
        "module ",
        "struct ",
        "enum ",
        "impl ",
        "class ",
        "interface ",
        "method ",
        "constant ",
        "variable ",
        "type ",
    ];
    if signature_prefixes.iter().any(|p| lower.starts_with(p)) {
        // It's a signature restatement — skip it entirely
        return None;
    }

    // Cap at 200 chars
    let result = if truncated.len() > 200 {
        truncated[..200]
            .rfind(';')
            .map_or(&truncated[..200], |pos| &truncated[..pos])
    } else {
        truncated
    };

    if result.is_empty() {
        None
    } else {
        Some(result.to_string())
    }
}

/// Build a structured block for a single chunk.
fn build_chunk_block(chunk: &ReadChunk, terms: &[String]) -> String {
    let mut block = String::new();

    // --- Header ---
    block.push_str(&chunk_header(chunk));
    block.push('\n');

    let mut has_content = false;

    // --- Docstring (first 2 sentences, 2-space indent) ---
    if let Some(ref docstring) = chunk.docstring {
        let excerpt = first_n_sentences(docstring, 2);
        if !excerpt.is_empty() {
            let _ = writeln!(block, "  {excerpt}");
            has_content = true;
        }
    }

    // --- NL summary (2-space indent, cleaned, skip if redundant with docstring) ---
    if let Some(ref summary) = chunk.summary
        && let Some(cleaned) = clean_summary(summary) {
            let already_shown = chunk
                .docstring
                .as_deref()
                .map(str::to_lowercase)
                .unwrap_or_default();
            let cleaned_lower = cleaned.to_lowercase();
            // Skip if the summary is a substring of the docstring (avoid redundancy)
            if !already_shown.contains(&cleaned_lower) {
                let _ = writeln!(block, "  {cleaned}");
                has_content = true;
            }
        }

    // --- Key content lines (4-space indent, top 3 by query term score) ---
    let mut matching_lines: Vec<(usize, String, usize)> = Vec::new();
    for (idx, line) in chunk.content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.len() < 3 {
            continue;
        }
        // Skip pure comments (but keep doc comments — they have useful content)
        if (trimmed.starts_with("//") && !trimmed.starts_with("///")) || trimmed.starts_with('#') {
            continue;
        }

        let term_count = count_query_terms(trimmed, terms);
        let is_sig = is_signature_line(trimmed);

        if term_count == 0 && !is_sig {
            continue;
        }

        // Signature bonus so fn headers appear even if fewer query terms
        let score = term_count + if is_sig { 2 } else { 0 };
        matching_lines.push((idx, trimmed.to_string(), score));
    }

    // Sort by score descending, take top 3, then restore source order
    matching_lines.sort_by(|a, b| b.2.cmp(&a.2));
    matching_lines.truncate(3);
    matching_lines.sort_by_key(|(idx, _, _)| *idx);

    for (_, line, _) in &matching_lines {
        // Truncate very long lines
        let display = if line.len() > 120 {
            &line[..120]
        } else {
            line.as_str()
        };
        let _ = writeln!(block, "    {display}");
        has_content = true;
    }

    // Only return the block if we found something useful beyond the header
    if has_content { block } else { String::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(file: &str, content: &str) -> ReadChunk {
        ReadChunk {
            file: file.to_string(),
            start_line: 1,
            end_line: 10,
            name: None,
            kind: None,
            content: content.to_string(),
            summary: None,
            docstring: None,
        }
    }

    fn make_named_chunk(file: &str, name: &str, kind: &str, content: &str) -> ReadChunk {
        ReadChunk {
            file: file.to_string(),
            start_line: 1,
            end_line: 10,
            name: Some(name.to_string()),
            kind: Some(kind.to_string()),
            content: content.to_string(),
            summary: None,
            docstring: None,
        }
    }

    #[test]
    fn test_empty_chunks_returns_empty() {
        assert_eq!(
            extractive_summarize("search query", &[]),
            "No relevant code found for this query."
        );
    }

    #[test]
    fn test_empty_query_terms_returns_empty() {
        let chunk = make_chunk("foo.rs", "fn example() {}");
        assert_eq!(extractive_summarize("the a an", &[chunk]), "");
    }

    #[test]
    fn test_signature_line_detected() {
        let chunk = make_chunk(
            "lib.rs",
            "pub fn search_index(query: &str) -> Vec<Result> { }",
        );
        let result = extractive_summarize("search index query", &[chunk]);
        assert!(
            !result.is_empty(),
            "Should produce output for signature line"
        );
        assert!(
            result.contains("pub fn search_index"),
            "Should include the signature: {result:?}"
        );
    }

    #[test]
    fn test_summary_field_included() {
        let mut chunk = make_chunk("lib.rs", "fn foo() {}");
        chunk.summary = Some("searches the index by query term".to_string());
        let result = extractive_summarize("search query", &[chunk]);
        assert!(
            result.contains("searches the index"),
            "Should include summary field: {result:?}"
        );
    }

    #[test]
    fn test_docstring_included() {
        let mut chunk = make_named_chunk(
            "lib.rs",
            "search_index",
            "fn",
            "pub fn search_index(query: &str) {}",
        );
        chunk.docstring = Some("Search the index using hybrid dense+sparse scoring.".to_string());
        let result = extractive_summarize("search index", &[chunk]);
        assert!(
            result.contains("Search the index"),
            "Should include docstring: {result:?}"
        );
    }

    #[test]
    fn test_multi_chunk_structure() {
        let chunk_a = make_named_chunk(
            "a.rs",
            "search_users",
            "fn",
            "pub fn search_users(query: &str) {}",
        );
        let chunk_b = make_named_chunk(
            "b.rs",
            "search_posts",
            "fn",
            "pub fn search_posts(query: &str) {}",
        );
        let result = extractive_summarize("search query", &[chunk_a, chunk_b]);
        // Should have separate blocks with file headers
        assert!(
            result.contains("a.rs:") && result.contains("b.rs:"),
            "Should have blocks for both files: {result:?}"
        );
    }

    #[test]
    fn test_chunk_header_format() {
        let chunk = make_named_chunk("src/lib.rs", "authenticate", "fn", "fn authenticate() {}");
        let header = chunk_header(&chunk);
        assert!(
            header.contains("lib.rs:1-10 authenticate [fn]"),
            "Header should have short file, range, name, kind: {header:?}"
        );
    }

    #[test]
    fn test_long_output_capped_at_3000_chars() {
        let chunks: Vec<ReadChunk> = (0..50)
            .map(|i| {
                let mut c = make_named_chunk(
                    &format!("file_{i}.rs"),
                    &format!("search_item_{i}"),
                    "fn",
                    &format!(
                        "pub fn search_item_{i}(query: &str) -> Option<Item> {{ query.search() }}"
                    ),
                );
                c.docstring = Some(format!(
                    "Search for item {i} in the database using the query."
                ));
                c
            })
            .collect();
        let result = extractive_summarize("search query item", &chunks);
        assert!(
            result.len() <= 3200, // small buffer for final block
            "Output should be roughly capped at 3000 chars, got {}",
            result.len()
        );
    }

    #[test]
    fn test_redundant_summary_docstring_dedup() {
        let mut chunk = make_named_chunk("lib.rs", "search", "fn", "pub fn search(q: &str) {}");
        chunk.docstring = Some("Search the index for matching results.".to_string());
        chunk.summary = Some("search the index for matching results".to_string());
        let result = extractive_summarize("search index", &[chunk]);
        // The summary is a lowercased version of the docstring — should appear only once
        let count = result.to_lowercase().matches("search the index").count();
        assert!(
            count <= 1,
            "Redundant summary/docstring should be deduped, got {count} occurrences: {result:?}"
        );
    }

    #[test]
    fn test_code_lines_indented() {
        let chunk = make_chunk(
            "x.rs",
            "pub fn authenticate_user(token: &str) -> bool { true }",
        );
        let result = extractive_summarize("authenticate token", &[chunk]);
        // Code lines should be 4-space indented
        assert!(
            result.contains("    pub fn authenticate_user"),
            "Code lines should be 4-space indented: {result:?}"
        );
    }

    #[test]
    fn test_first_n_sentences() {
        assert_eq!(
            first_n_sentences("First sentence. Second sentence. Third.", 2),
            "First sentence. Second sentence."
        );
        assert_eq!(first_n_sentences("No period", 2), "No period");
        assert_eq!(first_n_sentences("One. Two. Three.", 1), "One.");
    }

    #[test]
    fn test_clean_summary_skips_signature_restatements() {
        let summary = "function search; parameters: query: &str; returns Vec<Result>; calls tokenize, count_terms; called by deep_search; imports hash_map";
        assert_eq!(
            clean_summary(summary),
            None,
            "Signature restatements should be filtered out"
        );
    }

    #[test]
    fn test_clean_summary_keeps_descriptive() {
        let summary = "Searches the index for matching results using hybrid scoring.";
        assert_eq!(
            clean_summary(summary),
            Some("Searches the index for matching results using hybrid scoring.".to_string()),
        );
    }

    #[test]
    fn test_clean_summary_skips_module_enum() {
        let summary = "enum preferred kind; uses types preferred kind; imports hash map";
        assert_eq!(clean_summary(summary), None);

        let summary2 = "module tests; calls path_buf_from";
        assert_eq!(clean_summary(summary2), None);
    }

    #[test]
    fn test_clean_summary_strips_noise() {
        // A descriptive summary that happens to have call lists at the end
        let summary = "Run the full pipeline; calls search, triage, summarize; called by handler";
        assert_eq!(
            clean_summary(summary),
            Some("Run the full pipeline".to_string()),
        );
    }
}
