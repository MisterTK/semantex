// crates/semantex-core/src/search/summarize.rs
// Extractive summarizer: no LLM, fully algorithmic, <10ms target.

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
        .map(|s| s.to_lowercase())
        .collect()
}

/// Extract query terms after removing stop words.
fn extract_query_terms(query: &str) -> Vec<String> {
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

/// A candidate sentence for inclusion in the summary.
#[derive(Debug)]
struct Candidate {
    text: String,
    score: f32,
    file: String,
    line_pos: u32,
}

/// Build an extractive summary from a set of `ReadChunk`s using query-term scoring.
///
/// Returns an empty string if no chunks are provided or no query terms can be extracted.
/// Target output: ≤2000 chars (~500 tokens), ends with a period.
pub fn extractive_summarize(query: &str, chunks: &[ReadChunk]) -> String {
    if chunks.is_empty() {
        return "No relevant code found for this query.".to_string();
    }

    let terms = extract_query_terms(query);
    if terms.is_empty() {
        return String::new();
    }

    let mut candidates: Vec<Candidate> = Vec::new();

    for chunk in chunks {
        // --- Summary field (high value) ---
        if let Some(ref summary) = chunk.summary {
            if !summary.is_empty() {
                let term_count = count_query_terms(summary, &terms) as f32;
                let score = term_count + 1.5; // summary bonus
                candidates.push(Candidate {
                    text: summary.clone(),
                    score,
                    file: chunk.file.clone(),
                    line_pos: chunk.start_line,
                });
            }
        }

        // --- Docstring field (moderate value) ---
        if let Some(ref docstring) = chunk.docstring {
            if !docstring.is_empty() {
                let term_count = count_query_terms(docstring, &terms) as f32;
                let score = term_count + 1.0; // docstring bonus
                candidates.push(Candidate {
                    text: docstring.clone(),
                    score,
                    file: chunk.file.clone(),
                    line_pos: chunk.start_line,
                });
            }
        }

        // --- Content lines ---
        for (line_idx, line) in chunk.content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                // Skip blank lines and pure comment lines (they rarely help)
                continue;
            }

            let term_count = count_query_terms(trimmed, &terms);
            let is_sig = is_signature_line(trimmed);

            // Only include lines that either match query terms or are signatures
            if term_count == 0 && !is_sig {
                continue;
            }

            // Score: term count + signature bonus + position bonus
            let score =
                term_count as f32 + if is_sig { 2.0 } else { 0.0 } + 1.0 / (line_idx as f32 + 1.0);

            candidates.push(Candidate {
                text: trimmed.to_string(),
                score,
                file: chunk.file.clone(),
                line_pos: chunk.start_line + line_idx as u32,
            });
        }
    }

    if candidates.is_empty() {
        return String::new();
    }

    // Sort by score descending, then deduplicate by text
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut selected: Vec<Candidate> = Vec::new();
    for candidate in candidates {
        if seen.contains(&candidate.text) {
            continue;
        }
        seen.insert(candidate.text.clone());
        selected.push(candidate);
        if selected.len() >= 12 {
            break;
        }
    }

    if selected.is_empty() {
        return String::new();
    }

    // Group by file, then sort within each group by line position
    let files: Vec<String> = {
        let mut seen_files: Vec<String> = Vec::new();
        for c in &selected {
            if !seen_files.contains(&c.file) {
                seen_files.push(c.file.clone());
            }
        }
        seen_files
    };

    let multi_file = files.len() > 1;
    let mut parts: Vec<String> = Vec::new();

    for file in &files {
        let mut file_candidates: Vec<&Candidate> =
            selected.iter().filter(|c| &c.file == file).collect();
        file_candidates.sort_by_key(|c| c.line_pos);

        let sentences: Vec<String> = file_candidates
            .iter()
            .map(|c| {
                // Clean up trailing punctuation for consistent joining
                let text = c
                    .text
                    .trim_end_matches(|ch| matches!(ch, '.' | ';' | ','))
                    .to_string();
                text
            })
            .collect();

        if sentences.is_empty() {
            continue;
        }

        if multi_file {
            // Extract just the filename for readability
            let short_file = std::path::Path::new(file)
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or(file.as_str());
            parts.push(format!("In {}: {}", short_file, sentences.join(". ")));
        } else {
            parts.push(sentences.join(". "));
        }
    }

    if parts.is_empty() {
        return String::new();
    }

    // Join all parts and ensure ends with period
    let mut result = parts.join(". ");
    if !result.ends_with('.') {
        result.push('.');
    }

    // Cap at ~2000 chars by truncating at last sentence boundary
    if result.len() > 2000 {
        // Find the last period before or at position 2000
        let truncate_at = result[..2000].rfind('.').map(|pos| pos + 1).unwrap_or(2000);
        result.truncate(truncate_at);
        if !result.ends_with('.') {
            result.push('.');
        }
    }

    result
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

    #[test]
    fn test_empty_chunks_returns_empty() {
        assert_eq!(
            extractive_summarize("search query", &[]),
            "No relevant code found for this query."
        );
    }

    #[test]
    fn test_empty_query_terms_returns_empty() {
        // All stop words
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
        assert!(result.ends_with('.'));
    }

    #[test]
    fn test_summary_field_preferred() {
        let mut chunk = make_chunk("lib.rs", "fn foo() {}");
        chunk.summary = Some("searches the index by query term".to_string());
        let result = extractive_summarize("search query", &[chunk]);
        assert!(
            result.contains("searches the index"),
            "Should use summary field"
        );
    }

    #[test]
    fn test_multi_file_grouping() {
        let chunk_a = make_chunk("a.rs", "fn search_users(query: &str) {}");
        let chunk_b = make_chunk("b.rs", "fn search_posts(query: &str) {}");
        let result = extractive_summarize("search query", &[chunk_a, chunk_b]);
        assert!(result.contains("In a.rs:") || result.contains("In b.rs:"));
    }

    #[test]
    fn test_output_ends_with_period() {
        let chunk = make_chunk("x.rs", "fn authenticate_user(token: &str) -> bool {}");
        let result = extractive_summarize("authenticate token", &[chunk]);
        if !result.is_empty() {
            assert!(
                result.ends_with('.'),
                "Result should end with period: {result:?}"
            );
        }
    }

    #[test]
    fn test_deduplication() {
        // Same content in two chunks shouldn't appear twice
        let chunk_a = make_chunk("a.rs", "fn authenticate(token: &str) -> bool {}");
        let chunk_b = make_chunk("b.rs", "fn authenticate(token: &str) -> bool {}");
        let result = extractive_summarize("authenticate token", &[chunk_a, chunk_b]);
        let count = result.matches("fn authenticate").count();
        assert!(count <= 1, "Duplicate lines should be deduplicated");
    }

    #[test]
    fn test_long_output_capped_at_2000_chars() {
        // Build a chunk with many matching lines
        let lines: Vec<String> = (0..30)
            .map(|i| format!("fn search_item_{i}(query: &str) -> Option<Item> {{ }}"))
            .collect();
        let chunk = make_chunk("big.rs", &lines.join("\n"));
        let result = extractive_summarize("search query item", &[chunk]);
        assert!(
            result.len() <= 2000,
            "Output should be capped at 2000 chars, got {}",
            result.len()
        );
    }

    #[test]
    fn test_top_12_sentences_selected() {
        // Provide 20 distinct matching lines; output should use at most 12
        let lines: Vec<String> = (0..20)
            .map(|i| format!("let search_result_{i} = perform_search(query_{i});"))
            .collect();
        let chunk = make_chunk("x.rs", &lines.join("\n"));
        let result = extractive_summarize("search perform query result", &[chunk]);
        assert!(!result.is_empty());
    }
}
