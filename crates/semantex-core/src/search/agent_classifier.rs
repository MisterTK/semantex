use serde::{Deserialize, Serialize};

use super::query_classifier::is_camel_case;

/// Returns true for tokens like `HTMLParser`, `XMLHttpRequest`:
/// an all-uppercase run of 2+ chars followed by a mixed-case suffix.
fn has_caps_prefix_symbol(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 3 {
        return false;
    }
    // Find where the all-caps run ends
    let mut caps_run = 0usize;
    for &c in &chars {
        if c.is_ascii_uppercase() {
            caps_run += 1;
        } else {
            break;
        }
    }
    // Need at least 2 caps at start and at least 1 char following
    if caps_run < 2 || caps_run >= chars.len() {
        return false;
    }
    // The suffix after the caps run must contain at least one lowercase letter
    chars[caps_run..].iter().any(|c| c.is_lowercase())
}

/// High-level query intent for agent routing.
///
/// **Phase 4 additions** (`Architecture`, `ExhaustiveStructural`,
/// `DeepWithExamples`): replace the M1-M6 visible MCP tools that the
/// v0.3-visible release exposed and that regressed CCB by +20%. The agent
/// pipeline now routes architecture/exhaustive/deep-pattern queries
/// internally to the same logic those tools used, without inviting agents
/// to chain multiple structural tools additively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRoute {
    FilePattern,
    Regex,
    ExactSymbol,
    Structural,
    Deep,
    Analytical,
    Exhaustive,
    Semantic,
    /// "What are the main components?" / "Architecture overview" / "god nodes".
    /// Returns a compact ArchOverview (PageRank god nodes + communities +
    /// cross-directory boundaries) in one call.
    Architecture,
    /// "List all configuration options" / "every CLI flag" / "enumerate X".
    /// Wide-net search with structural enumeration of callers/imports for
    /// each candidate, in one response.
    ExhaustiveStructural,
    /// "Explain the most complex algorithm" / "deep dive into X with examples".
    /// Deep search enriched with pattern-catalog exemplars for any pattern
    /// names matched in the result set.
    DeepWithExamples,
    /// "If I wanted to add X, what files would change?" — v0.6 Item 10.
    /// Routes to the multi-step internal planner which decomposes the
    /// question into Architecture → ConventionLookup → ImpactedFiles
    /// sub-queries and merges the results into one response. Falls back
    /// to `Deep` if the planner errors or times out.
    FeaturePlanning,
}

impl std::fmt::Display for AgentRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FilePattern => write!(f, "file_pattern"),
            Self::Regex => write!(f, "regex"),
            Self::ExactSymbol => write!(f, "exact_symbol"),
            Self::Structural => write!(f, "structural"),
            Self::Deep => write!(f, "deep"),
            Self::Analytical => write!(f, "analytical"),
            Self::Exhaustive => write!(f, "exhaustive"),
            Self::Semantic => write!(f, "semantic"),
            Self::Architecture => write!(f, "architecture"),
            Self::ExhaustiveStructural => write!(f, "exhaustive_structural"),
            Self::DeepWithExamples => write!(f, "deep_with_examples"),
            Self::FeaturePlanning => write!(f, "feature_planning"),
        }
    }
}

/// Detect "feature_planning"-class questions for v0.6 Item 10 routing.
///
/// The input is expected to be already lowercased (the classifier calls
/// `query.to_lowercase()` once and reuses it). Matches the three patterns
/// documented in the spec:
///   1. "if I want(ed) to add ..."
///   2. "how do I add ..."
///   3. "what files would (change|need|be touched) (to|when) ..."
///
/// Kept as plain substring scans to stay aligned with the rest of the
/// keyword classifier — no regex crate use, no allocation beyond the
/// single lowercased copy the caller already owns.
fn is_feature_planning_query(lower: &str) -> bool {
    if lower.contains("if i want to add") || lower.contains("if i wanted to add") {
        return true;
    }
    if lower.contains("how do i add") {
        return true;
    }
    // "what files would <verb>" — a verb alone is a strong enough signal.
    // The original pattern required a verb-prep pair, which silently missed
    // the common short variant "what files would change?" (no prep).
    if lower.contains("what files would") {
        let after = lower.split("what files would").nth(1).unwrap_or("");
        // Look-ahead window: only consider the next ~80 chars to avoid
        // matching unrelated "would" + "change" elsewhere in the prompt.
        let window: String = after.chars().take(80).collect();
        let has_verb = window.contains("change")
            || window.contains("need")
            || window.contains("touched")
            || window.contains("update");
        if has_verb {
            return true;
        }
    }
    false
}

/// Returns true if the query looks like a regex pattern.
fn is_regex(query: &str) -> bool {
    // \b \B \d \D \s \S \w \W — backslash and these classes are all ASCII,
    // so byte iteration is safe and avoids allocating a Vec<char>.
    let bytes = query.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        if bytes[i] == b'\\'
            && matches!(
                bytes[i + 1],
                b'b' | b'B' | b'd' | b'D' | b's' | b'S' | b'w' | b'W'
            )
        {
            return true;
        }
    }

    // Pipe — unless entire query is quote-wrapped
    let trimmed = query.trim();
    let quote_wrapped = (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
        || (trimmed.starts_with('`') && trimmed.ends_with('`'));
    if !quote_wrapped && query.contains('|') {
        return true;
    }

    // Character class [...]
    if let Some(open) = query.find('[')
        && query[open..].contains(']')
    {
        return true;
    }

    // Group with quantifier (foo|bar), (foo?), (foo*), (foo+)
    if let Some(open) = query.find('(') {
        let after_open = &query[open + 1..];
        if let Some(close) = after_open.find(')') {
            let inner = &after_open[..close];
            if inner.contains('|')
                || inner.contains('?')
                || inner.contains('*')
                || inner.contains('+')
            {
                return true;
            }
        }
    }

    // Anchors
    if query.starts_with('^') || query.ends_with('$') {
        return true;
    }

    false
}

/// Classify a query into a high-level agent route.
pub fn classify_agent_query(query: &str) -> AgentRoute {
    // 1. FilePattern
    if query.contains('*') || query.contains("**/") {
        return AgentRoute::FilePattern;
    }
    // Check for mid-token `?` (not a trailing natural-language question mark)
    {
        let chars: Vec<char> = query.chars().collect();
        for i in 0..chars.len() {
            if chars[i] == '?' {
                let before_non_ws = i > 0 && !chars[i - 1].is_whitespace();
                let after_non_ws = i + 1 < chars.len() && !chars[i + 1].is_whitespace();
                if before_non_ws && after_non_ws {
                    return AgentRoute::FilePattern;
                }
            }
        }
    }

    // 2. Regex
    if is_regex(query) {
        return AgentRoute::Regex;
    }

    // 3. ExactSymbol
    {
        let trimmed = query.trim();
        let (was_wrapped, stripped) = if (trimmed.starts_with('`') && trimmed.ends_with('`'))
            || (trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
        {
            (true, &trimmed[1..trimmed.len() - 1])
        } else {
            (false, trimmed)
        };

        if !stripped.is_empty()
            && !stripped.contains(char::is_whitespace)
            && (was_wrapped
                || is_camel_case(stripped)
                || has_caps_prefix_symbol(stripped)
                || (stripped.contains('_') && stripped.len() > 2)
                || (stripped.contains('.') && stripped.len() > 2))
        {
            return AgentRoute::ExactSymbol;
        }
    }

    let lower = query.to_lowercase();

    // 3b. Architecture — Phase 4. Matches the v0.3 spec's M6 use case
    // ("primer at session start"). Triggered by overview-style language;
    // routes internally to ArchOverview (god nodes + communities + boundaries)
    // so the agent gets a complete map in one call instead of grep-spelunking.
    let architecture_keywords = [
        "main components",
        "main component",
        "primary components",
        "key components",
        "core components",
        "main modules",
        "primary subsystems",
        "key subsystems",
        "architecture overview",
        "architectural overview",
        "system architecture",
        "high-level architecture",
        "high level architecture",
        "god nodes",
        "god node",
        "entry points",
        "primary data flow",
        "how do they interact",
        "how do these interact",
        "overall structure",
        "overall organization",
        "project structure",
        "code organization",
        "main subsystems",
    ];
    for kw in &architecture_keywords {
        if lower.contains(kw) {
            return AgentRoute::Architecture;
        }
    }

    // 3c. FeaturePlanning — v0.6 Item 10. Catches "feature_planning"-class
    // questions ("if I wanted to add X, what files would change?") before
    // the Deep prefix scan claims them via the `how ` / `what ` openers.
    // Matched patterns:
    //   - "if I want(ed) to add ..."  → user is planning a change
    //   - "how do I add ..."           → explicit "how to add" form
    //   - "what files would change/need/be touched to/when ..." → impact query
    // Kept as cheap substring matches; the substrings are unlikely to appear
    // verbatim in unrelated NL queries.
    if is_feature_planning_query(&lower) {
        return AgentRoute::FeaturePlanning;
    }

    // 4. Structural — callers/callees/imports/type-refs intent.
    let structural_keywords = [
        "callers",
        "callees",
        "who calls",
        "what calls",
        "called by",
        "used by",
        "uses",
        "depends on",
        "references",
        "call graph",
        "type hierarchy",
        "imports",
        "imported by",
    ];
    for kw in &structural_keywords {
        if lower.contains(kw) {
            return AgentRoute::Structural;
        }
    }

    // 4b. DeepWithExamples — Phase 4. Checked BEFORE the Deep prefix scan
    // so "explain the most complex algorithm" doesn't fall through to plain
    // Deep. Routes to deep search enriched with pattern-catalog exemplars
    // for any pattern names matched in the result set, so the agent gets
    // concrete code alongside prose without a follow-up turn.
    let deep_with_examples_markers = [
        "explain the most complex",
        "most complex algorithm",
        "key algorithm",
        "main algorithm",
        "data transformation",
        "core algorithm",
        "step by step",
        "step-by-step",
        "deep dive",
        "show me how",
        "show me the pattern",
        "with examples",
        "give examples of",
    ];
    for marker in &deep_with_examples_markers {
        if lower.contains(marker) {
            return AgentRoute::DeepWithExamples;
        }
    }

    // 5. Deep — prefix matching
    let deep_prefixes = [
        "how ",
        "why ",
        "explain ",
        "describe ",
        "walk me through ",
        "what is the flow ",
        "trace the ",
    ];
    for prefix in &deep_prefixes {
        if lower.starts_with(prefix) {
            return AgentRoute::Deep;
        }
    }

    // 6. Analytical — keyword matching
    let analytical_keywords = [
        "most ",
        "least ",
        "biggest",
        "smallest",
        "longest",
        "shortest",
        "complex",
        "complicated",
        "important",
        "critical",
        "dangerous",
        "risky",
        "review",
        "assess",
        "evaluate",
        "analyze",
        "compare",
        "difference",
        "versus",
        " vs ",
    ];
    for kw in &analytical_keywords {
        if lower.contains(kw) {
            return AgentRoute::Analytical;
        }
    }

    // 7. Exhaustive — "list all X", "find all Y", "enumerate Z"
    let exhaustive_markers = [
        "list all",
        "list every",
        "find all",
        "find every",
        "show all",
        "show every",
        "what are all",
        "where are all",
        "enumerate all",
        "enumerate every",
        "enumerate ",
    ];
    let mut is_exhaustive = false;
    for marker in &exhaustive_markers {
        if lower.contains(marker) {
            is_exhaustive = true;
            break;
        }
    }
    if is_exhaustive {
        // 7b. ExhaustiveStructural — Phase 4. When an exhaustive query also
        // mentions config/env/cli/flag/option, route to the richer structural
        // enumeration that includes definitions + usages in one pass.
        let structural_exhaustive_markers = [
            "config",
            "env var",
            "environment variable",
            "cli flag",
            "command line",
            "option",
            "setting",
            "every flag",
            "every option",
            "every config",
        ];
        for kw in &structural_exhaustive_markers {
            if lower.contains(kw) {
                return AgentRoute::ExhaustiveStructural;
            }
        }
        return AgentRoute::Exhaustive;
    }

    // 8. Semantic — default
    AgentRoute::Semantic
}

/// LLM-augmented classification (v0.6 Item 9).
///
/// Wraps `classify_agent_query` with an opt-in `LlmClassifier` consulted
/// first. If the LLM returns `Ok(route)`, that route is used. On any error
/// — model not loaded, inference failure, tokenization issue — we silently
/// fall back to the deterministic keyword classifier. The keyword
/// classifier remains the single source of truth in any build that doesn't
/// load a model, including the default `cargo build` with no feature
/// flags.
///
/// This function is intentionally synchronous: the LLM inference itself is
/// a blocking call into the ONNX runtime, and adding a tokio executor just
/// to await a single blocking op would bloat the default build. If we
/// later want to overlap inference with other I/O, this signature can be
/// promoted to async without breaking callers (it returns `AgentRoute`,
/// not a route + cost tuple).
pub fn classify_with_llm(query: &str, llm: &dyn crate::llm::LlmClassifier) -> AgentRoute {
    match llm.classify(query) {
        Ok(route) => route,
        Err(_) => classify_agent_query(query),
    }
}

/// Extract the most relevant symbol from a query string.
///
/// Scans tokens right-to-left. Pass 1: wrapped symbols. Pass 2: code patterns.
pub fn extract_symbol(query: &str) -> Option<String> {
    let tokens: Vec<&str> = query.split_whitespace().collect();

    // Pass 1 — wrapped symbols (right-to-left)
    for &token in tokens.iter().rev() {
        if token.len() > 2 && token.starts_with('`') && token.ends_with('`') {
            return Some(token[1..token.len() - 1].to_string());
        }
        if token.len() > 2
            && ((token.starts_with('"') && token.ends_with('"'))
                || (token.starts_with('\'') && token.ends_with('\'')))
        {
            return Some(token[1..token.len() - 1].to_string());
        }
    }

    // Pass 2 — code patterns (right-to-left)
    for &token in tokens.iter().rev() {
        if is_camel_case(token) {
            return Some(token.to_string());
        }
        if token.contains('_') && token.len() > 2 {
            return Some(token.to_string());
        }
        if token.contains('.') && !token.contains(' ') && token.len() > 2 {
            return Some(token.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_star() {
        assert_eq!(
            classify_agent_query("**/*.test.ts"),
            AgentRoute::FilePattern
        );
    }
    #[test]
    fn test_glob_qmark() {
        assert_eq!(
            classify_agent_query("src/?/index.js"),
            AgentRoute::FilePattern
        );
    }
    #[test]
    fn test_glob_plain_star() {
        assert_eq!(classify_agent_query("*.rs"), AgentRoute::FilePattern);
    }
    #[test]
    fn test_trailing_qmark_not_glob() {
        assert_eq!(
            classify_agent_query("how does auth work?"),
            AgentRoute::Deep
        );
    }
    #[test]
    fn test_lib_qmark_glob() {
        assert_eq!(classify_agent_query("lib/a?.js"), AgentRoute::FilePattern);
    }

    #[test]
    fn test_regex_word_boundary() {
        assert_eq!(classify_agent_query(r"auth\w+Handler"), AgentRoute::Regex);
    }
    #[test]
    fn test_regex_pipe() {
        assert_eq!(classify_agent_query("TODO|FIXME|HACK"), AgentRoute::Regex);
    }
    #[test]
    fn test_regex_class() {
        assert_eq!(classify_agent_query(r"\bclass\s+Auth"), AgentRoute::Regex);
    }
    #[test]
    fn test_regex_parens() {
        assert_eq!(classify_agent_query("(foo|bar)"), AgentRoute::Regex);
    }

    #[test]
    fn test_camel_case() {
        assert_eq!(classify_agent_query("AuthService"), AgentRoute::ExactSymbol);
    }
    #[test]
    fn test_snake_case() {
        assert_eq!(
            classify_agent_query("handle_request"),
            AgentRoute::ExactSymbol
        );
    }
    #[test]
    fn test_backtick_wrap() {
        assert_eq!(
            classify_agent_query("`processPayment`"),
            AgentRoute::ExactSymbol
        );
    }
    #[test]
    fn test_dot_path() {
        assert_eq!(
            classify_agent_query("auth.middleware"),
            AgentRoute::ExactSymbol
        );
    }
    #[test]
    fn test_html_parser() {
        assert_eq!(classify_agent_query("HTMLParser"), AgentRoute::ExactSymbol);
    }
    #[test]
    fn test_xml_http() {
        assert_eq!(
            classify_agent_query("XMLHttpRequest"),
            AgentRoute::ExactSymbol
        );
    }
    #[test]
    fn test_single_lower() {
        assert_eq!(classify_agent_query("main"), AgentRoute::Semantic);
    }
    #[test]
    fn test_capitalized_word() {
        assert_eq!(classify_agent_query("Main"), AgentRoute::Semantic);
    }

    #[test]
    fn test_who_calls() {
        assert_eq!(
            classify_agent_query("who calls authenticate"),
            AgentRoute::Structural
        );
    }
    #[test]
    fn test_depends_on() {
        assert_eq!(
            classify_agent_query("what depends on DatabasePool"),
            AgentRoute::Structural
        );
    }
    #[test]
    fn test_callers_of() {
        assert_eq!(
            classify_agent_query("callers of handleRequest"),
            AgentRoute::Structural
        );
    }

    #[test]
    fn test_how_query() {
        assert_eq!(
            classify_agent_query("how does authentication work?"),
            AgentRoute::Deep
        );
    }
    #[test]
    fn test_explain() {
        assert_eq!(
            classify_agent_query("explain the payment pipeline"),
            AgentRoute::Deep
        );
    }
    #[test]
    fn test_why_query() {
        assert_eq!(
            classify_agent_query("why does the cache invalidate on deploy"),
            AgentRoute::Deep
        );
    }
    #[test]
    fn test_what_is_foo() {
        assert_eq!(classify_agent_query("what is foo?"), AgentRoute::Semantic);
    }

    #[test]
    fn test_most_complex() {
        assert_eq!(
            classify_agent_query("most complex functions in this repo"),
            AgentRoute::Analytical
        );
    }
    #[test]
    fn test_compare() {
        assert_eq!(
            classify_agent_query("compare auth approaches"),
            AgentRoute::Analytical
        );
    }
    #[test]
    fn test_review() {
        assert_eq!(
            classify_agent_query("review the error handling"),
            AgentRoute::Analytical
        );
    }

    #[test]
    fn test_exhaustive_list_all() {
        // Phase 4: "list all" + "configuration options" / "environment variables"
        // is now ExhaustiveStructural — the richer route that enumerates
        // definitions + usages in one response. The plain Exhaustive route is
        // for queries that don't name a configuration/CLI surface.
        assert_eq!(
            classify_agent_query("List all configuration options and environment variables"),
            AgentRoute::ExhaustiveStructural
        );
    }
    #[test]
    fn test_exhaustive_find_all() {
        assert_eq!(
            classify_agent_query("find all error types defined in this project"),
            AgentRoute::Exhaustive
        );
    }
    #[test]
    fn test_exhaustive_enumerate_cli() {
        // Phase 4: "enumerate" + "CLI flags" routes to ExhaustiveStructural.
        assert_eq!(
            classify_agent_query("enumerate the CLI flags this project supports"),
            AgentRoute::ExhaustiveStructural
        );
    }
    #[test]
    fn test_exhaustive_what_are_all() {
        assert_eq!(
            classify_agent_query("what are all the public API endpoints?"),
            AgentRoute::Exhaustive
        );
    }
    #[test]
    fn test_exhaustive_show_all() {
        assert_eq!(
            classify_agent_query("show all middleware registered in the app"),
            AgentRoute::Exhaustive
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Phase 4 — Architecture / ExhaustiveStructural / DeepWithExamples
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_architecture_main_components() {
        // The exact Q1 wording from agent_bench.py. v0.2 misclassified this as
        // Semantic (no architecture keyword) which led to ~40 baseline turns.
        // Phase 4 routes it to Architecture → one-call ArchOverview.
        assert_eq!(
            classify_agent_query(
                "What are the main components of this project and how do they interact? \
                 Trace the primary data flow from entry point through the core logic."
            ),
            AgentRoute::Architecture
        );
    }
    #[test]
    fn test_architecture_primary_subsystems() {
        assert_eq!(
            classify_agent_query("identify the primary subsystems"),
            AgentRoute::Architecture
        );
    }
    #[test]
    fn test_architecture_god_nodes() {
        assert_eq!(
            classify_agent_query("show the god nodes in this codebase"),
            AgentRoute::Architecture
        );
    }
    #[test]
    fn test_exhaustive_structural_config() {
        // Q4-class question; explicit config/env routing.
        assert_eq!(
            classify_agent_query(
                "list all configuration options, environment variables, \
                 and CLI flags this project supports"
            ),
            AgentRoute::ExhaustiveStructural
        );
    }
    #[test]
    fn test_exhaustive_plain_still_exhaustive() {
        // No config/CLI hint → stays as plain Exhaustive.
        assert_eq!(
            classify_agent_query("list all middleware registered in the app"),
            AgentRoute::Exhaustive
        );
    }
    #[test]
    fn test_deep_with_examples_most_complex() {
        // Q3-class question. Phase 4 routes to deep+exemplars so the agent
        // gets curated code blocks without a follow-up turn.
        assert_eq!(
            classify_agent_query(
                "explain the most complex algorithm or data transformation in \
                 this codebase step by step"
            ),
            AgentRoute::DeepWithExamples
        );
    }
    #[test]
    fn test_deep_with_examples_show_me_how() {
        assert_eq!(
            classify_agent_query("show me how the retry backoff is implemented"),
            AgentRoute::DeepWithExamples
        );
    }

    #[test]
    fn test_semantic_default() {
        assert_eq!(
            classify_agent_query("authentication middleware"),
            AgentRoute::Semantic
        );
    }
    #[test]
    fn test_semantic_multi() {
        assert_eq!(
            classify_agent_query("database connection pool"),
            AgentRoute::Semantic
        );
    }
    #[test]
    fn test_empty() {
        assert_eq!(classify_agent_query(""), AgentRoute::Semantic);
    }

    #[test]
    fn test_extract_backtick() {
        assert_eq!(
            extract_symbol("who calls `authenticate`"),
            Some("authenticate".into())
        );
    }
    #[test]
    fn test_extract_camel() {
        assert_eq!(
            extract_symbol("callers of AuthService"),
            Some("AuthService".into())
        );
    }
    #[test]
    fn test_extract_snake() {
        assert_eq!(
            extract_symbol("what uses handle_request"),
            Some("handle_request".into())
        );
    }
    #[test]
    fn test_extract_none() {
        assert_eq!(extract_symbol("show me the auth flow"), None);
    }
    #[test]
    fn test_extract_dot() {
        assert_eq!(
            extract_symbol("who calls `auth.service`"),
            Some("auth.service".into())
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // v0.3.1 Item 2 investigation — release-sequence §4.2
    // ────────────────────────────────────────────────────────────────────
    //
    // The v0.3.1 spec hypothesized that the platform Q2 +69% CCB regression
    // came from `Structural` over-matching on the multi-language repo. The
    // amended gate in `docs/RELEASE-SEQUENCE-2026-05.md` §4.2 requires running
    // the classifier on the EXACT Q2 wording from `benchmarks/agent_bench.py`
    // BEFORE writing any production code. The result determines whether the
    // proposed `detect_languages` override is warranted.
    //
    // Q2 wording (verbatim from `benchmarks/agent_bench.py::QUESTIONS`):
    //   "How does this project handle errors? What patterns are used for
    //    error propagation, reporting, and recovery?"
    //
    // Walking the classifier:
    //   - Not FilePattern (no `*`, no mid-token `?`).
    //   - Not Regex.
    //   - Not ExactSymbol (whitespace present).
    //   - Not Architecture (no architecture keyword matches).
    //   - Not Structural (none of: callers/callees/who calls/used by/uses/
    //     depends on/references/imports/etc. appear in the query).
    //   - Not DeepWithExamples (no marker matches).
    //   - Deep prefix `"how "` matches → AgentRoute::Deep.
    //
    // Conclusion (§4.2 branch (a)): The classifier ALREADY routes platform
    // Q2 to Deep. The proposed `detect_languages` override is NOT warranted
    // — the regression source is downstream (likely the Deep handler on
    // multi-language repos), and Tier 2 Item 5 already owns deep-audit work.
    // No production change in this workstream.
    #[test]
    fn q2_already_routes_to_deep_so_no_classifier_fix_needed() {
        let q2_exact_wording = "How does this project handle errors? What \
                                patterns are used for error propagation, \
                                reporting, and recovery?";
        assert_eq!(
            classify_agent_query(q2_exact_wording),
            AgentRoute::Deep,
            "Q2 must route to Deep; if this changes, re-evaluate v0.3.1 \
             Item 2 per release-sequence §4.2"
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // v0.6 Item 10 — FeaturePlanning classifier
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn fp_if_i_wanted_to_add() {
        assert_eq!(
            classify_agent_query("if I wanted to add logging, what would change?"),
            AgentRoute::FeaturePlanning
        );
    }

    #[test]
    fn fp_how_do_i_add() {
        assert_eq!(
            classify_agent_query("how do I add a new transport"),
            AgentRoute::FeaturePlanning
        );
    }

    #[test]
    fn fp_what_files_would_change_to() {
        assert_eq!(
            classify_agent_query("what files would change to support multi-tenant tables"),
            AgentRoute::FeaturePlanning
        );
    }

    #[test]
    fn fp_what_files_would_need_to() {
        assert_eq!(
            classify_agent_query("what files would need to be updated to add tracing"),
            AgentRoute::FeaturePlanning
        );
    }

    #[test]
    fn fp_does_not_match_unrelated_how_question() {
        // "how does X work" must keep routing to Deep, not FeaturePlanning.
        assert_eq!(
            classify_agent_query("how does the cache work"),
            AgentRoute::Deep
        );
    }

    #[test]
    fn fp_does_not_match_plain_architecture_question() {
        assert_eq!(
            classify_agent_query("what are the main components of this project"),
            AgentRoute::Architecture
        );
    }

    #[test]
    fn fp_does_not_match_structural_question() {
        assert_eq!(
            classify_agent_query("who calls authenticate"),
            AgentRoute::Structural
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // v0.6 Item 9 — classify_with_llm fallback behaviour
    // ────────────────────────────────────────────────────────────────────

    /// Stub LLM that always returns FeaturePlanning. Used to assert the
    /// LLM result wins when the LLM returns Ok.
    struct ConstClassifier(AgentRoute);
    impl crate::llm::LlmClassifier for ConstClassifier {
        fn classify(&self, _query: &str) -> anyhow::Result<AgentRoute> {
            Ok(self.0)
        }
    }
    /// Stub LLM that always errors. Used to assert keyword fallback.
    struct ErrClassifier;
    impl crate::llm::LlmClassifier for ErrClassifier {
        fn classify(&self, _query: &str) -> anyhow::Result<AgentRoute> {
            Err(anyhow::anyhow!("model not loaded"))
        }
    }

    #[test]
    fn classify_with_llm_uses_llm_result_when_ok() {
        let llm = ConstClassifier(AgentRoute::FeaturePlanning);
        // Query that would normally route to Semantic.
        assert_eq!(
            classify_with_llm("authentication middleware", &llm),
            AgentRoute::FeaturePlanning
        );
    }

    #[test]
    fn classify_with_llm_falls_back_on_error() {
        let llm = ErrClassifier;
        // Query that the keyword classifier routes to Deep.
        assert_eq!(
            classify_with_llm("how does authentication work?", &llm),
            AgentRoute::Deep
        );
    }
}
