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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRoute {
    FilePattern,
    Regex,
    ExactSymbol,
    Structural,
    Deep,
    Analytical,
    Semantic,
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
            Self::Semantic => write!(f, "semantic"),
        }
    }
}

/// Returns true if the query looks like a regex pattern.
fn is_regex(query: &str) -> bool {
    // \b \B \d \D \s \S \w \W
    let chars: Vec<char> = query.chars().collect();
    for i in 0..chars.len().saturating_sub(1) {
        if chars[i] == '\\' {
            if let Some(&next) = chars.get(i + 1) {
                if matches!(next, 'b' | 'B' | 'd' | 'D' | 's' | 'S' | 'w' | 'W') {
                    return true;
                }
            }
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
    if let Some(open) = query.find('[') {
        if query[open..].contains(']') {
            return true;
        }
    }

    // Group with quantifier (foo|bar), (foo?), (foo*), (foo+)
    if let Some(open) = query.find('(') {
        let after_open = &query[open + 1..];
        if let Some(close) = after_open.find(')') {
            let inner = &after_open[..close];
            if inner.contains('|') || inner.contains('?') || inner.contains('*') || inner.contains('+') {
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

        if !stripped.is_empty() && !stripped.contains(char::is_whitespace) {
            if was_wrapped
                || is_camel_case(stripped)
                || has_caps_prefix_symbol(stripped)
                || (stripped.contains('_') && stripped.len() > 2)
                || (stripped.contains('.') && stripped.len() > 2)
            {
                return AgentRoute::ExactSymbol;
            }
        }
    }

    // 4. Structural
    let lower = query.to_lowercase();
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

    // 7. Semantic — default
    AgentRoute::Semantic
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
        assert_eq!(classify_agent_query("**/*.test.ts"), AgentRoute::FilePattern);
    }
    #[test]
    fn test_glob_qmark() {
        assert_eq!(classify_agent_query("src/?/index.js"), AgentRoute::FilePattern);
    }
    #[test]
    fn test_glob_plain_star() {
        assert_eq!(classify_agent_query("*.rs"), AgentRoute::FilePattern);
    }
    #[test]
    fn test_trailing_qmark_not_glob() {
        assert_eq!(classify_agent_query("how does auth work?"), AgentRoute::Deep);
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
        assert_eq!(classify_agent_query("handle_request"), AgentRoute::ExactSymbol);
    }
    #[test]
    fn test_backtick_wrap() {
        assert_eq!(classify_agent_query("`processPayment`"), AgentRoute::ExactSymbol);
    }
    #[test]
    fn test_dot_path() {
        assert_eq!(classify_agent_query("auth.middleware"), AgentRoute::ExactSymbol);
    }
    #[test]
    fn test_html_parser() {
        assert_eq!(classify_agent_query("HTMLParser"), AgentRoute::ExactSymbol);
    }
    #[test]
    fn test_xml_http() {
        assert_eq!(classify_agent_query("XMLHttpRequest"), AgentRoute::ExactSymbol);
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
        assert_eq!(classify_agent_query("who calls authenticate"), AgentRoute::Structural);
    }
    #[test]
    fn test_depends_on() {
        assert_eq!(classify_agent_query("what depends on DatabasePool"), AgentRoute::Structural);
    }
    #[test]
    fn test_callers_of() {
        assert_eq!(classify_agent_query("callers of handleRequest"), AgentRoute::Structural);
    }

    #[test]
    fn test_how_query() {
        assert_eq!(classify_agent_query("how does authentication work?"), AgentRoute::Deep);
    }
    #[test]
    fn test_explain() {
        assert_eq!(classify_agent_query("explain the payment pipeline"), AgentRoute::Deep);
    }
    #[test]
    fn test_why_query() {
        assert_eq!(classify_agent_query("why does the cache invalidate on deploy"), AgentRoute::Deep);
    }
    #[test]
    fn test_what_is_foo() {
        assert_eq!(classify_agent_query("what is foo?"), AgentRoute::Semantic);
    }

    #[test]
    fn test_most_complex() {
        assert_eq!(classify_agent_query("most complex functions in this repo"), AgentRoute::Analytical);
    }
    #[test]
    fn test_compare() {
        assert_eq!(classify_agent_query("compare auth approaches"), AgentRoute::Analytical);
    }
    #[test]
    fn test_review() {
        assert_eq!(classify_agent_query("review the error handling"), AgentRoute::Analytical);
    }

    #[test]
    fn test_semantic_default() {
        assert_eq!(classify_agent_query("authentication middleware"), AgentRoute::Semantic);
    }
    #[test]
    fn test_semantic_multi() {
        assert_eq!(classify_agent_query("database connection pool"), AgentRoute::Semantic);
    }
    #[test]
    fn test_empty() {
        assert_eq!(classify_agent_query(""), AgentRoute::Semantic);
    }

    #[test]
    fn test_extract_backtick() {
        assert_eq!(extract_symbol("who calls `authenticate`"), Some("authenticate".into()));
    }
    #[test]
    fn test_extract_camel() {
        assert_eq!(extract_symbol("callers of AuthService"), Some("AuthService".into()));
    }
    #[test]
    fn test_extract_snake() {
        assert_eq!(extract_symbol("what uses handle_request"), Some("handle_request".into()));
    }
    #[test]
    fn test_extract_none() {
        assert_eq!(extract_symbol("show me the auth flow"), None);
    }
    #[test]
    fn test_extract_dot() {
        assert_eq!(extract_symbol("who calls `auth.service`"), Some("auth.service".into()));
    }
}
