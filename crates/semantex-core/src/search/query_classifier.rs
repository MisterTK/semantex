/// Query-adaptive search strategy: classify queries to tune RRF weights.
///
/// Identifier queries (e.g. `getUserById`, `get_user_by_id`) should heavily
/// favour the BM25 (sparse) path because exact token matches are critical.
/// Free-form natural-language queries benefit more from dense (semantic) search.
/// The kind of query the user issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// A single programming identifier (camelCase, snake_case, dot.path, etc.)
    Identifier,
    /// A short keyword/symbol query (1-2 tokens, mostly alphanumeric)
    Keyword,
    /// A natural-language / semantic question (3+ words, English prose)
    Semantic,
    /// Mix of identifiers and natural language
    Mixed,
}

/// Per-query weights for weighted RRF fusion.
#[derive(Debug, Clone, Copy)]
pub struct FusionWeights {
    pub w_dense: f32,
    pub w_sparse: f32,
}

impl QueryType {
    /// Return the recommended dense/sparse weights for RRF fusion.
    pub fn fusion_weights(self) -> FusionWeights {
        match self {
            QueryType::Identifier => FusionWeights {
                w_dense: 0.2,
                w_sparse: 1.0,
            },
            QueryType::Keyword => FusionWeights {
                w_dense: 0.4,
                w_sparse: 0.8,
            },
            QueryType::Semantic => FusionWeights {
                w_dense: 0.1,
                w_sparse: 0.9,
            },
            QueryType::Mixed => FusionWeights {
                w_dense: 0.6,
                w_sparse: 0.6,
            },
        }
    }
}

/// Classify a query string into a `QueryType`.
pub fn classify(query: &str) -> QueryType {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return QueryType::Keyword;
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();

    // Single token — check if it looks like a programming identifier
    if tokens.len() == 1 {
        let tok = tokens[0];
        if is_identifier_like(tok) {
            return QueryType::Identifier;
        }
        return QueryType::Keyword;
    }

    // Multi-token: count how many tokens look like identifiers
    let ident_count = tokens.iter().filter(|t| is_identifier_like(t)).count();
    let total = tokens.len();

    if ident_count == total {
        // All tokens are identifiers — still identifier-heavy
        return QueryType::Identifier;
    }

    if total <= 2 {
        if ident_count > 0 {
            return QueryType::Mixed;
        }
        return QueryType::Keyword;
    }

    // 3+ tokens
    if ident_count == 0 {
        return QueryType::Semantic;
    }

    // Some identifiers mixed with prose
    let ident_ratio = ident_count as f32 / total as f32;
    if ident_ratio >= 0.5 {
        QueryType::Mixed
    } else {
        QueryType::Semantic
    }
}

/// Does a token look like a programming identifier?
///
/// Heuristics:
/// - Contains an underscore (snake_case)
/// - Has interior uppercase after a lowercase (camelCase / PascalCase)
/// - Contains a dot separating segments (qualified.name)
/// - Contains `::` (Rust path separator)
/// - Contains `->` or `=>` (operator-like)
pub(crate) fn is_identifier_like(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }

    // Must contain at least one alphanumeric character
    if !token.chars().any(char::is_alphanumeric) {
        return false;
    }

    // snake_case: contains underscore between alnum
    if token.contains('_') && token.len() > 1 {
        return true;
    }

    // Qualified path: contains dot between alphanumeric segments
    if token.contains('.') && token.split('.').all(|s| !s.is_empty()) {
        return true;
    }

    // Rust/C++ path: contains ::
    if token.contains("::") {
        return true;
    }

    // camelCase / PascalCase: a lowercase letter followed by an uppercase letter
    if is_camel_case(token) {
        return true;
    }

    // ALL_CAPS constant (at least 2 chars, all uppercase + underscores + digits)
    if token.len() >= 2
        && token
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    {
        return true;
    }

    false
}

/// Check if a string uses camelCase or PascalCase.
pub(crate) fn is_camel_case(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    for i in 1..chars.len() {
        if chars[i - 1].is_lowercase() && chars[i].is_uppercase() {
            return true;
        }
    }
    false
}

/// True if the query expresses an *exhaustive* intent — "find every / all
/// occurrences / callers / usages everywhere". Such queries benefit from 2-hop
/// graph expansion so transitively related sites surface. Universal English
/// signals only (repo-agnostic).
#[must_use]
pub fn is_exhaustive_query(query: &str) -> bool {
    const SIGNALS: &[&str] = &[
        "all usages",
        "all callers",
        "all references",
        "all places",
        "all the places",
        "every place",
        "every usage",
        "every caller",
        "everywhere",
        "find all",
        "list all",
        "all occurrences",
    ];
    let q = query.to_lowercase();
    SIGNALS.iter().any(|s| q.contains(s))
}

/// True if the query expresses a *feature-planning / change-impact* intent —
/// "where should I add / implement / wire up X". These benefit from 2-hop
/// expansion to reveal the surrounding integration surface. Universal English
/// signals only (repo-agnostic).
#[must_use]
pub fn is_feature_planning_query(query: &str) -> bool {
    let q = query.to_lowercase();
    // Require a location/how interrogative paired with an add/implement intent
    // so we don't fire on every "how to" or "where is" question.
    let asks_where = q.contains("where should")
        || q.contains("where to")
        || q.contains("where would")
        || q.contains("where do i")
        || q.contains("where can i")
        || q.contains("how to")
        || q.contains("how do i");
    let intent_add = q.contains("add")
        || q.contains("implement")
        || q.contains("wire up")
        || q.contains("hook in")
        || q.contains("hook up")
        || q.contains("introduce")
        || q.contains("integrate");
    asks_where && intent_add
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- is_identifier_like ---

    #[test]
    fn test_snake_case() {
        assert!(is_identifier_like("get_user"));
        assert!(is_identifier_like("_private"));
        assert!(is_identifier_like("MAX_RETRIES"));
    }

    #[test]
    fn test_camel_case() {
        assert!(is_identifier_like("fhirBaseUrl"));
        assert!(is_identifier_like("getUserById"));
        assert!(is_identifier_like("StringBuilder"));
    }

    #[test]
    fn test_dot_path() {
        assert!(is_identifier_like("std.io"));
        assert!(is_identifier_like("com.example.App"));
    }

    #[test]
    fn test_rust_path() {
        assert!(is_identifier_like("std::io::Read"));
    }

    #[test]
    fn test_all_caps_constant() {
        assert!(is_identifier_like("HTTP"));
        assert!(is_identifier_like("MAX_SIZE"));
    }

    #[test]
    fn test_plain_words_not_identifiers() {
        assert!(!is_identifier_like("hello"));
        assert!(!is_identifier_like("search"));
        assert!(!is_identifier_like("the"));
        assert!(!is_identifier_like("a"));
    }

    #[test]
    fn test_empty() {
        assert!(!is_identifier_like(""));
    }

    // --- classify ---

    #[test]
    fn test_classify_single_identifier() {
        assert_eq!(classify("fhirBaseUrl"), QueryType::Identifier);
        assert_eq!(classify("get_user_by_id"), QueryType::Identifier);
        assert_eq!(classify("std::io::Read"), QueryType::Identifier);
        assert_eq!(classify("com.example.App"), QueryType::Identifier);
    }

    #[test]
    fn test_classify_single_keyword() {
        assert_eq!(classify("search"), QueryType::Keyword);
        assert_eq!(classify("error"), QueryType::Keyword);
    }

    #[test]
    fn test_classify_semantic() {
        assert_eq!(
            classify("how does the search engine work"),
            QueryType::Semantic
        );
        assert_eq!(
            classify("find all database connection errors"),
            QueryType::Semantic
        );
    }

    #[test]
    fn test_classify_mixed() {
        assert_eq!(classify("getUserById method"), QueryType::Mixed);
    }

    #[test]
    fn test_classify_multiple_identifiers() {
        assert_eq!(classify("fhirBaseUrl getPatient"), QueryType::Identifier);
    }

    #[test]
    fn test_classify_empty() {
        assert_eq!(classify(""), QueryType::Keyword);
        assert_eq!(classify("   "), QueryType::Keyword);
    }

    // --- FusionWeights ---

    #[test]
    fn test_identifier_weights_favour_sparse() {
        let w = QueryType::Identifier.fusion_weights();
        assert!(w.w_sparse > w.w_dense);
    }

    #[test]
    fn test_semantic_weights_favour_sparse() {
        let w = QueryType::Semantic.fusion_weights();
        assert!(w.w_sparse > w.w_dense);
    }

    #[test]
    fn test_mixed_weights_balanced() {
        let w = QueryType::Mixed.fusion_weights();
        assert!((w.w_dense - w.w_sparse).abs() < f32::EPSILON);
    }

    // --- route predicates (S4) ---

    #[test]
    fn test_exhaustive_query_detection() {
        assert!(is_exhaustive_query("find all usages of the retry helper"));
        assert!(is_exhaustive_query(
            "every place that reads the config file"
        ));
        assert!(is_exhaustive_query(
            "everywhere we open a database connection"
        ));
        assert!(is_exhaustive_query(
            "list all callers of the auth middleware"
        ));
    }

    #[test]
    fn test_non_exhaustive_query() {
        assert!(!is_exhaustive_query("login handler"));
        assert!(!is_exhaustive_query("getUserById"));
        assert!(!is_exhaustive_query("parse the request body"));
    }

    #[test]
    fn test_feature_planning_query_detection() {
        assert!(is_feature_planning_query(
            "where should I add rate limiting"
        ));
        assert!(is_feature_planning_query(
            "where to implement the new export endpoint"
        ));
        assert!(is_feature_planning_query(
            "how to wire up a second cache layer"
        ));
        assert!(is_feature_planning_query(
            "where would I hook in a metrics counter"
        ));
    }

    #[test]
    fn test_non_feature_planning_query() {
        assert!(!is_feature_planning_query("connection pool"));
        assert!(!is_feature_planning_query("error handling in the parser"));
        assert!(!is_feature_planning_query("std::io::Read"));
    }
}
