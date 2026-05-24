use std::collections::HashMap;
use std::sync::LazyLock;

/// Stem a word using the Snowball English stemmer.
fn stem(word: &str) -> String {
    use rust_stemmers::{Algorithm, Stemmer};
    let stemmer = Stemmer::create(Algorithm::English);
    stemmer.stem(word).into_owned()
}

/// Split a programming identifier into NL-ish lowercased subwords.
///
/// Handles camelCase, PascalCase, snake_case, kebab-case, dot.path, and
/// `::`-separated paths. Returns `None` if the token isn't an identifier
/// (no alpha-numeric content) or doesn't decompose into multiple subwords.
///
/// Examples:
/// - `getUserById`     → `["get", "user", "by", "id"]`
/// - `RetryPolicy`     → `["retry", "policy"]`
/// - `max_retry_count` → `["max", "retry", "count"]`
/// - `HTTPSConnection` → `["https", "connection"]`
/// - `user`            → `None` (single word — nothing to split)
/// - `the`             → `None` (no internal boundary signal)
///
/// E4: surfacing subwords lets the BM25 expanded-query channel match
/// NL phrasing that would otherwise miss compound identifiers.
pub fn split_identifier(token: &str) -> Option<Vec<String>> {
    // Need some alphanumeric content.
    if !token.chars().any(char::is_alphanumeric) {
        return None;
    }

    // Pre-split on common separators (snake_case, kebab-case, dots, ::, slashes).
    // We treat any sequence of non-alphanumerics as a boundary.
    let mut pieces: Vec<&str> = token
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();

    // If no separator-driven splits happened, the single token still needs
    // camelCase/PascalCase decomposition.
    let mut subwords: Vec<String> = Vec::new();
    if pieces.len() == 1 {
        subwords.extend(split_camel_case(pieces.remove(0)));
    } else {
        for piece in pieces {
            subwords.extend(split_camel_case(piece));
        }
    }

    if subwords.len() < 2 {
        return None;
    }
    Some(subwords)
}

/// Split a single mixed-case run into lowercased subwords.
/// Handles camelCase, PascalCase, and ALL-CAPS-followed-by-camelCase runs
/// (e.g. `HTTPSConnection` → `https connection`).
fn split_camel_case(run: &str) -> Vec<String> {
    if run.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = run.chars().collect();
    let mut boundaries: Vec<usize> = vec![0];

    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let curr = chars[i];

        // Standard camelCase: lowercase/digit followed by uppercase.
        let lower_to_upper = (prev.is_lowercase() || prev.is_ascii_digit()) && curr.is_uppercase();

        // Acronym boundary: uppercase followed by uppercase-then-lowercase
        // (e.g. `HTTPSConnection` → split between `HTTPS` and `Connection`).
        let acronym_boundary = prev.is_uppercase()
            && curr.is_uppercase()
            && i + 1 < chars.len()
            && chars[i + 1].is_lowercase();

        // Alpha-numeric transition: letter to digit or digit to letter.
        let alpha_digit = (prev.is_alphabetic() && curr.is_ascii_digit())
            || (prev.is_ascii_digit() && curr.is_alphabetic());

        if lower_to_upper || acronym_boundary || alpha_digit {
            boundaries.push(i);
        }
    }
    boundaries.push(chars.len());

    let mut out = Vec::with_capacity(boundaries.len().saturating_sub(1));
    for window in boundaries.windows(2) {
        let piece: String = chars[window[0]..window[1]]
            .iter()
            .collect::<String>()
            .to_lowercase();
        if !piece.is_empty() {
            out.push(piece);
        }
    }
    out
}

/// Build the stemmed synonym table: keys are stemmed (single words or
/// space-joined stemmed multi-word phrases) so that inflected query words
/// match after stemming.
fn build_stemmed_synonyms(
    raw: &'static [(&'static str, &'static [&'static str])],
) -> HashMap<String, Vec<&'static str>> {
    let mut m: HashMap<String, Vec<&'static str>> = HashMap::new();
    for &(key, values) in raw {
        let stemmed_key: String = key
            .split_whitespace()
            .map(stem)
            .collect::<Vec<_>>()
            .join(" ");
        m.insert(stemmed_key, values.to_vec());
    }
    m
}

/// Static synonym/expansion table mapping NL concepts to code tokens.
/// Used to bridge the vocabulary gap between natural-language queries and
/// code identifiers for BM25 retrieval.
/// Keys are stemmed at init time so inflected query words (e.g. "encrypting",
/// "classification") match after stemming.
static SYNONYMS: LazyLock<HashMap<String, Vec<&'static str>>> =
    LazyLock::new(|| build_stemmed_synonyms(RAW_SYNONYMS));

/// Raw synonym definitions. Keys are canonical NL forms; values are code tokens
/// to inject into the BM25 query.
const RAW_SYNONYMS: &[(&str, &[&str])] = &[
    // Concurrency patterns
    (
        "parallel",
        &[
            "concurrent",
            "async",
            "Promise.all",
            "allSettled",
            "tokio::spawn",
            "rayon",
            "par_iter",
            "ThreadPool",
            "ExecutorService",
        ],
    ),
    (
        "concurrent",
        &["parallel", "async", "thread", "mutex", "lock", "atomic"],
    ),
    (
        "async",
        &["await", "Future", "Promise", "Task", "coroutine"],
    ),
    // Error handling
    (
        "failure",
        &["error", "Error", "Result::Err", "reject", "anyhow"],
    ),
    (
        "parallel failure",
        &["Promise.allSettled", "allSettled", "Promise.all", "settled"],
    ),
    (
        "error handling",
        &[
            "try",
            "catch",
            "Result",
            "Option",
            "unwrap_or",
            "map_err",
            "context",
            "bail",
        ],
    ),
    (
        "recovery",
        &[
            "retry",
            "restart",
            "resume",
            "restore",
            "failover",
            "resilience",
            "fallback",
        ],
    ),
    (
        "fallback",
        &["default", "backup", "alternative", "rescue", "degraded"],
    ),
    (
        "retry",
        &[
            "backoff",
            "exponential",
            "retryWithBackoff",
            "max_retries",
            "attempt",
            "retry_policy",
        ],
    ),
    // Networking / connections
    (
        "connection",
        &["connect", "disconnect", "reconnect", "pool", "WebSocket"],
    ),
    (
        "lifecycle",
        &[
            "shutdown",
            "dispose",
            "cleanup",
            "close",
            "destroy",
            "initiate",
            "establish",
            "teardown",
            "open",
            "refresh",
            "sync",
        ],
    ),
    (
        "connection lifecycle",
        &[
            "connect",
            "disconnect",
            "reconnect",
            "connection_pool",
            "keep_alive",
            "heartbeat",
            "idle_timeout",
        ],
    ),
    // PII / sanitization
    (
        "pii",
        &[
            "sanitize",
            "redact",
            "mask",
            "scrub",
            "anonymize",
            "sensitive",
        ],
    ),
    (
        "redact",
        &[
            "sanitize",
            "mask",
            "scrub",
            "anonymize",
            "censor",
            "obfuscate",
            "strip",
        ],
    ),
    // Auth patterns
    (
        "authentication",
        &[
            "auth",
            "login",
            "logout",
            "session",
            "token",
            "JWT",
            "OAuth",
            "passport",
            "credential",
        ],
    ),
    (
        "authorization",
        &[
            "permission",
            "role",
            "access_control",
            "ACL",
            "RBAC",
            "guard",
            "policy",
        ],
    ),
    // Data patterns
    (
        "validation",
        &[
            "validate",
            "schema",
            "check",
            "verify",
            "sanitize",
            "constraint",
            "Zod",
            "Joi",
            "pydantic",
        ],
    ),
    (
        "serialization",
        &[
            "serialize",
            "deserialize",
            "marshal",
            "unmarshal",
            "encode",
            "decode",
            "serde",
            "JSON.parse",
            "JSON.stringify",
        ],
    ),
    (
        "caching",
        &[
            "cache",
            "memoize",
            "memo",
            "Redis",
            "LRU",
            "TTL",
            "invalidate",
            "cache_control",
        ],
    ),
    // Common programming patterns
    (
        "sql",
        &[
            "SELECT",
            "WHERE",
            "INSERT",
            "QueryBuilder",
            "ORM",
            "knex",
            "prisma",
        ],
    ),
    (
        "query builder",
        &["QueryBuilder", "buildQuery", "createQuery", "parameterized"],
    ),
    (
        "encryption",
        &["encrypt", "decrypt", "cipher", "KMS", "aes", "crypto"],
    ),
    (
        "token",
        &[
            "JWT",
            "accessToken",
            "refreshToken",
            "bearer",
            "OAuth",
            "credential",
        ],
    ),
    (
        "factory",
        &["Factory", "Provider", "builder", "AbstractFactory"],
    ),
    (
        "interceptor",
        &["middleware", "hook", "filter", "guard", "Interceptor"],
    ),
    ("sync", &["synchronize", "replicate", "mirror", "reconcile"]),
    // Architecture
    (
        "middleware",
        &[
            "interceptor",
            "filter",
            "guard",
            "pipe",
            "beforeEach",
            "use",
            "handler",
        ],
    ),
    (
        "dependency injection",
        &[
            "inject",
            "provider",
            "container",
            "IoC",
            "DI",
            "service_locator",
            "@Injectable",
        ],
    ),
    (
        "state management",
        &[
            "store", "reducer", "dispatch", "action", "selector", "Riverpod", "BLoC", "Redux",
            "Vuex",
        ],
    ),
    // Testing
    (
        "mock",
        &[
            "stub",
            "fake",
            "spy",
            "double",
            "mockito",
            "jest.mock",
            "patch",
            "monkeypatch",
        ],
    ),
    (
        "test",
        &[
            "spec",
            "assert",
            "expect",
            "should",
            "describe",
            "it",
            "#[test]",
            "def test_",
        ],
    ),
    // E4 — additional universal code↔NL pairs. Each entry passes:
    //   (1) Used across multiple languages
    //   (2) Helps a random project (not test-repo metadata)
    //   (3) Specific enough to avoid false positives
    //   (4) Not in the over-generic banned list (per CLAUDE.md)
    (
        "thread safety",
        &[
            "Mutex",
            "RwLock",
            "atomic",
            "Arc",
            "synchronized",
            "lock",
            "guarded",
            "thread_local",
        ],
    ),
    (
        "race condition",
        &[
            "atomic",
            "compare_and_swap",
            "CAS",
            "Mutex",
            "ordering",
            "fence",
            "happens_before",
        ],
    ),
    (
        "memory leak",
        &[
            "leak", "drop", "Drop", "dispose", "Weak", "finalize", "refcount", "free",
        ],
    ),
    (
        "timeout",
        &[
            "deadline",
            "expires",
            "TTL",
            "Duration",
            "cancel",
            "abort",
            "WithTimeout",
            "set_timeout",
        ],
    ),
    (
        "logging",
        &[
            "log",
            "logger",
            "info!",
            "debug!",
            "warn!",
            "tracing",
            "println",
            "console.log",
            "Logger",
            "slf4j",
        ],
    ),
    (
        "rate limit",
        &[
            "throttle",
            "debounce",
            "quota",
            "limiter",
            "RateLimiter",
            "max_per_second",
            "bucket",
        ],
    ),
    (
        "iterator",
        &[
            "iter",
            "Iterator",
            "next",
            "for_each",
            "foreach",
            "enumerate",
            "Iter",
            "yield",
        ],
    ),
    (
        "pagination",
        &[
            "paginate",
            "page",
            "offset",
            "limit",
            "cursor",
            "next_page",
            "page_size",
            "has_more",
        ],
    ),
    (
        "subscription",
        &[
            "subscribe",
            "unsubscribe",
            "publisher",
            "subscriber",
            "observable",
            "listen",
            "emit",
        ],
    ),
    (
        "event handler",
        &[
            "on_event",
            "handle",
            "listener",
            "callback",
            "addEventListener",
            "EventHandler",
            "subscribe",
        ],
    ),
    (
        "graceful shutdown",
        &[
            "shutdown",
            "SIGTERM",
            "SIGINT",
            "cleanup",
            "drain",
            "close",
            "abort_handle",
            "cancellation",
        ],
    ),
    (
        "feature flag",
        &[
            "feature_flag",
            "flag",
            "toggle",
            "rollout",
            "experiment",
            "if_enabled",
            "is_enabled",
        ],
    ),
    (
        "circuit breaker",
        &[
            "circuit_breaker",
            "CircuitBreaker",
            "breaker",
            "half_open",
            "trip",
            "failover",
        ],
    ),
    (
        "background job",
        &[
            "worker",
            "job",
            "queue",
            "scheduler",
            "cron",
            "tokio::spawn",
            "BackgroundService",
            "celery",
        ],
    ),
    (
        "telemetry",
        &[
            "trace",
            "span",
            "metric",
            "OpenTelemetry",
            "OTEL",
            "Histogram",
            "Counter",
            "instrument",
        ],
    ),
];

/// Expand a semantic query with relevant code synonyms.
///
/// Returns `None` if no expansion is applicable (query is a single word
/// with no internal identifier structure and no synonym matches).
///
/// Query words are stemmed before lookup so that inflected forms
/// (e.g. "encrypting", "classification", "tokens") match canonical keys
/// (e.g. "encryption", "classify", "token").
///
/// E4 — additionally splits identifier-like tokens (camelCase, snake_case)
/// into NL subwords (`getUserById` → `get user by id`) so that queries
/// containing compound identifiers also surface natural-language phrasing.
pub fn expand_query(query: &str) -> Option<String> {
    // Use the original (case-preserving) tokens for identifier splitting,
    // but lowercase for synonym lookup.
    let raw_words: Vec<&str> = query.split_whitespace().collect();
    let query_lower = query.to_lowercase();
    let words: Vec<&str> = query_lower.split_whitespace().collect();

    // E4 identifier-split additions — collected before the early-return so
    // single-token identifier-like queries (e.g. "getUserById") still expand.
    let mut id_expansions: Vec<String> = Vec::new();
    for raw in &raw_words {
        if let Some(subwords) = split_identifier(raw) {
            id_expansions.extend(subwords);
        }
    }

    // For single-word queries: only expand if identifier-splitting produced
    // something. Synonym lookup needs at least two tokens to consider
    // multi-word matches; single-word synonym lookup would be too broad.
    if words.len() < 2 && id_expansions.is_empty() {
        return None;
    }

    // Stem each word once for reuse in lookups
    let stemmed_words: Vec<String> = words.iter().map(|w| stem(w)).collect();

    let mut expansions: Vec<&str> = Vec::new();

    // Try multi-word matches first (higher priority), using stemmed words.
    // Also try the raw (unstemmed) form as a fallback — some multi-word keys
    // may not round-trip perfectly through per-word stemming.
    let max_window = stemmed_words.len().min(3);
    for window_size in (2..=max_window).rev() {
        for (i, window) in stemmed_words.windows(window_size).enumerate() {
            let phrase = window.join(" ");
            if let Some(synonyms) = SYNONYMS.get(&phrase) {
                expansions.extend(synonyms.iter());
            } else {
                // Fallback: try raw (unstemmed, lowercase) window
                let raw_phrase = words[i..i + window_size].join(" ");
                if let Some(synonyms) = SYNONYMS.get(&raw_phrase) {
                    expansions.extend(synonyms.iter());
                }
            }
        }
    }

    // Then single-word matches using stemmed words
    for stemmed in &stemmed_words {
        if let Some(synonyms) = SYNONYMS.get(stemmed.as_str()) {
            expansions.extend(synonyms.iter());
        }
    }

    if expansions.is_empty() && id_expansions.is_empty() {
        return None;
    }

    // Deduplicate string-slice expansions
    expansions.sort_unstable();
    expansions.dedup();

    // Deduplicate identifier subword expansions
    id_expansions.sort();
    id_expansions.dedup();

    // Combine: original query + synonym tokens + identifier subwords.
    // The order doesn't affect BM25 outcomes; we put synonyms before
    // subwords for readability.
    let mut tail: Vec<String> = expansions.iter().map(|s| (*s).to_string()).collect();
    tail.extend(id_expansions);

    Some(format!("{} {}", query, tail.join(" ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_parallel_failure_handling() {
        let result = expand_query("parallel failure handling").expect("should expand");
        // Multi-word "parallel failure" match (stemmed: "parallel failur")
        assert!(
            result.contains("Promise.allSettled"),
            "should contain Promise.allSettled"
        );
        assert!(result.contains("allSettled"), "should contain allSettled");
        assert!(result.contains("settled"), "should contain settled");
        // Single-word "parallel" match
        assert!(result.contains("concurrent"), "should contain concurrent");
        // Single-word "failure" match (stemmed "failur" matches key "failure" → "failur")
        assert!(result.contains("error"), "should contain error");
        assert!(result.contains("reject"), "should contain reject");
    }

    #[test]
    fn expand_connection_lifecycle() {
        let result = expand_query("connection lifecycle").expect("should expand");
        assert!(result.contains("connect"), "should contain connect");
        assert!(result.contains("disconnect"), "should contain disconnect");
        assert!(
            result.contains("heartbeat"),
            "should contain heartbeat from multi-word match"
        );
    }

    #[test]
    fn single_word_returns_none() {
        assert!(expand_query("x").is_none());
    }

    #[test]
    fn no_matching_synonyms_returns_none() {
        assert!(expand_query("find the bug").is_none());
    }

    #[test]
    fn expansions_are_deduplicated() {
        let result = expand_query("parallel concurrent tasks").expect("should expand");
        // "async" appears in both "parallel" and "concurrent" synonym lists
        let async_count = result.matches("async").count();
        // The original query doesn't contain "async", so it should appear exactly once
        assert_eq!(
            async_count, 1,
            "async should appear exactly once (deduplicated)"
        );
    }

    #[test]
    fn stemmed_lookup_encrypting_matches_encryption() {
        // "encrypting" stems to "encrypt", same as "encryption" → key matches
        let result = expand_query("encrypting sensitive data").expect("should expand");
        assert!(
            result.contains("KMS"),
            "should contain KMS from encryption synonyms"
        );
        assert!(result.contains("cipher"), "should contain cipher");
        assert!(result.contains("decrypt"), "should contain decrypt");
    }

    #[test]
    fn stemmed_lookup_tokens_matches_token() {
        // "tokens" stems to "token", matching the "token" key
        let result = expand_query("OAuth tokens refresh").expect("should expand");
        assert!(
            result.contains("JWT"),
            "should contain JWT from token synonyms"
        );
        assert!(
            result.contains("refreshToken"),
            "should contain refreshToken"
        );
    }

    #[test]
    fn stemmed_lookup_classifying_matches_synonyms() {
        // "classifying" stems to "classifi" — no direct key, but validates no crash
        // "validation" stems to "valid" which matches "validation" key → "valid"
        let result = expand_query("validating input data");
        assert!(
            result.is_some(),
            "validating should match validation key via stemming"
        );
        let r = result.expect("should expand");
        assert!(
            r.contains("sanitize"),
            "should contain sanitize from validation synonyms"
        );
    }

    // =================================================================
    // E4 — split_identifier (camelCase / snake_case decomposition)
    // =================================================================

    #[test]
    fn split_camel_case_basic() {
        let parts = split_identifier("getUserById").expect("should split");
        assert_eq!(parts, vec!["get", "user", "by", "id"]);
    }

    #[test]
    fn split_pascal_case() {
        let parts = split_identifier("RetryPolicy").expect("should split");
        assert_eq!(parts, vec!["retry", "policy"]);
    }

    #[test]
    fn split_snake_case() {
        let parts = split_identifier("max_retry_count").expect("should split");
        assert_eq!(parts, vec!["max", "retry", "count"]);
    }

    #[test]
    fn split_acronym_then_word() {
        // HTTPS|Connection — the boundary is between the last upper of the
        // acronym run and the first upper of the next word.
        let parts = split_identifier("HTTPSConnection").expect("should split");
        assert_eq!(parts, vec!["https", "connection"]);
    }

    #[test]
    fn split_alpha_digit() {
        // Words containing digits should split at the alpha-digit boundary.
        let parts = split_identifier("OAuth2Provider").expect("should split");
        // O|Auth|2|Provider — alpha-digit and lower-upper boundaries
        assert!(parts.contains(&"o".to_string()));
        assert!(parts.contains(&"auth".to_string()));
        assert!(parts.contains(&"2".to_string()));
        assert!(parts.contains(&"provider".to_string()));
    }

    #[test]
    fn split_kebab_case() {
        let parts = split_identifier("retry-with-backoff").expect("should split");
        assert_eq!(parts, vec!["retry", "with", "backoff"]);
    }

    #[test]
    fn split_dot_path() {
        let parts = split_identifier("com.example.UserService").expect("should split");
        assert_eq!(parts, vec!["com", "example", "user", "service"]);
    }

    #[test]
    fn split_double_colon_path() {
        let parts = split_identifier("std::io::Read").expect("should split");
        assert_eq!(parts, vec!["std", "io", "read"]);
    }

    #[test]
    fn split_single_lowercase_word_returns_none() {
        // A plain English word doesn't decompose — must return None
        assert!(split_identifier("user").is_none());
        assert!(split_identifier("the").is_none());
    }

    #[test]
    fn split_non_alphanumeric_returns_none() {
        assert!(split_identifier("---").is_none());
        assert!(split_identifier("").is_none());
    }

    // =================================================================
    // E4 — expand_query: identifier splitting integration
    // =================================================================

    #[test]
    fn expand_query_splits_identifier_in_multiword() {
        // A query mixing prose and a compound identifier should include the
        // subwords as expansions.
        let result = expand_query("how does getUserById work").expect("should expand");
        // Identifier subwords appear
        assert!(result.contains(" get "), "subword 'get' should appear");
        assert!(result.contains(" user"), "subword 'user' should appear");
        assert!(result.contains(" by "), "subword 'by' should appear");
        assert!(result.contains(" id"), "subword 'id' should appear");
    }

    #[test]
    fn expand_query_splits_single_identifier_only() {
        // Single-word non-identifier still returns None.
        assert!(expand_query("hello").is_none());
        // Single-word identifier IS now expanded (camelCase subwords).
        let result = expand_query("RetryPolicy");
        assert!(
            result.is_some(),
            "single-word identifier should expand via subwords"
        );
        let r = result.expect("should expand");
        assert!(r.contains("retry"));
        assert!(r.contains("policy"));
    }

    // =================================================================
    // E4 — new universal NL pairs (sanity)
    // =================================================================

    #[test]
    fn expand_thread_safety_pair() {
        let r = expand_query("thread safety concerns").expect("should expand");
        assert!(r.contains("Mutex"));
        assert!(r.contains("atomic"));
    }

    #[test]
    fn expand_rate_limit_pair() {
        let r = expand_query("rate limit api calls").expect("should expand");
        assert!(r.contains("throttle"));
        assert!(r.contains("RateLimiter"));
    }

    #[test]
    fn expand_graceful_shutdown_pair() {
        let r = expand_query("graceful shutdown sequence").expect("should expand");
        assert!(r.contains("SIGTERM"));
        assert!(r.contains("drain"));
    }

    #[test]
    fn expand_memory_leak_pair() {
        let r = expand_query("debug memory leak").expect("should expand");
        assert!(r.contains("Weak"));
        assert!(r.contains("dispose"));
    }

    #[test]
    fn expand_telemetry_pair() {
        let r = expand_query("add telemetry to handler").expect("should expand");
        assert!(r.contains("trace"));
        assert!(r.contains("OpenTelemetry"));
    }

    #[test]
    fn expand_dual_route_differs_from_single_route() {
        // E4 design intent: expansion meaningfully changes the BM25 query
        // text. Verify that the expanded form is a strict superset of the
        // original (original is preserved as prefix) and adds tokens not
        // present in the input.
        let original = "thread safety check";
        let expanded = expand_query(original).expect("should expand");
        assert!(
            expanded.starts_with(original),
            "expanded should preserve original prefix"
        );
        // Expanded has more than just the original
        assert!(
            expanded.len() > original.len(),
            "expanded should be strictly longer than original"
        );
        // Specific token from synonyms
        assert!(expanded.contains("Mutex"));
    }
}
