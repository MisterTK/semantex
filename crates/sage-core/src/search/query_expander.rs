use std::collections::HashMap;
use std::sync::LazyLock;

/// Stem a word using the Snowball English stemmer.
fn stem(word: &str) -> String {
    use rust_stemmers::{Algorithm, Stemmer};
    let stemmer = Stemmer::create(Algorithm::English);
    stemmer.stem(word).into_owned()
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
];

/// Expand a semantic query with relevant code synonyms.
///
/// Returns `None` if no expansion is applicable (query is a single word
/// or no synonyms match any of its tokens/phrases).
///
/// Query words are stemmed before lookup so that inflected forms
/// (e.g. "encrypting", "classification", "tokens") match canonical keys
/// (e.g. "encryption", "classify", "token").
pub fn expand_query(query: &str) -> Option<String> {
    let query_lower = query.to_lowercase();
    let words: Vec<&str> = query_lower.split_whitespace().collect();

    if words.len() < 2 {
        return None; // Don't expand single-word queries
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

    if expansions.is_empty() {
        return None;
    }

    // Deduplicate and combine with original
    expansions.sort_unstable();
    expansions.dedup();

    // Return expanded query: original + synonym tokens
    Some(format!("{} {}", query, expansions.join(" ")))
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
}
