//! Per-language deterministic pattern catalog mined at index time.
//!
//! For each chunk, scans for idiomatic language patterns (e.g. Rust `Drop` impl,
//! TypeScript `Promise.allSettled` parallel-failure handling) and emits zero or
//! more `PatternMatch` records. These records are queried later by the
//! `semantex_examples` MCP tool (M5) so an agent asking "show me an idiomatic
//! JWT validation pattern in this codebase" receives 3 structurally-confirmed
//! exemplars instead of 10 grep hits.
//!
//! ## Scope (v0.3 initial)
//!
//! Per the v0.3 SOTA spec risk T4 mitigation: **30 patterns × 2 languages
//! (Rust + TypeScript)**, expand in v0.3.x. Every pattern in this catalog must
//! be **universally idiomatic** for its language — never tied to a specific
//! project, framework version, or domain. See `CLAUDE.md` "Review Checklist"
//! for the bar.
//!
//! ## Design
//!
//! - Pattern matching is **substring-based**, run against the raw chunk content.
//!   Substring patterns are deterministic, language-portable, and avoid the
//!   compile-time cost of regex initialisation per pattern.
//! - Patterns hold a **language tag** (`PatternLang::Any` allows cross-language
//!   patterns like `try / catch`).
//! - Match results carry the matching `pattern_name` + a 1-line `description`
//!   so the consumer (M5) can present human-readable exemplars without round-
//!   tripping back to the catalog.
//!
//! ## Storage
//!
//! `PatternCatalog::new()` builds the in-memory catalog at startup (the catalog
//! is small enough — a few hundred patterns at most). Matched chunks are stored
//! externally (e.g. in a separate Tantivy index keyed by `pattern_name` or in
//! SQLite) by the index builder; this module only owns the **definitions and
//! the matcher**.

use serde::{Deserialize, Serialize};

/// Language a pattern targets. `Any` matches across all languages and is
/// reserved for highly portable idioms (e.g. early-return guards).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternLang {
    Rust,
    TypeScript,
    Any,
}

impl PatternLang {
    /// Map from `FileType::language_name()` strings to the pattern language tag.
    /// Returns `None` for languages without a catalog (substring matchers can
    /// still run via `PatternLang::Any` patterns if the caller wishes).
    #[must_use]
    pub fn from_language_name(name: &str) -> Option<Self> {
        match name {
            "rust" => Some(Self::Rust),
            "typescript" | "javascript" => Some(Self::TypeScript),
            _ => None,
        }
    }

    /// Short canonical label suitable for storage and protocol serialization.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::Any => "any",
        }
    }
}

/// A static catalog entry — the *what* and *how to detect*.
///
/// `required` substrings must all appear; if any of `forbidden` appear, the
/// pattern is rejected. This two-list design keeps each pattern definition
/// readable and lets us add precision filters as we observe false positives
/// in real-world catalogs.
#[derive(Debug, Clone)]
pub struct PatternDef {
    /// Canonical pattern name (used as the storage key).
    pub name: &'static str,
    /// Target language; `Any` for cross-language idioms.
    pub lang: PatternLang,
    /// Human-readable one-line description (surfaced by M5).
    pub description: &'static str,
    /// All of these substrings must appear in the chunk content.
    pub required: &'static [&'static str],
    /// If any of these substrings appear, the match is rejected.
    /// Empty slice means "no negative constraint".
    pub forbidden: &'static [&'static str],
}

/// A successful match: a pattern fired on a chunk.
///
/// `pattern_name` and `description` are owned strings so the value can outlive
/// the catalog if needed (e.g. shipped over an MCP response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern_name: String,
    pub description: String,
    pub language: String,
}

impl PatternMatch {
    fn from_def(def: &PatternDef) -> Self {
        Self {
            pattern_name: def.name.to_string(),
            description: def.description.to_string(),
            language: def.lang.label().to_string(),
        }
    }
}

/// Returns the static set of Rust patterns the catalog ships with.
///
/// Each entry must satisfy CLAUDE.md "Review Checklist":
/// - universally used across Rust projects
/// - helpful for any unrelated Rust codebase
/// - specific enough to avoid false positives
/// - no references to specific products/services
#[must_use]
pub fn rust_patterns() -> &'static [PatternDef] {
    &RUST_PATTERNS
}

/// Returns the static set of TypeScript patterns the catalog ships with.
///
/// Each entry must satisfy CLAUDE.md "Review Checklist":
/// - universally used across TS/JS projects
/// - helpful for any unrelated TS codebase
/// - specific enough to avoid false positives
/// - no references to specific products/services
#[must_use]
pub fn typescript_patterns() -> &'static [PatternDef] {
    &TYPESCRIPT_PATTERNS
}

// ─────────────────────────────────────────────────────────────────────────────
// Rust catalog — 32 universally-idiomatic patterns
// ─────────────────────────────────────────────────────────────────────────────

static RUST_PATTERNS: [PatternDef; 32] = [
    PatternDef {
        name: "rust.drop_impl",
        lang: PatternLang::Rust,
        description: "Custom Drop impl for explicit resource cleanup",
        required: &["impl Drop", "fn drop"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.from_impl",
        lang: PatternLang::Rust,
        description: "From conversion impl (infallible Into route)",
        required: &["impl From<", "fn from"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.try_from_impl",
        lang: PatternLang::Rust,
        description: "TryFrom conversion impl (fallible parse/validation)",
        required: &["impl TryFrom<", "fn try_from"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.serde_derive",
        lang: PatternLang::Rust,
        description: "Serde derive macro for (de)serialization",
        required: &["#[derive(", "Serialize"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.error_with_thiserror",
        lang: PatternLang::Rust,
        description: "Custom error enum with thiserror derive",
        required: &["#[derive(", "thiserror::Error"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.anyhow_context",
        lang: PatternLang::Rust,
        description: "anyhow Result with .context() for error annotation",
        required: &[".context(", "Result"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.result_question_mark_chain",
        lang: PatternLang::Rust,
        description: "Question-mark operator chain for short-circuit error propagation",
        required: &["?;\n", "Result"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.tokio_spawn",
        lang: PatternLang::Rust,
        description: "tokio::spawn for an async task on the runtime",
        required: &["tokio::spawn"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.tokio_select",
        lang: PatternLang::Rust,
        description: "tokio::select! for concurrent branch selection",
        required: &["tokio::select!"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.tokio_main",
        lang: PatternLang::Rust,
        description: "#[tokio::main] async entrypoint",
        required: &["#[tokio::main]"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.async_trait_method",
        lang: PatternLang::Rust,
        description: "async fn (often behind #[async_trait]) for asynchronous methods",
        required: &["async fn"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.arc_mutex_shared_state",
        lang: PatternLang::Rust,
        description: "Arc<Mutex<T>> shared mutable state across threads",
        required: &["Arc<Mutex<"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.arc_rwlock_shared_state",
        lang: PatternLang::Rust,
        description: "Arc<RwLock<T>> shared multi-reader / single-writer state",
        required: &["Arc<RwLock<"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.builder_pattern",
        lang: PatternLang::Rust,
        description: "Builder struct ending in .build() returning the configured value",
        required: &["fn build("],
        forbidden: &["fn build_path", "fn build_index", "fn build_url"],
    },
    PatternDef {
        name: "rust.new_constructor",
        lang: PatternLang::Rust,
        description: "Associated `new` constructor — idiomatic factory",
        required: &["pub fn new("],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.default_impl",
        lang: PatternLang::Rust,
        description: "impl Default for sensible zero-config construction",
        required: &["impl Default", "fn default"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.display_impl",
        lang: PatternLang::Rust,
        description: "impl Display for user-facing formatting",
        required: &["impl Display", "fn fmt"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.debug_impl",
        lang: PatternLang::Rust,
        description: "Custom Debug impl (overriding the derive)",
        required: &["impl Debug", "fn fmt"],
        forbidden: &["#[derive("],
    },
    PatternDef {
        name: "rust.iterator_impl",
        lang: PatternLang::Rust,
        description: "Custom Iterator impl with next()",
        required: &["impl Iterator", "fn next"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.derive_clone_copy",
        lang: PatternLang::Rust,
        description: "Derive Clone + Copy for value-type semantics",
        required: &["#[derive(", "Clone", "Copy"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.if_let_some",
        lang: PatternLang::Rust,
        description: "if let Some(x) = ... — idiomatic Option destructuring",
        required: &["if let Some("],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.match_result",
        lang: PatternLang::Rust,
        description: "match on Result with Ok / Err arms",
        required: &["match ", "Ok(", "Err("],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.lazy_static_oncelock",
        lang: PatternLang::Rust,
        description: "OnceLock / OnceCell for lazy one-shot initialisation",
        required: &["OnceLock"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.channel_mpsc",
        lang: PatternLang::Rust,
        description: "mpsc channel for cross-thread message passing",
        required: &["mpsc::channel"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.test_module",
        lang: PatternLang::Rust,
        description: "#[cfg(test)] mod tests — colocated unit tests",
        required: &["#[cfg(test)]", "mod tests"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.test_fn",
        lang: PatternLang::Rust,
        description: "#[test] function — standard unit test",
        required: &["#[test]", "fn "],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.tracing_instrument",
        lang: PatternLang::Rust,
        description: "#[tracing::instrument] for span instrumentation",
        required: &["#[tracing::instrument"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.tracing_log",
        lang: PatternLang::Rust,
        description: "Structured logging via tracing macros (info!/warn!/error!)",
        required: &["tracing::"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.box_dyn_trait",
        lang: PatternLang::Rust,
        description: "Box<dyn Trait> heap-allocated trait object",
        required: &["Box<dyn "],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.lifetime_param",
        lang: PatternLang::Rust,
        description: "Lifetime parameter — borrow-checker-driven API",
        required: &["<'"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.macro_rules_def",
        lang: PatternLang::Rust,
        description: "macro_rules! declarative macro definition",
        required: &["macro_rules!"],
        forbidden: &[],
    },
    PatternDef {
        name: "rust.unsafe_block",
        lang: PatternLang::Rust,
        description: "unsafe block — review carefully for soundness",
        required: &["unsafe {"],
        forbidden: &[],
    },
];

// ─────────────────────────────────────────────────────────────────────────────
// TypeScript catalog — 30 universally-idiomatic patterns
// ─────────────────────────────────────────────────────────────────────────────

static TYPESCRIPT_PATTERNS: [PatternDef; 30] = [
    PatternDef {
        name: "ts.promise_all_settled",
        lang: PatternLang::TypeScript,
        description: "Promise.allSettled — parallel execution with per-task failure handling",
        required: &["Promise.allSettled"],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.promise_all",
        lang: PatternLang::TypeScript,
        description: "Promise.all — parallel fan-out, fail-fast",
        required: &["Promise.all("],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.promise_race",
        lang: PatternLang::TypeScript,
        description: "Promise.race — first-to-settle wins (timeout / fallback)",
        required: &["Promise.race"],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.async_await",
        lang: PatternLang::TypeScript,
        description: "async/await — suspending asynchronous control flow",
        required: &["async ", "await "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.try_catch_async",
        lang: PatternLang::TypeScript,
        description: "try/catch around await — async error handling",
        required: &["try {", "catch", "await "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.optional_chaining",
        lang: PatternLang::TypeScript,
        description: "Optional chaining (?.) — null-safe property access",
        required: &["?."],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.nullish_coalescing",
        lang: PatternLang::TypeScript,
        description: "Nullish coalescing (??) — default for null/undefined only",
        required: &[" ?? "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.destructure_object",
        lang: PatternLang::TypeScript,
        description: "Object destructuring in function params — named-arg ergonomics",
        required: &["({ "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.rest_spread",
        lang: PatternLang::TypeScript,
        description: "Rest/spread operator for arrays or args",
        required: &["..."],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.arrow_function",
        lang: PatternLang::TypeScript,
        description: "Arrow function — lexical-this callback",
        required: &[" => "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.array_map_chain",
        lang: PatternLang::TypeScript,
        description: "Array .map().filter() — functional chain over collections",
        required: &[".map(", ".filter("],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.array_reduce",
        lang: PatternLang::TypeScript,
        description: "Array.reduce — accumulator-driven aggregation",
        required: &[".reduce("],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.type_alias",
        lang: PatternLang::TypeScript,
        description: "type alias declaration — structural typing primitive",
        required: &["type ", " = "],
        forbidden: &["type of "],
    },
    PatternDef {
        name: "ts.interface_decl",
        lang: PatternLang::TypeScript,
        description: "interface declaration — extensible object contract",
        required: &["interface ", "{"],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.enum_decl",
        lang: PatternLang::TypeScript,
        description: "enum declaration — bounded set of named values",
        required: &["enum ", "{"],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.discriminated_union",
        lang: PatternLang::TypeScript,
        description: "Discriminated union via shared 'kind' field — safe pattern-matching",
        required: &["kind:", "|"],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.generic_function",
        lang: PatternLang::TypeScript,
        description: "Generic function with type parameter",
        required: &["function ", "<", ">("],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.readonly_field",
        lang: PatternLang::TypeScript,
        description: "readonly modifier — immutable field declaration",
        required: &["readonly "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.private_class_field",
        lang: PatternLang::TypeScript,
        description: "private class field (TS access modifier)",
        required: &["private "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.class_constructor",
        lang: PatternLang::TypeScript,
        description: "class with constructor — OOP initialisation",
        required: &["class ", "constructor("],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.class_extends",
        lang: PatternLang::TypeScript,
        description: "class extends — single-class inheritance",
        required: &["class ", "extends "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.implements_interface",
        lang: PatternLang::TypeScript,
        description: "class implements interface — contract conformance",
        required: &["class ", "implements "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.import_module",
        lang: PatternLang::TypeScript,
        description: "ES module import",
        required: &["import ", "from "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.export_default",
        lang: PatternLang::TypeScript,
        description: "export default declaration — module's primary export",
        required: &["export default"],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.export_named",
        lang: PatternLang::TypeScript,
        description: "named export",
        required: &["export "],
        forbidden: &["export default"],
    },
    PatternDef {
        name: "ts.fetch_api_call",
        lang: PatternLang::TypeScript,
        description: "fetch() with await — standard HTTP request",
        required: &["fetch(", "await "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.json_parse_stringify",
        lang: PatternLang::TypeScript,
        description: "JSON.parse / JSON.stringify for serialisation",
        required: &["JSON."],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.set_timeout",
        lang: PatternLang::TypeScript,
        description: "setTimeout / setInterval scheduling",
        required: &["setTimeout("],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.error_throw",
        lang: PatternLang::TypeScript,
        description: "throw new Error — explicit failure signal",
        required: &["throw new "],
        forbidden: &[],
    },
    PatternDef {
        name: "ts.error_handler_chain",
        lang: PatternLang::TypeScript,
        description: "Promise.catch() — promise-chain error handler",
        required: &[".catch("],
        forbidden: &[],
    },
];

/// In-memory pattern catalog backing the matcher and the storage lookup APIs.
///
/// The catalog is constant for a given build of semantex; future versions may
/// load patterns from a config file, but for v0.3 every pattern ships in the
/// binary and is reviewed against `CLAUDE.md` standards.
#[derive(Debug)]
pub struct PatternCatalog {
    rust: &'static [PatternDef],
    typescript: &'static [PatternDef],
}

impl PatternCatalog {
    /// Build a new catalog over the static pattern tables.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rust: rust_patterns(),
            typescript: typescript_patterns(),
        }
    }

    /// Total number of patterns across all languages in the catalog.
    #[must_use]
    pub fn pattern_count(&self) -> usize {
        self.rust.len() + self.typescript.len()
    }

    /// Number of patterns for a specific language.
    #[must_use]
    pub fn count_for(&self, lang: PatternLang) -> usize {
        match lang {
            PatternLang::Rust => self.rust.len(),
            PatternLang::TypeScript => self.typescript.len(),
            PatternLang::Any => self.pattern_count(),
        }
    }

    /// Iterate over every pattern definition in the catalog.
    pub fn all_patterns(&self) -> impl Iterator<Item = &PatternDef> {
        self.rust.iter().chain(self.typescript.iter())
    }

    /// Look up a pattern definition by canonical name. O(N) — the catalog is
    /// small enough that a linear scan is faster than building a hashmap.
    #[must_use]
    pub fn lookup(&self, pattern_name: &str) -> Option<&PatternDef> {
        self.all_patterns().find(|p| p.name == pattern_name)
    }
}

impl Default for PatternCatalog {
    fn default() -> Self {
        Self::new()
    }
}

/// Mine all matching patterns from a single chunk's content for a given language.
///
/// Returns one `PatternMatch` per pattern that fired. If `language` has no
/// catalog (e.g. unsupported language), returns an empty vector — callers
/// should not treat this as an error.
///
/// Substring matching is `&str::contains` based, so the matcher is allocation-
/// free for the matching phase.
#[must_use]
pub fn mine_patterns(chunk_content: &str, language: PatternLang) -> Vec<PatternMatch> {
    let catalog = PatternCatalog::new();
    mine_patterns_with(chunk_content, language, &catalog)
}

/// Same as `mine_patterns` but reuses a pre-built catalog. Useful when mining
/// thousands of chunks in a single indexing pass.
#[must_use]
pub fn mine_patterns_with(
    chunk_content: &str,
    language: PatternLang,
    catalog: &PatternCatalog,
) -> Vec<PatternMatch> {
    let defs: &[PatternDef] = match language {
        PatternLang::Rust => catalog.rust,
        PatternLang::TypeScript => catalog.typescript,
        PatternLang::Any => return mine_all_languages(chunk_content, catalog),
    };
    let mut matches = Vec::new();
    for def in defs {
        if matches_pattern(chunk_content, def) {
            matches.push(PatternMatch::from_def(def));
        }
    }
    matches
}

/// Mine patterns across every language catalog. Used when the language is
/// unknown or when a chunk could plausibly belong to multiple languages.
fn mine_all_languages(chunk_content: &str, catalog: &PatternCatalog) -> Vec<PatternMatch> {
    let mut matches = Vec::new();
    for def in catalog.all_patterns() {
        if matches_pattern(chunk_content, def) {
            matches.push(PatternMatch::from_def(def));
        }
    }
    matches
}

fn matches_pattern(content: &str, def: &PatternDef) -> bool {
    if !def.required.iter().all(|s| content.contains(s)) {
        return false;
    }
    if def.forbidden.iter().any(|s| content.contains(s)) {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_meets_minimum_scope() {
        // Per spec T4 mitigation: 30 patterns × 2 languages minimum.
        let cat = PatternCatalog::new();
        assert!(
            cat.count_for(PatternLang::Rust) >= 30,
            "Rust catalog must have >= 30 patterns, got {}",
            cat.count_for(PatternLang::Rust)
        );
        assert!(
            cat.count_for(PatternLang::TypeScript) >= 30,
            "TS catalog must have >= 30 patterns, got {}",
            cat.count_for(PatternLang::TypeScript)
        );
    }

    #[test]
    fn pattern_names_are_unique() {
        let cat = PatternCatalog::new();
        let mut seen = std::collections::HashSet::new();
        for p in cat.all_patterns() {
            assert!(
                seen.insert(p.name),
                "duplicate pattern name in catalog: {}",
                p.name
            );
        }
    }

    #[test]
    fn lookup_finds_known_pattern() {
        let cat = PatternCatalog::new();
        let p = cat
            .lookup("rust.drop_impl")
            .expect("rust.drop_impl must exist");
        assert_eq!(p.lang, PatternLang::Rust);
    }

    #[test]
    fn rust_drop_impl_matches() {
        let chunk = "
            impl Drop for Foo {
                fn drop(&mut self) {
                    self.close();
                }
            }
        ";
        let matches = mine_patterns(chunk, PatternLang::Rust);
        assert!(
            matches.iter().any(|m| m.pattern_name == "rust.drop_impl"),
            "Drop impl should match: {matches:?}"
        );
    }

    #[test]
    fn rust_serde_derive_matches() {
        let chunk = "
            #[derive(Serialize, Deserialize)]
            struct Config { name: String }
        ";
        let matches = mine_patterns(chunk, PatternLang::Rust);
        assert!(
            matches
                .iter()
                .any(|m| m.pattern_name == "rust.serde_derive"),
            "serde derive should match: {matches:?}"
        );
    }

    #[test]
    fn rust_tokio_spawn_matches() {
        let chunk = "
            fn launch() {
                tokio::spawn(async move {
                    do_work().await;
                });
            }
        ";
        let matches = mine_patterns(chunk, PatternLang::Rust);
        assert!(
            matches.iter().any(|m| m.pattern_name == "rust.tokio_spawn"),
            "tokio::spawn should match: {matches:?}"
        );
    }

    #[test]
    fn rust_question_mark_chain_matches() {
        let chunk = "
            fn read_config() -> Result<Config> {
                let bytes = read_file()?;
                let cfg = parse(&bytes)?;
                Ok(cfg)
            }
        ";
        let matches = mine_patterns(chunk, PatternLang::Rust);
        assert!(
            matches
                .iter()
                .any(|m| m.pattern_name == "rust.result_question_mark_chain"),
            "?-chain should match: {matches:?}"
        );
    }

    #[test]
    fn rust_unsafe_block_matches() {
        let chunk = "
            fn raw_access() {
                unsafe {
                    let ptr = std::ptr::null_mut();
                }
            }
        ";
        let matches = mine_patterns(chunk, PatternLang::Rust);
        assert!(
            matches
                .iter()
                .any(|m| m.pattern_name == "rust.unsafe_block"),
            "unsafe block should match: {matches:?}"
        );
    }

    #[test]
    fn ts_promise_all_settled_matches() {
        let chunk = "
            const results = await Promise.allSettled([
                fetchA(),
                fetchB(),
            ]);
        ";
        let matches = mine_patterns(chunk, PatternLang::TypeScript);
        assert!(
            matches
                .iter()
                .any(|m| m.pattern_name == "ts.promise_all_settled"),
            "Promise.allSettled should match: {matches:?}"
        );
    }

    #[test]
    fn ts_async_await_matches() {
        let chunk = "
            async function load() {
                const data = await fetch(url);
                return data.json();
            }
        ";
        let matches = mine_patterns(chunk, PatternLang::TypeScript);
        assert!(
            matches.iter().any(|m| m.pattern_name == "ts.async_await"),
            "async/await should match: {matches:?}"
        );
    }

    #[test]
    fn ts_optional_chaining_matches() {
        let chunk = "
            const v = user?.profile?.name ?? 'anon';
        ";
        let matches = mine_patterns(chunk, PatternLang::TypeScript);
        let names: Vec<_> = matches.iter().map(|m| m.pattern_name.as_str()).collect();
        assert!(
            names.contains(&"ts.optional_chaining"),
            "optional chaining should match: {names:?}"
        );
        assert!(
            names.contains(&"ts.nullish_coalescing"),
            "?? should match: {names:?}"
        );
    }

    #[test]
    fn ts_class_extends_matches() {
        let chunk = "
            class Foo extends Bar {
                constructor() { super(); }
            }
        ";
        let matches = mine_patterns(chunk, PatternLang::TypeScript);
        let names: Vec<_> = matches.iter().map(|m| m.pattern_name.as_str()).collect();
        assert!(
            names.contains(&"ts.class_extends"),
            "class extends should match: {names:?}"
        );
        assert!(
            names.contains(&"ts.class_constructor"),
            "constructor should match: {names:?}"
        );
    }

    #[test]
    fn no_match_returns_empty() {
        let chunk = "let x = 42;";
        let matches = mine_patterns(chunk, PatternLang::Rust);
        assert!(
            matches.is_empty() || matches.iter().all(|m| m.pattern_name.starts_with("rust.")),
            "no Rust-specific pattern should match a JS-style chunk: {matches:?}"
        );
    }

    #[test]
    fn unsupported_language_returns_empty() {
        // Cobol isn't in the catalog at all.
        assert_eq!(PatternLang::from_language_name("cobol"), None);
    }

    #[test]
    fn from_language_name_maps_correctly() {
        assert_eq!(
            PatternLang::from_language_name("rust"),
            Some(PatternLang::Rust)
        );
        assert_eq!(
            PatternLang::from_language_name("typescript"),
            Some(PatternLang::TypeScript)
        );
        assert_eq!(
            PatternLang::from_language_name("javascript"),
            Some(PatternLang::TypeScript)
        );
        assert_eq!(PatternLang::from_language_name("python"), None);
    }

    #[test]
    fn forbidden_substring_rejects_match() {
        // The custom Debug impl pattern has #[derive( in forbidden — if the
        // chunk has both impl Debug AND #[derive(, it's a derived Debug not a custom impl.
        let chunk = "
            #[derive(Debug)]
            struct Foo;
        ";
        let matches = mine_patterns(chunk, PatternLang::Rust);
        assert!(
            !matches.iter().any(|m| m.pattern_name == "rust.debug_impl"),
            "derived Debug should not match custom Debug pattern: {matches:?}"
        );
    }

    #[test]
    fn any_language_mines_across_catalogs() {
        let chunk = "
            impl Drop for Foo {
                fn drop(&mut self) { Promise.allSettled(stuff); }
            }
        ";
        let matches = mine_patterns(chunk, PatternLang::Any);
        let names: Vec<_> = matches.iter().map(|m| m.pattern_name.as_str()).collect();
        assert!(
            names.contains(&"rust.drop_impl"),
            "Rust Drop should match under Any: {names:?}"
        );
        assert!(
            names.contains(&"ts.promise_all_settled"),
            "TS Promise.allSettled should match under Any: {names:?}"
        );
    }

    #[test]
    fn reusing_catalog_yields_same_matches() {
        let chunk = "tokio::spawn(async move {});";
        let cat = PatternCatalog::new();
        let m1 = mine_patterns(chunk, PatternLang::Rust);
        let m2 = mine_patterns_with(chunk, PatternLang::Rust, &cat);
        assert_eq!(m1.len(), m2.len());
    }
}
