//! Path-derived ranking signals ported from colgrep.
//!
//! These functions translate `chunk.file_path` and `chunk.meta` data into
//! score adjustments applied AFTER fusion. They are language-agnostic and
//! repo-agnostic; no project-specific tuning is permitted here.

use crate::search::code_tokenizer::expand_identifiers;
use crate::types::{Chunk, ScoredChunkId};
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

// Penalty / boost constants — per spec §4.2.

/// Strong multiplicative penalty for test / compat / example paths.
pub const STRONG_PENALTY: f32 = 0.30; // per spec §4.2
/// Moderate multiplicative penalty for re-export barrel files.
pub const MODERATE_PENALTY: f32 = 0.50; // per spec §4.2
/// Mild multiplicative penalty for declaration stubs (e.g. `.d.ts`).
pub const MILD_PENALTY: f32 = 0.70; // per spec §4.2

/// Additive boost fraction for an exact query-token / path-stem-token match.
pub const STEM_EXACT_BOOST_FRAC: f32 = 0.40; // per spec §4.2
/// Additive boost fraction for a prefix query-token / path-stem-token match.
pub const STEM_PREFIX_BOOST_FRAC: f32 = 0.20; // per spec §4.2
/// Minimum shared prefix length to count for the stem prefix boost.
pub const STEM_PREFIX_MIN_LEN: usize = 3; // per spec §4.2

/// Additive boost fraction when query tokens overlap with a chunk's symbol name.
pub const DEFINITION_BOOST_FRAC: f32 = 0.20; // per spec §4.2

/// Additive boost fraction for the top-scoring chunk of a multi-chunk file.
pub const FILE_COHERENCE_BOOST_FRAC: f32 = 0.20; // per spec §4.2

// ---- Item 6: file path penalty ---------------------------------------------

/// Query tokens that, when present, disable the path penalty so users
/// actively searching for tests / benchmarks / examples can still see them.
const PENALTY_DISABLE_TOKENS: &[&str] = &[
    "test",
    "tests",
    "spec",
    "specs",
    "benchmark",
    "benchmarks",
    "bench",
    "example",
    "examples",
    "demo",
];

/// Path component / extension patterns whose presence triggers the strong
/// "test directory" penalty.
const TEST_DIR_PATTERNS: &[&str] = &["/tests/", "/test/", "/__tests__/", "/spec/"];

/// Path component patterns whose presence triggers the strong
/// "compat / legacy / deprecated / polyfill" penalty.
const COMPAT_DIR_PATTERNS: &[&str] = &["/compat/", "/legacy/", "/deprecated/", "/polyfill/"];

/// Path component patterns whose presence triggers the strong
/// "examples / demos / samples" penalty.
const EXAMPLE_DIR_PATTERNS: &[&str] = &[
    "/examples/",
    "/example/",
    "/demos/",
    "/demo/",
    "/samples/",
    "/sample/",
];

/// Exact filenames that count as re-export barrels.
const BARREL_FILENAMES: &[&str] = &[
    "__init__.py",
    "mod.rs",
    "index.ts",
    "index.js",
    "package-info.java",
];

/// Filename regex matching common test naming conventions across languages.
///
/// Covers Go (`_test.go`), Ruby (`_spec.rb`), Dart (`_test.dart`), Vue
/// (`.spec.vue`), JS/TS (`.test.ts`, `.spec.ts`), etc.
///
/// IMPORTANT: the bare-stem alternatives (`test`, `tests`, `spec`, `specs`)
/// require an EXPLICIT `_` / `.` / `-` / `/` boundary on the left. They do
/// NOT match at the start of the filename — otherwise a legitimate
/// top-level file literally named `tests.rs` (or `spec.dart`, etc.) would
/// be penalised even though it's a real source module. See defect #10.
///
/// Pytest-style `test_X.py` and RSpec-style `spec_X.rb` filenames are
/// matched by a SEPARATE prefix regex (`TEST_FILENAME_PREFIX_RE`), so they
/// can legitimately anchor at the start of the filename.
static TEST_FILENAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)([_./-])(test|tests|spec|specs|_test|\.spec|\.test)\.[A-Za-z0-9]+$")
        .expect("path_signals: test filename regex MUST compile")
});

/// Filename prefix regex for pytest (`test_*.py`) and RSpec
/// (`spec_*.rb`) conventions, which legitimately anchor at the start of
/// the filename. Kept separate from `TEST_FILENAME_RE` so the bare-stem
/// alternatives there can require an explicit left boundary.
static TEST_FILENAME_PREFIX_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(test_|spec_)[A-Za-z0-9_]+\.[A-Za-z0-9]+$")
        .expect("path_signals: test prefix regex MUST compile")
});

/// Returns `true` when the path penalty SHOULD be applied to the given query.
///
/// The penalty is disabled when the lowercased query contains any token
/// related to tests, benchmarks, or examples — see `PENALTY_DISABLE_TOKENS`.
pub fn should_apply_path_penalty(query: &str) -> bool {
    let lower = query.to_lowercase();
    for tok in PENALTY_DISABLE_TOKENS {
        // Require a non-alphanumeric boundary on at least one side to avoid
        // spurious matches on tokens that appear inside longer identifiers
        // (e.g. "attestation" contains "test").
        if has_word_boundary_match(&lower, tok) {
            return false;
        }
    }
    true
}

/// Returns `true` if `needle` appears in `haystack` with a word boundary
/// (start/end of string or a non-alphanumeric char) on both sides.
fn has_word_boundary_match(haystack: &str, needle: &str) -> bool {
    let nlen = needle.len();
    let bytes = haystack.as_bytes();
    let nbytes = needle.as_bytes();
    if nlen == 0 || nlen > bytes.len() {
        return false;
    }
    let mut i = 0;
    while i + nlen <= bytes.len() {
        if &bytes[i..i + nlen] == nbytes {
            let before_ok = i == 0 || !(bytes[i - 1] as char).is_ascii_alphanumeric();
            let after_idx = i + nlen;
            let after_ok =
                after_idx == bytes.len() || !(bytes[after_idx] as char).is_ascii_alphanumeric();
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Compute the multiplicative path penalty for a chunk's file path.
///
/// The result is in `(0.0, 1.0]`. Categories compound multiplicatively;
/// within each category only the FIRST matching pattern contributes.
/// A path with no matching category returns `1.0` (no penalty).
pub fn file_path_penalty(path: &Path) -> f32 {
    // Normalize to forward-slash-separated lowercase string for pattern
    // matching. Pad with leading/trailing slashes so `/tests/` patterns
    // still match when the path starts or ends with the test segment.
    let raw = path.to_string_lossy();
    let lower = raw.to_lowercase();
    let normalized = lower.replace('\\', "/");
    let padded = format!("/{}/", normalized.trim_matches('/'));

    let mut factor: f32 = 1.0;

    // Category: test directory.
    if TEST_DIR_PATTERNS.iter().any(|p| padded.contains(p)) {
        factor *= STRONG_PENALTY;
    }

    // Category: test file name. Match against the filename only.
    // We use two regexes: TEST_FILENAME_RE for boundary-anchored matches
    // (e.g. `foo_test.go`, `bar.spec.ts`) and TEST_FILENAME_PREFIX_RE for
    // start-anchored matches (e.g. `test_auth.py`, `spec_helper.rb`). The
    // split exists so legitimate non-test files literally named `tests.rs`
    // / `spec.dart` are NOT penalized — see defect #10.
    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if !filename.is_empty()
        && (TEST_FILENAME_RE.is_match(&filename) || TEST_FILENAME_PREFIX_RE.is_match(&filename))
    {
        factor *= STRONG_PENALTY;
    }

    // Category: compat / legacy / deprecated / polyfill.
    if COMPAT_DIR_PATTERNS.iter().any(|p| padded.contains(p)) {
        factor *= STRONG_PENALTY;
    }

    // Category: examples / demos / samples.
    if EXAMPLE_DIR_PATTERNS.iter().any(|p| padded.contains(p)) {
        factor *= STRONG_PENALTY;
    }

    // Category: `.d.ts` declaration stubs.
    if filename.to_lowercase().ends_with(".d.ts") {
        factor *= MILD_PENALTY;
    }

    // Category: re-export barrel filenames.
    if BARREL_FILENAMES
        .iter()
        .any(|b| filename.eq_ignore_ascii_case(b))
    {
        factor *= MODERATE_PENALTY;
    }

    factor
}

// ---- Shared helper for Items 7 + 8 -----------------------------------------

/// Stopwords filtered out of query token extraction for path / definition
/// boost lookups. Per spec §7.3.2.
const QUERY_STOPWORDS: &[&str] = &[
    "how", "the", "for", "of", "in", "on", "at", "to", "is", "are", "and", "or", "what", "where",
    "when", "why",
];

/// Extract lowercased identifier-style tokens from a free-form query string
/// for use by the path stem boost (Item 7) and definition boost (Item 8).
///
/// The query is first run through `code_tokenizer::expand_identifiers` to
/// pick up camelCase / snake_case decomposition consistent with BM25
/// tokenization, then through a raw identifier-span pass to catch
/// single-word tokens that `expand_identifiers` skips (it only emits
/// multi-part splits). Tokens are lowercased, de-duplicated (preserving
/// first occurrence), and filtered against `QUERY_STOPWORDS`.
///
/// Item-16 bigram tokens (e.g. `get_user`, `by_id`) emitted by
/// `expand_identifiers` are an INDEX-time BM25 signal only — they MUST NOT
/// participate in path/definition stem-match logic. If they did, a stem
/// subtoken like `gener` (>=3 chars) would prefix-match the bigram
/// `gener_ate` and falsely award STEM_PREFIX_BOOST_FRAC. We strip any
/// expansion token containing `_`; the raw-identifier-span pass below
/// still supplies the clean bare sub-tokens we need.
pub fn query_tokens_for_path_signals(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let expanded = expand_identifiers(text);
    for tok in expanded.split_whitespace() {
        // Skip Item-16 adjacent-pair bigrams (`a_b`, `b_c`, ...). The
        // raw-identifier pass below provides bare sub-tokens.
        if tok.contains('_') {
            continue;
        }
        push_token(tok, &mut out, &mut seen);
    }

    for raw in split_identifier_spans(text) {
        for sub in split_into_subtokens(&raw) {
            push_token(&sub, &mut out, &mut seen);
        }
    }

    out
}

fn push_token(tok: &str, out: &mut Vec<String>, seen: &mut std::collections::HashSet<String>) {
    if tok.is_empty() {
        return;
    }
    let lower = tok.to_lowercase();
    if QUERY_STOPWORDS.contains(&lower.as_str()) {
        return;
    }
    if seen.insert(lower.clone()) {
        out.push(lower);
    }
}

/// Split `text` on any non-identifier character (anything outside
/// `[A-Za-z0-9_]`). Empty runs are dropped.
fn split_identifier_spans(text: &str) -> Vec<String> {
    let mut spans: Vec<String> = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            current.push(c);
        } else if !current.is_empty() {
            spans.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        spans.push(current);
    }
    spans
}

/// Split a single identifier span into snake_case + camelCase sub-tokens.
fn split_into_subtokens(span: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for piece in span.split('_').filter(|s| !s.is_empty()) {
        for cp in split_camel_case_local(piece) {
            if !cp.is_empty() {
                out.push(cp.to_lowercase());
            }
        }
    }
    out
}

/// Local copy of camelCase splitting (kept private to avoid widening the
/// `code_tokenizer` public surface for v0.4).
///
/// MUST stay byte-for-byte identical to `code_tokenizer::split_camel_case`
/// — any divergence here causes path_signals tokenization to disagree with
/// BM25 tokenization, producing score-composition mismatches on identifiers
/// like `OAuth`, `XMLParser`, `HTTPServer`. See defect #5 (v0.4.1).
///
/// Rules (mirroring `code_tokenizer::split_camel_case`):
/// - `[a-z][A-Z]` or `[0-9][A-Z]` boundary: `getUserById` -> `get|User|By|Id`.
/// - `[A-Z][A-Z][a-z]` boundary: `XMLParser` -> `XML|Parser`. Only splits
///   when the uppercase prefix is at least 2 chars, so `OAuth` stays whole.
fn split_camel_case_local(word: &str) -> Vec<&str> {
    let bytes = word.as_bytes();
    if bytes.is_empty() {
        return vec![];
    }

    let mut parts = Vec::new();
    let mut start = 0;

    for i in 1..bytes.len() {
        let prev = bytes[i - 1];
        let cur = bytes[i];

        // lowerUpper or digitUpper boundary
        if (prev.is_ascii_lowercase() || prev.is_ascii_digit()) && cur.is_ascii_uppercase() {
            parts.push(&word[start..i]);
            start = i;
            continue;
        }

        // UPPERLower boundary (e.g., XMLParser -> XML|Parser)
        // Only split if the uppercase prefix is at least 2 chars (so "OAuth" stays whole)
        if i >= 2
            && bytes[i - 2].is_ascii_uppercase()
            && prev.is_ascii_uppercase()
            && cur.is_ascii_lowercase()
            && (i - 1 - start) >= 2
        {
            parts.push(&word[start..i - 1]);
            start = i - 1;
        }
    }

    parts.push(&word[start..]);
    parts
}

// ---- Item 7: path stem boost -----------------------------------------------

/// Compute the additive path-stem boost fraction for a chunk path against a
/// set of query tokens. The returned fraction is one of:
/// - `STEM_EXACT_BOOST_FRAC` for any exact match between a query token and
///   an identifier sub-token of the file stem;
/// - `STEM_PREFIX_BOOST_FRAC` for any prefix match of at least
///   `STEM_PREFIX_MIN_LEN` characters;
/// - `0.0` otherwise.
///
/// Both sides are lowercased; plural normalization is intentionally skipped
/// per spec §7.3.2.
pub fn path_stem_boost_factor(path: &Path, query_tokens: &[String]) -> f32 {
    if query_tokens.is_empty() {
        return 0.0;
    }

    // Repeatedly strip extensions so `foo.spec.vue` → `foo.spec` → `foo`.
    let mut stem_str = match path.file_stem().and_then(|s| s.to_str()) {
        Some(s) => s.to_string(),
        None => return 0.0,
    };
    while let Some(inner) = Path::new(&stem_str).file_stem().and_then(|s| s.to_str()) {
        if inner == stem_str {
            break;
        }
        stem_str = inner.to_string();
    }

    let mut stem_tokens: Vec<String> = Vec::new();
    for span in split_identifier_spans(&stem_str) {
        for sub in split_into_subtokens(&span) {
            if !sub.is_empty() {
                stem_tokens.push(sub);
            }
        }
    }
    if stem_tokens.is_empty() {
        return 0.0;
    }

    let mut best: f32 = 0.0;
    for qt in query_tokens {
        for st in &stem_tokens {
            if qt == st {
                return STEM_EXACT_BOOST_FRAC;
            }
            // Prefix match: one string is a prefix of the other AND the
            // shared prefix length is at least STEM_PREFIX_MIN_LEN.
            let shared = qt.len().min(st.len());
            if shared >= STEM_PREFIX_MIN_LEN
                && (qt.starts_with(st.as_str()) || st.starts_with(qt.as_str()))
            {
                best = best.max(STEM_PREFIX_BOOST_FRAC);
            }
        }
    }
    best
}

// ---- Item 8: definition boost ----------------------------------------------

/// Compute the additive definition-boost fraction for a chunk's symbol name
/// against a set of query tokens.
///
/// Returns `DEFINITION_BOOST_FRAC` when any query token (case-insensitive)
/// equals any identifier sub-token of `symbol_name`, else `0.0`. Sub-token
/// extraction uses the same snake_case + camelCase split as the query side
/// so `getUserById` decomposes into `get`, `user`, `by`, `id`.
pub fn definition_boost_factor(symbol_name: &str, query_tokens: &[String]) -> f32 {
    if symbol_name.is_empty() || query_tokens.is_empty() {
        return 0.0;
    }

    let mut sym_tokens: Vec<String> = Vec::new();
    for span in split_identifier_spans(symbol_name) {
        for sub in split_into_subtokens(&span) {
            if !sub.is_empty() {
                sym_tokens.push(sub);
            }
        }
    }
    if sym_tokens.is_empty() {
        return 0.0;
    }

    for qt in query_tokens {
        let lower = qt.to_lowercase();
        if sym_tokens.iter().any(|s| s == &lower) {
            return DEFINITION_BOOST_FRAC;
        }
    }
    0.0
}

// ---- Item 9: file coherence ------------------------------------------------

/// Compute per-chunk file-coherence boost fractions.
///
/// For every file in `fused` that contributes more than one chunk:
/// - `file_sum` := sum of those chunks' current scores;
/// - `top_chunk_id` := chunk in that file with the highest score.
///
/// Then `max_file_sum` is taken across all multi-chunk files, and the
/// returned map sends each per-file `top_chunk_id` to
/// `FILE_COHERENCE_BOOST_FRAC * (file_sum / max_file_sum)`.
///
/// Chunks belonging to single-chunk files are not present in the map; the
/// caller MUST treat absence as "no boost".
pub fn file_coherence_boosts<S: std::hash::BuildHasher>(
    fused: &[ScoredChunkId],
    chunk_map: &HashMap<u64, Chunk, S>,
) -> HashMap<u64, f32> {
    #[derive(Default)]
    struct Group {
        sum: f32,
        top_id: u64,
        top_score: f32,
        count: usize,
    }
    let mut groups: HashMap<String, Group> = HashMap::new();
    for s in fused {
        let Some(chunk) = chunk_map.get(&s.chunk_id) else {
            continue;
        };
        let key = chunk.file_path.to_string_lossy().into_owned();
        let entry = groups.entry(key).or_default();
        entry.sum += s.score;
        entry.count += 1;
        if entry.count == 1 || s.score > entry.top_score {
            entry.top_score = s.score;
            entry.top_id = s.chunk_id;
        }
    }

    groups.retain(|_, g| g.count >= 2);
    if groups.is_empty() {
        return HashMap::new();
    }

    let max_file_sum = groups.values().map(|g| g.sum).fold(f32::MIN, f32::max);
    // Guard: only proceed when max_file_sum is a finite positive number.
    // NaN propagates through arithmetic but compares false in every
    // direction, so use partial_cmp to disambiguate.
    if !matches!(
        max_file_sum.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return HashMap::new();
    }

    let mut out: HashMap<u64, f32> = HashMap::with_capacity(groups.len());
    for g in groups.values() {
        let frac = FILE_COHERENCE_BOOST_FRAC * (g.sum / max_file_sum);
        if frac > 0.0 {
            out.insert(g.top_id, frac);
        }
    }
    out
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Chunk, ChunkType, ScoredChunkId};
    use std::path::PathBuf;

    fn pb(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn make_chunk(id: u64, file_path: &str) -> Chunk {
        Chunk {
            id,
            file_path: pb(file_path),
            start_line: 1,
            end_line: 10,
            content: String::new(),
            chunk_type: ChunkType::TextWindow { window_index: 0 },
        }
    }

    // ---- Defect #5 regression: split_camel_case_local must match
    //      code_tokenizer::split_camel_case byte-for-byte. ----

    #[test]
    fn split_camel_case_local_matches_code_tokenizer() {
        // Each case below mirrors the contract of
        // `code_tokenizer::split_camel_case` (which is module-private and so
        // can't be imported directly). The expected values ARE the
        // ground-truth output of that function — if the BM25 splitter changes
        // these, this test must change in lockstep with the BM25 side.
        let cases: &[(&str, &[&str])] = &[
            // 2-char uppercase prefix is preserved as a single piece: "OAuth"
            // stays whole because the (i - 1 - start) >= 2 guard rejects the
            // UPPER+lower split when the uppercase prefix is only 1 char.
            ("OAuth", &["OAuth"]),
            ("OAuthClient", &["OAuth", "Client"]),
            // 3+ char uppercase run splits before the final upper: XMLParser
            // -> XML|Parser; HTTPServer -> HTTP|Server.
            ("XMLParser", &["XML", "Parser"]),
            ("HTTPServer", &["HTTP", "Server"]),
            // Lowercase->uppercase boundary.
            ("getUserById", &["get", "User", "By", "Id"]),
            // Mixed lowercase+UPPER+lowercase: "MyHTTPHandler" -> My|HTTP|Handler.
            ("MyHTTPHandler", &["My", "HTTP", "Handler"]),
            // Digit->uppercase boundary; uppercase run too short to split.
            ("id3Tag", &["id3", "Tag"]),
            // Digit-in-middle case ("HTML5Parser"): "HTML5" stays together
            // (no boundary between digit and preceding upper); "Parser"
            // splits off because digit->upper triggers the boundary.
            ("HTML5Parser", &["HTML5", "Parser"]),
        ];
        for (input, expected) in cases {
            let got = split_camel_case_local(input);
            assert_eq!(
                got, *expected,
                "split_camel_case_local({input:?}) = {got:?}, expected {expected:?}"
            );
        }
    }

    // ---- Item 6 tests ----

    #[test]
    fn item6_pure_test_file_path_gets_strong_penalty() {
        // Filename matches the test regex (`_test.go`); no test directory.
        let p = pb("src/auth/auth_test.go");
        let f = file_path_penalty(&p);
        assert!(approx(f, STRONG_PENALTY), "got {f}");
    }

    #[test]
    fn item6_python_pytest_test_prefix_strong_penalty() {
        // pytest convention: filename starts with `test_`. Regression test for
        // the v0.4 followup fix to TEST_FILENAME_RE (commit on v0.4-integration).
        let p = pb("tests/unit/test_auth.py");
        let f = file_path_penalty(&p);
        // /tests/ dir (0.30) × test_*.py filename (0.30) = 0.09
        assert!(approx(f, STRONG_PENALTY * STRONG_PENALTY), "got {f}");

        let p2 = pb("src/auth/test_login.py");
        let f2 = file_path_penalty(&p2);
        // No test dir; just the prefix-form test_*.py filename = 0.30
        assert!(approx(f2, STRONG_PENALTY), "got {f2}");

        let p3 = pb("spec_helper.rb");
        let f3 = file_path_penalty(&p3);
        // Ruby RSpec convention: spec_*.rb
        assert!(approx(f3, STRONG_PENALTY), "got {f3}");
    }

    #[test]
    fn item6_pure_test_dir_strong_penalty() {
        let p = pb("crates/foo/tests/integration_smoke.rs");
        let f = file_path_penalty(&p);
        assert!(approx(f, STRONG_PENALTY), "got {f}");
    }

    #[test]
    fn item6_compat_plus_test_file_compounds() {
        let p = pb("vendor/compat/legacy_auth_test.py");
        let f = file_path_penalty(&p);
        // compat dir (0.30) × test filename (0.30) = 0.09
        assert!(approx(f, STRONG_PENALTY * STRONG_PENALTY), "got {f}");
    }

    #[test]
    fn item6_d_ts_only_mild_penalty() {
        let p = pb("types/foo.d.ts");
        let f = file_path_penalty(&p);
        assert!(approx(f, MILD_PENALTY), "got {f}");
    }

    #[test]
    fn item6_mod_rs_barrel_moderate_penalty() {
        let p = pb("crates/semantex-core/src/search/mod.rs");
        let f = file_path_penalty(&p);
        assert!(approx(f, MODERATE_PENALTY), "got {f}");
    }

    #[test]
    fn item6_index_ts_barrel_moderate_penalty() {
        let p = pb("packages/foo/src/index.ts");
        let f = file_path_penalty(&p);
        assert!(approx(f, MODERATE_PENALTY), "got {f}");
    }

    #[test]
    fn item6_no_match_returns_one() {
        let p = pb("crates/semantex-core/src/search/hybrid.rs");
        let f = file_path_penalty(&p);
        assert!(approx(f, 1.0), "got {f}");
    }

    #[test]
    fn item6_examples_dir_strong_penalty() {
        let p = pb("examples/quickstart/main.rs");
        let f = file_path_penalty(&p);
        assert!(approx(f, STRONG_PENALTY), "got {f}");
    }

    // ---- Defect #10 regression: bare-stem `tests.rs` at filename root
    //      must NOT be flagged as a test file (no left boundary). ----

    #[test]
    #[allow(non_snake_case)] // Intentional: `NOT` emphasises the assertion.
    fn item6_bare_tests_rs_at_root_does_NOT_match() {
        // `tests.rs` is a legitimate top-level source module name in many
        // Rust projects. The old regex matched it via the `^|` alternation
        // on the left-boundary group, falsely awarding STRONG_PENALTY.
        // Without an explicit `_./-` boundary on the left, the new regex
        // must NOT match.
        let p = pb("crates/foo/src/tests.rs");
        let f = file_path_penalty(&p);
        assert!(
            approx(f, 1.0),
            "tests.rs at filename root must not be penalised, got {f}"
        );

        // Same invariant for `spec.dart`, `test.go`, `specs.rb`.
        for name in &["crates/foo/src/spec.dart", "src/test.go", "lib/specs.rb"] {
            let f = file_path_penalty(&pb(name));
            assert!(
                approx(f, 1.0),
                "{name} (no left boundary) must not be penalised, got {f}"
            );
        }
    }

    #[test]
    fn item6_underscore_separated_tests_still_matches() {
        // The fix to the regex must NOT regress the canonical Go/Ruby/Dart
        // conventions (`<base>_test.<ext>`, `<base>_spec.<ext>`) — the `_`
        // boundary is still a left-anchor for the alternation.
        let p = pb("src/foo_tests.rs");
        let f = file_path_penalty(&p);
        assert!(
            approx(f, STRONG_PENALTY),
            "foo_tests.rs must still be penalised, got {f}"
        );

        // Same for the `.test.<ext>` / `.spec.<ext>` JS conventions.
        let f2 = file_path_penalty(&pb("src/foo.test.ts"));
        assert!(
            approx(f2, STRONG_PENALTY),
            "foo.test.ts must still be penalised, got {f2}"
        );
        let f3 = file_path_penalty(&pb("src/foo.spec.vue"));
        assert!(
            approx(f3, STRONG_PENALTY),
            "foo.spec.vue must still be penalised, got {f3}"
        );
    }

    #[test]
    fn item6_penalty_disabled_when_query_mentions_test() {
        assert!(!should_apply_path_penalty("auth middleware test"));
        assert!(!should_apply_path_penalty("Test integration flow"));
        assert!(!should_apply_path_penalty("how do I write unit tests"));
        assert!(!should_apply_path_penalty("show me an example"));
        assert!(!should_apply_path_penalty("benchmark harness"));
    }

    #[test]
    fn item6_penalty_enabled_for_normal_queries() {
        assert!(should_apply_path_penalty("authentication middleware"));
        assert!(should_apply_path_penalty("code tokenizer"));
        assert!(should_apply_path_penalty("how does graph propagation work"));
    }

    #[test]
    fn item6_penalty_not_disabled_by_substring_inside_word() {
        // "attestation" contains "test" but should NOT disable the penalty.
        assert!(should_apply_path_penalty("attestation flow"));
        // "specification" contains "spec" but is not a test/spec hint.
        assert!(should_apply_path_penalty("specification document"));
    }

    // ---- Item 7 tests ----

    #[test]
    fn item7_exact_match_returns_exact_frac() {
        let tokens = query_tokens_for_path_signals("parse request");
        let f = path_stem_boost_factor(&pb("src/parse_request.py"), &tokens);
        assert!(
            approx(f, STEM_EXACT_BOOST_FRAC),
            "got {f}, tokens={tokens:?}"
        );
    }

    #[test]
    fn item7_prefix_match_returns_prefix_frac() {
        // Query "toke" — strict prefix of stem token "tokenizer" (shared=4 >= 3).
        let tokens = query_tokens_for_path_signals("toke");
        let f = path_stem_boost_factor(&pb("src/code_tokenizer.rs"), &tokens);
        assert!(
            approx(f, STEM_PREFIX_BOOST_FRAC),
            "got {f}, tokens={tokens:?}"
        );
    }

    #[test]
    fn item7_no_match_returns_zero() {
        let tokens = query_tokens_for_path_signals("authentication middleware");
        let f = path_stem_boost_factor(&pb("src/unrelated.rs"), &tokens);
        assert!(approx(f, 0.0), "got {f}");
    }

    #[test]
    fn item7_stopwords_filtered() {
        let tokens = query_tokens_for_path_signals("how the why of for");
        assert!(tokens.is_empty(), "got {tokens:?}");
    }

    #[test]
    fn item7_short_prefix_below_min_len_does_not_match() {
        // Query token "co" is shorter than STEM_PREFIX_MIN_LEN (3) — must
        // not produce a prefix boost against "code_tokenizer.rs".
        let tokens = vec!["co".to_string()];
        let f = path_stem_boost_factor(&pb("src/code_tokenizer.rs"), &tokens);
        assert!(approx(f, 0.0), "got {f}");
    }

    #[test]
    fn item7_acceptance_code_tokenizer_query() {
        // Acceptance smoke: query "code tokenizer" must produce an exact
        // stem boost for `code_tokenizer.rs` since both query tokens are
        // also stem subtokens.
        let tokens = query_tokens_for_path_signals("code tokenizer");
        let f = path_stem_boost_factor(
            &pb("crates/semantex-core/src/search/code_tokenizer.rs"),
            &tokens,
        );
        assert!(
            approx(f, STEM_EXACT_BOOST_FRAC),
            "got {f}, tokens={tokens:?}"
        );
    }

    #[test]
    fn item7_camel_case_query_extracts_subtokens() {
        // "getUserById" should decompose into get/user/by/id, so we'd match
        // a stem like "user_by_id".
        let tokens = query_tokens_for_path_signals("getUserById");
        let f = path_stem_boost_factor(&pb("src/user_by_id.rs"), &tokens);
        assert!(
            approx(f, STEM_EXACT_BOOST_FRAC),
            "got {f}, tokens={tokens:?}"
        );
    }

    // ---- Defect #6 regression: Item-16 bigrams must NOT leak into
    //      query_tokens_for_path_signals output. ----

    #[test]
    fn query_tokens_excludes_bigrams_for_camelcase_input() {
        // `expand_identifiers("getUserById")` now emits adjacent-pair bigrams
        // (`get_user`, `user_by`, `by_id`) as part of its space-separated
        // output. Those bigrams are an index-time BM25 signal only — they
        // must be filtered out before the stem-prefix logic sees them, or
        // short query tokens like "by" / "id" will spuriously prefix-match
        // longer bigrams during path_stem_boost_factor.
        let tokens = query_tokens_for_path_signals("getUserById");

        // All four bare sub-tokens must be present.
        for expected in ["get", "user", "by", "id"] {
            assert!(
                tokens.iter().any(|t| t == expected),
                "missing expected token {expected:?}; got {tokens:?}"
            );
        }

        // No token may contain `_` — that would mean a bigram leaked through.
        for tok in &tokens {
            assert!(
                !tok.contains('_'),
                "bigram token leaked into query tokens: {tok:?} (full: {tokens:?})"
            );
        }
    }

    #[test]
    fn query_tokens_excludes_bigrams_for_snake_case_input() {
        // snake_case identifiers go through the same expand_identifiers path
        // and so emit the same bigram set as camelCase.
        let tokens = query_tokens_for_path_signals("get_user_by_id");
        for tok in &tokens {
            assert!(
                !tok.contains('_'),
                "bigram token leaked from snake_case input: {tok:?} (full: {tokens:?})"
            );
        }
    }

    // ---- Item 8 tests ----

    #[test]
    fn item8_symbol_match_returns_definition_frac() {
        let tokens = query_tokens_for_path_signals("expand identifiers");
        let f = definition_boost_factor("expand_identifiers", &tokens);
        assert!(
            approx(f, DEFINITION_BOOST_FRAC),
            "got {f}, tokens={tokens:?}"
        );
    }

    #[test]
    fn item8_camel_case_symbol_match() {
        let tokens = query_tokens_for_path_signals("retry handler");
        let f = definition_boost_factor("RetryHandler", &tokens);
        assert!(approx(f, DEFINITION_BOOST_FRAC), "got {f}");
    }

    #[test]
    fn item8_no_match_returns_zero() {
        let tokens = query_tokens_for_path_signals("authentication flow");
        let f = definition_boost_factor("expand_identifiers", &tokens);
        assert!(approx(f, 0.0), "got {f}");
    }

    #[test]
    fn item8_empty_symbol_returns_zero() {
        let tokens = vec!["foo".to_string()];
        let f = definition_boost_factor("", &tokens);
        assert!(approx(f, 0.0), "got {f}");
    }

    #[test]
    fn item8_empty_tokens_returns_zero() {
        let f = definition_boost_factor("RetryHandler", &[]);
        assert!(approx(f, 0.0), "got {f}");
    }

    // ---- Item 9 tests ----

    #[test]
    fn item9_multi_chunk_file_boosts_only_top_chunk() {
        let fused = vec![
            ScoredChunkId::new(1, 1.0),
            ScoredChunkId::new(2, 0.8),
            ScoredChunkId::new(3, 0.6),
        ];
        let mut chunk_map: HashMap<u64, Chunk> = HashMap::new();
        chunk_map.insert(1, make_chunk(1, "src/foo.rs"));
        chunk_map.insert(2, make_chunk(2, "src/foo.rs"));
        chunk_map.insert(3, make_chunk(3, "src/foo.rs"));

        let boosts = file_coherence_boosts(&fused, &chunk_map);
        // Only id=1 (highest score within foo.rs) gets a boost.
        assert_eq!(boosts.len(), 1);
        let frac = *boosts.get(&1).expect("top chunk should be present");
        // Single multi-chunk file: file_sum == max_file_sum, so frac == FRAC.
        assert!(approx(frac, FILE_COHERENCE_BOOST_FRAC), "got {frac}");
        assert!(!boosts.contains_key(&2));
        assert!(!boosts.contains_key(&3));
    }

    #[test]
    fn item9_single_chunk_files_get_no_boost() {
        let fused = vec![
            ScoredChunkId::new(1, 1.0),
            ScoredChunkId::new(2, 0.9),
            ScoredChunkId::new(3, 0.8),
        ];
        let mut chunk_map: HashMap<u64, Chunk> = HashMap::new();
        chunk_map.insert(1, make_chunk(1, "src/a.rs"));
        chunk_map.insert(2, make_chunk(2, "src/b.rs"));
        chunk_map.insert(3, make_chunk(3, "src/c.rs"));

        let boosts = file_coherence_boosts(&fused, &chunk_map);
        assert!(boosts.is_empty(), "got {boosts:?}");
    }

    #[test]
    fn item9_multiple_files_normalized_by_max_sum() {
        let fused = vec![
            ScoredChunkId::new(1, 1.0),
            ScoredChunkId::new(2, 0.8),
            ScoredChunkId::new(3, 0.5),
            ScoredChunkId::new(4, 0.4),
        ];
        let mut chunk_map: HashMap<u64, Chunk> = HashMap::new();
        chunk_map.insert(1, make_chunk(1, "src/foo.rs"));
        chunk_map.insert(2, make_chunk(2, "src/foo.rs"));
        chunk_map.insert(3, make_chunk(3, "src/bar.rs"));
        chunk_map.insert(4, make_chunk(4, "src/bar.rs"));

        let boosts = file_coherence_boosts(&fused, &chunk_map);
        assert_eq!(boosts.len(), 2);
        // foo.rs sum=1.8, bar.rs sum=0.9, max_file_sum=1.8.
        // foo top=id 1, boost=FRAC * 1.0 = FRAC.
        // bar top=id 3, boost=FRAC * 0.5.
        let foo = *boosts.get(&1).expect("foo top");
        let bar = *boosts.get(&3).expect("bar top");
        assert!(approx(foo, FILE_COHERENCE_BOOST_FRAC), "foo: {foo}");
        assert!(approx(bar, FILE_COHERENCE_BOOST_FRAC * 0.5), "bar: {bar}");
    }
}
