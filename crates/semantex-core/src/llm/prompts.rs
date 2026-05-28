//! Shared prompt constants and helpers used by every LLM backend.
//!
//! Both `GenaiBackend` and `SubscriptionCliBackend` call the same functions so
//! classification and HyDE behaviour are identical regardless of backend.

use crate::search::agent_classifier::AgentRoute;

// ─────────────────────────────────────────────────────────────────────────────
// Classifier
// ─────────────────────────────────────────────────────────────────────────────

/// System prompt for the route-classifier LLM call.
///
/// The LLM must reply with exactly one snake_case route name — nothing else.
/// The response cap is 8 tokens; even the longest route (`deep_with_examples`)
/// fits comfortably.
pub(crate) const CLASSIFIER_SYSTEM_PROMPT: &str = "\
You are a code-search query classifier. Given a developer's query, pick the \
single best search strategy from the list below and respond with its name in \
snake_case. Respond with EXACTLY ONE route name, no other text.

Routes:
  file_pattern       — query is a glob / file-name pattern (e.g. **/*.rs)
  regex              — query is a regex (contains \\b, \\d, |, [...], anchors)
  exact_symbol       — query is a single identifier or symbol name (CamelCase, snake_case, dot.path)
  structural         — callers / callees / imports / type references / call graph
  deep               — open-ended \"how\" / \"why\" / \"explain\" questions
  analytical         — comparative or quality analysis (most, least, compare, review)
  exhaustive         — enumerate all occurrences of something (list all, find all, show every)
  semantic           — natural-language concept search with no structural intent
  architecture       — high-level overview: main components, subsystems, god nodes, data flow
  exhaustive_structural — exhaustive enumeration of config options, env vars, or CLI flags
  deep_with_examples — deep explanation requested alongside code examples or step-by-step walkthrough
  feature_planning   — impact analysis for adding a feature (\"if I wanted to add X\", \"what files would change\")

Respond with exactly one route name in snake_case, no other text.";

/// Wrap the user query into the classifier prompt body.
pub(crate) fn build_classify_prompt(query: &str) -> String {
    format!("Classify this code-search query:\n\n{query}\n\nRoute:")
}

/// Parse a single route name emitted by the LLM.
///
/// Defensively strips a leading `route:` / `route =` / `answer:` prefix
/// (the classify prompt ends with `Route:` so a chatty LLM that echoes it
/// — e.g. `"Route: deep"` — still parses cleanly), then delegates to
/// [`AgentRoute::from_str`] which handles the snake_case / no-separator
/// matrix once.
///
/// Returns `Err` with a message containing "unrecognized" on unknown input.
pub(crate) fn parse_route_from_llm_output(text: &str) -> anyhow::Result<AgentRoute> {
    let lowered = text.trim().to_ascii_lowercase();
    let body = strip_label_prefix(&lowered);
    body.parse::<AgentRoute>()
        .map_err(|_| anyhow::anyhow!("LLM returned unrecognized route: {text:?}"))
}

/// Strip an optional leading `route:` / `route =` / `answer:` label from an
/// already-trimmed, already-lowercased response. Returns the remaining body,
/// also trimmed. If no label is present, returns the input unchanged.
fn strip_label_prefix(lowered: &str) -> &str {
    for label in ["route:", "route =", "route=", "answer:", "answer ="] {
        if let Some(rest) = lowered.strip_prefix(label) {
            return rest.trim();
        }
    }
    lowered
}

// ─────────────────────────────────────────────────────────────────────────────
// HyDE (Hypothetical Document Embedding)
// ─────────────────────────────────────────────────────────────────────────────

/// System prompt for the HyDE synthesis LLM call.
///
/// The LLM writes a short code snippet (1–30 lines) that would directly answer
/// the user's question. The snippet is then embedded and used as the HyDE
/// query vector, closing the NL→code semantic gap.
pub(crate) const HYDE_SYSTEM_PROMPT: &str = "\
You are a code-synthesis assistant. Given a developer's question about a \
codebase, write a short code snippet (1-30 lines) that would directly answer \
their question. Output ONLY the code, no markdown fences, no explanation.";

/// Wrap the user query into the HyDE synthesis prompt body.
///
/// Appends a language hint when the query explicitly names one, so the LLM
/// targets the right syntax without guessing.
pub(crate) fn build_hyde_prompt(query: &str) -> String {
    let lang_hint = detect_language_hint(query);
    if let Some(lang) = lang_hint {
        format!("Write a short {lang} code snippet that would directly answer:\n\n{query}")
    } else {
        format!("Write a short code snippet that would directly answer:\n\n{query}")
    }
}

/// Detect an explicit language mention in the query for the HyDE prompt hint.
///
/// Ordering is critical: substrings that overlap MUST be checked
/// most-specific first. e.g. "javascript" before "java", "typescript" before
/// any plain-"script" check, "c++" / "c#" / "csharp" / "objective-c" all
/// before any short "c" match. Each fix here needs a corresponding test in
/// `detect_language_hint_ordering` below — substring-overlap bugs are silent
/// until exercised.
fn detect_language_hint(query: &str) -> Option<&'static str> {
    let lower = query.to_ascii_lowercase();

    // ── Languages whose name is a substring of another language's name ──
    // TypeScript before Java/Script collisions
    if lower.contains("typescript") || lower.contains("in ts ") || lower.ends_with("in ts") {
        return Some("TypeScript");
    }
    // JavaScript before Java (CRITICAL: "in java" matches "in javascript")
    if lower.contains("javascript") || lower.contains("in js ") || lower.ends_with("in js") {
        return Some("JavaScript");
    }
    // Objective-C before plain "C"
    if lower.contains("objective-c") || lower.contains("objc") {
        return Some("Objective-C");
    }
    // C# / C++ before plain "C"
    if lower.contains("c#") || lower.contains("csharp") || lower.contains("c-sharp") {
        return Some("C#");
    }
    if lower.contains("c++") || lower.contains("cplusplus") || lower.contains("cpp ") {
        return Some("C++");
    }

    // ── Languages with no substring overlap ──
    if lower.contains("python") {
        return Some("Python");
    }
    if lower.contains("rust") {
        return Some("Rust");
    }
    if lower.contains("golang") || lower.contains("in go ") || lower.ends_with("in go") {
        return Some("Go");
    }
    if lower.contains("kotlin") {
        return Some("Kotlin");
    }
    if lower.contains("java") {
        return Some("Java");
    }
    if lower.contains("swift") {
        return Some("Swift");
    }
    if lower.contains("dart") {
        return Some("Dart");
    }
    if lower.contains("scala") {
        return Some("Scala");
    }

    // Plain "C" — only as a clearly-delimited word. We accept "in c"
    // (trailing space/newline or end-of-string) and " c code".
    if lower.contains("in c ")
        || lower.contains("in c\n")
        || lower.ends_with("in c")
        || lower.contains(" c code")
    {
        return Some("C");
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "llm", test))]
mod tests {
    use super::*;

    // 1. Every variant round-trips through parse_route_from_llm_output via
    //    its Display form (snake_case).
    #[test]
    fn parse_route_each_variant_round_trips() {
        let variants = [
            AgentRoute::FilePattern,
            AgentRoute::Regex,
            AgentRoute::ExactSymbol,
            AgentRoute::Structural,
            AgentRoute::Deep,
            AgentRoute::Analytical,
            AgentRoute::Exhaustive,
            AgentRoute::Semantic,
            AgentRoute::Architecture,
            AgentRoute::ExhaustiveStructural,
            AgentRoute::DeepWithExamples,
            AgentRoute::FeaturePlanning,
        ];
        for variant in variants {
            let display = format!("{variant}");
            let parsed = parse_route_from_llm_output(&display)
                .unwrap_or_else(|e| panic!("parse failed for {display:?}: {e}"));
            assert_eq!(parsed, variant, "round-trip failed for {display:?}");
        }
    }

    // 2. Accepts snake_case, PascalCase, UPPER, and trimmed whitespace.
    #[test]
    fn parse_route_accepts_both_cases() {
        assert_eq!(
            parse_route_from_llm_output("Deep").unwrap(),
            AgentRoute::Deep
        );
        assert_eq!(
            parse_route_from_llm_output("deep").unwrap(),
            AgentRoute::Deep
        );
        assert_eq!(
            parse_route_from_llm_output("DEEP").unwrap(),
            AgentRoute::Deep
        );
        assert_eq!(
            parse_route_from_llm_output("  deep\n").unwrap(),
            AgentRoute::Deep
        );
    }

    // 3. Unknown strings return Err with "unrecognized" in the message.
    #[test]
    fn parse_route_rejects_unknown() {
        let err = parse_route_from_llm_output("not a route").unwrap_err();
        assert!(
            err.to_string().contains("unrecognized"),
            "expected 'unrecognized' in error, got: {err}"
        );
    }

    // 4. Both the snake_case and no-separator forms are accepted.
    #[test]
    fn parse_route_accepts_both_naming_forms() {
        assert_eq!(
            parse_route_from_llm_output("exact_symbol").unwrap(),
            AgentRoute::ExactSymbol
        );
        assert_eq!(
            parse_route_from_llm_output("exactsymbol").unwrap(),
            AgentRoute::ExactSymbol
        );

        assert_eq!(
            parse_route_from_llm_output("file_pattern").unwrap(),
            AgentRoute::FilePattern
        );
        assert_eq!(
            parse_route_from_llm_output("filepattern").unwrap(),
            AgentRoute::FilePattern
        );

        assert_eq!(
            parse_route_from_llm_output("exhaustive_structural").unwrap(),
            AgentRoute::ExhaustiveStructural
        );
        assert_eq!(
            parse_route_from_llm_output("exhaustivestructural").unwrap(),
            AgentRoute::ExhaustiveStructural
        );

        assert_eq!(
            parse_route_from_llm_output("deep_with_examples").unwrap(),
            AgentRoute::DeepWithExamples
        );
        assert_eq!(
            parse_route_from_llm_output("deepwithexamples").unwrap(),
            AgentRoute::DeepWithExamples
        );

        assert_eq!(
            parse_route_from_llm_output("feature_planning").unwrap(),
            AgentRoute::FeaturePlanning
        );
        assert_eq!(
            parse_route_from_llm_output("featureplanning").unwrap(),
            AgentRoute::FeaturePlanning
        );
    }

    // 5. The classify prompt ends with "Route:" so a chatty LLM may echo it
    //    back. `parse_route_from_llm_output` must strip the label gracefully.
    #[test]
    fn parse_route_strips_label_prefix() {
        assert_eq!(
            parse_route_from_llm_output("Route: deep").unwrap(),
            AgentRoute::Deep
        );
        assert_eq!(
            parse_route_from_llm_output("route: deep").unwrap(),
            AgentRoute::Deep
        );
        assert_eq!(
            parse_route_from_llm_output("Route = structural").unwrap(),
            AgentRoute::Structural
        );
        assert_eq!(
            parse_route_from_llm_output("Answer: feature_planning").unwrap(),
            AgentRoute::FeaturePlanning
        );
        // Combined: prefix + trailing whitespace + camelCase variant
        assert_eq!(
            parse_route_from_llm_output("  Route:  exactSymbol  ").unwrap(),
            AgentRoute::ExactSymbol
        );
        // No prefix is still fine
        assert_eq!(
            parse_route_from_llm_output("architecture").unwrap(),
            AgentRoute::Architecture
        );
    }

    // 6. Language-hint ordering guard. The most common bug here is shorter
    //    language names shadowing longer ones via substring matching
    //    (e.g. "in javascript" → Some("Java")). Every overlap pair must have
    //    an assertion below — if you add a new language, add its overlap
    //    test too.
    #[test]
    fn detect_language_hint_ordering() {
        // The bug this fix exists for.
        assert_eq!(
            detect_language_hint("write a server in javascript"),
            Some("JavaScript")
        );
        assert_eq!(
            detect_language_hint("write a server in typescript"),
            Some("TypeScript")
        );
        assert_eq!(detect_language_hint("write a class in java"), Some("Java"));
        // Short forms
        assert_eq!(detect_language_hint("write this in js"), Some("JavaScript"));
        assert_eq!(detect_language_hint("write this in ts"), Some("TypeScript"));

        // C-family overlap pairs
        assert_eq!(detect_language_hint("write me a class in c#"), Some("C#"));
        assert_eq!(
            detect_language_hint("write me a class in csharp"),
            Some("C#")
        );
        assert_eq!(detect_language_hint("write this in c++"), Some("C++"));
        assert_eq!(detect_language_hint("write this in cplusplus"), Some("C++"));
        assert_eq!(
            detect_language_hint("write a function in objective-c"),
            Some("Objective-C")
        );
        assert_eq!(detect_language_hint("write a function in c "), Some("C"));
        assert_eq!(detect_language_hint("a snippet in c"), Some("C"));

        // Plain Go variants
        assert_eq!(detect_language_hint("write this in golang"), Some("Go"));
        assert_eq!(detect_language_hint("write this in go "), Some("Go"));

        // Non-collision languages still match
        assert_eq!(detect_language_hint("in python please"), Some("Python"));
        assert_eq!(detect_language_hint("rust example"), Some("Rust"));
        assert_eq!(detect_language_hint("kotlin example"), Some("Kotlin"));
        assert_eq!(detect_language_hint("swift example"), Some("Swift"));
        assert_eq!(detect_language_hint("dart example"), Some("Dart"));
        assert_eq!(detect_language_hint("scala example"), Some("Scala"));

        // No language hint → None
        assert_eq!(detect_language_hint("no language hint here"), None);
        assert_eq!(detect_language_hint("how does auth work"), None);
    }

    // 7. Hidden exhaustiveness test: if a new AgentRoute variant is added, this
    //    test fails to compile (no catch-all `_` arm), forcing the author to
    //    update `parse_route_from_llm_output` and this file too.
    #[test]
    fn all_variants_covered_by_parse_exhaustiveness_guard() {
        // This function is never called at runtime — it exists only for the
        // exhaustiveness check at compile time.
        #[allow(dead_code)]
        fn assert_all_covered(route: AgentRoute) {
            // NOTE: No `_ => {}` wildcard. Adding a new variant breaks compile,
            // which is the desired outcome.
            match route {
                AgentRoute::FilePattern => {}
                AgentRoute::Regex => {}
                AgentRoute::ExactSymbol => {}
                AgentRoute::Structural => {}
                AgentRoute::Deep => {}
                AgentRoute::Analytical => {}
                AgentRoute::Exhaustive => {}
                AgentRoute::Semantic => {}
                AgentRoute::Architecture => {}
                AgentRoute::ExhaustiveStructural => {}
                AgentRoute::DeepWithExamples => {}
                AgentRoute::FeaturePlanning => {}
            }
        }
    }
}
