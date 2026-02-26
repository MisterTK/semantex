use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

/// A code-aware tokenizer that splits identifiers by camelCase, PascalCase,
/// snake_case, dot.paths, and Rust::paths, then emits each sub-token lowercased
/// plus a joined lowercased form at the same position as the first sub-token.
///
/// Identifier boundaries: characters that are NOT alphanumeric and NOT underscore
/// separate distinct identifiers. Underscores split within an identifier but
/// the parts are still considered one identifier (so a joined form is emitted).
#[derive(Clone, Default)]
pub struct CodeTokenizer;

pub struct CodeTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

impl Tokenizer for CodeTokenizer {
    type TokenStream<'a> = CodeTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> CodeTokenStream {
        let tokens = tokenize_code(text);
        CodeTokenStream { tokens, index: 0 }
    }
}

impl TokenStream for CodeTokenStream {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.index - 1]
    }
}

/// Split a camelCase/PascalCase word into sub-words.
///
/// Rules:
/// - `[a-z][A-Z]` boundary: `getUserById` -> `get|User|By|Id`
/// - `[A-Z][A-Z][a-z]` boundary: `XMLParser` -> `XML|Parser`
fn split_camel_case(word: &str) -> Vec<&str> {
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

/// Expands identifiers in code content for BM25 indexing.
/// Splits camelCase/snake_case identifiers into component words.
/// Returns space-separated expansion string to prepend to BM25 content.
pub fn expand_identifiers(content: &str) -> String {
    let mut expansions = Vec::new();

    let spans = extract_identifier_spans(content);
    for (span, _, _) in spans {
        if span.len() < 4 {
            continue;
        }

        // Split on underscores, then camelCase within each part
        let underscore_parts: Vec<&str> = span.split('_').filter(|s| !s.is_empty()).collect();
        let mut sub_tokens: Vec<String> = Vec::new();
        for part in &underscore_parts {
            let camel_parts = split_camel_case(part);
            for cp in camel_parts {
                if !cp.is_empty() {
                    sub_tokens.push(cp.to_lowercase());
                }
            }
        }

        // Only include if the token actually splits into multiple parts
        if sub_tokens.len() > 1 {
            expansions.push(sub_tokens.join(" "));
        }
    }

    expansions.sort();
    expansions.dedup();
    expansions.join(" ")
}

/// Returns true if the character is part of an identifier (alphanumeric or underscore).
fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Extract identifier spans from text. An identifier span is a maximal run of
/// alphanumeric and underscore characters. Returns (span_text, byte_offset_from, byte_offset_to).
fn extract_identifier_spans(text: &str) -> Vec<(&str, usize, usize)> {
    let mut result = Vec::new();
    let mut span_start: Option<usize> = None;

    for (i, c) in text.char_indices() {
        if is_ident_char(c) {
            if span_start.is_none() {
                span_start = Some(i);
            }
        } else if let Some(start) = span_start.take() {
            result.push((&text[start..i], start, i));
        }
    }

    if let Some(start) = span_start {
        result.push((&text[start..], start, text.len()));
    }

    result
}

/// Tokenize code text into a list of tokens with positions.
fn tokenize_code(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut position: usize = 0;

    let spans = extract_identifier_spans(text);

    for (span, offset_from, offset_to) in spans {
        // Split on underscores first, then camelCase within each part
        let underscore_parts: Vec<&str> = span.split('_').filter(|s| !s.is_empty()).collect();

        // Collect all sub-tokens from this identifier
        let mut sub_tokens: Vec<String> = Vec::new();

        for part in &underscore_parts {
            let camel_parts = split_camel_case(part);
            for cp in camel_parts {
                if !cp.is_empty() {
                    sub_tokens.push(cp.to_lowercase());
                }
            }
        }

        if sub_tokens.is_empty() {
            continue;
        }

        let first_position = position;

        // Emit each sub-token
        for st in &sub_tokens {
            tokens.push(Token {
                offset_from,
                offset_to,
                position,
                text: st.clone(),
                position_length: 1,
            });
            position += 1;
        }

        // Emit joined form if there are multiple sub-tokens
        if sub_tokens.len() > 1 {
            let joined: String = sub_tokens.concat();
            tokens.push(Token {
                offset_from,
                offset_to,
                position: first_position,
                text: joined,
                position_length: sub_tokens.len(),
            });
        }
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use tantivy::tokenizer::{TextAnalyzer, Token};

    fn tokenize(text: &str) -> Vec<Token> {
        let mut analyzer = TextAnalyzer::from(CodeTokenizer);
        let mut stream = analyzer.token_stream(text);
        let mut tokens = Vec::new();
        stream.process(&mut |token: &Token| {
            tokens.push(token.clone());
        });
        tokens
    }

    fn texts(tokens: &[Token]) -> Vec<&str> {
        tokens.iter().map(|t| t.text.as_str()).collect()
    }

    #[test]
    fn test_camel_case() {
        let tokens = tokenize("getUserById");
        assert_eq!(
            texts(&tokens),
            vec!["get", "user", "by", "id", "getuserbyid"]
        );
    }

    #[test]
    fn test_snake_case() {
        let tokens = tokenize("get_user_by_id");
        assert_eq!(
            texts(&tokens),
            vec!["get", "user", "by", "id", "getuserbyid"]
        );
    }

    #[test]
    fn test_pascal_case() {
        let tokens = tokenize("ConnectionServiceFactory");
        assert_eq!(
            texts(&tokens),
            vec![
                "connection",
                "service",
                "factory",
                "connectionservicefactory"
            ]
        );
    }

    #[test]
    fn test_all_caps() {
        let tokens = tokenize("MAX_RETRY_COUNT");
        assert_eq!(
            texts(&tokens),
            vec!["max", "retry", "count", "maxretrycount"]
        );
    }

    #[test]
    fn test_dot_path() {
        let tokens = tokenize("com.example.MyClass");
        assert_eq!(
            texts(&tokens),
            vec!["com", "example", "my", "class", "myclass"]
        );
    }

    #[test]
    fn test_rust_path() {
        let tokens = tokenize("std::collections::HashMap");
        assert_eq!(
            texts(&tokens),
            vec!["std", "collections", "hash", "map", "hashmap"]
        );
    }

    #[test]
    fn test_mixed_natural() {
        let tokens = tokenize("OAuth PKCE token refresh");
        // OAuth -> single token (uppercase prefix too short to split)
        // PKCE -> single all-caps -> [pkce]
        // token -> single -> [token]
        // refresh -> single -> [refresh]
        let token_texts = texts(&tokens);
        assert_eq!(token_texts, vec!["oauth", "pkce", "token", "refresh"]);
    }

    #[test]
    fn test_xml_parser() {
        let tokens = tokenize("XMLParser");
        assert_eq!(texts(&tokens), vec!["xml", "parser", "xmlparser"]);
    }

    #[test]
    fn test_fhir_base_url() {
        let tokens = tokenize("fhirBaseUrl");
        assert_eq!(texts(&tokens), vec!["fhir", "base", "url", "fhirbaseurl"]);
    }

    #[test]
    fn test_single_word() {
        let tokens = tokenize("hello");
        assert_eq!(texts(&tokens), vec!["hello"]);
        assert_eq!(tokens[0].position, 0);
    }

    #[test]
    fn test_single_component_identifier() {
        // Single-component identifiers should still emit the token
        // (no separate joined form since there's only one part)
        let tokens = tokenize("count");
        assert_eq!(texts(&tokens), vec!["count"]);
    }

    #[test]
    fn test_empty_string() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_positions() {
        let tokens = tokenize("getUserById");
        // Sub-tokens: get(0), user(1), by(2), id(3)
        // Joined: getuserbyid(0) -- same position as first sub-token
        assert_eq!(tokens[0].position, 0); // get
        assert_eq!(tokens[1].position, 1); // user
        assert_eq!(tokens[2].position, 2); // by
        assert_eq!(tokens[3].position, 3); // id
        assert_eq!(tokens[4].position, 0); // getuserbyid (same as "get")
    }

    #[test]
    fn test_multiple_identifiers() {
        let tokens = tokenize("getUserById maxRetryCount");
        let token_texts = texts(&tokens);
        assert_eq!(
            token_texts,
            vec![
                "get",
                "user",
                "by",
                "id",
                "getuserbyid",
                "max",
                "retry",
                "count",
                "maxretrycount"
            ]
        );
        // Second group starts at position 4
        assert_eq!(tokens[5].position, 4); // max
        assert_eq!(tokens[6].position, 5); // retry
        assert_eq!(tokens[7].position, 6); // count
        assert_eq!(tokens[8].position, 4); // maxretrycount
    }

    #[test]
    fn test_arrow_separator() {
        let tokens = tokenize("self->getValue");
        assert_eq!(texts(&tokens), vec!["self", "get", "value", "getvalue"]);
    }

    #[test]
    fn test_slash_separator() {
        let tokens = tokenize("api/v2/users");
        assert_eq!(texts(&tokens), vec!["api", "v2", "users"]);
    }

    #[test]
    fn test_parens_braces() {
        let tokens = tokenize("foo(bar, baz)");
        assert_eq!(texts(&tokens), vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn test_numeric_in_identifier() {
        let tokens = tokenize("base64Encode");
        assert_eq!(texts(&tokens), vec!["base64", "encode", "base64encode"]);
    }

    #[test]
    fn test_leading_underscores() {
        // Leading underscores produce empty splits that are filtered
        let tokens = tokenize("__init__");
        assert_eq!(texts(&tokens), vec!["init"]);
    }

    #[test]
    fn test_expand_identifiers_camel_case() {
        let result = expand_identifiers("retryWithBackoff(maxRetries, delay)");
        assert!(result.contains("retry with backoff"));
        assert!(result.contains("max retries"));
        assert!(!result.contains("delay")); // too short (<4 chars)
    }

    #[test]
    fn test_expand_identifiers_snake_case() {
        let result = expand_identifiers("get_user_by_id(user_name)");
        // "get_user_by_id" splits to "get user by id"
        // "user_name" splits to "user name"
        assert!(result.contains("get user by id"));
        assert!(result.contains("user name"));
    }

    #[test]
    fn test_expand_identifiers_pascal_case() {
        let result = expand_identifiers("class ConnectionServiceFactory");
        assert!(result.contains("connection service factory"));
    }

    #[test]
    fn test_expand_identifiers_short_tokens_skipped() {
        let result = expand_identifiers("if (a == b) { foo(); }");
        assert!(result.is_empty()); // all tokens < 4 chars
    }

    #[test]
    fn test_expand_identifiers_no_split_needed() {
        let result = expand_identifiers("simple token here");
        assert!(result.is_empty()); // single-component tokens don't need expansion
    }
}
