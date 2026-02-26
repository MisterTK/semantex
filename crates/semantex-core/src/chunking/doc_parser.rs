//! Structured docstring/doc-comment parsing for NL summary enrichment.
//!
//! Extracts `@param`, `@returns`, `@throws`, `@deprecated` and similar tags from
//! JSDoc, Javadoc, Python Google-style, and Rust doc comment formats.

use super::structured_meta::DocTag;

/// Parse doc comment into structured tags.
///
/// Supported formats:
/// - Universal: `@param`, `@returns`, `@throws`, `@deprecated`, `@see`, `@example`
/// - Python Google-style: `Args:`, `Returns:`, `Raises:` sections
/// - Rust: `# Arguments`, `# Returns`, `# Errors`, `# Panics` headings
#[must_use]
pub fn parse_doc_tags(docstring: &str, language: &str) -> Vec<DocTag> {
    let mut tags = Vec::new();

    // Universal @tag parsing
    parse_at_tags(docstring, &mut tags);

    // Language-specific formats
    match language {
        "python" => parse_google_docstring(docstring, &mut tags),
        "rust" => parse_rust_doc_sections(docstring, &mut tags),
        _ => {}
    }

    tags
}

/// Strip leading comment decoration from a line (`*`, `/`, `#`, whitespace).
fn strip_decoration(line: &str) -> &str {
    let trimmed = line.trim_start();
    // Strip Javadoc/JSDoc leading `* ` or `*/`
    if let Some(rest) = trimmed.strip_prefix("* ") {
        return rest;
    }
    if let Some(rest) = trimmed.strip_prefix('*') {
        return rest.trim_start();
    }
    // Strip `///` or `//!` (Rust doc comments)
    if let Some(rest) = trimmed.strip_prefix("///") {
        return rest.strip_prefix(' ').unwrap_or(rest);
    }
    if let Some(rest) = trimmed.strip_prefix("//!") {
        return rest.strip_prefix(' ').unwrap_or(rest);
    }
    // Strip `# ` (Python doc comments in some styles)
    if let Some(rest) = trimmed.strip_prefix("# ") {
        return rest;
    }
    trimmed
}

/// Parse universal `@tag` annotations (JSDoc, Javadoc, PHPDoc, etc.).
///
/// Recognises patterns like:
/// - `@param {Type} name - description`
/// - `@param name description`
/// - `@returns description`
/// - `@deprecated description`
fn parse_at_tags(docstring: &str, tags: &mut Vec<DocTag>) {
    for line in docstring.lines() {
        let cleaned = strip_decoration(line);
        let Some(rest) = cleaned.strip_prefix('@') else {
            continue;
        };

        // Split into tag name and the remainder.
        let (tag_name, remainder) = match rest.find(char::is_whitespace) {
            Some(pos) => (&rest[..pos], rest[pos..].trim_start()),
            None => (rest, ""),
        };

        let tag_name = tag_name.to_lowercase();

        // For @param / @arg / @argument, try to extract the parameter name.
        if tag_name == "param" || tag_name == "arg" || tag_name == "argument" {
            let (name, text) = parse_param_remainder(remainder);
            tags.push(DocTag {
                tag: "param".to_string(),
                name: Some(name.to_string()),
                text: text.to_string(),
            });
        } else {
            tags.push(DocTag {
                tag: tag_name,
                name: None,
                text: remainder.to_string(),
            });
        }
    }
}

/// Parse the remainder after `@param` to extract the parameter name and description.
///
/// Handles:
/// - `{Type} name - description`
/// - `{Type} name description`
/// - `name - description`
/// - `name description`
fn parse_param_remainder(remainder: &str) -> (&str, &str) {
    let mut s = remainder;

    // Skip optional `{Type}` block.
    #[allow(clippy::collapsible_if)]
    if s.starts_with('{') {
        if let Some(close) = s.find('}') {
            s = s[close + 1..].trim_start();
        }
    }

    // Next token is the parameter name.
    let (name, rest) = match s.find(char::is_whitespace) {
        Some(pos) => (&s[..pos], s[pos..].trim_start()),
        None => (s, ""),
    };

    // Strip optional leading `- ` from description.
    let text = rest.strip_prefix("- ").unwrap_or(rest);

    (name, text)
}

/// Parse Python Google-style docstring sections.
///
/// Looks for section headers like `Args:`, `Returns:`, `Raises:`, `Yields:`,
/// `Note:`, `Example:`. Indented lines after a header become the content.
fn parse_google_docstring(docstring: &str, tags: &mut Vec<DocTag>) {
    let lines: Vec<&str> = docstring.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        let section_tag = match trimmed {
            "Args:" | "Arguments:" | "Parameters:" => Some("param"),
            "Returns:" | "Return:" => Some("returns"),
            "Raises:" | "Except:" | "Exceptions:" => Some("throws"),
            "Yields:" | "Yield:" => Some("yields"),
            "Note:" | "Notes:" => Some("note"),
            "Example:" | "Examples:" => Some("example"),
            _ => None,
        };

        let Some(section) = section_tag else {
            i += 1;
            continue;
        };

        i += 1;

        // Collect indented lines belonging to this section.
        if section == "param" {
            // For Args, each indented line with `name: desc` or `name (type): desc`
            // becomes a separate param tag.
            while i < lines.len() && is_indented(lines[i]) {
                let entry = lines[i].trim();
                if entry.is_empty() {
                    i += 1;
                    continue;
                }

                let (name, text) = parse_google_arg_entry(entry);
                // Collect continuation lines (more deeply indented).
                let mut full_text = text.to_string();
                let entry_indent = indent_level(lines[i]);
                i += 1;
                while i < lines.len()
                    && is_indented(lines[i])
                    && indent_level(lines[i]) > entry_indent
                {
                    let cont = lines[i].trim();
                    if !cont.is_empty() {
                        full_text.push(' ');
                        full_text.push_str(cont);
                    }
                    i += 1;
                }

                tags.push(DocTag {
                    tag: "param".to_string(),
                    name: Some(name.to_string()),
                    text: full_text,
                });
            }
        } else {
            // For non-param sections, collect all indented lines as the text body.
            let mut body_parts: Vec<&str> = Vec::new();
            while i < lines.len() && is_indented(lines[i]) {
                let content = lines[i].trim();
                if !content.is_empty() {
                    body_parts.push(content);
                }
                i += 1;
            }
            if !body_parts.is_empty() {
                tags.push(DocTag {
                    tag: section.to_string(),
                    name: None,
                    text: body_parts.join(" "),
                });
            }
        }
    }
}

/// Check whether a line is indented (starts with whitespace).
fn is_indented(line: &str) -> bool {
    line.starts_with(' ') || line.starts_with('\t')
}

/// Return the number of leading spaces (tabs count as 4).
fn indent_level(line: &str) -> usize {
    let mut level = 0;
    for ch in line.chars() {
        match ch {
            ' ' => level += 1,
            '\t' => level += 4,
            _ => break,
        }
    }
    level
}

/// Parse a single Google-style arg entry: `name (type): description` or `name: description`.
fn parse_google_arg_entry(entry: &str) -> (&str, &str) {
    // Try `name (type): desc` or `name: desc`.
    if let Some(colon_pos) = entry.find(':') {
        let before_colon = entry[..colon_pos].trim();
        let desc = entry[colon_pos + 1..].trim();

        // The name is the first token before any parenthesized type.
        let name = match before_colon.find('(') {
            Some(paren_pos) => before_colon[..paren_pos].trim(),
            None => before_colon,
        };

        (name, desc)
    } else {
        // No colon found; treat the whole thing as name with no description.
        (entry, "")
    }
}

/// Parse Rust doc comment sections (`# Arguments`, `# Returns`, etc.).
fn parse_rust_doc_sections(docstring: &str, tags: &mut Vec<DocTag>) {
    let lines: Vec<&str> = docstring.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        let section_tag = match trimmed {
            "# Arguments" | "# Params" | "# Parameters" => Some("param"),
            "# Returns" | "# Return" => Some("returns"),
            "# Errors" => Some("errors"),
            "# Panics" => Some("panics"),
            "# Safety" => Some("safety"),
            "# Examples" | "# Example" => Some("example"),
            _ => None,
        };

        let Some(section) = section_tag else {
            i += 1;
            continue;
        };

        i += 1;

        // Skip blank lines after heading.
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }

        if section == "param" {
            // Look for `* \`name\` - description` bullet items.
            while i < lines.len() {
                let bullet = lines[i].trim();
                if bullet.is_empty() {
                    i += 1;
                    continue;
                }
                // Stop at next heading.
                if bullet.starts_with("# ") {
                    break;
                }

                #[allow(clippy::collapsible_if)]
                if let Some(rest) = bullet
                    .strip_prefix("* ")
                    .or_else(|| bullet.strip_prefix("- "))
                {
                    if let Some((name, desc)) = parse_rust_bullet_param(rest) {
                        tags.push(DocTag {
                            tag: "param".to_string(),
                            name: Some(name.to_string()),
                            text: desc.to_string(),
                        });
                    }
                }
                i += 1;
            }
        } else {
            // Collect body until next heading or end.
            let mut body_parts: Vec<&str> = Vec::new();
            while i < lines.len() {
                let content = lines[i].trim();
                if content.starts_with("# ") {
                    break;
                }
                if !content.is_empty() {
                    body_parts.push(content);
                }
                i += 1;
            }
            if !body_parts.is_empty() {
                tags.push(DocTag {
                    tag: section.to_string(),
                    name: None,
                    text: body_parts.join(" "),
                });
            }
        }
    }
}

/// Parse a Rust doc bullet item: `` `name` - description `` -> `(name, description)`.
fn parse_rust_bullet_param(item: &str) -> Option<(&str, &str)> {
    let s = item.trim();

    // Try backtick-quoted name: `name` - desc
    #[allow(clippy::collapsible_if)]
    if let Some(rest) = s.strip_prefix('`') {
        if let Some(end_tick) = rest.find('`') {
            let name = &rest[..end_tick];
            let after = rest[end_tick + 1..].trim();
            let desc = after.strip_prefix('-').map_or(after, |d| d.trim_start());
            return Some((name, desc));
        }
    }

    // Fallback: name - desc (no backticks)
    if let Some(dash_pos) = s.find(" - ") {
        let name = s[..dash_pos].trim();
        let desc = s[dash_pos + 3..].trim();
        return Some((name, desc));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsdoc_param() {
        let doc = "@param {string} name - The user's name\n@returns {User} The created user";
        let tags = parse_doc_tags(doc, "javascript");
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].tag, "param");
        assert_eq!(tags[0].name.as_deref(), Some("name"));
        assert!(tags[0].text.contains("user"));
        assert_eq!(tags[1].tag, "returns");
    }

    #[test]
    fn test_jsdoc_multiple_params() {
        let doc = "/**\n * @param {number} x - The X coordinate\n * @param {number} y - The Y coordinate\n * @returns {Point} A new point\n */";
        let tags = parse_doc_tags(doc, "javascript");
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0].name.as_deref(), Some("x"));
        assert_eq!(tags[1].name.as_deref(), Some("y"));
        assert_eq!(tags[2].tag, "returns");
    }

    #[test]
    fn test_javadoc_param_no_type() {
        let doc = "@param name the user name\n@param age the user age";
        let tags = parse_doc_tags(doc, "java");
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].name.as_deref(), Some("name"));
        assert!(tags[0].text.contains("user name"));
        assert_eq!(tags[1].name.as_deref(), Some("age"));
    }

    #[test]
    fn test_python_google_style() {
        let doc = "Create a new user.\n\nArgs:\n    name: The user's name\n    age: The user's age\n\nReturns:\n    A new User object";
        let tags = parse_doc_tags(doc, "python");
        assert!(
            tags.iter()
                .any(|t| t.tag == "param" && t.name.as_deref() == Some("name"))
        );
        assert!(
            tags.iter()
                .any(|t| t.tag == "param" && t.name.as_deref() == Some("age"))
        );
        assert!(tags.iter().any(|t| t.tag == "returns"));
    }

    #[test]
    fn test_python_google_with_types() {
        let doc = "Args:\n    name (str): The user's name\n    age (int): The user's age";
        let tags = parse_doc_tags(doc, "python");
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].name.as_deref(), Some("name"));
        assert_eq!(tags[1].name.as_deref(), Some("age"));
    }

    #[test]
    fn test_python_raises() {
        let doc = "Raises:\n    ValueError: If input is invalid\n    TypeError: If type mismatch";
        let tags = parse_doc_tags(doc, "python");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].tag, "throws");
        assert!(tags[0].text.contains("ValueError"));
    }

    #[test]
    fn test_rust_doc_sections() {
        let doc = "# Arguments\n\n* `config` - The connection configuration\n* `timeout` - Optional timeout in seconds\n\n# Returns\n\nA new connection handle\n\n# Errors\n\nReturns error if connection fails";
        let tags = parse_doc_tags(doc, "rust");
        assert!(
            tags.iter()
                .any(|t| t.tag == "param" && t.name.as_deref() == Some("config"))
        );
        assert!(
            tags.iter()
                .any(|t| t.tag == "param" && t.name.as_deref() == Some("timeout"))
        );
        assert!(tags.iter().any(|t| t.tag == "returns"));
        assert!(tags.iter().any(|t| t.tag == "errors"));
    }

    #[test]
    fn test_rust_panics_and_safety() {
        let doc = "# Panics\n\nPanics if index is out of bounds\n\n# Safety\n\nCaller must ensure pointer is valid";
        let tags = parse_doc_tags(doc, "rust");
        assert!(tags.iter().any(|t| t.tag == "panics"));
        assert!(tags.iter().any(|t| t.tag == "safety"));
    }

    #[test]
    fn test_deprecated() {
        let doc = "@deprecated Use newMethod instead";
        let tags = parse_doc_tags(doc, "java");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].tag, "deprecated");
        assert!(tags[0].text.contains("newMethod"));
    }

    #[test]
    fn test_throws() {
        let doc = "@throws {Error} If connection fails\n@throws {TypeError} If arg is invalid";
        let tags = parse_doc_tags(doc, "javascript");
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].tag, "throws");
        assert_eq!(tags[1].tag, "throws");
    }

    #[test]
    fn test_see_tag() {
        let doc = "@see OtherClass#method";
        let tags = parse_doc_tags(doc, "java");
        assert_eq!(tags[0].tag, "see");
        assert!(tags[0].text.contains("OtherClass"));
    }

    #[test]
    fn test_empty_docstring() {
        let tags = parse_doc_tags("", "rust");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_no_tags_in_plain_text() {
        let doc = "This is a plain docstring with no tags.";
        let tags = parse_doc_tags(doc, "javascript");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_strip_decoration_javadoc() {
        assert_eq!(
            strip_decoration("   * @param x the value"),
            "@param x the value"
        );
    }

    #[test]
    fn test_strip_decoration_rust() {
        assert_eq!(strip_decoration("/// # Arguments"), "# Arguments");
    }

    #[test]
    fn test_mixed_at_tags_in_decorated_block() {
        let doc = "/**\n * Create a user.\n * @param {string} name - User name\n * @returns {boolean} success\n */";
        let tags = parse_doc_tags(doc, "javascript");
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].tag, "param");
        assert_eq!(tags[1].tag, "returns");
    }

    #[test]
    fn test_arg_alias() {
        let doc = "@arg x - the input value";
        let tags = parse_doc_tags(doc, "javascript");
        assert_eq!(tags[0].tag, "param");
        assert_eq!(tags[0].name.as_deref(), Some("x"));
    }

    #[test]
    fn test_rust_dash_bullets() {
        let doc = "# Arguments\n\n- `x` - first value\n- `y` - second value";
        let tags = parse_doc_tags(doc, "rust");
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].name.as_deref(), Some("x"));
        assert_eq!(tags[1].name.as_deref(), Some("y"));
    }
}
