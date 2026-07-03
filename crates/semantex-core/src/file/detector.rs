use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    C,
    Cpp,
    Ruby,
    Dart,
    CSharp,
    Scala,
    Php,
    Lua,
    Haskell,
    OCaml,
    Zig,
    R,
    Html,
    Swift,
    Elixir,
    Kotlin,
    Sql,
    Vue,
    Svelte,
    Markdown,
    Toml,
    Yaml,
    Json,
    Pdf,
    PlainText,
    Binary,
    Unknown,
}

impl FileType {
    /// Detect file type from file path extension.
    pub fn detect(path: &Path) -> Self {
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_lowercase(),
            None => return FileType::Unknown,
        };

        match ext.as_str() {
            "rs" => FileType::Rust,
            "py" | "pyi" => FileType::Python,
            "js" | "jsx" | "mjs" => FileType::JavaScript,
            "ts" | "tsx" => FileType::TypeScript,
            "go" => FileType::Go,
            "java" => FileType::Java,
            "c" | "h" => FileType::C,
            "cpp" | "cc" | "cxx" | "hpp" => FileType::Cpp,
            "rb" => FileType::Ruby,
            "dart" => FileType::Dart,
            "cs" => FileType::CSharp,
            "scala" => FileType::Scala,
            "php" => FileType::Php,
            "lua" => FileType::Lua,
            "hs" | "lhs" => FileType::Haskell,
            "ml" | "mli" => FileType::OCaml,
            "zig" => FileType::Zig,
            "r" => FileType::R,
            "html" | "htm" => FileType::Html,
            "swift" => FileType::Swift,
            "ex" | "exs" => FileType::Elixir,
            "kt" | "kts" => FileType::Kotlin,
            "sql" => FileType::Sql,
            "vue" => FileType::Vue,
            "svelte" => FileType::Svelte,
            "md" | "markdown" => FileType::Markdown,
            "toml" => FileType::Toml,
            "yml" | "yaml" => FileType::Yaml,
            "json" => FileType::Json,
            "pdf" => FileType::Pdf,
            "txt" | "log" | "cfg" | "ini" | "env" => FileType::PlainText,
            _ => FileType::Unknown,
        }
    }

    /// Whether this file type supports AST chunking.
    pub fn supports_ast(&self) -> bool {
        matches!(
            self,
            FileType::Rust
                | FileType::Python
                | FileType::JavaScript
                | FileType::TypeScript
                | FileType::Go
                | FileType::Java
                | FileType::C
                | FileType::Cpp
                | FileType::Ruby
                | FileType::Dart
                | FileType::CSharp
                | FileType::Scala
                | FileType::Php
                | FileType::Lua
                | FileType::Haskell
                | FileType::OCaml
                | FileType::Zig
                | FileType::R
                | FileType::Html
                | FileType::Swift
                | FileType::Elixir
                | FileType::Svelte
                | FileType::Vue
                | FileType::Kotlin
                | FileType::Sql
                | FileType::Markdown
                | FileType::Toml
                | FileType::Json
        )
    }

    /// Whether this is a text-based file type.
    pub fn is_text(&self) -> bool {
        !matches!(self, FileType::Binary | FileType::Pdf)
    }

    /// Get the language name string for this file type.
    pub fn language_name(&self) -> &'static str {
        match self {
            FileType::Rust => "rust",
            FileType::Python => "python",
            FileType::JavaScript => "javascript",
            FileType::TypeScript => "typescript",
            FileType::Go => "go",
            FileType::Java => "java",
            FileType::C => "c",
            FileType::Cpp => "cpp",
            FileType::Ruby => "ruby",
            FileType::Dart => "dart",
            FileType::CSharp => "csharp",
            FileType::Scala => "scala",
            FileType::Php => "php",
            FileType::Lua => "lua",
            FileType::Haskell => "haskell",
            FileType::OCaml => "ocaml",
            FileType::Zig => "zig",
            FileType::R => "r",
            FileType::Html => "html",
            FileType::Swift => "swift",
            FileType::Elixir => "elixir",
            FileType::Kotlin => "kotlin",
            FileType::Sql => "sql",
            FileType::Vue => "vue",
            FileType::Svelte => "svelte",
            FileType::Markdown => "markdown",
            FileType::Toml => "toml",
            FileType::Yaml => "yaml",
            FileType::Json => "json",
            FileType::Pdf => "pdf",
            FileType::PlainText => "plaintext",
            FileType::Binary => "binary",
            FileType::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_known_types() {
        assert_eq!(FileType::detect(Path::new("main.rs")), FileType::Rust);
        assert_eq!(FileType::detect(Path::new("app.py")), FileType::Python);
        assert_eq!(
            FileType::detect(Path::new("index.js")),
            FileType::JavaScript
        );
        assert_eq!(FileType::detect(Path::new("app.tsx")), FileType::TypeScript);
        assert_eq!(FileType::detect(Path::new("main.go")), FileType::Go);
        assert_eq!(FileType::detect(Path::new("App.java")), FileType::Java);
        assert_eq!(FileType::detect(Path::new("lib.c")), FileType::C);
        assert_eq!(FileType::detect(Path::new("lib.h")), FileType::C);
        assert_eq!(FileType::detect(Path::new("lib.cpp")), FileType::Cpp);
        assert_eq!(FileType::detect(Path::new("lib.hpp")), FileType::Cpp);
        assert_eq!(FileType::detect(Path::new("app.rb")), FileType::Ruby);
        assert_eq!(FileType::detect(Path::new("README.md")), FileType::Markdown);
        assert_eq!(FileType::detect(Path::new("Cargo.toml")), FileType::Toml);
        assert_eq!(FileType::detect(Path::new("config.yaml")), FileType::Yaml);
        assert_eq!(FileType::detect(Path::new("data.json")), FileType::Json);
        assert_eq!(FileType::detect(Path::new("doc.pdf")), FileType::Pdf);
        assert_eq!(
            FileType::detect(Path::new("notes.txt")),
            FileType::PlainText
        );
    }

    #[test]
    fn test_dart_detection() {
        assert_eq!(FileType::detect(Path::new("lib/main.dart")), FileType::Dart);
        assert_eq!(
            FileType::detect(Path::new("test/widget_test.dart")),
            FileType::Dart
        );
        assert!(FileType::Dart.supports_ast());
        assert!(FileType::Dart.is_text());
        assert_eq!(FileType::Dart.language_name(), "dart");
    }

    #[test]
    fn test_detect_unknown() {
        assert_eq!(FileType::detect(Path::new("noext")), FileType::Unknown);
        assert_eq!(FileType::detect(Path::new("file.xyz")), FileType::Unknown);
    }

    #[test]
    fn test_supports_ast() {
        assert!(FileType::Rust.supports_ast());
        assert!(FileType::Python.supports_ast());
        assert!(FileType::Sql.supports_ast());
        assert!(!FileType::Pdf.supports_ast());
        assert!(!FileType::Binary.supports_ast());
        assert!(!FileType::PlainText.supports_ast());
        assert!(!FileType::Yaml.supports_ast());
    }

    #[test]
    fn test_is_text() {
        assert!(FileType::Rust.is_text());
        assert!(FileType::PlainText.is_text());
        assert!(FileType::Unknown.is_text());
        assert!(!FileType::Binary.is_text());
        assert!(!FileType::Pdf.is_text());
    }

    #[test]
    fn test_language_name() {
        assert_eq!(FileType::Rust.language_name(), "rust");
        assert_eq!(FileType::Cpp.language_name(), "cpp");
        assert_eq!(FileType::Unknown.language_name(), "unknown");
    }
}
