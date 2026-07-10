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
    // Config-as-code formats.
    Terraform,
    Bash,
    PowerShell,
    Protobuf,
    GraphQl,
    Starlark,
    Cmake,
    Groovy,
    Ini,
    Unknown,
}

impl FileType {
    /// Detect file type from file path extension, or from a small set of
    /// conventionally bare/ambiguous filenames checked first (e.g.
    /// `CMakeLists.txt`, `BUILD`) — mirrors `lightonai/colgrep`'s
    /// `detect_language()` (Apache-2.0), which checks filename before
    /// falling through to the same extension match every other language uses.
    pub fn detect(path: &Path) -> Self {
        if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
            match filename.to_lowercase().as_str() {
                "cmakelists.txt" => return FileType::Cmake,
                "build" | "build.bazel" | "workspace" | "workspace.bazel" | "module.bazel" => {
                    return FileType::Starlark;
                }
                _ => {}
            }
        }

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
            "txt" | "log" | "cfg" | "env" => FileType::PlainText,
            "tf" | "tfvars" | "hcl" => FileType::Terraform,
            "sh" | "bash" | "zsh" => FileType::Bash,
            "ps1" | "psm1" | "psd1" => FileType::PowerShell,
            "proto" => FileType::Protobuf,
            "graphql" | "gql" => FileType::GraphQl,
            "bzl" | "star" => FileType::Starlark,
            "cmake" => FileType::Cmake,
            "groovy" | "gradle" => FileType::Groovy,
            "ini" => FileType::Ini,
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
                | FileType::Terraform
                | FileType::Bash
                | FileType::PowerShell
                | FileType::Protobuf
                | FileType::GraphQl
                | FileType::Starlark
                | FileType::Cmake
                | FileType::Groovy
                | FileType::Ini
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
            FileType::Terraform => "terraform",
            FileType::Bash => "bash",
            FileType::PowerShell => "powershell",
            FileType::Protobuf => "protobuf",
            FileType::GraphQl => "graphql",
            FileType::Starlark => "starlark",
            FileType::Cmake => "cmake",
            FileType::Groovy => "groovy",
            FileType::Ini => "ini",
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

    #[test]
    fn test_detect_config_as_code_extensions() {
        assert_eq!(FileType::detect(Path::new("main.tf")), FileType::Terraform);
        assert_eq!(
            FileType::detect(Path::new("vars.tfvars")),
            FileType::Terraform
        );
        assert_eq!(
            FileType::detect(Path::new("network.hcl")),
            FileType::Terraform
        );
        assert_eq!(FileType::detect(Path::new("deploy.sh")), FileType::Bash);
        assert_eq!(FileType::detect(Path::new("setup.bash")), FileType::Bash);
        assert_eq!(FileType::detect(Path::new("profile.zsh")), FileType::Bash);
        assert_eq!(
            FileType::detect(Path::new("Install.ps1")),
            FileType::PowerShell
        );
        assert_eq!(
            FileType::detect(Path::new("Module.psm1")),
            FileType::PowerShell
        );
        assert_eq!(FileType::detect(Path::new("api.proto")), FileType::Protobuf);
        assert_eq!(
            FileType::detect(Path::new("schema.graphql")),
            FileType::GraphQl
        );
        assert_eq!(FileType::detect(Path::new("query.gql")), FileType::GraphQl);
        assert_eq!(FileType::detect(Path::new("rules.bzl")), FileType::Starlark);
        assert_eq!(FileType::detect(Path::new("defs.star")), FileType::Starlark);
        assert_eq!(
            FileType::detect(Path::new("modules.cmake")),
            FileType::Cmake
        );
        assert_eq!(FileType::detect(Path::new("Task.groovy")), FileType::Groovy);
        assert_eq!(
            FileType::detect(Path::new("build.gradle")),
            FileType::Groovy
        );
        assert_eq!(FileType::detect(Path::new("app.ini")), FileType::Ini);
    }

    #[test]
    fn test_detect_config_as_code_bare_filenames() {
        assert_eq!(
            FileType::detect(Path::new("CMakeLists.txt")),
            FileType::Cmake
        );
        assert_eq!(
            FileType::detect(Path::new("modules/cmakelists.txt")),
            FileType::Cmake,
            "filename match must be case-insensitive"
        );
        assert_eq!(FileType::detect(Path::new("BUILD")), FileType::Starlark);
        assert_eq!(
            FileType::detect(Path::new("BUILD.bazel")),
            FileType::Starlark
        );
        assert_eq!(FileType::detect(Path::new("WORKSPACE")), FileType::Starlark);
        assert_eq!(
            FileType::detect(Path::new("WORKSPACE.bazel")),
            FileType::Starlark
        );
        assert_eq!(
            FileType::detect(Path::new("MODULE.bazel")),
            FileType::Starlark
        );
        // .cfg/.env stay on PlainText — only .ini gets structural parsing.
        assert_eq!(
            FileType::detect(Path::new("setup.cfg")),
            FileType::PlainText
        );
        // NOTE: a bare `.env` dotfile has no extension per `Path::extension()`
        // semantics (a filename starting with '.' and containing no other '.'
        // yields None) — this was true before this task too, since the old
        // "env" match arm was only ever reachable for e.g. `foo.env`, never a
        // bare `.env`. Verified unchanged pre/post this task's edit; see
        // task-2-report.md for the repro. Behaviorally moot in practice: the
        // real file walker skips all hidden files before detect() ever runs.
        assert_eq!(FileType::detect(Path::new(".env")), FileType::Unknown);
    }

    #[test]
    fn test_config_as_code_supports_ast_and_language_name() {
        for (ft, name) in [
            (FileType::Terraform, "terraform"),
            (FileType::Bash, "bash"),
            (FileType::PowerShell, "powershell"),
            (FileType::Protobuf, "protobuf"),
            (FileType::GraphQl, "graphql"),
            (FileType::Starlark, "starlark"),
            (FileType::Cmake, "cmake"),
            (FileType::Groovy, "groovy"),
            (FileType::Ini, "ini"),
        ] {
            assert!(ft.supports_ast(), "{ft:?} must support AST chunking");
            assert!(ft.is_text(), "{ft:?} must be a text file type");
            assert_eq!(ft.language_name(), name);
        }
    }

    #[test]
    fn test_starlark_bare_filenames_are_ast_capable() {
        for filename in ["BUILD", "BUILD.bazel", "WORKSPACE", "MODULE.bazel"] {
            let ft = FileType::detect(Path::new(filename));
            assert_eq!(ft, FileType::Starlark, "{filename} must detect as Starlark");
            assert!(ft.supports_ast(), "{filename} must support AST chunking");
        }
    }
}
