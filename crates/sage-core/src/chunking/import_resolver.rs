//! Import/use declaration extraction and file path resolution.
//!
//! Extracts import statements from tree-sitter AST nodes and resolves
//! them to relative file paths within the project when possible.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Extract import declarations from a file's AST root node.
///
/// Returns the raw import text for each import statement found.
pub fn extract_imports(root: &tree_sitter::Node, source: &[u8], language: &str) -> Vec<String> {
    let mut imports = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        let kind = child.kind();
        let is_import = match language {
            "rust" => kind == "use_declaration",
            "python" => kind == "import_statement" || kind == "import_from_statement",
            "javascript" | "typescript" | "tsx" | "jsx" => {
                kind == "import_statement" || kind == "import_declaration"
            }
            "go" | "java" => kind == "import_declaration",
            "c" | "cpp" => kind == "preproc_include",
            "dart" => kind == "import_directive" || kind == "export_directive",
            "c_sharp" => kind == "using_directive",
            _ => false,
        };

        #[allow(clippy::collapsible_if)]
        if is_import {
            if let Ok(text) = child.utf8_text(source) {
                imports.push(text.to_string());
            }
        }

        // Go: individual import specs are nested inside import_declaration
        if language == "go" && kind == "import_declaration" {
            let mut inner_cursor = child.walk();
            for inner in child.children(&mut inner_cursor) {
                #[allow(clippy::collapsible_if)]
                if inner.kind() == "import_spec" || inner.kind() == "import_spec_list" {
                    if let Ok(text) = inner.utf8_text(source) {
                        imports.push(text.to_string());
                    }
                }
            }
        }
    }

    imports
}

/// Attempt to resolve an import statement to a relative file path within the project.
///
/// Returns `None` for external/stdlib imports or unresolvable paths.
#[allow(clippy::implicit_hasher)]
pub fn resolve_import_path(
    import_text: &str,
    language: &str,
    current_file: &Path,
    project_files: &HashSet<PathBuf>,
) -> Option<PathBuf> {
    match language {
        "rust" => resolve_rust_import(import_text, current_file, project_files),
        "python" => resolve_python_import(import_text, current_file, project_files),
        "javascript" | "typescript" | "tsx" | "jsx" => {
            resolve_js_import(import_text, current_file, project_files)
        }
        "go" => resolve_go_import(import_text, project_files),
        "java" => resolve_java_import(import_text, project_files),
        "c" | "cpp" => resolve_c_include(import_text, current_file, project_files),
        "dart" => resolve_dart_import(import_text, project_files),
        "c_sharp" => resolve_csharp_using(import_text, project_files),
        _ => None,
    }
}

/// Resolve a Rust `use` declaration to a project-relative file path.
///
/// Handles `use crate::...` and `use super::...` forms.
/// Skips stdlib (`std`, `core`, `alloc`) and external crate imports.
fn resolve_rust_import(
    import_text: &str,
    current_file: &Path,
    project_files: &HashSet<PathBuf>,
) -> Option<PathBuf> {
    // Strip "use " prefix and trailing ";"
    let trimmed = import_text
        .trim()
        .strip_prefix("use ")?
        .trim_end_matches(';')
        .trim();

    // Strip any trailing brace-group (e.g. `crate::foo::{Bar, Baz}` -> `crate::foo`)
    let path_part = if let Some(brace_pos) = trimmed.find('{') {
        trimmed[..brace_pos].trim_end_matches("::")
    } else {
        // Also strip the last segment if it looks like a type/function import
        // e.g. `crate::foo::Bar` -> try `crate::foo`
        trimmed
    };

    if path_part.starts_with("std::")
        || path_part.starts_with("core::")
        || path_part.starts_with("alloc::")
    {
        return None;
    }

    if let Some(module_path) = path_part.strip_prefix("crate::") {
        // Convention: Rust crate roots live under `src/`. This covers the vast
        // majority of real-world crate layouts. Custom `[lib] path` settings in
        // Cargo.toml are not detectable without parsing the manifest and are
        // exceedingly rare, so we accept this default.
        return resolve_rust_module_path(module_path, Path::new("src"), project_files);
    }

    if let Some(module_path) = path_part.strip_prefix("self::") {
        // `self::foo` resolves relative to the current module directory
        let current_dir = current_file.parent()?;
        return resolve_rust_module_path(module_path, current_dir, project_files);
    }

    if path_part.starts_with("super::") {
        // In Rust, `super` from `src/chunking/ast_chunker.rs` refers to the
        // `chunking` module (= `src/chunking/`). Each additional `super::`
        // goes up one more directory.
        let current_dir = current_file.parent()?;
        let mut base = current_dir.to_path_buf();
        let mut remaining = path_part;

        // First `super::` stays at current_dir (the parent module).
        remaining = remaining.strip_prefix("super::").unwrap_or(remaining);
        // Subsequent `super::` go up one directory each.
        while let Some(rest) = remaining.strip_prefix("super::") {
            base = base.parent()?.to_path_buf();
            remaining = rest;
        }

        return resolve_rust_module_path(remaining, &base, project_files);
    }

    // External crate import -- not resolvable within the project
    None
}

/// Try to resolve a `::` separated Rust module path relative to a base directory.
///
/// Tries both `base/a/b.rs` and `base/a/b/mod.rs` forms.
fn resolve_rust_module_path(
    module_path: &str,
    base: &Path,
    project_files: &HashSet<PathBuf>,
) -> Option<PathBuf> {
    let segments: Vec<&str> = module_path.split("::").collect();
    if segments.is_empty() {
        return None;
    }

    // Try progressively shorter prefixes (the tail segments may be types/functions)
    for take in (1..=segments.len()).rev() {
        let dir_path: PathBuf = segments[..take].iter().collect();
        let as_file = base.join(dir_path.with_extension("rs"));
        if project_files.contains(&as_file) {
            return Some(as_file);
        }

        let as_mod = base.join(&dir_path).join("mod.rs");
        if project_files.contains(&as_mod) {
            return Some(as_mod);
        }
    }

    None
}

/// Resolve a Python import to a project-relative file path.
///
/// Handles relative imports (`from .foo import ...`, `from ..bar import ...`)
/// and absolute imports (`from mypackage.utils import ...`).
/// Skips standard-library and obviously external packages.
fn resolve_python_import(
    import_text: &str,
    current_file: &Path,
    project_files: &HashSet<PathBuf>,
) -> Option<PathBuf> {
    let trimmed = import_text.trim();

    if let Some(rest) = trimmed.strip_prefix("from ") {
        // "from <module> import <names>"
        let module_part = rest.split_whitespace().next()?;

        if module_part.starts_with('.') {
            // Relative import
            let current_dir = current_file.parent()?;
            let dots = module_part.bytes().take_while(|&b| b == b'.').count();
            let mut base = current_dir.to_path_buf();
            // Each dot beyond the first means "go up one more directory"
            for _ in 1..dots {
                base = base.parent()?.to_path_buf();
            }

            let module_name = &module_part[dots..];
            if module_name.is_empty() {
                // "from . import foo" — look in current directory
                return None;
            }

            return resolve_python_module(module_name, &base, project_files);
        }

        // Absolute import -- try resolving from project root
        return resolve_python_module(module_part, Path::new(""), project_files);
    }

    if let Some(rest) = trimmed.strip_prefix("import ") {
        let module_part = rest.split_whitespace().next()?;
        // Simple `import foo` -- only resolve if it looks like a local package
        return resolve_python_module(module_part, Path::new(""), project_files);
    }

    None
}

/// Try to resolve a dotted Python module name to a file path.
fn resolve_python_module(
    module_name: &str,
    base: &Path,
    project_files: &HashSet<PathBuf>,
) -> Option<PathBuf> {
    let segments: Vec<&str> = module_name.split('.').collect();
    if segments.is_empty() {
        return None;
    }

    // Try progressively shorter prefixes
    for take in (1..=segments.len()).rev() {
        let dir_path: PathBuf = segments[..take].iter().collect();
        let as_file = base.join(dir_path.with_extension("py"));
        if project_files.contains(&as_file) {
            return Some(as_file);
        }

        let as_init = base.join(&dir_path).join("__init__.py");
        if project_files.contains(&as_init) {
            return Some(as_init);
        }
    }

    None
}

/// Resolve a JS/TS import to a project-relative file path.
///
/// Only resolves relative imports (starting with `./` or `../`).
/// Bare specifiers like `"express"` or `"@angular/core"` are external.
fn resolve_js_import(
    import_text: &str,
    current_file: &Path,
    project_files: &HashSet<PathBuf>,
) -> Option<PathBuf> {
    // Extract the module specifier from the import statement.
    // Handles: import ... from "path"  /  import ... from 'path'
    //          import "path"  /  require("path")
    let specifier = extract_js_specifier(import_text)?;

    if !specifier.starts_with("./") && !specifier.starts_with("../") {
        // Bare specifier (npm package or node built-in) -- not resolvable
        return None;
    }

    let current_dir = current_file.parent()?;
    let resolved_base = normalize_path(&current_dir.join(specifier));

    // Try exact path first, then with extensions, then as directory index
    let extensions = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"];
    let index_names = ["index.ts", "index.tsx", "index.js", "index.jsx"];

    // Exact match (e.g. `./foo.ts`)
    if project_files.contains(&resolved_base) {
        return Some(resolved_base);
    }

    // With extensions (e.g. `./foo` -> `./foo.ts`)
    for ext in &extensions {
        let mut candidate = resolved_base.as_os_str().to_owned();
        candidate.push(ext);
        let candidate = PathBuf::from(candidate);
        if project_files.contains(&candidate) {
            return Some(candidate);
        }
    }

    // As directory index (e.g. `./foo` -> `./foo/index.ts`)
    for idx in &index_names {
        let candidate = resolved_base.join(idx);
        if project_files.contains(&candidate) {
            return Some(candidate);
        }
    }

    None
}

/// Resolve a Go import to a project-relative file path.
///
/// Go uses package paths (e.g. `"mypackage/utils"`). We try to match them to
/// directories or `.go` files within the project. Standard library and remote
/// module paths (containing a dot in the first segment) are skipped.
fn resolve_go_import(import_text: &str, project_files: &HashSet<PathBuf>) -> Option<PathBuf> {
    // Extract the quoted import path
    let specifier = extract_js_specifier(import_text)?;

    // Skip stdlib (no slash) and remote modules (first segment contains '.')
    let first_segment = specifier.split('/').next().unwrap_or(specifier);
    if !specifier.contains('/') || first_segment.contains('.') {
        return None;
    }

    // Try as a directory containing .go files (Go packages are directories)
    let pkg_dir = PathBuf::from(specifier);
    // Check if any project file lives in that directory
    for pf in project_files {
        if pf.starts_with(&pkg_dir) && pf.extension().is_some_and(|e| e == "go") {
            return Some(pkg_dir);
        }
    }

    None
}

/// Resolve a Java import to a project-relative file path.
///
/// Maps `import com.example.Foo` to `com/example/Foo.java`.
/// Skips wildcard imports (`import com.example.*`).
fn resolve_java_import(import_text: &str, project_files: &HashSet<PathBuf>) -> Option<PathBuf> {
    let trimmed = import_text
        .trim()
        .strip_prefix("import ")?
        .trim_end_matches(';')
        .trim()
        .trim_start_matches("static ");

    // Skip wildcard imports
    if trimmed.ends_with(".*") {
        return None;
    }

    // Convert dots to path separators: com.example.Foo -> com/example/Foo.java
    let path_str: String = trimmed.replace('.', "/");
    let candidate = PathBuf::from(path_str).with_extension("java");

    // Try with common source prefixes
    for prefix in ["", "src/main/java/", "src/"] {
        let full = PathBuf::from(prefix).join(&candidate);
        if project_files.contains(&full) {
            return Some(full);
        }
    }

    None
}

/// Resolve a C/C++ `#include "..."` to a project-relative file path.
///
/// Only resolves quoted includes (project-local). Angle-bracket includes
/// (`<stdio.h>`) are system headers and skipped.
fn resolve_c_include(
    import_text: &str,
    current_file: &Path,
    project_files: &HashSet<PathBuf>,
) -> Option<PathBuf> {
    let trimmed = import_text.trim();

    // Only resolve quoted includes: #include "foo.h"
    // Skip angle-bracket includes: #include <stdio.h>
    let path_str = if let Some(rest) = trimmed.strip_prefix("#include") {
        let rest = rest.trim();
        if rest.starts_with('"') {
            rest.trim_matches('"')
        } else {
            return None; // angle-bracket or malformed
        }
    } else {
        return None;
    };

    // Try relative to current file's directory first
    if let Some(current_dir) = current_file.parent() {
        let candidate = normalize_path(&current_dir.join(path_str));
        if project_files.contains(&candidate) {
            return Some(candidate);
        }
    }

    // Try from project root
    let candidate = PathBuf::from(path_str);
    if project_files.contains(&candidate) {
        return Some(candidate);
    }

    // Try common include directories
    for prefix in ["include/", "src/"] {
        let full = PathBuf::from(prefix).join(path_str);
        if project_files.contains(&full) {
            return Some(full);
        }
    }

    None
}

/// Resolve a Dart import to a project-relative file path.
///
/// Handles relative imports (`'../utils.dart'`) and package imports
/// (`'package:myapp/utils.dart'`). `dart:` SDK imports are skipped.
fn resolve_dart_import(import_text: &str, project_files: &HashSet<PathBuf>) -> Option<PathBuf> {
    let specifier = extract_js_specifier(import_text)?;

    // Skip dart: SDK imports
    if specifier.starts_with("dart:") {
        return None;
    }

    // Package imports: package:app_name/path.dart -> lib/path.dart
    if let Some(rest) = specifier.strip_prefix("package:") {
        // Strip the package name segment: "myapp/utils.dart" -> "utils.dart"
        let after_pkg = rest.split_once('/')?.1;
        let candidate = PathBuf::from("lib").join(after_pkg);
        if project_files.contains(&candidate) {
            return Some(candidate);
        }
        return None;
    }

    // Relative imports are not common but possible — treated as project-root-relative
    let candidate = PathBuf::from(specifier);
    if project_files.contains(&candidate) {
        return Some(candidate);
    }

    None
}

/// Resolve a C# `using` directive to a project-relative file path.
///
/// Maps `using MyNamespace.SubNs` to `MyNamespace/SubNs.cs`.
/// Skips `using static` and `using alias = ...` forms.
fn resolve_csharp_using(import_text: &str, project_files: &HashSet<PathBuf>) -> Option<PathBuf> {
    let trimmed = import_text
        .trim()
        .strip_prefix("using ")?
        .trim_end_matches(';')
        .trim();

    // Skip `using static ...` and alias forms `using Alias = ...`
    if trimmed.starts_with("static ") || trimmed.contains('=') {
        return None;
    }

    // Skip common System namespaces (stdlib)
    if trimmed.starts_with("System") {
        return None;
    }

    // Convert dots to path separators: MyNamespace.Utils -> MyNamespace/Utils.cs
    let path_str: String = trimmed.replace('.', "/");
    let candidate = PathBuf::from(path_str).with_extension("cs");

    if project_files.contains(&candidate) {
        return Some(candidate);
    }

    None
}

/// Extract the string specifier from a JS/TS import statement.
///
/// Looks for the first quoted string (single or double quotes).
fn extract_js_specifier(import_text: &str) -> Option<&str> {
    // Try double quotes first, then single quotes
    for quote in ['"', '\''] {
        if let Some(start) = import_text.find(quote) {
            let rest = &import_text[start + 1..];
            if let Some(end) = rest.find(quote) {
                return Some(&rest[..end]);
            }
        }
    }
    None
}

/// Normalize a path by resolving `.` and `..` components without filesystem access.
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::CurDir => {}
            other => components.push(other),
        }
    }
    components.iter().collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_file_set(paths: &[&str]) -> HashSet<PathBuf> {
        paths.iter().map(PathBuf::from).collect()
    }

    // -----------------------------------------------------------------------
    // Rust import resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_crate_import_file() {
        let files = make_file_set(&["src/config.rs", "src/search/mod.rs"]);
        let result = resolve_import_path(
            "use crate::config::Config;",
            "rust",
            Path::new("src/main.rs"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/config.rs")));
    }

    #[test]
    fn test_rust_crate_import_mod() {
        let files = make_file_set(&["src/search/mod.rs", "src/search/hybrid.rs"]);
        let result = resolve_import_path(
            "use crate::search::hybrid::HybridSearch;",
            "rust",
            Path::new("src/main.rs"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/search/hybrid.rs")));
    }

    #[test]
    fn test_rust_crate_import_nested_mod() {
        let files = make_file_set(&["src/search/mod.rs"]);
        let result = resolve_import_path(
            "use crate::search;",
            "rust",
            Path::new("src/main.rs"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/search/mod.rs")));
    }

    #[test]
    fn test_rust_brace_import() {
        let files = make_file_set(&["src/types.rs"]);
        let result = resolve_import_path(
            "use crate::types::{Chunk, ChunkType};",
            "rust",
            Path::new("src/main.rs"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/types.rs")));
    }

    #[test]
    fn test_rust_super_import() {
        let files = make_file_set(&["src/chunking/text_chunker.rs"]);
        let result = resolve_import_path(
            "use super::text_chunker::TextChunker;",
            "rust",
            Path::new("src/chunking/ast_chunker.rs"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/chunking/text_chunker.rs")));
    }

    #[test]
    fn test_rust_std_import_skipped() {
        let files = make_file_set(&["src/main.rs"]);
        let result = resolve_import_path(
            "use std::collections::HashMap;",
            "rust",
            Path::new("src/main.rs"),
            &files,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_rust_external_crate_skipped() {
        let files = make_file_set(&["src/main.rs"]);
        let result = resolve_import_path(
            "use anyhow::Result;",
            "rust",
            Path::new("src/main.rs"),
            &files,
        );
        assert_eq!(result, None);
    }

    // -----------------------------------------------------------------------
    // Python import resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_python_relative_import() {
        let files = make_file_set(&["mypackage/utils.py", "mypackage/__init__.py"]);
        let result = resolve_import_path(
            "from .utils import helper",
            "python",
            Path::new("mypackage/main.py"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("mypackage/utils.py")));
    }

    #[test]
    fn test_python_double_dot_import() {
        // from ..models in mypackage/sub/views.py:
        //   one dot = mypackage/sub/, two dots = mypackage/
        //   -> resolves to mypackage/models.py
        let files = make_file_set(&["mypackage/models.py"]);
        let result = resolve_import_path(
            "from ..models import User",
            "python",
            Path::new("mypackage/sub/views.py"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("mypackage/models.py")));
    }

    #[test]
    fn test_python_absolute_import() {
        let files = make_file_set(&["mypackage/utils.py"]);
        let result = resolve_import_path(
            "from mypackage.utils import helper",
            "python",
            Path::new("main.py"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("mypackage/utils.py")));
    }

    #[test]
    fn test_python_package_import() {
        let files = make_file_set(&["mypackage/__init__.py"]);
        let result =
            resolve_import_path("import mypackage", "python", Path::new("main.py"), &files);
        assert_eq!(result, Some(PathBuf::from("mypackage/__init__.py")));
    }

    #[test]
    fn test_python_stdlib_not_found() {
        let files = make_file_set(&["main.py"]);
        let result = resolve_import_path("import os", "python", Path::new("main.py"), &files);
        assert_eq!(result, None);
    }

    // -----------------------------------------------------------------------
    // JS/TS import resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_js_relative_import_with_ext() {
        let files = make_file_set(&["src/utils.ts", "src/main.ts"]);
        let result = resolve_import_path(
            "import { helper } from \"./utils.ts\";",
            "typescript",
            Path::new("src/main.ts"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/utils.ts")));
    }

    #[test]
    fn test_js_relative_import_without_ext() {
        let files = make_file_set(&["src/utils.ts"]);
        let result = resolve_import_path(
            "import { helper } from './utils';",
            "typescript",
            Path::new("src/main.ts"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/utils.ts")));
    }

    #[test]
    fn test_js_relative_import_jsx() {
        let files = make_file_set(&["src/components/Button.tsx"]);
        let result = resolve_import_path(
            "import Button from '../components/Button';",
            "tsx",
            Path::new("src/pages/Home.tsx"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/components/Button.tsx")));
    }

    #[test]
    fn test_js_directory_index() {
        let files = make_file_set(&["src/components/index.ts"]);
        let result = resolve_import_path(
            "import { Button } from './components';",
            "typescript",
            Path::new("src/app.ts"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/components/index.ts")));
    }

    #[test]
    fn test_js_bare_specifier_skipped() {
        let files = make_file_set(&["src/main.ts"]);
        let result = resolve_import_path(
            "import express from 'express';",
            "typescript",
            Path::new("src/main.ts"),
            &files,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_js_scoped_package_skipped() {
        let files = make_file_set(&["src/main.ts"]);
        let result = resolve_import_path(
            "import { Component } from '@angular/core';",
            "typescript",
            Path::new("src/main.ts"),
            &files,
        );
        assert_eq!(result, None);
    }

    // -----------------------------------------------------------------------
    // Rust self:: imports
    // -----------------------------------------------------------------------

    #[test]
    fn test_rust_self_import() {
        let files = make_file_set(&["src/chunking/text_chunker.rs"]);
        let result = resolve_import_path(
            "use self::text_chunker::TextChunker;",
            "rust",
            Path::new("src/chunking/ast_chunker.rs"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/chunking/text_chunker.rs")));
    }

    // -----------------------------------------------------------------------
    // Go import resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_go_local_package_import() {
        let files = make_file_set(&["internal/auth/handler.go"]);
        let result =
            resolve_import_path("\"internal/auth\"", "go", Path::new("cmd/main.go"), &files);
        assert_eq!(result, Some(PathBuf::from("internal/auth")));
    }

    #[test]
    fn test_go_stdlib_skipped() {
        let files = make_file_set(&["main.go"]);
        let result = resolve_import_path("\"fmt\"", "go", Path::new("main.go"), &files);
        assert_eq!(result, None);
    }

    #[test]
    fn test_go_remote_module_skipped() {
        let files = make_file_set(&["main.go"]);
        let result = resolve_import_path(
            "\"github.com/user/repo/pkg\"",
            "go",
            Path::new("main.go"),
            &files,
        );
        assert_eq!(result, None);
    }

    // -----------------------------------------------------------------------
    // Java import resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_java_import_simple() {
        let files = make_file_set(&["com/example/Foo.java"]);
        let result = resolve_import_path(
            "import com.example.Foo;",
            "java",
            Path::new("com/example/Main.java"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("com/example/Foo.java")));
    }

    #[test]
    fn test_java_import_maven_layout() {
        let files = make_file_set(&["src/main/java/com/example/Foo.java"]);
        let result = resolve_import_path(
            "import com.example.Foo;",
            "java",
            Path::new("src/main/java/com/example/Main.java"),
            &files,
        );
        assert_eq!(
            result,
            Some(PathBuf::from("src/main/java/com/example/Foo.java"))
        );
    }

    #[test]
    fn test_java_wildcard_skipped() {
        let files = make_file_set(&["com/example/Foo.java"]);
        let result = resolve_import_path(
            "import com.example.*;",
            "java",
            Path::new("Main.java"),
            &files,
        );
        assert_eq!(result, None);
    }

    // -----------------------------------------------------------------------
    // C/C++ include resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_c_quoted_include_relative() {
        let files = make_file_set(&["src/config.h", "src/main.c"]);
        let result = resolve_import_path(
            "#include \"config.h\"",
            "c",
            Path::new("src/main.c"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("src/config.h")));
    }

    #[test]
    fn test_c_angle_bracket_skipped() {
        let files = make_file_set(&["src/main.c"]);
        let result =
            resolve_import_path("#include <stdio.h>", "c", Path::new("src/main.c"), &files);
        assert_eq!(result, None);
    }

    #[test]
    fn test_cpp_include_from_include_dir() {
        let files = make_file_set(&["include/mylib/types.h"]);
        let result = resolve_import_path(
            "#include \"mylib/types.h\"",
            "cpp",
            Path::new("src/main.cpp"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("include/mylib/types.h")));
    }

    // -----------------------------------------------------------------------
    // Dart import resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_dart_package_import() {
        let files = make_file_set(&["lib/utils.dart"]);
        let result = resolve_import_path(
            "import 'package:myapp/utils.dart';",
            "dart",
            Path::new("lib/main.dart"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("lib/utils.dart")));
    }

    #[test]
    fn test_dart_sdk_import_skipped() {
        let files = make_file_set(&["lib/main.dart"]);
        let result = resolve_import_path(
            "import 'dart:async';",
            "dart",
            Path::new("lib/main.dart"),
            &files,
        );
        assert_eq!(result, None);
    }

    // -----------------------------------------------------------------------
    // C# using resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_csharp_using_resolved() {
        let files = make_file_set(&["MyNamespace/Utils.cs"]);
        let result = resolve_import_path(
            "using MyNamespace.Utils;",
            "c_sharp",
            Path::new("Program.cs"),
            &files,
        );
        assert_eq!(result, Some(PathBuf::from("MyNamespace/Utils.cs")));
    }

    #[test]
    fn test_csharp_system_skipped() {
        let files = make_file_set(&["Program.cs"]);
        let result = resolve_import_path(
            "using System.Collections.Generic;",
            "c_sharp",
            Path::new("Program.cs"),
            &files,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_csharp_static_using_skipped() {
        let files = make_file_set(&["Program.cs"]);
        let result = resolve_import_path(
            "using static System.Math;",
            "c_sharp",
            Path::new("Program.cs"),
            &files,
        );
        assert_eq!(result, None);
    }

    // -----------------------------------------------------------------------
    // Unsupported language
    // -----------------------------------------------------------------------

    #[test]
    fn test_unsupported_language_returns_none() {
        let files = make_file_set(&["main.rb"]);
        let result = resolve_import_path("require 'json'", "ruby", Path::new("main.rb"), &files);
        assert_eq!(result, None);
    }

    // -----------------------------------------------------------------------
    // extract_js_specifier
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_js_specifier_double_quotes() {
        assert_eq!(
            extract_js_specifier("import x from \"./foo\""),
            Some("./foo")
        );
    }

    #[test]
    fn test_extract_js_specifier_single_quotes() {
        assert_eq!(extract_js_specifier("import x from './bar'"), Some("./bar"));
    }

    #[test]
    fn test_extract_js_specifier_none() {
        assert_eq!(extract_js_specifier("import x from foo"), None);
    }
}
