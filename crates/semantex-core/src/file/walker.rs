use anyhow::{Result, bail};
use ignore::WalkBuilder;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use std::path::{Path, PathBuf};
use tracing::debug;

/// Files that are never useful for semantic search (lock files, generated code, etc.)
const DEFAULT_IGNORE_NAMES: &[&str] = &[
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Gemfile.lock",
    "poetry.lock",
    "composer.lock",
    "go.sum",
    "flake.lock",
];

/// Gitignore-style patterns applied to every walked tree before any user
/// `.gitignore` / `.semantexignore` is consulted.
///
/// These cover the output directories of widely-used code-intelligence tools
/// whose artifacts are JSON dumps of every symbol in the repo. A single such
/// file matches almost any semantic query and ranks high, poisoning results.
/// (See `benchmarks/results/run28-v0.6` → `run29-v0.6-clean` for the run that
/// established 16–46% of indexable content in real repos can be tool output.)
///
/// These defaults are baked in and cannot be negated by a user
/// `.semantexignore` in the current release; file an issue if you need to
/// index one of these patterns.
const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
    // Code-graph generators (graphify and similar)
    "graphify-out",
    ".graphify_*",
    // SCIP indexes (Sourcegraph, scip-typescript, scip-python, ...)
    "*.scip",
    "index.scip",
    ".scip-typescript",
    // LSIF indexes (Microsoft Language Server Index Format)
    "*.lsif",
    "*.lsif.json",
    "dump.lsif",
    ".lsif-tsc",
    // Sourcegraph workspace metadata
    ".sourcegraph",
];

fn build_default_ignore(root: &Path) -> Result<Gitignore> {
    let mut builder = GitignoreBuilder::new(root);
    for pattern in DEFAULT_IGNORE_PATTERNS {
        builder
            .add_line(None, pattern)
            .map_err(|e| anyhow::anyhow!("invalid default ignore pattern {pattern:?}: {e}"))?;
    }
    builder
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build default ignore set: {e}"))
}

pub struct FileWalker {
    max_file_size: u64,
    max_file_count: usize,
}

impl FileWalker {
    pub fn new(max_file_size: u64, max_file_count: usize) -> Self {
        Self {
            max_file_size,
            max_file_count,
        }
    }

    /// Walk directory, respecting .gitignore and .semantexignore, returning discovered file paths.
    pub fn walk(&self, root: &Path) -> Result<Vec<PathBuf>> {
        if !root.is_dir() {
            bail!("Root path is not a directory: {}", root.display());
        }

        let mut builder = WalkBuilder::new(root);
        builder
            .hidden(true) // skip hidden files/dirs
            .git_ignore(true) // respect .gitignore
            .git_global(true) // respect global gitignore
            .git_exclude(true) // respect .git/info/exclude
            .require_git(false) // walk even if not a git repo
            .max_filesize(Some(self.max_file_size));

        // Add .semantexignore support
        builder.add_custom_ignore_filename(".semantexignore");

        // Apply built-in patterns at directory enumeration time so the walker
        // doesn't descend into known tool-output trees like `graphify-out/`.
        // (For file-pattern entries like `*.scip`, the same Gitignore is also
        // checked post-walk below — `filter_entry` in this crate only fires on
        // directories.)
        let default_gi = build_default_ignore(root)?;
        let dir_gi = default_gi.clone();
        builder.filter_entry(move |entry| {
            let is_dir = entry.file_type().is_some_and(|ft| ft.is_dir());
            !dir_gi.matched(entry.path(), is_dir).is_ignore()
        });

        let walker = builder.build();

        let mut paths = Vec::new();
        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(err) => {
                    debug!("Skipping entry due to error: {}", err);
                    continue;
                }
            };

            // Only include files (not directories or symlinks to dirs)
            let Some(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_file() {
                continue;
            }

            let path = entry.into_path();

            // Skip lock files and other default-ignore filenames
            if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && DEFAULT_IGNORE_NAMES.contains(&name)
            {
                debug!("Skipping default-ignored file: {}", path.display());
                continue;
            }

            // Skip files matching built-in patterns (e.g. *.scip, *.lsif).
            // Directory-level patterns are also pruned at descent time via
            // filter_entry above; this check covers leaf-file patterns that
            // filter_entry never sees.
            if default_gi.matched(&path, false).is_ignore() {
                debug!("Skipping default-ignored pattern: {}", path.display());
                continue;
            }

            // Skip binary files by checking for null bytes in the first 8KB
            if is_binary(&path) {
                debug!("Skipping binary file: {}", path.display());
                continue;
            }

            paths.push(path);

            if paths.len() >= self.max_file_count {
                debug!(
                    "Reached max file count limit ({}), stopping walk",
                    self.max_file_count
                );
                break;
            }
        }

        paths.sort();
        Ok(paths)
    }
}

/// Check if a file appears to be binary by reading the first 8KB and looking for null bytes.
fn is_binary(path: &Path) -> bool {
    use std::fs::File;
    use std::io::Read;

    // Allow PDF files through even though they contain binary data
    if let Some(ext) = path.extension()
        && ext == "pdf"
    {
        return false;
    }

    let Ok(mut file) = File::open(path) else {
        return false;
    };

    let mut buf = [0u8; 8192];
    let Ok(n) = file.read(&mut buf) else {
        return false;
    };

    buf[..n].contains(&0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn touch(p: &Path, body: &str) {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, body).unwrap();
    }

    #[test]
    fn default_ignores_skip_code_intel_outputs() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // legitimate content that must survive
        touch(&root.join("src/main.rs"), "fn main() {}");
        touch(&root.join("README.md"), "# repo");

        // graphify pollution
        touch(&root.join("graphify-out/graph.json"), "{}");
        touch(&root.join("graphify-out/cache/abc.json"), "{}");
        touch(&root.join(".graphify_analysis.json"), "{}");
        touch(&root.join(".graphify_semantic_marker"), "ok");

        // SCIP / LSIF tool output
        touch(&root.join("index.scip"), "scipdata");
        touch(&root.join("dump.lsif"), "{}");
        touch(&root.join("subdir/local.lsif.json"), "{}");

        let walker = FileWalker::new(10_000_000, 1000);
        let paths = walker.walk(root).unwrap();
        let rel: Vec<String> = paths
            .iter()
            .map(|p| {
                // Normalize separators so assertions hold on Windows too.
                p.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();

        // legitimate files included
        assert!(
            rel.iter().any(|p| p == "src/main.rs"),
            "missing main.rs: {rel:?}"
        );
        assert!(
            rel.iter().any(|p| p == "README.md"),
            "missing README.md: {rel:?}"
        );

        // tool outputs excluded
        for forbidden in [
            "graphify-out/graph.json",
            "graphify-out/cache/abc.json",
            ".graphify_analysis.json",
            ".graphify_semantic_marker",
            "index.scip",
            "dump.lsif",
            "subdir/local.lsif.json",
        ] {
            assert!(
                !rel.iter().any(|p| p == forbidden),
                "tool output leaked into walk: {forbidden} (got {rel:?})"
            );
        }
    }

    #[test]
    fn semantexignore_still_applies_for_user_patterns() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        touch(&root.join("src/keep.rs"), "ok");
        touch(&root.join("vendor/skip.rs"), "skip me");
        touch(&root.join(".semantexignore"), "vendor/\n");

        let walker = FileWalker::new(10_000_000, 1000);
        let paths = walker.walk(root).unwrap();
        let rel: Vec<String> = paths
            .iter()
            .map(|p| {
                // Normalize separators so assertions hold on Windows too.
                p.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();

        assert!(
            rel.iter().any(|p| p == "src/keep.rs"),
            "missing keep.rs: {rel:?}"
        );
        assert!(
            !rel.iter().any(|p| p.starts_with("vendor/")),
            ".semantexignore stopped working: {rel:?}"
        );
    }
}
