use anyhow::{Result, bail};
use ignore::WalkBuilder;
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

            // Skip lock files and other default-ignore patterns
            if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && DEFAULT_IGNORE_NAMES.contains(&name)
            {
                debug!("Skipping default-ignored file: {}", path.display());
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
