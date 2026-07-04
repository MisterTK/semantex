use anyhow::{Context, Result};
use notify::RecursiveMode;
use notify_debouncer_full::{DebounceEventResult, Debouncer, RecommendedCache, new_debouncer};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;
use tracing::debug;

pub struct FileWatcher {
    debouncer: Option<Debouncer<notify::RecommendedWatcher, RecommendedCache>>,
}

impl FileWatcher {
    /// Create a new file watcher.
    pub fn new() -> Result<Self> {
        Ok(Self { debouncer: None })
    }

    /// Watch a directory for changes, returning a receiver for changed file paths.
    /// Events are debounced (500ms).
    pub fn watch(&mut self, root: &Path) -> Result<mpsc::Receiver<Vec<PathBuf>>> {
        let (tx, rx) = mpsc::channel();

        let mut debouncer = new_debouncer(
            Duration::from_millis(500),
            None,
            move |result: DebounceEventResult| {
                let events = match result {
                    Ok(events) => events,
                    Err(errs) => {
                        for err in errs {
                            debug!("File watcher error: {:?}", err);
                        }
                        return;
                    }
                };

                let mut paths = Vec::new();
                for event in &events {
                    use notify::EventKind;
                    match event.kind {
                        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {
                            for path in &event.paths {
                                // Only include files, not directories
                                if path.is_file() || !path.exists() {
                                    paths.push(path.clone());
                                }
                            }
                        }
                        _ => {}
                    }
                }

                paths.sort();
                paths.dedup();

                if !paths.is_empty() {
                    let _ = tx.send(paths);
                }
            },
        )
        .context("Failed to create file watcher")?;

        debouncer
            .watch(root, RecursiveMode::Recursive)
            .with_context(|| format!("Failed to watch directory: {}", root.display()))?;

        self.debouncer = Some(debouncer);

        Ok(rx)
    }

    /// Add another path to the SAME watch session (must be called after
    /// [`watch`](Self::watch) — reuses its debouncer/channel rather than
    /// creating a second one). Events from `path` are funneled into the
    /// same receiver `watch` already returned. `recursive` selects
    /// [`RecursiveMode::Recursive`] vs [`RecursiveMode::NonRecursive`] (a
    /// `bool` rather than the `notify` type itself, so callers outside this
    /// crate don't need a direct `notify` dependency just to call this).
    ///
    /// Used by `semantex watch` to additionally watch the git `HEAD` file
    /// (which, for a linked worktree, lives OUTSIDE the project root and so
    /// wouldn't be seen by the recursive project-root watch at all) so a
    /// branch switch is detected even when it changes no tracked file
    /// (e.g. `git switch -c` from an identical tree).
    pub fn watch_additional(&mut self, path: &Path, recursive: bool) -> Result<()> {
        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        let debouncer = self
            .debouncer
            .as_mut()
            .context("watch_additional called before watch")?;
        debouncer
            .watch(path, mode)
            .with_context(|| format!("Failed to watch path: {}", path.display()))
    }

    /// Stop watching.
    pub fn stop(&mut self) {
        self.debouncer.take();
    }
}

impl Drop for FileWatcher {
    fn drop(&mut self) {
        self.stop();
    }
}
