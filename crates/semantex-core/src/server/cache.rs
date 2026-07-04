//! Multi-client, branch-aware searcher lifecycle for the `semantex serve` daemon
//! (Wave 2 batch 2, contract §E/§F).
//!
//! # Background — the two gaps this closes
//!
//! Before this module, `SemantexServer::run` (see `server/mod.rs`) opened a
//! single [`HybridSearcher`] once at daemon startup and [`Listener`] served it
//! for the daemon's whole lifetime (up to the idle timeout, historically up to
//! 24h under `semantex watch`). Two consequences, both previously documented
//! inline at the `serve` command's branch-switch call site:
//!
//! 1. A `git switch` made *while the daemon is running* was invisible to it —
//!    every other entry point (`semantex index`/`watch`, MCP) reconciles a
//!    branch switch via [`branches::detect_and_handle_branch_switch`] on its
//!    own hot path, but the daemon's accept loop had no hook to do the same,
//!    so it kept serving the OLD branch's content until restarted.
//! 2. The daemon had no way to notice its OWN index being rebuilt underneath
//!    it (e.g. `semantex index` run again in another terminal for the same
//!    project) — the single long-lived `HybridSearcher` just kept its
//!    originally-opened view.
//!
//! # Design — mirrors `semantex-mcp`'s proven LRU searcher cache
//!
//! `semantex-mcp/src/server.rs` already solves an equivalent problem for the
//! in-process MCP path: a small `HashMap<PathBuf, Arc<CachedSearcher>>` LRU,
//! evicted by recency + a memory-pressure valve, with entries wrapped in `Arc`
//! so an in-flight query keeps a searcher alive even after it's evicted from
//! the map. [`SearcherCache`] mirrors that shape almost exactly, keyed by
//! `(project_root, branch_key)` instead of just `project_root` (Wave 2
//! §Multi-branch's `layout::current_branch_key`) so that a branch switch
//! naturally misses the cache under its new key rather than requiring the
//! caller to reason about generation counters.
//!
//! Per-request flow ([`SearcherCache::get`]):
//!
//! 1. **Branch-switch probe** (cheap: two small file reads, per
//!    [`branches::branch_switch_pending`]). If pending, reconcile via
//!    [`branches::detect_and_handle_branch_switch`] — which takes its OWN
//!    exclusive `.semantex.lock` internally and blocks until it's free; we do
//!    NOT hold any lock of ours across that call — then drop every cached
//!    entry for this project (the live root's content just changed, or for
//!    the `SnapshottedOutgoing` case still needs a real incremental rebuild
//!    via `semantex index`/`watch` before it fully reflects the new branch;
//!    either way an old cached instance must not keep being served under a
//!    reused key).
//! 2. **Staleness probe** (cheap: one small `meta.json` read). Compares the
//!    index's `updated_at` stamp against the value recorded when the cached
//!    entry was opened. A mismatch means the index was rebuilt beneath us
//!    (another terminal ran `semantex index`, or a background catch-up
//!    finished) — evict and reopen.
//! 3. Cache hit → clone the `Arc` and touch `last_used`. Cache miss → evict
//!    the least-recently-used entry if at capacity, open a fresh
//!    `HybridSearcher` (outside the map lock — I/O + model load is slow), and
//!    insert.
//!
//! Concurrency safety: every returned handle is `Arc<CachedSearcher>`. A
//! request holds its own clone for the whole connection, so even if another
//! thread's `get()` call evicts that entry from the map in the meantime, the
//! in-flight query keeps a live, valid `HybridSearcher` until it finishes and
//! drops its `Arc` — identical to the MCP cache's guarantee.
//!
//! # Known residual limitation (documented, not silently papered over)
//!
//! Immediately after a `SnapshottedOutgoing` reconcile (HEAD moved to a branch
//! with no saved snapshot yet), the live root's *content* is still the
//! OUTGOING branch's — only the sidecar bookkeeping changed — until a real
//! incremental build runs (`semantex index`/`watch`, unchanged from before
//! this module). A search issued in that narrow window is served under the
//! new branch's key but old content. The daemon has no in-protocol "building"
//! signal to guard this today (unlike the MCP path, which reports
//! `IndexState::Building` and falls back to ripgrep); documenting it here
//! matches this module's brief, which is scoped to the CHEAP reconcile hook,
//! not a synchronous full rebuild on the search hot path.

use crate::config::SemantexConfig;
use crate::index::branches;
use crate::index::layout;
use crate::search::hybrid::HybridSearcher;
use crate::types::IndexMeta;
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Default cap on resident `HybridSearcher` instances, overridable via
/// `SEMANTEX_DAEMON_MAX_SEARCHERS`. Kept small (2) — enough to keep the
/// current branch's searcher plus one recently-switched-away-from branch warm
/// (a quick `git switch main; ...; git switch feature` round trip is a common
/// dev workflow), without unbounded RAM growth for a daemon that may run for
/// hours.
const DEFAULT_MAX_SEARCHERS: usize = 2;

/// A cached, ready-to-query searcher plus the bookkeeping the LRU/staleness
/// logic needs. `Arc`-wrapped by [`SearcherCache::get`] so an in-flight query
/// survives eviction from the map (mirrors `semantex-mcp`'s `CachedSearcher`).
pub struct CachedSearcher {
    pub searcher: HybridSearcher,
    last_used: Mutex<Instant>,
    /// The index's `meta.json::updated_at` stamp at open time — the cheap
    /// staleness marker compared on every [`SearcherCache::get`] call.
    marker: String,
}

/// Cache key: a project can have at most one *live* branch at a time (the
/// storage layout keeps a single `.semantex/` root plus `indexes/<key>/`
/// snapshots), but the daemon may cycle through several during its lifetime
/// as the user switches branches — hence branch_key is part of the key rather
/// than an invalidation side-channel.
type CacheKey = (PathBuf, String);

/// LRU cache of [`HybridSearcher`] instances for the `semantex serve` daemon.
/// See the module docs for the full per-request reconcile/staleness flow.
pub struct SearcherCache {
    config: SemantexConfig,
    entries: Mutex<HashMap<CacheKey, Arc<CachedSearcher>>>,
    max_cached: usize,
    /// Serializes concurrent branch-switch reconcile attempts WITHIN this
    /// process. `detect_and_handle_branch_switch` is itself safe to call
    /// concurrently (it takes the exclusive index lock and re-checks under
    /// it), but there's no reason for every simultaneously-arriving request
    /// to redundantly race into it — the first one through does the work,
    /// the rest see `branch_switch_pending() == false` once they acquire
    /// this gate right behind it.
    reconcile_gate: Mutex<()>,
}

impl SearcherCache {
    /// Construct an empty cache, reading `SEMANTEX_DAEMON_MAX_SEARCHERS` for
    /// the LRU cap (default [`DEFAULT_MAX_SEARCHERS`]; any non-positive or
    /// unparseable value falls back to the default).
    pub fn new(config: SemantexConfig) -> Self {
        Self {
            config,
            entries: Mutex::new(HashMap::new()),
            max_cached: max_searchers_from_env(),
            reconcile_gate: Mutex::new(()),
        }
    }

    /// Construct a cache pre-seeded with an already-open searcher for
    /// `project_root`'s CURRENT branch. Used by callers (`semantex serve`,
    /// `semantex watch`) that already pay the eager-open cost upfront (so
    /// startup fails fast on a broken index) and would otherwise waste that
    /// work by opening a second time on the first `get()` call.
    pub fn seeded(config: SemantexConfig, project_root: &Path, searcher: HybridSearcher) -> Self {
        let cache = Self::new(config);
        let canonical = canonicalize(project_root);
        let key = (canonical.clone(), layout::current_branch_key(&canonical));
        let marker = index_marker(&canonical);
        cache.entries.lock().insert(
            key,
            Arc::new(CachedSearcher {
                searcher,
                last_used: Mutex::new(Instant::now()),
                marker,
            }),
        );
        cache
    }

    /// Number of resident searchers (test/diagnostic hook).
    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    /// Whether the cache currently holds no resident searchers.
    pub fn is_empty(&self) -> bool {
        self.entries.lock().is_empty()
    }

    /// Resolve the ready-to-query searcher for `project_root`'s CURRENT
    /// branch, reconciling a pending branch switch and reopening on
    /// detected staleness first. See module docs for the full flow.
    pub fn get(&self, project_root: &Path) -> Result<Arc<CachedSearcher>> {
        let canonical = canonicalize(project_root);

        // Item 2: cheap per-request branch-switch probe. Only the (rare)
        // pending case pays the reconcile-gate lock + `detect_and_handle_
        // branch_switch`'s own exclusive-lock wait; the overwhelmingly common
        // unchanged case is two small file reads and nothing else.
        if branches::branch_switch_pending(&canonical) {
            let _gate = self.reconcile_gate.lock();
            // Re-check now that we hold the gate — another thread may have
            // already reconciled this exact switch while we were waiting.
            if branches::branch_switch_pending(&canonical) {
                match branches::detect_and_handle_branch_switch(&canonical) {
                    Ok(action) if action.switched() => {
                        tracing::info!(
                            path = %canonical.display(),
                            ?action,
                            "Daemon: branch switch reconciled"
                        );
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(
                            path = %canonical.display(),
                            err = %e,
                            "Daemon: branch switch check failed (continuing with cached/current index)"
                        );
                    }
                }
                // Drop every cached entry for this project: its live root
                // just changed identity (or, for SnapshottedOutgoing, is
                // pending a real rebuild — see module docs) so nothing
                // currently cached for it is trustworthy under a fresh key.
                self.entries
                    .lock()
                    .retain(|(root, _), _| root != &canonical);
            }
        }

        let branch_key = layout::current_branch_key(&canonical);
        let marker = index_marker(&canonical);
        let key = (canonical.clone(), branch_key);

        // Fast path: cache hit with a matching staleness marker.
        {
            let mut entries = self.entries.lock();
            if let Some(entry) = entries.get(&key) {
                if entry.marker == marker {
                    *entry.last_used.lock() = Instant::now();
                    return Ok(Arc::clone(entry));
                }
                // Item 3: index regenerated beneath us since this entry was
                // opened (another terminal's `semantex index`, or a
                // background catch-up completed) — drop, fall through to
                // reopen.
                tracing::info!(
                    path = %canonical.display(),
                    "Daemon: index changed beneath cached searcher — reopening"
                );
                entries.remove(&key);
            }
            evict_lru_locked(&mut entries, self.max_cached);
        }

        // Drop the lock while opening — I/O + model load is slow, and other
        // projects'/branches' lookups must not block on it.
        let index_dir = SemantexConfig::project_index_dir(&canonical);
        let searcher = HybridSearcher::open(&index_dir, &self.config)?;
        let entry = Arc::new(CachedSearcher {
            searcher,
            last_used: Mutex::new(Instant::now()),
            marker,
        });

        self.entries.lock().insert(key, Arc::clone(&entry));
        Ok(entry)
    }

    /// Drop every cached entry for `project_root`, regardless of branch key.
    /// Exposed for `semantex watch`'s own re-index loop, which already knows
    /// exactly when a rebuild just completed and can invalidate immediately
    /// rather than waiting for the next `get()`'s marker check.
    pub fn invalidate_project(&self, project_root: &Path) {
        let canonical = canonicalize(project_root);
        self.entries
            .lock()
            .retain(|(root, _), _| root != &canonical);
    }
}

fn canonicalize(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Cheap staleness marker: the index's `meta.json::updated_at` stamp.
/// Best-effort — an unreadable/missing/unparseable `meta.json` yields an
/// empty string, which simply means the FIRST open always "misses" (there is
/// nothing to compare against yet); it never panics or blocks the daemon.
fn index_marker(project_root: &Path) -> String {
    let meta_path = SemantexConfig::project_index_dir(project_root).join("meta.json");
    std::fs::read_to_string(meta_path)
        .ok()
        .and_then(|s| serde_json::from_str::<IndexMeta>(&s).ok())
        .map(|m| m.updated_at)
        .unwrap_or_default()
}

fn max_searchers_from_env() -> usize {
    std::env::var("SEMANTEX_DAEMON_MAX_SEARCHERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_SEARCHERS)
}

/// Evict the least-recently-used entry if the map is at (or over) capacity.
/// Called with the map lock already held; never touches disk.
fn evict_lru_locked(entries: &mut HashMap<CacheKey, Arc<CachedSearcher>>, max_cached: usize) {
    while entries.len() >= max_cached {
        let Some(lru_key) = entries
            .iter()
            .min_by_key(|(_, v)| *v.last_used.lock())
            .map(|(k, _)| k.clone())
        else {
            break;
        };
        tracing::info!(
            project = %lru_key.0.display(),
            branch = %lru_key.1,
            "Daemon: evicting cached searcher (cache full)"
        );
        entries.remove(&lru_key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::storage::ChunkStore;
    use crate::types::{Chunk, ChunkType, FileEntry, IndexMeta};
    use tempfile::TempDir;

    /// Test config that pins the dense backend alias to `coderank-hnsw` so
    /// `HybridSearcher::open` resolves it directly (`DenseBackendKind::parse`)
    /// without needing to walk the embedder/model registry — deterministic
    /// and independent of the shipped default embedder, matching
    /// `sample_meta()`'s `dense_backend` stamp below.
    fn test_config() -> SemantexConfig {
        SemantexConfig {
            dense_backend: "coderank-hnsw".to_string(),
            ..SemantexConfig::default()
        }
    }

    fn sample_meta() -> IndexMeta {
        IndexMeta {
            schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: PathBuf::from("/x"),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 1,
            chunk_count: 1,
            embedding_model: "test".to_string(),
            embedding_dim: 48,
            use_bm25_stemmer: true,
            dense_backend: "coderank-hnsw".to_string(),
            embedder_fingerprint: "fp".to_string(),
        }
    }

    /// Build a minimal, directly-searchable index at `<project>/.semantex/`
    /// (no real embedder/tantivy build — mirrors `index/branches.rs`'s own
    /// synthetic test harness so these tests stay fast and model-free).
    fn build_index(project: &Path, content: &str, updated_at: &str) {
        let container = layout::container_dir(project);
        std::fs::create_dir_all(&container).unwrap();
        let store = ChunkStore::open(&container.join("chunks.db")).unwrap();
        store
            .insert_chunk(
                &Chunk {
                    id: 0,
                    file_path: PathBuf::from("src/a.rs"),
                    start_line: 1,
                    end_line: 1,
                    content: content.to_string(),
                    chunk_type: ChunkType::TextWindow { window_index: 0 },
                },
                1,
                0,
            )
            .unwrap();
        store
            .set_file_entry(&FileEntry {
                path: PathBuf::from("src/a.rs"),
                hash: 1,
                size: content.len() as u64,
                mtime: 0,
            })
            .unwrap();
        let mut meta = sample_meta();
        meta.updated_at = updated_at.to_string();
        std::fs::write(
            container.join("meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
    }

    fn write_fake_git_head(project: &Path, branch: &str) {
        let git = project.join(".git");
        std::fs::create_dir_all(git.join("refs").join("heads")).unwrap();
        std::fs::write(git.join("HEAD"), format!("ref: refs/heads/{branch}\n")).unwrap();
        std::fs::write(git.join("refs").join("heads").join(branch), "deadbeef\n").unwrap();
    }

    /// Read back the single chunk's content via `with_store`, without
    /// assuming a specific autoincrement id (each test opens a fresh
    /// `chunks.db`, but relying on the DB's actual id assignment is safer
    /// than hardcoding one).
    fn read_chunk_content(searcher: &HybridSearcher) -> String {
        searcher.with_store(|store| {
            let ids = store.get_all_chunk_ids().unwrap();
            let id = *ids.first().expect("test index must have exactly one chunk");
            store.get_chunk(id).unwrap().content
        })
    }

    #[test]
    fn max_searchers_env_default_and_override() {
        unsafe {
            std::env::remove_var("SEMANTEX_DAEMON_MAX_SEARCHERS");
        }
        assert_eq!(max_searchers_from_env(), DEFAULT_MAX_SEARCHERS);
        unsafe {
            std::env::set_var("SEMANTEX_DAEMON_MAX_SEARCHERS", "5");
        }
        assert_eq!(max_searchers_from_env(), 5);
        unsafe {
            std::env::set_var("SEMANTEX_DAEMON_MAX_SEARCHERS", "not-a-number");
        }
        assert_eq!(max_searchers_from_env(), DEFAULT_MAX_SEARCHERS);
        unsafe {
            std::env::set_var("SEMANTEX_DAEMON_MAX_SEARCHERS", "0");
        }
        assert_eq!(max_searchers_from_env(), DEFAULT_MAX_SEARCHERS);
        unsafe {
            std::env::remove_var("SEMANTEX_DAEMON_MAX_SEARCHERS");
        }
    }

    /// Three distinct projects with a cap of 2: the least-recently-touched
    /// project must be evicted when a third is opened.
    #[test]
    fn lru_evicts_oldest_entry_beyond_cap() {
        let tmp_a = TempDir::new().unwrap();
        let tmp_b = TempDir::new().unwrap();
        let tmp_c = TempDir::new().unwrap();
        for (tmp, content) in [(&tmp_a, "a"), (&tmp_b, "b"), (&tmp_c, "c")] {
            write_fake_git_head(tmp.path(), "main");
            build_index(tmp.path(), content, "0");
        }

        let cache = SearcherCache::new(test_config());
        // cap comes from env; force a small one for this test.
        let cache = SearcherCache {
            max_cached: 2,
            ..cache
        };

        let a = cache.get(tmp_a.path()).unwrap();
        assert_eq!(cache.len(), 1);
        let _b = cache.get(tmp_b.path()).unwrap();
        assert_eq!(cache.len(), 2);
        // Touch `a` again so `b` becomes the least-recently-used entry.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let _a_again = cache.get(tmp_a.path()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Opening `c` must evict `b` (LRU), not `a` (most recently touched).
        let _c = cache.get(tmp_c.path()).unwrap();
        assert_eq!(cache.len(), 2);

        // `a`'s original Arc handle must still be valid/usable even though
        // the map may have long since dropped/replaced its map entry —
        // concurrent-safety guarantee: an in-flight holder survives eviction.
        assert_eq!(read_chunk_content(&a.searcher), "a");
    }

    /// The concurrent-client safety guarantee this module exists to provide:
    /// a thread holding an `Arc<CachedSearcher>` keeps working correctly even
    /// while OTHER threads are concurrently opening new entries and evicting
    /// old ones out from under the map.
    #[test]
    fn concurrent_clients_survive_eviction_via_arc() {
        let tmp_a = TempDir::new().unwrap();
        let tmp_b = TempDir::new().unwrap();
        write_fake_git_head(tmp_a.path(), "main");
        build_index(tmp_a.path(), "alpha", "0");
        write_fake_git_head(tmp_b.path(), "main");
        build_index(tmp_b.path(), "beta", "0");

        let cache = Arc::new(SearcherCache {
            max_cached: 1,
            ..SearcherCache::new(test_config())
        });

        // Thread 1: grab and hold project A's searcher (simulating an
        // in-flight query) across a barrage of evictions from project B.
        let cache_1 = Arc::clone(&cache);
        let path_a = tmp_a.path().to_path_buf();
        let holder = std::thread::spawn(move || {
            let cached = cache_1.get(&path_a).unwrap();
            // Hold it while other threads hammer the cache with a different
            // project key (cap=1 guarantees every B lookup evicts A's entry
            // from the map).
            std::thread::sleep(std::time::Duration::from_millis(50));
            read_chunk_content(&cached.searcher)
        });

        let cache_2 = Arc::clone(&cache);
        let path_b = tmp_b.path().to_path_buf();
        let evictors: Vec<_> = (0..8)
            .map(|_| {
                let cache_2 = Arc::clone(&cache_2);
                let path_b = path_b.clone();
                std::thread::spawn(move || {
                    for _ in 0..5 {
                        let _ = cache_2.get(&path_b);
                    }
                })
            })
            .collect();

        for e in evictors {
            e.join().unwrap();
        }
        let content = holder.join().unwrap();
        assert_eq!(
            content, "alpha",
            "in-flight holder must keep serving project A's content even after its \
             map entry was evicted by concurrent project B lookups"
        );
    }

    /// Branch-switch reconcile round trip: index branch "a", snapshot,
    /// switch to "b" (also snapshotted), switch back to "a" — `get()` must
    /// serve each branch's OWN content after the switch, proving the
    /// per-request probe actually reconciles rather than serving a stale
    /// cached (project_root, old_branch_key) entry.
    #[test]
    fn branch_switch_reconcile_round_trip_serves_new_branch_content() {
        let tmp = TempDir::new().unwrap();
        let project = tmp.path();

        write_fake_git_head(project, "a");
        build_index(project, "fn on_a() {}", "100");
        layout::sync_v13_layout(project, "proj").unwrap();

        write_fake_git_head(project, "b");
        // Fresh chunks.db for "b" — mirroring `index/branches.rs`'s own
        // round-trip test, which removes the file before writing the next
        // branch's synthetic content so the store doesn't accumulate BOTH
        // branches' chunks (an artifact of this lightweight ChunkStore-only
        // harness, not something a real incremental build would do).
        std::fs::remove_file(layout::container_dir(project).join("chunks.db")).unwrap();
        build_index(project, "fn on_b() {}", "200");
        layout::sync_v13_layout(project, "proj").unwrap();
        // Ensure a snapshot for "b" exists at this content before we ever
        // switch back to "a" below.
        assert!(
            layout::branch_index_dir(project, &layout::branch_key_for_branch("b"))
                .join("chunks.db")
                .exists()
        );

        // Switch HEAD back to "a" — "a" has a snapshot (from the very first
        // sync above via `mirror_root_as`/`mirror_into_branch_dir`), so this
        // must resolve as a Restore, not a fresh in-place build.
        write_fake_git_head(project, "a");
        assert!(branches::branch_switch_pending(project));

        let cache = SearcherCache::new(test_config());
        let cached = cache.get(project).unwrap();
        assert_eq!(
            read_chunk_content(&cached.searcher),
            "fn on_a() {}",
            "cache.get() must reconcile the pending switch and serve branch a's restored content"
        );
        assert!(
            !branches::branch_switch_pending(project),
            "reconcile must have cleared the pending switch"
        );

        // Switch forward to "b" again — must serve b's content, proving the
        // reconcile isn't a one-shot fluke and the cache key correctly
        // differentiates branches.
        write_fake_git_head(project, "b");
        let cached_b = cache.get(project).unwrap();
        assert_eq!(read_chunk_content(&cached_b.searcher), "fn on_b() {}");
    }

    /// Staleness reload: simulate "someone ran `semantex index` again in
    /// another terminal" by rewriting `chunks.db`/`meta.json` directly (no
    /// branch switch involved) and bumping `updated_at`. The next `get()`
    /// must notice the marker changed and reopen rather than keep serving
    /// the old cached searcher's content.
    #[test]
    fn staleness_marker_reload_serves_fresh_content_after_rebuild() {
        let tmp = TempDir::new().unwrap();
        let project = tmp.path();
        write_fake_git_head(project, "main");
        build_index(project, "fn before() {}", "1");

        let cache = SearcherCache::new(test_config());
        let first = cache.get(project).unwrap();
        assert_eq!(read_chunk_content(&first.searcher), "fn before() {}");

        // Simulate an out-of-band rebuild: new content, bumped updated_at.
        // Must drop the old sqlite handle before overwriting the file, same
        // as a real rebuild replacing chunks.db.
        drop(first);
        std::fs::remove_file(layout::container_dir(project).join("chunks.db")).unwrap();
        build_index(project, "fn after() {}", "2");

        let second = cache.get(project).unwrap();
        assert_eq!(
            read_chunk_content(&second.searcher),
            "fn after() {}",
            "get() must detect the updated_at marker change and reopen"
        );
    }

    /// `SEMANTEX_DAEMON_MAX_SEARCHERS` cap override actually bounds the
    /// resident entry count end-to-end (not just the parsing unit test
    /// above).
    #[test]
    fn cap_env_override_bounds_resident_entries() {
        let tmp_a = TempDir::new().unwrap();
        let tmp_b = TempDir::new().unwrap();
        let tmp_c = TempDir::new().unwrap();
        for tmp in [&tmp_a, &tmp_b, &tmp_c] {
            write_fake_git_head(tmp.path(), "main");
            build_index(tmp.path(), "x", "0");
        }

        let cache = SearcherCache {
            max_cached: 1,
            ..SearcherCache::new(test_config())
        };
        let _a = cache.get(tmp_a.path()).unwrap();
        assert_eq!(cache.len(), 1);
        let _b = cache.get(tmp_b.path()).unwrap();
        assert_eq!(
            cache.len(),
            1,
            "cap=1 must evict before inserting the second entry"
        );
        let _c = cache.get(tmp_c.path()).unwrap();
        assert_eq!(cache.len(), 1);
    }

    /// `seeded` must reuse the given searcher (not open a second one) and
    /// make it immediately available under the project's current branch key.
    #[test]
    fn seeded_reuses_the_given_searcher() {
        let tmp = TempDir::new().unwrap();
        write_fake_git_head(tmp.path(), "main");
        build_index(tmp.path(), "seeded content", "0");

        let index_dir = layout::container_dir(tmp.path());
        let searcher = HybridSearcher::open(&index_dir, &test_config()).unwrap();
        let cache = SearcherCache::seeded(test_config(), tmp.path(), searcher);
        assert_eq!(cache.len(), 1);

        let cached = cache.get(tmp.path()).unwrap();
        assert_eq!(read_chunk_content(&cached.searcher), "seeded content");
    }

    /// `invalidate_project` drops every entry for a project regardless of
    /// branch key.
    #[test]
    fn invalidate_project_drops_all_branch_entries() {
        let tmp = TempDir::new().unwrap();
        write_fake_git_head(tmp.path(), "main");
        build_index(tmp.path(), "x", "0");

        let cache = SearcherCache::new(test_config());
        let _ = cache.get(tmp.path()).unwrap();
        assert_eq!(cache.len(), 1);
        cache.invalidate_project(tmp.path());
        assert_eq!(cache.len(), 0);
    }
}
