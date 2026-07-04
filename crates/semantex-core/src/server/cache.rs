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
//!    [`branches::branch_switch_pending`]). If pending — and not already
//!    handled per the tombstone below — reconcile via
//!    [`branches::detect_and_handle_branch_switch`] — which takes its OWN
//!    exclusive `.semantex.lock` internally and blocks until it's free; we do
//!    NOT hold any lock of ours across that call — then drop every cached
//!    entry for this project (the live root's content just changed, or for
//!    the `SnapshottedOutgoing` case still needs a real incremental rebuild
//!    via `semantex index`/`watch` before it fully reflects the new branch;
//!    either way an old cached instance must not keep being served under a
//!    reused key).
//!
//!    **F1 — reconcile-once tombstone.** A `SnapshottedOutgoing` reconcile
//!    deliberately leaves the root sidecar recording the OLD branch (that
//!    pending-ness is the signal every other entry point uses to force the
//!    catch-up build), so `branch_switch_pending` stays `true` until a real
//!    build runs — which the daemon itself never does. Without a guard,
//!    EVERY subsequent request would re-enter the reconcile: take the
//!    exclusive index flock, drop all cached entries, and pay a full
//!    searcher reopen — permanent thrash. So after a `SnapshottedOutgoing`
//!    result we record a per-project `(from_key, to_key)` tombstone and skip
//!    re-entering the reconcile while the root sidecar still records
//!    `from_key` and HEAD still points at `to_key`. The tombstone
//!    self-invalidates the moment either side changes (HEAD moved again, or
//!    a real build restamped the sidecar). We do NOT stamp the sidecar with
//!    the new identity from the daemon — that would silently clear the
//!    pending signal the CLI/MCP entry points rely on.
//! 2. **Staleness probe** (cheap: one `stat` of `meta.json`). Compares the
//!    file's nanosecond mtime + length against the value recorded when the
//!    cached entry was opened. A mismatch means the index was rebuilt
//!    beneath us (another terminal ran `semantex index`, or a background
//!    catch-up finished) — evict and reopen. mtime-ns rather than the
//!    parsed `updated_at` field: the builder stamps `updated_at` in whole
//!    SECONDS, so two rebuilds within the same second (routine under
//!    `semantex watch` save-bursts) would be indistinguishable by content
//!    while the file mtime still moves at nanosecond resolution (F4).
//! 3. Cache hit → clone the `Arc` and touch `last_used`. Cache miss →
//!    **single-flight** per key (F3): concurrent misses for the SAME key
//!    serialize on a per-key in-progress marker so exactly ONE
//!    `HybridSearcher::open` runs and the racers reuse its result — N
//!    same-key racers must not pay N slow opens and N× transient searcher
//!    RAM (which could trip the daemon's RSS guard). The winner evicts the
//!    least-recently-used entry BEFORE opening (bounding peak memory, as
//!    the MCP cache does) and re-runs eviction again AT INSERT (distinct-key
//!    racers may have refilled the map while the slow open ran — without the
//!    re-check the map can end up permanently over cap, since hits never
//!    evict).
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
//! OUTGOING branch's — only the snapshot bookkeeping changed — until a real
//! incremental build runs (`semantex index`/`watch`, unchanged from before
//! this module). Searches during that window (i.e. while the F1 tombstone is
//! active) are served under the new branch's key but old content. The daemon
//! has no in-protocol "building" signal to guard this today (unlike the MCP
//! path, which reports `IndexState::Building` and falls back to ripgrep);
//! documenting it here matches this module's brief, which is scoped to the
//! CHEAP reconcile hook, not a synchronous full rebuild on the search hot
//! path. When the catch-up build eventually runs, the staleness probe (step
//! 2) notices the fresh `meta.json` and reopens automatically.

use crate::config::SemantexConfig;
use crate::index::branches;
use crate::index::layout;
use crate::search::hybrid::HybridSearcher;
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// The index's `meta.json` mtime (nanoseconds) + length at open time —
    /// the cheap staleness marker compared on every [`SearcherCache::get`]
    /// call (see [`index_marker`] for why not the parsed `updated_at`).
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
    /// the rest see either `branch_switch_pending() == false` or the F1
    /// tombstone once they acquire this gate right behind it.
    reconcile_gate: Mutex<()>,
    /// F1: per-project "already reconciled" tombstone for the
    /// `SnapshottedOutgoing` case — `project → (from_key, to_key)` of the
    /// exact switch this daemon already handled. `branch_switch_pending`
    /// deliberately stays `true` after that reconcile (the root sidecar keeps
    /// the OLD identity so other entry points force the catch-up build);
    /// without this tombstone the daemon would re-enter the reconcile —
    /// exclusive flock + full cache drop + full searcher reopen — on EVERY
    /// request until a manual `semantex index`. See the module docs.
    reconciled_outgoing: Mutex<HashMap<PathBuf, (String, String)>>,
    /// F3: per-key single-flight locks. A cache miss takes its key's lock
    /// and holds it across the slow `HybridSearcher::open`, so same-key
    /// racers wait for the winner's entry (re-checking the cache once they
    /// acquire the lock) instead of each opening their own — N slow opens +
    /// transient N× searcher RAM would risk tripping the daemon's RSS guard.
    /// Entries are never removed: the table is bounded by the distinct
    /// `(project, branch)` keys ever requested (a few bytes each), and a
    /// permanent per-key lock avoids the remove-vs-recreate races a
    /// self-cleaning map would need to reason about.
    flight_locks: Mutex<HashMap<CacheKey, Arc<Mutex<()>>>>,
    /// Diagnostic count of underlying `HybridSearcher::open` calls this cache
    /// has performed. Read via [`SearcherCache::open_count`]; the F1/F3 tests
    /// use it to prove thrash/duplicate opens can't happen.
    opens: AtomicU64,
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
            reconciled_outgoing: Mutex::new(HashMap::new()),
            flight_locks: Mutex::new(HashMap::new()),
            opens: AtomicU64::new(0),
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

    /// Total number of underlying `HybridSearcher::open` calls this cache has
    /// performed (test/diagnostic hook). The F1 test uses it to prove a
    /// `SnapshottedOutgoing` switch reconciles exactly once; the F3 test uses
    /// it to prove same-key racing misses collapse to a single open.
    pub fn open_count(&self) -> u64 {
        self.opens.load(Ordering::Relaxed)
    }

    /// Resolve the ready-to-query searcher for `project_root`'s CURRENT
    /// branch, reconciling a pending branch switch and reopening on
    /// detected staleness first. See module docs for the full flow.
    pub fn get(&self, project_root: &Path) -> Result<Arc<CachedSearcher>> {
        let canonical = canonicalize(project_root);

        // Item 2: cheap per-request branch-switch probe + reconcile (with the
        // F1 reconcile-once tombstone for the SnapshottedOutgoing case).
        self.reconcile_pending_branch_switch(&canonical);

        let branch_key = layout::current_branch_key(&canonical);
        let marker = index_marker(&canonical);
        let key = (canonical.clone(), branch_key);

        // Fast path: cache hit with a matching staleness marker.
        if let Some(entry) = self.lookup(&key, &marker) {
            return Ok(entry);
        }

        // F3 single-flight: take this key's flight lock so same-key racers
        // wait for the first opener instead of each paying a slow open +
        // transient duplicate searcher RAM.
        let flight = Arc::clone(
            self.flight_locks
                .lock()
                .entry(key.clone())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        );
        let _flight_guard = flight.lock();

        // Re-check under the single-flight guard: if we were a racer, the
        // winner has inserted the entry by the time we acquire the guard.
        // (On a FAILED winner open, racers fall through and retry the open
        // themselves — serially, still one at a time.)
        if let Some(entry) = self.lookup(&key, &marker) {
            return Ok(entry);
        }

        // Evict BEFORE opening to bound PEAK memory (the MCP cache's
        // rationale, kept), then open outside the map lock — I/O + model
        // load is slow and other keys' lookups must not block on it.
        evict_lru_locked(&mut self.entries.lock(), self.max_cached);
        self.open_and_insert(&canonical, &key, marker)
    }

    /// Slow path of [`SearcherCache::get`]: open a fresh searcher and insert
    /// it, re-running eviction AT INSERT TIME (F3) — while our slow open ran,
    /// distinct-key racers may have inserted their own entries, and without
    /// this re-check the map would exceed the cap permanently (cache hits
    /// never evict).
    fn open_and_insert(
        &self,
        canonical: &Path,
        key: &CacheKey,
        marker: String,
    ) -> Result<Arc<CachedSearcher>> {
        let index_dir = SemantexConfig::project_index_dir(canonical);
        self.opens.fetch_add(1, Ordering::Relaxed);
        let searcher = HybridSearcher::open(&index_dir, &self.config)?;
        let entry = Arc::new(CachedSearcher {
            searcher,
            last_used: Mutex::new(Instant::now()),
            marker,
        });

        let mut entries = self.entries.lock();
        evict_lru_locked(&mut entries, self.max_cached);
        entries.insert(key.clone(), Arc::clone(&entry));
        Ok(entry)
    }

    /// Cache lookup with the staleness-marker check: a hit with a matching
    /// marker touches `last_used` and returns the entry; a hit with a STALE
    /// marker (index rebuilt beneath us — another terminal's `semantex
    /// index`, or a background catch-up completed) removes the entry and
    /// returns `None` so the caller reopens.
    fn lookup(&self, key: &CacheKey, marker: &str) -> Option<Arc<CachedSearcher>> {
        let mut entries = self.entries.lock();
        let entry = entries.get(key)?;
        if entry.marker == marker {
            *entry.last_used.lock() = Instant::now();
            return Some(Arc::clone(entry));
        }
        tracing::info!(
            path = %key.0.display(),
            "Daemon: index changed beneath cached searcher — reopening"
        );
        entries.remove(key);
        None
    }

    /// Item 2 + F1: probe for a pending branch switch and reconcile it at
    /// most ONCE per actual switch. Only the (rare) pending case pays the
    /// reconcile-gate lock + `detect_and_handle_branch_switch`'s own
    /// exclusive-lock wait; the overwhelmingly common unchanged case is two
    /// small file reads and nothing else, and an already-tombstoned
    /// `SnapshottedOutgoing` switch costs two more small reads (no gate, no
    /// flock, no cache drop).
    fn reconcile_pending_branch_switch(&self, canonical: &Path) {
        if !branches::branch_switch_pending(canonical)
            || self.outgoing_switch_already_reconciled(canonical)
        {
            return;
        }
        let _gate = self.reconcile_gate.lock();
        // Re-check now that we hold the gate — another thread may have
        // already reconciled this exact switch while we were waiting.
        if !branches::branch_switch_pending(canonical)
            || self.outgoing_switch_already_reconciled(canonical)
        {
            return;
        }
        match branches::detect_and_handle_branch_switch(canonical) {
            Ok(action) => {
                match &action {
                    branches::BranchSwitchAction::SnapshottedOutgoing {
                        from_branch_key,
                        to_branch_key,
                    } => {
                        // F1: `branch_switch_pending` stays true for this
                        // switch until a real build restamps the sidecar —
                        // record the tombstone so we don't re-enter the
                        // reconcile (flock + cache drop + reopen) on every
                        // request until then.
                        self.reconciled_outgoing.lock().insert(
                            canonical.to_path_buf(),
                            (from_branch_key.clone(), to_branch_key.clone()),
                        );
                    }
                    _ => {
                        // Restored / Unchanged / FirstBuild all leave (or put)
                        // the sidecar in agreement with HEAD — any previous
                        // tombstone for this project is obsolete.
                        self.reconciled_outgoing.lock().remove(canonical);
                    }
                }
                if action.switched() {
                    tracing::info!(
                        path = %canonical.display(),
                        ?action,
                        "Daemon: branch switch reconciled"
                    );
                    // Drop every cached entry for this project: its live root
                    // just changed identity (or, for SnapshottedOutgoing, is
                    // pending a real rebuild — see module docs) so nothing
                    // currently cached for it is trustworthy under a fresh
                    // key.
                    self.entries.lock().retain(|(root, _), _| root != canonical);
                }
            }
            Err(e) => {
                tracing::warn!(
                    path = %canonical.display(),
                    err = %e,
                    "Daemon: branch switch check failed (continuing with cached/current index)"
                );
            }
        }
    }

    /// F1 tombstone check: `true` iff this daemon already ran the
    /// `SnapshottedOutgoing` reconcile for EXACTLY the switch that is
    /// currently pending — the root sidecar still records the tombstone's
    /// `from_key` AND HEAD still points at its `to_key`. The moment either
    /// side moves (HEAD switched again; a real build restamped the sidecar)
    /// this returns `false` and the normal reconcile path runs.
    fn outgoing_switch_already_reconciled(&self, canonical: &Path) -> bool {
        let tombstones = self.reconciled_outgoing.lock();
        let Some((from_key, to_key)) = tombstones.get(canonical) else {
            return false;
        };
        let Some(root_meta) = layout::read_root_branch_meta(canonical) else {
            return false;
        };
        root_meta.branch_key == *from_key && layout::current_branch_key(canonical) == *to_key
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

/// Cheap staleness marker: the index's `meta.json` mtime in NANOSECONDS since
/// epoch, plus the file length (one `stat`, no read/parse).
///
/// F4: deliberately NOT the parsed `meta.json::updated_at` field — the
/// builder stamps that in whole epoch SECONDS, so two rebuilds landing within
/// the same second (routine under `semantex watch` save-bursts) would produce
/// an identical marker and an interleaved reopen would serve the first
/// rebuild's content indefinitely. The file's mtime moves at nanosecond
/// resolution on every rewrite; length is appended as a cheap extra
/// discriminator for filesystems with coarse timestamps.
///
/// Best-effort — a missing/unstattable `meta.json` yields an empty string,
/// which simply means the FIRST open always "misses" (there is nothing to
/// compare against yet); it never panics or blocks the daemon.
fn index_marker(project_root: &Path) -> String {
    let meta_path = SemantexConfig::project_index_dir(project_root).join("meta.json");
    match std::fs::metadata(&meta_path) {
        Ok(md) => {
            let mtime_ns = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            format!("{mtime_ns}:{}", md.len())
        }
        Err(_) => String::new(),
    }
}

fn max_searchers_from_env() -> usize {
    parse_max_searchers(
        std::env::var("SEMANTEX_DAEMON_MAX_SEARCHERS")
            .ok()
            .as_deref(),
    )
}

/// Pure parsing core of [`max_searchers_from_env`] — split out (mirroring
/// `index/branches.rs`'s `_with_cap` idiom) so tests can exercise the parsing
/// rules without mutating process-global environment variables.
fn parse_max_searchers(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
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

    /// `SEMANTEX_DAEMON_MAX_SEARCHERS` parsing rules, exercised through the
    /// pure `parse_max_searchers` core — no process-global env mutation
    /// (the `_with_cap`/`_from` split idiom from `index/branches.rs` /
    /// `index/registry.rs`).
    #[test]
    fn max_searchers_parsing_accepts_override_and_rejects_garbage() {
        assert_eq!(parse_max_searchers(None), DEFAULT_MAX_SEARCHERS);
        assert_eq!(parse_max_searchers(Some("5")), 5);
        assert_eq!(parse_max_searchers(Some(" 3 ")), 3);
        assert_eq!(
            parse_max_searchers(Some("not-a-number")),
            DEFAULT_MAX_SEARCHERS
        );
        assert_eq!(parse_max_searchers(Some("0")), DEFAULT_MAX_SEARCHERS);
        assert_eq!(parse_max_searchers(Some("")), DEFAULT_MAX_SEARCHERS);
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

    /// F1 (blocker regression): a switch to a branch with NO snapshot
    /// (`SnapshottedOutgoing` — the `git switch -c` case) leaves
    /// `branch_switch_pending` TRUE (only a real build clears it, and the
    /// daemon never builds). The reconcile must run exactly ONCE: the second
    /// `get()` must reuse the SAME cached searcher (`Arc::ptr_eq`), must not
    /// reopen (open_count stays 1), and must NOT re-enter
    /// `detect_and_handle_branch_switch` — proven by holding the exclusive
    /// index flock during the second call: were the reconcile re-entered, it
    /// would block on that lock and the call would time out.
    #[test]
    fn snapshotted_outgoing_reconciles_once_then_reuses_cached_searcher() {
        let tmp = TempDir::new().unwrap();
        let project = tmp.path().to_path_buf();

        write_fake_git_head(&project, "a");
        build_index(&project, "fn on_a() {}", "100");
        layout::sync_v13_layout(&project, "proj").unwrap();

        // Switch HEAD to a brand-new branch with no snapshot.
        write_fake_git_head(&project, "b");
        assert!(branches::branch_switch_pending(&project));

        let cache = Arc::new(SearcherCache::new(test_config()));
        let first = cache.get(&project).unwrap();
        assert_eq!(cache.open_count(), 1);
        // The pending flag is INTENTIONALLY still set — the root sidecar
        // keeps the old identity so `semantex index`/`watch`/MCP force the
        // catch-up build. That persistence is exactly what made the pre-F1
        // code thrash.
        assert!(
            branches::branch_switch_pending(&project),
            "test precondition: SnapshottedOutgoing must leave the switch pending"
        );

        // Hold the exclusive index lock, as a concurrent `semantex index`
        // build would. A buggy second get() that re-enters the reconcile
        // blocks here forever; the fixed one never touches the lock.
        let lock_path = layout::container_dir(&project).join(".semantex.lock");
        let lock_file = std::fs::File::create(&lock_path).unwrap();
        lock_file.lock().unwrap();

        let (tx, rx) = std::sync::mpsc::channel();
        let cache_2 = Arc::clone(&cache);
        let project_2 = project.clone();
        std::thread::spawn(move || {
            let _ = tx.send(cache_2.get(&project_2));
        });
        let second = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect(
                "second get() after a SnapshottedOutgoing reconcile must not re-enter \
                 detect_and_handle_branch_switch (it would block on the index flock) — \
                 the F1 tombstone should have skipped the reconcile entirely",
            )
            .unwrap();
        drop(lock_file);

        assert!(
            Arc::ptr_eq(&first, &second),
            "second get() must reuse the same cached searcher, not drop + reopen"
        );
        assert_eq!(
            cache.open_count(),
            1,
            "no additional searcher open may happen while the tombstone is active"
        );

        // Tombstone self-invalidation: switching back to "a" agrees with the
        // (never-restamped) root sidecar, so pending clears without any
        // reconcile; switching to a THIRD branch is a different (from, to)
        // pair and must reconcile again rather than being masked.
        write_fake_git_head(&project, "a");
        assert!(!branches::branch_switch_pending(&project));
        write_fake_git_head(&project, "c");
        assert!(branches::branch_switch_pending(&project));
        let third = cache.get(&project).unwrap();
        assert!(
            !Arc::ptr_eq(&first, &third),
            "a switch to a different branch must not be masked by the old tombstone"
        );
    }

    /// F3: N threads racing a cache miss for the SAME key must collapse to
    /// exactly ONE underlying `HybridSearcher::open` (single-flight), all
    /// receiving the same entry. Without single-flight each racer pays its
    /// own slow open and transiently holds its own full searcher — N× RAM,
    /// enough to trip the daemon's RSS guard.
    #[test]
    fn racing_same_key_misses_open_exactly_once() {
        let tmp = TempDir::new().unwrap();
        write_fake_git_head(tmp.path(), "main");
        build_index(tmp.path(), "raced", "0");

        let cache = Arc::new(SearcherCache::new(test_config()));
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let barrier = Arc::clone(&barrier);
                let path = tmp.path().to_path_buf();
                std::thread::spawn(move || {
                    barrier.wait();
                    cache.get(&path).unwrap()
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(
            cache.open_count(),
            1,
            "8 same-key racers must be collapsed into exactly one searcher open"
        );
        for r in &results[1..] {
            assert!(
                Arc::ptr_eq(&results[0], r),
                "all racers must receive the single-flight winner's entry"
            );
        }
    }

    /// Staleness reload: simulate "someone ran `semantex index` again in
    /// another terminal" by rewriting `chunks.db`/`meta.json` directly (no
    /// branch switch involved). The next `get()` must notice the marker
    /// (meta.json mtime-ns + length, F4) changed and reopen rather than keep
    /// serving the old cached searcher's content.
    #[test]
    fn staleness_marker_reload_serves_fresh_content_after_rebuild() {
        let tmp = TempDir::new().unwrap();
        let project = tmp.path();
        write_fake_git_head(project, "main");
        build_index(project, "fn before() {}", "1");

        let cache = SearcherCache::new(test_config());
        let first = cache.get(project).unwrap();
        assert_eq!(read_chunk_content(&first.searcher), "fn before() {}");

        // Simulate an out-of-band rebuild: new content, rewritten meta.json.
        // Deliberately the SAME updated_at value ("1") — the F4 marker must
        // catch the rebuild from the file's mtime alone, exactly the two-
        // rebuilds-in-one-second case the parsed updated_at (whole seconds)
        // could not distinguish. Must drop the old sqlite handle before
        // overwriting the file, same as a real rebuild replacing chunks.db.
        drop(first);
        std::fs::remove_file(layout::container_dir(project).join("chunks.db")).unwrap();
        build_index(project, "fn after() {}", "1");

        let second = cache.get(project).unwrap();
        assert_eq!(
            read_chunk_content(&second.searcher),
            "fn after() {}",
            "get() must detect the meta.json mtime marker change and reopen"
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
