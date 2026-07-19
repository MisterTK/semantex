//! Cinder centroid shortlists (SXCS format, spec §4.1 #4): per-vocab-token
//! shortlists of nearby centroid ids, derived offline from a
//! [`StaticTokenTable`] (Ember Plan A) and a trained centroid set (Ember
//! Plan B / `centroid_train`). Cinder's compiled index path uses these to
//! replace an exhaustive per-token argmax over every centroid with a
//! union-of-shortlists argmax over just the few centroids near each window
//! token's typical embedding.

use anyhow::{Context, Result};
use ndarray::ArrayView2;
use std::path::Path;

use super::static_table::StaticTokenTable;

const MAGIC: &[u8; 4] = b"SXCS";
const VERSION: u32 = 1;
const MAX_VOCAB_SIZE: usize = 10_000_000;
const MIN_M: usize = 1;
const MAX_M: usize = 1024;
/// u16 centroid ids can address at most this many centroids.
const MAX_CENTROIDS: usize = u16::MAX as usize + 1;

/// Per-vocab-token centroid shortlists (SXCS v1).
///
/// `data` is laid out as `vocab_size + 1` rows of `m` centroid ids each; the
/// last row (index `vocab_size`) is the global fallback shortlist used for
/// zero-row and out-of-vocabulary token ids.
#[derive(Debug, Clone)]
pub struct CentroidShortlists {
    /// Entries per token (and in the fallback row).
    pub m: usize,
    /// Number of per-token rows (excludes the fallback row).
    pub vocab_size: usize,
    data: Vec<u16>,
}

impl CentroidShortlists {
    /// Derive shortlists for every token in `table` against `centroids`.
    ///
    /// For each nonzero table row (per `StaticTokenTable::lookup`), the
    /// exact top-`m` centroids by dot product are stored, ties broken by
    /// ascending centroid id. Zero rows get the global fallback row: the
    /// top-`m` centroids of the mean of all nonzero rows (an all-zero mean,
    /// when the table has no nonzero rows at all, falls back to centroids
    /// `0..m` by the same tie-break rule).
    ///
    /// `m` is capped at `centroids.nrows()`. `centroids.nrows()` must be
    /// nonzero and fit a u16 id (`<= 65536`), otherwise this errors.
    pub fn derive(table: &StaticTokenTable, centroids: &ArrayView2<f32>, m: usize) -> Result<Self> {
        let n_centroids = centroids.nrows();
        anyhow::ensure!(
            n_centroids > 0 && n_centroids <= MAX_CENTROIDS,
            "derive: centroid count {n_centroids} must be in 1..={MAX_CENTROIDS} to fit u16 ids"
        );
        anyhow::ensure!(
            centroids.ncols() == table.dims,
            "derive: centroid dims {} != table dims {}",
            centroids.ncols(),
            table.dims
        );
        let m = m.min(n_centroids);
        let vocab_size = table.vocab_size();

        let mut data = vec![0u16; (vocab_size + 1) * m];
        let mut mean_sum = vec![0.0f64; table.dims];
        let mut mean_count: u64 = 0;
        let mut missing_ids: Vec<usize> = Vec::new();

        for token_id in 0..vocab_size {
            match table.lookup(token_id as u32) {
                Some(row) => {
                    for (acc, &v) in mean_sum.iter_mut().zip(row.iter()) {
                        *acc += f64::from(v);
                    }
                    mean_count += 1;
                    let top = top_m_by_dot(row, centroids, m);
                    data[token_id * m..(token_id + 1) * m].copy_from_slice(&top);
                }
                None => missing_ids.push(token_id),
            }
        }

        let fallback_row: Vec<f32> = if mean_count > 0 {
            let inv = 1.0 / mean_count as f64;
            mean_sum.iter().map(|&v| (v * inv) as f32).collect()
        } else {
            vec![0.0f32; table.dims]
        };
        let fallback_top = top_m_by_dot(&fallback_row, centroids, m);
        data[vocab_size * m..(vocab_size + 1) * m].copy_from_slice(&fallback_top);
        for token_id in missing_ids {
            data[token_id * m..(token_id + 1) * m].copy_from_slice(&fallback_top);
        }

        Ok(Self {
            m,
            vocab_size,
            data,
        })
    }

    /// The shortlist for `token_id`: `m` centroid ids, nearest-first (as
    /// produced by `derive`'s descending-dot sort). Zero-row and
    /// out-of-vocabulary token ids return the global fallback row.
    pub fn for_token(&self, token_id: u32) -> &[u16] {
        let idx = token_id as usize;
        if idx >= self.vocab_size {
            return self.fallback_slice();
        }
        &self.data[idx * self.m..(idx + 1) * self.m]
    }

    fn fallback_slice(&self) -> &[u16] {
        &self.data[self.vocab_size * self.m..(self.vocab_size + 1) * self.m]
    }

    /// Save as SXCS v1, atomically (temp file in the same directory, then
    /// rename) — mirrors `MicroMixer::save`.
    pub fn save(&self, path: &Path) -> Result<()> {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join(format!(
            ".{}.{}.tmp",
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("shortlists"),
            std::process::id()
        ));
        let mut buf = Vec::with_capacity(16 + self.data.len() * 2);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.vocab_size as u32).to_le_bytes());
        buf.extend_from_slice(&(self.m as u32).to_le_bytes());
        for &v in &self.data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&tmp, &buf)
            .with_context(|| format!("writing shortlists temp file {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("saving shortlists to {}", path.display()))?;
        Ok(())
    }

    /// Load from SXCS v1, with a fully checked header (magic, version,
    /// plausibility bounds, checked size arithmetic, exact-size match) —
    /// mirrors `MicroMixer::load` / `StaticTokenTable::load`'s forged-header
    /// defenses.
    pub fn load(path: &Path) -> Result<Self> {
        let buf = std::fs::read(path)?;
        anyhow::ensure!(buf.len() >= 16, "shortlists file too short for header");
        anyhow::ensure!(&buf[0..4] == MAGIC, "invalid shortlists magic");
        let rd = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        anyhow::ensure!(rd(4) == VERSION, "unsupported shortlists version {}", rd(4));
        let vocab_size = rd(8) as usize;
        let m = rd(12) as usize;

        // Checked total size BEFORE allocation. A forged header can put
        // arbitrary u32s in vocab_size/m; unchecked (vocab_size+1)*m*2 can
        // wrap, letting a size/bounds check pass incorrectly.
        let n_entries = vocab_size
            .checked_add(1)
            .and_then(|v| v.checked_mul(m))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "implausibly large shortlists header: (vocab_size {vocab_size} + 1) × m {m} overflows"
                )
            })?;
        let expected_bytes = n_entries.checked_mul(2).ok_or_else(|| {
            anyhow::anyhow!(
                "implausibly large shortlists header: entry count {n_entries} × 2 overflows"
            )
        })?;

        anyhow::ensure!(
            vocab_size <= MAX_VOCAB_SIZE,
            "implausible shortlists vocab_size {vocab_size} (max {MAX_VOCAB_SIZE})"
        );
        anyhow::ensure!(
            (MIN_M..=MAX_M).contains(&m),
            "implausible shortlists m {m} (must be {MIN_M}..={MAX_M})"
        );

        let remaining = buf.len() - 16;
        anyhow::ensure!(
            remaining == expected_bytes,
            "shortlists file wrong size: {remaining} data bytes, want {expected_bytes}"
        );

        let mut data = Vec::with_capacity(n_entries);
        let mut cursor = 16;
        for _ in 0..n_entries {
            data.push(u16::from_le_bytes([buf[cursor], buf[cursor + 1]]));
            cursor += 2;
        }

        Ok(Self {
            m,
            vocab_size,
            data,
        })
    }
}

/// Exact top-`m` centroid ids by descending dot product with `row`, ties
/// broken by ascending id. Exhaustive over all centroids — this is an
/// offline derivation step (`O(n_centroids × dims)` per call — no accidental
/// `O(vocab²)` across the caller's `0..vocab_size` loop).
///
/// Precondition (checked by `derive`'s caller): `centroids.nrows()` fits a
/// u16 id and `m <= centroids.nrows()`.
fn top_m_by_dot(row: &[f32], centroids: &ArrayView2<f32>, m: usize) -> Vec<u16> {
    let mut scored: Vec<(f32, u16)> = centroids
        .rows()
        .into_iter()
        .enumerate()
        .map(|(id, c)| {
            let dot: f32 = row.iter().zip(c.iter()).map(|(&a, &b)| a * b).sum();
            (dot, id as u16)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.truncate(m);
    scored.into_iter().map(|(_, id)| id).collect()
}

/// Reusable per-thread scratch for [`shortlist_argmax`]. Holds an
/// epoch-stamped "seen" marker over centroid ids so each unique candidate is
/// scored exactly once WITHOUT sorting/deduping the ~`window·m` gathered ids —
/// `seen[cid] == epoch` means "already scored during the current call", and
/// bumping `epoch` once per call retires the previous marks in O(1) instead of
/// re-clearing the whole array. One instance per worker (created via
/// `map_init`), reused across every token that worker handles.
///
/// This replaced the previous `Vec<u16>` gather-then-`sort_unstable`+`dedup`
/// scratch: at the production window (9 taps) × `m` (128) that was a ~1152-wide
/// sort per token, and profiling (Task 11) showed the assign stage — of which
/// that sort is a large part — dominates the whole dense build (~74%). The
/// marker dedup is O(window·m) with no sort, and the argmax it drives is
/// BYTE-IDENTICAL to the old sorted scan (see `shortlist_argmax`'s tie-break
/// note and the `marker_argmax_matches_sorted_reference` differential test).
#[derive(Default)]
pub struct ArgmaxScratch {
    /// `seen[cid]` holds the epoch during which `cid` was last scored.
    seen: Vec<u32>,
    /// Current call's epoch; `seen[cid] == epoch` ⇒ already scored this call.
    epoch: u32,
}

impl ArgmaxScratch {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a new call: grow the marker to cover `n_centroids` and advance the
    /// epoch. On the (practically unreachable) `u32` wrap back to 0 the whole
    /// marker is reset so a freshly-`resize`d `0` can never be mistaken for the
    /// live epoch.
    #[inline]
    fn begin(&mut self, n_centroids: usize) {
        if self.seen.len() < n_centroids {
            self.seen.resize(n_centroids, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.seen.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
    }

    /// `true` the FIRST time `cid` is offered during the current call, `false`
    /// on every later repeat (the union of window shortlists overlaps heavily).
    #[inline]
    fn first_time(&mut self, cid: usize) -> bool {
        if self.seen[cid] == self.epoch {
            false
        } else {
            self.seen[cid] = self.epoch;
            true
        }
    }
}

/// `argmax_{c in union of window shortlists} (e · centroid_c)`, ties broken by
/// lowest id. `scratch` (see [`ArgmaxScratch`]) dedups the candidate ids via an
/// epoch marker so each unique centroid is scored once; `window_ids` may repeat
/// the same id (edge replication), which the marker absorbs.
///
/// Byte-identity with the previous sort-based implementation: the set of unique
/// candidate centroids scored, and each individual dot product `e · centroid_c`
/// (same 48-wide sequential sum, unchanged), are identical — only the ORDER in
/// which candidates are scored differs (marker/first-seen order vs ascending
/// id). The explicit `dot > best || (dot == best && cid < best_id)` tie-break
/// makes the winner independent of that order — max dot, lowest id on an exact
/// tie — reproducing the old "sorted ascending + strict `>`" result bit-for-bit
/// (proven empirically by `marker_argmax_matches_sorted_reference`).
pub fn shortlist_argmax(
    e: &[f32],
    window_ids: &[u32],
    shortlists: &CentroidShortlists,
    centroids: &ArrayView2<f32>,
    scratch: &mut ArgmaxScratch,
) -> usize {
    scratch.begin(centroids.nrows());

    let mut best_id: usize = 0;
    let mut best_dot = f32::NEG_INFINITY;
    for &id in window_ids {
        for &cid in shortlists.for_token(id) {
            let cid = cid as usize;
            if !scratch.first_time(cid) {
                continue;
            }
            let row = centroids.row(cid);
            let dot: f32 = e.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
            if dot > best_dot || (dot == best_dot && cid < best_id) {
                best_dot = dot;
                best_id = cid;
            }
        }
    }
    best_id
}

/// Exhaustive argmax over every centroid, with the same tie-break rule as
/// [`shortlist_argmax`] — the ground truth [`shortlist_agreement`] compares
/// against.
fn exhaustive_argmax(e: &[f32], centroids: &ArrayView2<f32>) -> usize {
    let mut best_id = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for (id, row) in centroids.rows().into_iter().enumerate() {
        let dot: f32 = e.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
        if dot > best_dot {
            best_dot = dot;
            best_id = id;
        }
    }
    best_id
}

/// Diagnostic for gate C4: fraction of `samples` whose `shortlist_argmax`
/// equals the exhaustive argmax over all centroids. `1.0` (vacuous
/// agreement) if `samples` is empty.
pub fn shortlist_agreement(
    samples: &[(Vec<f32>, Vec<u32>)],
    shortlists: &CentroidShortlists,
    centroids: &ArrayView2<f32>,
) -> f64 {
    if samples.is_empty() {
        return 1.0;
    }
    let mut scratch = ArgmaxScratch::new();
    let matches = samples
        .iter()
        .filter(|(e, window_ids)| {
            shortlist_argmax(e, window_ids, shortlists, centroids, &mut scratch)
                == exhaustive_argmax(e, centroids)
        })
        .count();
    matches as f64 / samples.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Minimal splitmix64 PRNG for deterministic pseudo-random test fixtures.
    /// Copied from `mixer_train::SplitMix64` / `centroid_train::SplitMix64`
    /// (both private to their own modules) — no cryptographic randomness
    /// needed here, just a reproducible sequence.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn next_below(&mut self, bound: u64) -> u64 {
            self.next_u64() % bound
        }

        fn next_unit_f32(&mut self) -> f32 {
            let bits = (self.next_u64() >> 40) as u32;
            (bits as f32) / (1u32 << 24) as f32
        }

        fn next_range_f32(&mut self, lo: f32, hi: f32) -> f32 {
            lo + self.next_unit_f32() * (hi - lo)
        }
    }

    // ── (a)/(b)/(e) hand-built fixture: 6 centroids around a 2D "clock",
    //    4 tokens each clearly nearest a distinct centroid. ────────────────

    /// c0=+x, c1=+y, c2=-x, c3=-y, c4≈+x+y, c5≈-x-y.
    fn hand_built_centroids() -> Array2<f32> {
        Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 0.0, // c0
                0.0, 1.0, // c1
                -1.0, 0.0, // c2
                0.0, -1.0, // c3
                0.6, 0.8, // c4
                -0.6, -0.8, // c5
            ],
        )
        .unwrap()
    }

    /// 4 tokens, each clearly nearest a distinct centroid: 0->c0, 1->c1,
    /// 2->c2, 3->c4. All rows nonzero.
    fn hand_built_table_all_set() -> StaticTokenTable {
        let mut t = StaticTokenTable::new(4, 2, [0.2; 5]);
        t.set_row(0, &[2.0, 0.1]);
        t.set_row(1, &[0.1, 2.0]);
        t.set_row(2, &[-2.0, -0.1]);
        t.set_row(3, &[1.2, 1.6]);
        t
    }

    /// Same as above but token 3's row is left zero (never set) — the
    /// `StaticTokenTable::lookup` `None` convention.
    fn hand_built_table_with_zero_row() -> StaticTokenTable {
        let mut t = StaticTokenTable::new(4, 2, [0.2; 5]);
        t.set_row(0, &[2.0, 0.1]);
        t.set_row(1, &[0.1, 2.0]);
        t.set_row(2, &[-2.0, -0.1]);
        t
    }

    #[test]
    fn derive_puts_true_nearest_centroid_first_per_token() {
        let centroids = hand_built_centroids();
        let table = hand_built_table_all_set();
        let shortlists = CentroidShortlists::derive(&table, &centroids.view(), 3).unwrap();

        assert_eq!(shortlists.m, 3);
        assert_eq!(shortlists.vocab_size, 4);
        // Hand-computed dot products against all 6 centroids, descending:
        //   token0 [2.0,0.1]:  c0=2.0, c4=1.28, c1=0.1, c3=-0.1, c5=-1.28, c2=-2.0
        //   token1 [0.1,2.0]:  c1=2.0, c4=1.66, c0=0.1, c2=-0.1, c5=-1.66, c3=-2.0
        //   token2 [-2.0,-0.1]:c2=2.0, c5=1.28, c3=0.1, c1=-0.1, c4=-1.28, c0=-2.0
        //   token3 [1.2,1.6]:  c4=2.0, c1=1.6,  c0=1.2, c2=-1.2, c3=-1.6,  c5=-2.0
        assert_eq!(shortlists.for_token(0), &[0, 4, 1]);
        assert_eq!(shortlists.for_token(1), &[1, 4, 0]);
        assert_eq!(shortlists.for_token(2), &[2, 5, 3]);
        assert_eq!(shortlists.for_token(3), &[4, 1, 0]);
    }

    #[test]
    fn zero_row_token_gets_fallback_row_equal_to_mean_of_nonzero_rows_top_m() {
        let centroids = hand_built_centroids();
        let table = hand_built_table_with_zero_row();
        let shortlists = CentroidShortlists::derive(&table, &centroids.view(), 3).unwrap();

        // Mean of the 3 nonzero rows [2.0,0.1],[0.1,2.0],[-2.0,-0.1]:
        //   ((2.0-2.0+0.1)/3, (0.1-0.1+2.0)/3) = (0.1/3, 2.0/3) ≈ (0.0333, 0.6667)
        // Dots against the 6 centroids:
        //   c0≈0.0333, c1≈0.6667, c2≈-0.0333, c3≈-0.6667,
        //   c4 ≈ 0.0333*0.6 + 0.6667*0.8 ≈ 0.5533, c5≈-0.5533
        // Descending: c1, c4, c0, c2, c5, c3 -> top-3 = [1, 4, 0].
        let fallback = shortlists.for_token(3);
        assert_eq!(fallback, &[1, 4, 0]);

        // Out-of-vocabulary ids map to the same fallback row.
        assert_eq!(shortlists.for_token(999), fallback);
    }

    #[test]
    fn save_load_round_trip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("cinder_shortlists.bin");
        let centroids = hand_built_centroids();
        let table = hand_built_table_all_set();
        let original = CentroidShortlists::derive(&table, &centroids.view(), 3).unwrap();

        original.save(&path).unwrap();
        let loaded = CentroidShortlists::load(&path).unwrap();
        assert_eq!(loaded.m, original.m);
        assert_eq!(loaded.vocab_size, original.vocab_size);
        assert_eq!(loaded.data, original.data);
    }

    // ── (c)/(d1) full-coverage fixture: random centroids/table, m equal to
    //    the full centroid count so every window's shortlist union always
    //    contains the globally best centroid. ──────────────────────────────

    const FULL_DIMS: usize = 8;
    const FULL_N_CENTROIDS: usize = 16;
    const FULL_VOCAB_SIZE: usize = 5;

    fn random_centroids(n: usize, dims: usize, seed: u64) -> Array2<f32> {
        let mut rng = SplitMix64::new(seed);
        Array2::from_shape_fn((n, dims), |_| rng.next_range_f32(-1.0, 1.0))
    }

    fn random_table(vocab_size: usize, dims: usize, seed: u64) -> StaticTokenTable {
        let mut rng = SplitMix64::new(seed);
        let mut t = StaticTokenTable::new(vocab_size, dims, [0.2; 5]);
        for id in 0..vocab_size as u32 {
            let row: Vec<f32> = (0..dims).map(|_| rng.next_range_f32(-1.0, 1.0)).collect();
            t.set_row(id, &row);
        }
        t
    }

    /// 20 deterministic pseudo-random `(embedding, window_ids)` samples;
    /// window length 1..=3, ids may repeat (edge-replication convention).
    fn random_samples(vocab_size: usize, dims: usize, seed: u64) -> Vec<(Vec<f32>, Vec<u32>)> {
        let mut rng = SplitMix64::new(seed);
        (0..20)
            .map(|_| {
                let e: Vec<f32> = (0..dims).map(|_| rng.next_range_f32(-1.0, 1.0)).collect();
                let window_len = 1 + rng.next_below(3) as usize;
                let window_ids: Vec<u32> = (0..window_len)
                    .map(|_| rng.next_below(vocab_size as u64) as u32)
                    .collect();
                (e, window_ids)
            })
            .collect()
    }

    fn full_coverage_fixture() -> (Array2<f32>, CentroidShortlists, Vec<(Vec<f32>, Vec<u32>)>) {
        let centroids = random_centroids(FULL_N_CENTROIDS, FULL_DIMS, 0xC1DE_A001);
        let table = random_table(FULL_VOCAB_SIZE, FULL_DIMS, 0xC1DE_A002);
        // m == centroid count: the shortlist union always covers every
        // centroid, so this isolates shortlist_argmax's own scan/tie-break
        // logic against the exhaustive ground truth.
        let shortlists =
            CentroidShortlists::derive(&table, &centroids.view(), FULL_N_CENTROIDS).unwrap();
        let samples = random_samples(FULL_VOCAB_SIZE, FULL_DIMS, 0xC1DE_A003);
        (centroids, shortlists, samples)
    }

    #[test]
    fn shortlist_argmax_matches_exhaustive_when_shortlist_covers_all_centroids() {
        let (centroids, shortlists, samples) = full_coverage_fixture();
        let mut scratch = ArgmaxScratch::new();
        for (e, window_ids) in &samples {
            let got = shortlist_argmax(e, window_ids, &shortlists, &centroids.view(), &mut scratch);
            let want = exhaustive_argmax(e, &centroids.view());
            assert_eq!(got, want, "e={e:?} window={window_ids:?}");
        }
    }

    #[test]
    fn parallel_map_init_matches_serial_argmax() {
        // Proves the build-path parallelization strategy used in
        // `build_cinder`'s id-aware assigner — `(0..n).into_par_iter()
        // .map_init(Vec::new, ...)` over the chunk's token rows — produces
        // BYTE-IDENTICAL codes to the serial loop, regardless of thread
        // scheduling. This is the determinism guarantee the Cinder
        // byte-identity contract depends on: each row's argmax reads only its
        // own inputs, and rayon's indexed `collect` restores row order.
        use rayon::prelude::*;
        let centroids = random_centroids(FULL_N_CENTROIDS, FULL_DIMS, 0xD37E_0001);
        let table = random_table(64, FULL_DIMS, 0xD37E_0002);
        // m=4 << 16 centroids so the shortlist union is a strict subset and the
        // argmax genuinely depends on the union (not a trivial full scan).
        let shortlists = CentroidShortlists::derive(&table, &centroids.view(), 4).unwrap();
        let samples = random_samples(64, FULL_DIMS, 0xD37E_0003);
        let cview = centroids.view();

        let mut scratch = ArgmaxScratch::new();
        let serial: Vec<usize> = samples
            .iter()
            .map(|(e, w)| shortlist_argmax(e, w, &shortlists, &cview, &mut scratch))
            .collect();

        let parallel: Vec<usize> = (0..samples.len())
            .into_par_iter()
            .map_init(ArgmaxScratch::new, |scratch, r| {
                let (e, w) = &samples[r];
                shortlist_argmax(e, w, &shortlists, &cview, scratch)
            })
            .collect();

        assert_eq!(
            serial, parallel,
            "parallel argmax must match serial exactly"
        );
    }

    /// The pre–Task-11 `shortlist_argmax` body: gather the window's shortlist
    /// ids, `sort_unstable`+`dedup`, then scan ascending with strict `>`
    /// (lowest-id tie-break for free). Kept ONLY as a byte-identity oracle for
    /// the epoch-marker rewrite.
    fn sorted_reference_argmax(
        e: &[f32],
        window_ids: &[u32],
        shortlists: &CentroidShortlists,
        centroids: &ArrayView2<f32>,
    ) -> usize {
        let mut scratch: Vec<u16> = Vec::new();
        for &id in window_ids {
            scratch.extend_from_slice(shortlists.for_token(id));
        }
        scratch.sort_unstable();
        scratch.dedup();
        let mut best_id: u16 = 0;
        let mut best_dot = f32::NEG_INFINITY;
        for &cid in scratch.iter() {
            let row = centroids.row(cid as usize);
            let dot: f32 = e.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
            if dot > best_dot {
                best_dot = dot;
                best_id = cid;
            }
        }
        best_id as usize
    }

    #[test]
    fn marker_argmax_matches_sorted_reference() {
        // Byte-identity oracle for the Task-11 sort→epoch-marker rewrite of
        // shortlist_argmax. Over thousands of pseudo-random (embedding, window)
        // cases — spanning strict-subset shortlists (m < n_centroids),
        // near-full unions (large m), and 9-tap (production MIXER_WINDOW)
        // windows with freely-repeated ids — the marker implementation must
        // return the IDENTICAL centroid id the old sort+dedup+ascending-scan
        // implementation did. Different m values give different-sized deduped
        // candidate sets, exercising the marker dedup across the range the real
        // build hits.
        let dims = 24;
        let n_centroids = 512;
        let centroids = random_centroids(n_centroids, dims, 0x7A11_0001);
        let cview = centroids.view();
        let vocab = 256;
        let table = random_table(vocab, dims, 0x7A11_0002);
        for &m in &[1usize, 4, 32, 128] {
            let shortlists = CentroidShortlists::derive(&table, &cview, m).unwrap();
            let mut rng = SplitMix64::new(0x7A11_1000 + m as u64);
            let mut scratch = ArgmaxScratch::new();
            for _ in 0..2000 {
                let e: Vec<f32> = (0..dims).map(|_| rng.next_range_f32(-1.0, 1.0)).collect();
                let window: Vec<u32> = (0..9)
                    .map(|_| rng.next_below(vocab as u64) as u32)
                    .collect();
                let got = shortlist_argmax(&e, &window, &shortlists, &cview, &mut scratch);
                let want = sorted_reference_argmax(&e, &window, &shortlists, &cview);
                assert_eq!(got, want, "m={m} window={window:?}");
            }
        }

        // Exact-dot tie: centroids 1 and 2 are identical rows, so any e scores
        // them equally. The lowest id (1) must win in BOTH implementations,
        // exercising the marker path's explicit `dot == best && cid < best_id`
        // tie-break (the `==` branch the reference resolves via ascending order).
        let tie_centroids =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0]).unwrap();
        let tcv = tie_centroids.view();
        let mut tie_table = StaticTokenTable::new(1, 2, [0.2; 5]);
        tie_table.set_row(0, &[0.5, 0.5]);
        let tie_short = CentroidShortlists::derive(&tie_table, &tcv, 4).unwrap();
        let mut scratch = ArgmaxScratch::new();
        for e in [vec![0.3f32, 0.3], vec![0.9, 0.1], vec![-0.2, 0.7]] {
            let got = shortlist_argmax(&e, &[0, 0, 0], &tie_short, &tcv, &mut scratch);
            let want = sorted_reference_argmax(&e, &[0, 0, 0], &tie_short, &tcv);
            assert_eq!(got, want, "exact-tie case e={e:?} must match reference");
        }
    }

    #[test]
    fn shortlist_agreement_is_1_when_shortlists_cover_all_centroids() {
        let (centroids, shortlists, samples) = full_coverage_fixture();
        let agreement = shortlist_agreement(&samples, &shortlists, &centroids.view());
        assert_eq!(agreement, 1.0);
    }

    #[test]
    fn shortlist_agreement_drops_below_1_with_adversarial_truncated_shortlists() {
        // Token A nearest centroid 0, token B nearest centroid 1; m=1 so
        // each token's shortlist holds only its own nearest centroid.
        let dims = 2;
        let centroids = Array2::from_shape_vec((2, dims), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let mut table = StaticTokenTable::new(2, dims, [0.2; 5]);
        table.set_row(0, &[1.0, 0.0]); // token A
        table.set_row(1, &[0.0, 1.0]); // token B
        let shortlists = CentroidShortlists::derive(&table, &centroids.view(), 1).unwrap();
        assert_eq!(shortlists.for_token(0), &[0]);
        assert_eq!(shortlists.for_token(1), &[1]);

        // Query embedding is nearest centroid 1 globally, but the window
        // contains only token A, whose truncated shortlist is {centroid 0}.
        let e = vec![0.0f32, 1.0];
        let samples = vec![(e, vec![0u32])];
        let agreement = shortlist_agreement(&samples, &shortlists, &centroids.view());
        assert_eq!(agreement, 0.0, "agreement should drop below 1.0");
    }

    // ── (f) forged-header rejection ────────────────────────────────────────

    fn forged_header(vocab_size: u32, m: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SXCS");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&vocab_size.to_le_bytes());
        buf.extend_from_slice(&m.to_le_bytes());
        buf
    }

    #[test]
    fn load_rejects_forged_headers() {
        let tmp = tempfile::TempDir::new().unwrap();
        for (name, bytes) in [
            ("overflow", forged_header(u32::MAX, u32::MAX)),
            ("truncated", forged_header(4, 3)), // claims 15 entries, supplies 0
            ("m-zero", forged_header(4, 0)),
            ("m-too-large", forged_header(4, 2000)),
        ] {
            let p = tmp.path().join(name);
            std::fs::write(&p, bytes).unwrap();
            assert!(
                CentroidShortlists::load(&p).is_err(),
                "{name} must be rejected"
            );
        }
        let p = tmp.path().join("bad-magic");
        std::fs::write(&p, b"NOPE").unwrap();
        assert!(CentroidShortlists::load(&p).is_err());
    }
}
