//! Streaming distillation of a [`StaticTokenTable`] from a real (or fake, for
//! tests) contextual document encoder.
//!
//! The table is built in one pass over a corpus, without ever materializing
//! more than `O(vocab_size × dims)` floats for the mean accumulators plus a
//! bounded reservoir sample used to fit the five-position mixing weights.
//! See `docs/superpowers/plans/2026-07-17-ember-plan-a-gate1.md` (Task 3) for
//! the algorithm this implements.

use crate::embedding::colbert::{ColbertEmbedder, TokenEmbeddings};
use crate::embedding::static_table::StaticTokenTable;
use anyhow::Result;

/// Width of the mixing window: two tokens of left context, the center token,
/// two tokens of right context.
const WINDOW_LEN: usize = 5;
/// Index of the center token within a window.
const CENTER_OFFSET: usize = 2;
/// Cap on the number of `(window, true center embedding)` pairs retained for
/// the mix-weight least-squares fit. Bounds memory independent of corpus
/// size; large enough to give the 5×5 normal equations a stable fit.
const RESERVOIR_CAPACITY: usize = 200_000;
/// Fixed seed for the reservoir-sampling PRNG. Distillation is meant to be a
/// reproducible, offline batch job — a fixed seed means re-running `distill`
/// over the same corpus produces byte-identical tables.
const RESERVOIR_SEED: u64 = 0xE813_9B0C_5A70_C13B;
/// Below this pivot magnitude, the 5×5 normal-equation solve is treated as
/// singular (see [`solve_5x5`]).
const SINGULAR_EPS: f64 = 1e-9;
/// Weights the mix-weight fit falls back to when the reservoir's normal
/// equations are singular: pure center-token lookup, ignoring context.
const FALLBACK_MIX_WEIGHTS: [f32; WINDOW_LEN] = [0.0, 0.0, 1.0, 0.0, 0.0];

/// A document encoder that can hand back, for each input text, the token ids
/// aligned 1:1 with its per-token embedding rows, plus its vocabulary size.
///
/// This is the seam that makes [`distill`] hermetically testable: production
/// code uses the [`ColbertEmbedder`] impl below; tests use a fake that
/// fabricates deterministic ids/embeddings with no model or network.
pub trait DocTokenEncoder {
    /// Encode a batch of documents, returning one `(token_ids, embeddings)`
    /// pair per input text with `token_ids.len() == embeddings.nrows()`.
    fn encode_with_ids(&self, texts: &[String]) -> Result<Vec<(Vec<u32>, TokenEmbeddings)>>;

    /// Size of the id space this encoder can emit into `encode_with_ids`'
    /// token ids (every id returned must be `< vocab_size`).
    fn vocab_size(&self) -> usize;
}

impl DocTokenEncoder for ColbertEmbedder {
    fn encode_with_ids(&self, texts: &[String]) -> Result<Vec<(Vec<u32>, TokenEmbeddings)>> {
        self.encode_documents_with_ids(texts)
    }

    /// # Panics
    ///
    /// Panics if the tokenizer/config backing this embedder cannot be
    /// loaded. `vocab_size` is an infallible trait method, but the only way
    /// this can fail is a broken model directory — the exact same failure
    /// [`ColbertEmbedder::encode_with_ids`] would hit on its own first call
    /// via the same underlying `id_alignment()`. There is no scenario where
    /// the model directory is valid at construction and then becomes
    /// invalid mid-process, so a silent fallback (e.g. `0`) would just
    /// convert a real configuration bug into a corrupt/empty static table
    /// with no error anywhere downstream (Task 4's CLI, Task 7's Gate-1
    /// run) — failing loudly here is more honest than that.
    fn vocab_size(&self) -> usize {
        self.tokenizer_vocab_size().unwrap_or_else(|e| {
            panic!("ColbertEmbedder::vocab_size: failed to load tokenizer/config: {e}")
        })
    }
}

/// One retained sample for the mix-weight fit: the five token ids
/// surrounding (and including) a position, and the true contextual
/// embedding the encoder produced at that position.
struct ReservoirItem {
    window_ids: [u32; WINDOW_LEN],
    center_embedding: Vec<f32>,
}

/// Minimal splitmix64 PRNG, used only to draw reservoir-sampling indices.
/// Reservoir sampling needs approximately uniform draws, not cryptographic
/// randomness, so a tiny inline generator avoids pulling in the `rand` crate
/// for this one call site (see `RESERVOIR_SEED` for why the seed is fixed).
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

    /// Uniform-ish integer in `[0, bound)`. `bound` must be > 0. Uses a
    /// modulo reduction rather than Lemire's unbiased method — reservoir
    /// sampling's correctness only needs approximately uniform draws, and
    /// the bias from the modulo is immaterial at our scale.
    fn next_below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// Streaming accumulator state for one [`distill`] run.
struct Accumulator {
    vocab_size: usize,
    /// Embedding width, discovered from the first document that emits at
    /// least one token (see the struct-level note on why this can't be
    /// known up front).
    dims: Option<usize>,
    /// `vocab_size × dims` running sum of embeddings per token id, flat
    /// row-major (`token_id * dims .. token_id * dims + dims`). Allocated
    /// lazily once `dims` is known.
    sums: Option<Vec<f32>>,
    /// Per-token occurrence count.
    counts: Vec<u64>,
    reservoir: Vec<ReservoirItem>,
    /// Total number of token positions seen so far, across all documents —
    /// the "stream index" reservoir sampling (Algorithm R) needs to decide
    /// whether/where a new item displaces an existing reservoir entry.
    seen: u64,
    rng: SplitMix64,
}

impl Accumulator {
    fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            dims: None,
            sums: None,
            counts: vec![0u64; vocab_size],
            reservoir: Vec::new(),
            seen: 0,
            rng: SplitMix64::new(RESERVOIR_SEED),
        }
    }

    fn ingest_batch(&mut self, encoder: &dyn DocTokenEncoder, texts: &[String]) -> Result<()> {
        for (ids, emb) in encoder.encode_with_ids(texts)? {
            self.ingest_document(&ids, &emb)?;
        }
        Ok(())
    }

    fn ingest_document(&mut self, ids: &[u32], emb: &TokenEmbeddings) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        anyhow::ensure!(
            ids.len() == emb.nrows(),
            "distill: token id/embedding row mismatch ({} ids vs {} rows); \
             DocTokenEncoder contract violated",
            ids.len(),
            emb.nrows()
        );

        let ncols = emb.ncols();
        match self.dims {
            None => {
                self.dims = Some(ncols);
                self.sums = Some(vec![0.0f32; self.vocab_size * ncols]);
            }
            Some(d) => anyhow::ensure!(
                d == ncols,
                "distill: inconsistent embedding width across batches ({d} vs {ncols}); \
                 DocTokenEncoder must report a constant width"
            ),
        }
        let dims = self.dims.expect("just set above");

        for i in 0..ids.len() {
            let id = ids[i];
            let idx = id as usize;
            anyhow::ensure!(
                idx < self.vocab_size,
                "distill: token id {id} out of bounds for vocab_size {}",
                self.vocab_size
            );

            let row = emb.row(i);
            {
                let sums = self.sums.as_mut().expect("just set above");
                let base = idx * dims;
                for (s, v) in sums[base..base + dims].iter_mut().zip(row.iter()) {
                    *s += v;
                }
            }
            self.counts[idx] += 1;

            // Five-window centered at i; positions past either edge of this
            // document reuse the center id (per the plan's mixing-weight
            // algorithm) rather than reading into a neighboring document or
            // padding with an out-of-vocab sentinel.
            let mut window_ids = [id; WINDOW_LEN];
            for offset in 1..=CENTER_OFFSET {
                if i >= offset {
                    window_ids[CENTER_OFFSET - offset] = ids[i - offset];
                }
                if i + offset < ids.len() {
                    window_ids[CENTER_OFFSET + offset] = ids[i + offset];
                }
            }
            self.reservoir_sample(ReservoirItem {
                window_ids,
                center_embedding: row.to_vec(),
            });
        }
        Ok(())
    }

    /// Textbook reservoir sampling (Algorithm R): the k-th item (0-indexed)
    /// is kept unconditionally while the reservoir has spare capacity;
    /// afterwards it replaces a uniformly-chosen existing slot with
    /// probability `RESERVOIR_CAPACITY / (k + 1)`, which is what keeps the
    /// final sample uniform over the entire stream regardless of its length.
    fn reservoir_sample(&mut self, item: ReservoirItem) {
        if self.reservoir.len() < RESERVOIR_CAPACITY {
            self.reservoir.push(item);
        } else {
            let j = self.rng.next_below(self.seen + 1);
            if (j as usize) < RESERVOIR_CAPACITY {
                self.reservoir[j as usize] = item;
            }
        }
        self.seen += 1;
    }
}

/// Solve a 5×5 linear system `a·x = b` via partial-pivot Gaussian
/// elimination. Returns `None` if the system is singular (no usable pivot),
/// in which case the caller falls back to [`FALLBACK_MIX_WEIGHTS`].
fn solve_5x5(
    mut a: [[f64; WINDOW_LEN]; WINDOW_LEN],
    mut b: [f64; WINDOW_LEN],
) -> Option<[f32; WINDOW_LEN]> {
    for col in 0..WINDOW_LEN {
        let mut pivot_row = col;
        let mut pivot_val = a[col][col].abs();
        for (row, arow) in a.iter().enumerate().skip(col + 1) {
            if arow[col].abs() > pivot_val {
                pivot_val = arow[col].abs();
                pivot_row = row;
            }
        }
        if pivot_val < SINGULAR_EPS {
            return None;
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
        // The pivot row is read while other rows are mutated in place, so it
        // is copied out first (cheap: `[f64; WINDOW_LEN]` is `Copy`) rather
        // than borrowed — that keeps the borrow checker happy without index
        // gymnastics on `a` itself.
        let pivot_row_vals = a[col];
        let pivot_b = b[col];
        for (row, arow) in a.iter_mut().enumerate().skip(col + 1) {
            let factor = arow[col] / pivot_row_vals[col];
            if factor == 0.0 {
                continue;
            }
            for (k, &pv) in pivot_row_vals.iter().enumerate().skip(col) {
                arow[k] -= factor * pv;
            }
            b[row] -= factor * pivot_b;
        }
    }

    let mut x = [0.0f64; WINDOW_LEN];
    for row in (0..WINDOW_LEN).rev() {
        let mut sum = b[row];
        for (k, &xk) in x.iter().enumerate().skip(row + 1) {
            sum -= a[row][k] * xk;
        }
        x[row] = sum / a[row][row];
    }
    Some(std::array::from_fn(|i| x[i] as f32))
}

/// Fit the five mixing weights minimizing
/// `Σ ||Σ_k w_k · mean_table[window_ids[k]] − center_embedding||²` over the
/// reservoir sample, via the 5×5 least-squares normal equations. Falls back
/// to [`FALLBACK_MIX_WEIGHTS`] (pure center lookup) if the system is
/// singular — e.g. a degenerate corpus where every window is identical.
fn fit_mix_weights(reservoir: &[ReservoirItem], mean_table: &[Vec<f32>]) -> [f32; WINDOW_LEN] {
    let mut a = [[0.0f64; WINDOW_LEN]; WINDOW_LEN];
    let mut b = [0.0f64; WINDOW_LEN];

    // Dot product in f64 (accumulation precision matters for a normal-
    // equations fit); relies on `u`/`v` sharing a length, which holds here
    // because every `mean_table` row and every `center_embedding` is `dims`
    // wide by construction.
    let dot = |u: &[f32], v: &[f32]| -> f64 {
        u.iter()
            .zip(v.iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum()
    };

    for item in reservoir {
        let xs: [&[f32]; WINDOW_LEN] =
            std::array::from_fn(|k| mean_table[item.window_ids[k] as usize].as_slice());
        for k in 0..WINDOW_LEN {
            b[k] += dot(xs[k], &item.center_embedding);
            for l in k..WINDOW_LEN {
                let akl = dot(xs[k], xs[l]);
                a[k][l] += akl;
                if l != k {
                    a[l][k] += akl;
                }
            }
        }
    }

    solve_5x5(a, b).unwrap_or(FALLBACK_MIX_WEIGHTS)
}

/// Distill a [`StaticTokenTable`] from `corpus`, streaming it through
/// `encoder` in chunks of `batch` documents at a time.
///
/// Memory is bounded: `O(vocab_size × dims)` for the running mean
/// accumulators, plus the fixed-size reservoir sample used to fit
/// `mix_weights`. See the module docs for the full algorithm.
///
/// # Errors
///
/// Returns an error if `batch == 0`, if `encoder` reports a `vocab_size` of
/// `0`, if any batch's `encode_with_ids` call fails, if the encoder ever
/// hands back a token id `>= vocab_size` or an inconsistent embedding width
/// across batches, or if `corpus` (or every document in it) produces no
/// tokens at all — an empty corpus must fail loudly rather than silently
/// producing an all-zero table.
pub fn distill(
    encoder: &dyn DocTokenEncoder,
    corpus: impl Iterator<Item = String>,
    batch: usize,
) -> Result<StaticTokenTable> {
    anyhow::ensure!(batch > 0, "distill: `batch` must be greater than zero");
    let vocab_size = encoder.vocab_size();
    anyhow::ensure!(vocab_size > 0, "distill: encoder reports vocab_size = 0");

    let mut acc = Accumulator::new(vocab_size);
    let mut batch_buf: Vec<String> = Vec::with_capacity(batch);
    for text in corpus {
        batch_buf.push(text);
        if batch_buf.len() == batch {
            acc.ingest_batch(encoder, &batch_buf)?;
            batch_buf.clear();
        }
    }
    if !batch_buf.is_empty() {
        acc.ingest_batch(encoder, &batch_buf)?;
    }

    let dims = acc
        .dims
        .ok_or_else(|| anyhow::anyhow!("distill: empty corpus produced no tokens"))?;
    let sums = acc.sums.expect("sums is set alongside dims");

    // Step 3: mean + L2-normalize each observed row.
    let mut mean_table: Vec<Vec<f32>> = vec![vec![0.0; dims]; vocab_size];
    for (token_id, (&count, mean_row)) in acc.counts.iter().zip(mean_table.iter_mut()).enumerate() {
        if count == 0 {
            continue;
        }
        let base = token_id * dims;
        let inv = 1.0 / count as f32;
        let mut row: Vec<f32> = sums[base..base + dims].iter().map(|&v| v * inv).collect();
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut row {
                *v /= norm;
            }
        }
        *mean_row = row;
    }

    // Step 4: fit mix_weights from the reservoir sample.
    let mix_weights = fit_mix_weights(&acc.reservoir, &mean_table);

    // Step 5: materialize the table.
    let mut table = StaticTokenTable::new(vocab_size, dims, mix_weights);
    for (token_id, (&count, mean_row)) in acc.counts.iter().zip(mean_table.iter()).enumerate() {
        if count > 0 {
            table.set_row(token_id as u32, mean_row);
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Deterministic fake encoder for hermetic tests: maps each ASCII
    /// lowercase letter `'a'..='z'` to a fixed token id (`byte - b'a'`),
    /// with a "contextual" embedding that is a one-hot(token) vector plus a
    /// small, fixed-size perturbation ("leak") whose position depends on
    /// the *previous* token in the same document (or on the token itself,
    /// at the start of a document — mirroring the same "reuse the center"
    /// edge convention `distill`'s five-window uses). The leak is
    /// deliberately tiny relative to the one-hot component so a
    /// least-squares fit over five-token windows should assign nearly all
    /// weight to the center position.
    struct FakeEncoder {
        dims: usize,
        vocab_size: usize,
    }

    impl FakeEncoder {
        /// Magnitude of the previous-token-dependent perturbation, relative
        /// to the one-hot component's magnitude of 1.0.
        const LEAK: f32 = 0.05;

        fn new(vocab_size: usize, dims: usize) -> Self {
            Self { dims, vocab_size }
        }

        /// The formula under test: shared by the encoder impl and by tests
        /// that need to independently recompute expected sums.
        fn embed_row(&self, token_id: u32, prev_id: u32) -> Vec<f32> {
            let mut row = vec![0.0f32; self.dims];
            row[token_id as usize % self.dims] += 1.0;
            let leak_pos = (token_id as usize + 1 + prev_id as usize) % self.dims;
            row[leak_pos] += Self::LEAK;
            row
        }
    }

    impl DocTokenEncoder for FakeEncoder {
        fn encode_with_ids(&self, texts: &[String]) -> Result<Vec<(Vec<u32>, TokenEmbeddings)>> {
            Ok(texts
                .iter()
                .map(|text| {
                    let bytes = text.as_bytes();
                    let mut ids: Vec<u32> = Vec::with_capacity(bytes.len());
                    let mut emb = Array2::<f32>::zeros((bytes.len(), self.dims));
                    for (i, &b) in bytes.iter().enumerate() {
                        let id = u32::from(b - b'a');
                        let prev = if i == 0 { id } else { ids[i - 1] };
                        let row = self.embed_row(id, prev);
                        for (d, v) in row.into_iter().enumerate() {
                            emb[[i, d]] = v;
                        }
                        ids.push(id);
                    }
                    (ids, emb)
                })
                .collect())
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    /// (a) + (b): every observed token's table row equals the L2-normalized
    /// mean of its emitted embeddings.
    #[test]
    fn table_rows_are_the_normalized_mean_of_emitted_embeddings() {
        let dims = 4;
        let vocab = 4; // letters a..d
        let encoder = FakeEncoder::new(vocab, dims);
        let corpus = vec!["abcd".to_string(), "dcba".to_string(), "aabb".to_string()];

        // Independently accumulate the expected raw sum/count per token id,
        // using the same embed_row formula the fake encoder itself uses (the
        // system under test is distill()'s aggregation, not this formula).
        let mut sums = vec![vec![0.0f32; dims]; vocab];
        let mut counts = vec![0u64; vocab];
        for text in &corpus {
            let bytes = text.as_bytes();
            let mut ids: Vec<u32> = Vec::new();
            for (i, &b) in bytes.iter().enumerate() {
                let id = u32::from(b - b'a');
                let prev = if i == 0 { id } else { ids[i - 1] };
                let row = encoder.embed_row(id, prev);
                for (s, v) in sums[id as usize].iter_mut().zip(row.iter()) {
                    *s += v;
                }
                counts[id as usize] += 1;
                ids.push(id);
            }
        }

        let table = distill(&encoder, corpus.into_iter(), 2).expect("non-empty corpus distills");

        for token_id in 0..vocab as u32 {
            let count = counts[token_id as usize];
            assert!(count > 0, "test corpus must exercise every token id");
            let mean: Vec<f32> = sums[token_id as usize]
                .iter()
                .map(|&s| s / count as f32)
                .collect();
            let norm = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
            let expected: Vec<f32> = mean.iter().map(|&v| v / norm).collect();

            let row = table
                .lookup(token_id)
                .unwrap_or_else(|| panic!("token {token_id} missing from table"));
            for (got, want) in row.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < 1e-4,
                    "token {token_id}: got {row:?}, want {expected:?}"
                );
            }
        }
    }

    #[test]
    fn table_rows_are_l2_normalized() {
        let dims = 4;
        let vocab = 4;
        let encoder = FakeEncoder::new(vocab, dims);
        let corpus = vec!["abcd".to_string(), "dcba".to_string(), "aabb".to_string()];

        let table = distill(&encoder, corpus.into_iter(), 2).expect("non-empty corpus distills");

        for token_id in 0..vocab as u32 {
            let row = table
                .lookup(token_id)
                .unwrap_or_else(|| panic!("token {token_id} missing from table"));
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "token {token_id} row norm {norm} is not ~1.0 ({row:?})"
            );
        }
    }

    /// (c) The fake's context effect (`LEAK = 0.05`) is small relative to
    /// its one-hot identity component, so the least-squares fit over
    /// five-token windows should put nearly all weight on the center
    /// position.
    #[test]
    fn center_mix_weight_dominates_when_context_effect_is_small() {
        let dims = 6;
        let vocab = 5; // letters a..e
        let encoder = FakeEncoder::new(vocab, dims);

        // Repeat a shifted rotation of the alphabet many times so a wide
        // variety of (token, prev-token) contexts occur and the reservoir
        // fit has plenty of five-token windows to work with.
        let base = "abcdeabcdeabcdeabcdeabcde";
        let corpus: Vec<String> = (0..50)
            .map(|i| {
                let start = i % base.len();
                format!("{}{}", &base[start..], &base[..start])
            })
            .collect();

        let table = distill(&encoder, corpus.into_iter(), 8).expect("non-empty corpus distills");
        let w = table.mix_weights;
        assert_ne!(
            w, FALLBACK_MIX_WEIGHTS,
            "weights exactly match the singular-system fallback; the reservoir fit \
             likely didn't run (this corpus should produce a well-conditioned system)"
        );
        for (k, &wk) in w.iter().enumerate() {
            if k != 2 {
                assert!(
                    w[2] > wk,
                    "center weight w[2]={} should dominate w[{k}]={wk}; full weights: {w:?}",
                    w[2]
                );
            }
        }
    }

    /// When every reservoir sample has an identical five-window (a corpus
    /// with only one distinct token), the normal-equations matrix is
    /// exactly rank-1 (every row is a scalar multiple of every other), so
    /// the fit must hit the singular branch and fall back to pure
    /// center-token lookup.
    #[test]
    fn singular_reservoir_falls_back_to_pure_center_lookup() {
        let encoder = FakeEncoder::new(4, 4);
        let corpus: Vec<String> = vec!["a".repeat(20); 5];

        let table = distill(&encoder, corpus.into_iter(), 8).expect("non-empty corpus distills");
        assert_eq!(
            table.mix_weights, FALLBACK_MIX_WEIGHTS,
            "a single-token corpus produces a rank-1 normal matrix and must fall back exactly"
        );
    }

    /// (d) An empty corpus must error, not silently produce a zero table.
    #[test]
    fn empty_corpus_errors_instead_of_producing_a_zero_table() {
        let encoder = FakeEncoder::new(4, 4);
        let corpus: Vec<String> = Vec::new();
        let err = distill(&encoder, corpus.into_iter(), 8).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("empty"),
            "expected an 'empty corpus' error, got: {err}"
        );
    }

    /// A non-empty corpus whose every document tokenizes to zero ids (all
    /// empty strings) must fail the same way as a literally empty corpus —
    /// `dims` is never discovered either way.
    #[test]
    fn corpus_of_only_empty_documents_errors_like_an_empty_corpus() {
        let encoder = FakeEncoder::new(4, 4);
        let corpus = vec![String::new(), String::new()];
        let err = distill(&encoder, corpus.into_iter(), 8).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("empty"),
            "expected an 'empty corpus' error, got: {err}"
        );
    }

    /// A `DocTokenEncoder` that violates the ids/rows alignment contract
    /// must be rejected with an error, not panic via an out-of-bounds
    /// index or silently corrupt the table.
    #[test]
    fn mismatched_ids_and_embedding_rows_errors() {
        struct BrokenEncoder;
        impl DocTokenEncoder for BrokenEncoder {
            fn encode_with_ids(
                &self,
                texts: &[String],
            ) -> Result<Vec<(Vec<u32>, TokenEmbeddings)>> {
                Ok(texts
                    .iter()
                    .map(|_| (vec![0, 1, 2], Array2::<f32>::zeros((2, 4))))
                    .collect())
            }
            fn vocab_size(&self) -> usize {
                4
            }
        }

        let err = distill(&BrokenEncoder, vec!["x".to_string()].into_iter(), 8).unwrap_err();
        assert!(
            err.to_string().contains("mismatch"),
            "expected a row/id mismatch error, got: {err}"
        );
    }

    /// A `DocTokenEncoder` that emits a token id outside its declared
    /// `vocab_size` must be rejected rather than panicking on an
    /// out-of-bounds accumulator write.
    #[test]
    fn out_of_bounds_token_id_errors() {
        struct BrokenEncoder;
        impl DocTokenEncoder for BrokenEncoder {
            fn encode_with_ids(
                &self,
                texts: &[String],
            ) -> Result<Vec<(Vec<u32>, TokenEmbeddings)>> {
                Ok(texts
                    .iter()
                    .map(|_| (vec![99], Array2::<f32>::zeros((1, 4))))
                    .collect())
            }
            fn vocab_size(&self) -> usize {
                4
            }
        }

        let err = distill(&BrokenEncoder, vec!["x".to_string()].into_iter(), 8).unwrap_err();
        assert!(
            err.to_string().contains("out of bounds"),
            "expected an out-of-bounds token id error, got: {err}"
        );
    }
}
