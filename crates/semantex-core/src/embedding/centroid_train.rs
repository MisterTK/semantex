//! Offline training of frozen universal PLAID centroids (Ember Plan B).
//!
//! Streams corpus chunks through a [`DocTokenEncoder`], reservoir-samples
//! token embeddings to a fixed memory bound, then runs next-plaid's k-means
//! (`compute_centroids`) over the sample. The result ships as a model-dir
//! artifact (`frozen_centroids.npy`) so per-repo indexing can skip k-means
//! entirely — see the Ember design spec, "Frozen universal centroids".

use crate::embedding::static_distill::DocTokenEncoder;
use anyhow::{Context, Result};
use ndarray::Array2;
use std::path::Path;

/// Fixed seed for the token-embedding reservoir; training is an offline,
/// reproducible batch job (same rationale as static_distill::RESERVOIR_SEED).
const RESERVOIR_SEED: u64 = 0x51D3_C0DE_F02E_A7B1;
/// Fixed seed passed to next-plaid's k-means.
const KMEANS_SEED: u64 = 42;

/// Minimal splitmix64 PRNG, used only to draw reservoir-sampling indices.
/// Copied from `static_distill::SplitMix64` (private to that module) —
/// reservoir sampling needs approximately uniform draws, not cryptographic
/// randomness, so a tiny inline generator avoids pulling in the `rand` crate
/// for this one call site.
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

pub struct CentroidTrainOptions {
    /// Number of centroids to train (clamped to the sample size).
    pub num_centroids: usize,
    /// Max token embeddings retained for k-means (bounds memory:
    /// capacity × dims × 4 bytes; 1M × 48d ≈ 192 MB).
    pub sample_capacity: usize,
    /// Documents per encoder batch.
    pub batch: usize,
}

impl Default for CentroidTrainOptions {
    fn default() -> Self {
        Self {
            num_centroids: 8192,
            sample_capacity: 1_000_000,
            batch: 32,
        }
    }
}

/// Streaming reservoir-sample accumulator for token embeddings, mirroring
/// `static_distill::Accumulator`'s shape: one `ingest` call per encoded
/// batch, bounded memory regardless of corpus size.
struct Accumulator {
    sample_capacity: usize,
    reservoir: Vec<Vec<f32>>,
    seen: u64,
    rng: SplitMix64,
    dims: Option<usize>,
}

impl Accumulator {
    fn new(sample_capacity: usize) -> Self {
        Self {
            sample_capacity,
            reservoir: Vec::new(),
            seen: 0,
            rng: SplitMix64::new(RESERVOIR_SEED),
            dims: None,
        }
    }

    fn ingest_batch(&mut self, encoder: &dyn DocTokenEncoder, texts: &[String]) -> Result<()> {
        for (_ids, emb) in encoder.encode_with_ids(texts)? {
            self.ingest_document(&emb)?;
        }
        Ok(())
    }

    fn ingest_document(&mut self, emb: &crate::embedding::colbert::TokenEmbeddings) -> Result<()> {
        if emb.nrows() == 0 {
            return Ok(());
        }
        let ncols = emb.ncols();
        match self.dims {
            None => self.dims = Some(ncols),
            Some(d) => anyhow::ensure!(
                d == ncols,
                "train_centroids: inconsistent embedding width ({d} vs {ncols})"
            ),
        }

        for row in emb.rows() {
            self.reservoir_sample(row.to_vec());
        }
        Ok(())
    }

    /// Textbook reservoir sampling (Algorithm R): see
    /// `static_distill::Accumulator::reservoir_sample` for the invariant.
    fn reservoir_sample(&mut self, item: Vec<f32>) {
        if self.reservoir.len() < self.sample_capacity {
            self.reservoir.push(item);
        } else {
            let j = self.rng.next_below(self.seen + 1);
            if (j as usize) < self.sample_capacity {
                self.reservoir[j as usize] = item;
            }
        }
        self.seen += 1;
    }
}

/// Train frozen universal centroids from `corpus`, streaming it through
/// `encoder` in chunks of `opts.batch` documents at a time.
///
/// Token embeddings are reservoir-sampled down to `opts.sample_capacity`
/// before k-means, bounding memory independent of corpus size. `k` is
/// clamped to the sample size, so a tiny corpus with a large requested
/// `num_centroids` clamps rather than errors.
///
/// # Errors
///
/// Returns an error if `opts.batch`, `opts.num_centroids`, or
/// `opts.sample_capacity` is `0`, if any batch's `encode_with_ids` call
/// fails, if the encoder hands back an inconsistent embedding width across
/// batches, if `corpus` produces no tokens at all, or if next-plaid's
/// k-means fails.
pub fn train_centroids(
    encoder: &dyn DocTokenEncoder,
    corpus: impl Iterator<Item = String>,
    opts: &CentroidTrainOptions,
) -> Result<Array2<f32>> {
    anyhow::ensure!(opts.batch > 0, "train_centroids: `batch` must be > 0");
    anyhow::ensure!(
        opts.num_centroids > 0,
        "train_centroids: `num_centroids` must be > 0"
    );
    anyhow::ensure!(
        opts.sample_capacity > 0,
        "train_centroids: `sample_capacity` must be > 0"
    );

    let mut acc = Accumulator::new(opts.sample_capacity);
    let mut batch_buf: Vec<String> = Vec::with_capacity(opts.batch);
    for text in corpus {
        batch_buf.push(text);
        if batch_buf.len() == opts.batch {
            acc.ingest_batch(encoder, &batch_buf)?;
            batch_buf.clear();
        }
    }
    if !batch_buf.is_empty() {
        acc.ingest_batch(encoder, &batch_buf)?;
    }

    let dims = acc
        .dims
        .ok_or_else(|| anyhow::anyhow!("train_centroids: empty corpus produced no tokens"))?;

    // Flatten the sample into one [n, dims] matrix for k-means.
    let n = acc.reservoir.len();
    let mut flat = Array2::<f32>::zeros((n, dims));
    for (i, row) in acc.reservoir.iter().enumerate() {
        flat.row_mut(i)
            .assign(&ndarray::ArrayView1::from(row.as_slice()));
    }
    drop(acc.reservoir);

    let k = opts.num_centroids.min(n);
    if k < opts.num_centroids {
        tracing::warn!(
            requested = opts.num_centroids,
            sample = n,
            "train_centroids: clamping k to sample size"
        );
    }
    let mut kmeans_config = next_plaid::kmeans::default_config(k);
    kmeans_config.seed = KMEANS_SEED;
    let centroids = next_plaid::compute_centroids(&flat.view(), k, Some(kmeans_config), true)
        .map_err(|e| anyhow::anyhow!("k-means over {n} sampled token embeddings failed: {e}"))?;
    Ok(centroids)
}

/// Write `centroids` to `path` as a `.npy` file, atomically: the array is
/// serialized to a temp file in the same directory, then renamed into place
/// (mirrors `write_mapping_atomic` in `colbert_plaid_backend.rs`), so a
/// concurrent reader never observes a partially-written file.
pub fn save_centroids_npy(path: &Path, centroids: &Array2<f32>) -> Result<()> {
    use ndarray_npy::WriteNpyExt;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("centroids"),
        std::process::id()
    ));
    let file = std::fs::File::create_new(&tmp).with_context(|| {
        format!(
            "failed to create temp file {} (leftover from a crashed run? remove it and retry)",
            tmp.display()
        )
    })?;
    if let Err(e) = centroids.write_npy(file) {
        let _ = std::fs::remove_file(&tmp);
        return Err(anyhow::Error::new(e).context("failed to write centroids npy"));
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        anyhow::Error::new(e).context(format!(
            "failed to move centroids into place at {}",
            path.display()
        ))
    })?;
    Ok(())
}

/// Read a `.npy` file previously written by [`save_centroids_npy`] back into
/// an `Array2<f32>`. Kept alongside `save_centroids_npy` so Task 4's CLI
/// `--verify` flag can round-trip without pulling its own `ndarray` deps.
pub fn load_centroids_npy(path: &Path) -> Result<Array2<f32>> {
    use ndarray_npy::ReadNpyExt;
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open centroids file {}", path.display()))?;
    Array2::<f32>::read_npy(file)
        .with_context(|| format!("failed to read centroids npy at {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::colbert::TokenEmbeddings;
    use crate::embedding::static_distill::DocTokenEncoder;
    use anyhow::Result as AnyResult;
    use ndarray::Array2 as NdArray2;

    /// Minimal deterministic fake: token id = byte - b'a'; embedding =
    /// one-hot(id) in `dims`. Enough structure for k-means to find obvious
    /// clusters.
    struct FakeEncoder {
        dims: usize,
        vocab_size: usize,
    }

    impl DocTokenEncoder for FakeEncoder {
        fn encode_with_ids(&self, texts: &[String]) -> AnyResult<Vec<(Vec<u32>, TokenEmbeddings)>> {
            Ok(texts
                .iter()
                .map(|text| {
                    let bytes = text.as_bytes();
                    let mut emb = NdArray2::<f32>::zeros((bytes.len(), self.dims));
                    let ids: Vec<u32> = bytes
                        .iter()
                        .enumerate()
                        .map(|(i, &b)| {
                            let id = u32::from(b - b'a');
                            emb[[i, id as usize % self.dims]] = 1.0;
                            id
                        })
                        .collect();
                    (ids, emb)
                })
                .collect())
        }
        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    #[test]
    fn trains_requested_number_of_centroids_with_right_dim() {
        let encoder = FakeEncoder {
            dims: 4,
            vocab_size: 4,
        };
        let corpus: Vec<String> = (0..50).map(|_| "abcdabcdabcd".to_string()).collect();
        let opts = CentroidTrainOptions {
            num_centroids: 4,
            sample_capacity: 10_000,
            batch: 8,
        };
        let c = train_centroids(&encoder, corpus.into_iter(), &opts).unwrap();
        assert_eq!(c.ncols(), 4);
        assert_eq!(c.nrows(), 4);
    }

    #[test]
    fn clamps_k_to_sample_size() {
        // 1 doc × 4 tokens but k=64 → must clamp rather than error.
        let encoder = FakeEncoder {
            dims: 4,
            vocab_size: 4,
        };
        let corpus = vec!["abcd".to_string()];
        let opts = CentroidTrainOptions {
            num_centroids: 64,
            sample_capacity: 10_000,
            batch: 8,
        };
        let c = train_centroids(&encoder, corpus.into_iter(), &opts).unwrap();
        assert!(
            c.nrows() >= 1 && c.nrows() <= 4,
            "got {} centroids",
            c.nrows()
        );
        assert_eq!(c.ncols(), 4);
    }

    #[test]
    fn empty_corpus_errors() {
        let encoder = FakeEncoder {
            dims: 4,
            vocab_size: 4,
        };
        let err = train_centroids(
            &encoder,
            Vec::<String>::new().into_iter(),
            &CentroidTrainOptions::default(),
        )
        .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("empty"),
            "got: {err}"
        );
    }

    #[test]
    fn save_is_atomic_and_readable_back() {
        use ndarray_npy::ReadNpyExt;
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("frozen_centroids.npy");
        let c = NdArray2::<f32>::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f32);
        save_centroids_npy(&path, &c).unwrap();
        let back: NdArray2<f32> = NdArray2::read_npy(std::fs::File::open(&path).unwrap()).unwrap();
        assert_eq!(back, c);
        // No stray temp files left behind.
        let leftovers: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path() != path)
            .collect();
        assert!(leftovers.is_empty(), "temp files left: {leftovers:?}");

        // Round-trip via the load helper too (Task 4's CLI --verify seam).
        let loaded = load_centroids_npy(&path).unwrap();
        assert_eq!(loaded, c);
    }

    #[test]
    fn training_is_deterministic() {
        let encoder = FakeEncoder {
            dims: 4,
            vocab_size: 4,
        };
        let corpus: Vec<String> = (0..30).map(|i| "abcd".repeat(1 + i % 3)).collect();
        let opts = CentroidTrainOptions {
            num_centroids: 3,
            sample_capacity: 1_000,
            batch: 8,
        };
        let a = train_centroids(&encoder, corpus.clone().into_iter(), &opts).unwrap();
        let b = train_centroids(&encoder, corpus.into_iter(), &opts).unwrap();
        assert_eq!(a, b, "fixed seeds must make training reproducible");
    }
}
