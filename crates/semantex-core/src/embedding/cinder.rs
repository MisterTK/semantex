//! Cinder compiled-indexing document encoder (spec §4.1, plan Task 6).
//!
//! Composes the three Cinder artifacts — the Ember Plan A
//! [`StaticTokenTable`] (Task 1), the distilled [`MicroMixer`] (Tasks 1/2),
//! and the per-vocab [`CentroidShortlists`] (Task 3) — plus the shared
//! tokenization/id-alignment plumbing ([`build_doc_token_ids`] /
//! [`load_doc_id_alignment`]) into a document encoder that produces per-token
//! UNIT-NORM embeddings with NO neural-network evaluation: every position is a
//! handful of table lookups fed through the tiny depthwise-mix + GELU + linear
//! [`MicroMixer::forward`] (the only floating-point compute in the path).
//!
//! # Assignment path
//!
//! [`CinderEncoder::encode_documents`] produces plain per-token embeddings (the
//! generic embedder contract). [`CinderEncoder::encode_documents_with_window_ids`]
//! additionally returns, per token, the 9-tap window ids the mix consumed — the
//! `(embedding, window)` inputs a shortlist-union argmax needs. Cinder's build
//! path (`ColbertPlaidIndexBuilder::build_cinder`) feeds those to
//! [`next_plaid::CompiledIndexWriter::add_document_with_ids`] with an
//! [`next_plaid::IdAwareCodeAssigner`] that calls
//! [`crate::embedding::shortlists::shortlist_argmax`] over the loaded
//! [`CentroidShortlists`] — replacing the writer's default exhaustive
//! per-centroid scan with the O(~m·|window|) shortlist union. Residual
//! computation and quantization are untouched, so the on-disk layout stays
//! byte-compatible with the reference PLAID format. [`CinderEncoder::shortlists`]
//! exposes the loaded shortlists to that build path.

use crate::embedding::colbert::{
    DocIdAlignment, TokenEmbeddings, build_doc_token_ids, load_doc_id_alignment,
};
use crate::embedding::mixer::{MIXER_CENTER, MIXER_WINDOW, MicroMixer};
use crate::embedding::model_manager;
use crate::embedding::shortlists::CentroidShortlists;
use crate::embedding::static_table::StaticTokenTable;
use crate::embedding::static_token::window_ids_at;
use anyhow::{Context, Result};
use ndarray::Array2;
use rayon::prelude::*;
use std::path::Path;

/// Encoder-free document embedder for Cinder's compiled index path.
///
/// See the module docs for the composition and v1 scope. Construct with
/// [`CinderEncoder::new`]; produce per-document embeddings with
/// [`CinderEncoder::encode_documents`].
pub struct CinderEncoder {
    /// Static per-token embedding table (Ember Plan A). Read on every token.
    table: StaticTokenTable,
    /// Distilled contextualization operator. Read on every token.
    mixer: MicroMixer,
    /// Per-vocab centroid shortlists (SXCS). Loaded/validated here; consumed at
    /// build time by the shortlist-union assigner (see module docs), exposed via
    /// [`Self::shortlists`].
    shortlists: CentroidShortlists,
    /// Tokenization + id-alignment (shared with `StaticTokenEmbedder` /
    /// `ColbertEmbedder`), so Cinder's input ids can never drift from the ids
    /// the table/mixer were calibrated against.
    alignment: DocIdAlignment,
}

impl CinderEncoder {
    /// Load all Cinder artifacts from `model_dir`.
    ///
    /// # Errors
    ///
    /// Errors — naming the specific artifact — if ANY of the three
    /// Cinder-specific artifacts is missing or corrupt: the static token table
    /// (`static_token_table.bin`), the mixer (`cinder_mixer.bin`), or the
    /// centroid shortlists (`cinder_shortlists.bin`); or if the tokenizer/config
    /// backing the id-alignment can't be loaded; or if the mixer's dimensionality
    /// disagrees with the table's. The caller (`cinder_for_build`) decides the
    /// fallback — this constructor just fails cleanly.
    pub fn new(model_dir: &Path) -> Result<Self> {
        let table_path = model_manager::static_token_table_path(model_dir);
        let table = StaticTokenTable::load(&table_path).with_context(|| {
            format!(
                "failed to load Cinder static token table {}",
                table_path.display()
            )
        })?;

        let mixer_path = model_manager::cinder_mixer_path(model_dir);
        let mixer = MicroMixer::load(&mixer_path)
            .with_context(|| format!("failed to load Cinder mixer {}", mixer_path.display()))?;
        anyhow::ensure!(
            mixer.dims == table.dims,
            "Cinder mixer dims {} != static token table dims {}",
            mixer.dims,
            table.dims
        );

        let shortlists_path = model_manager::cinder_shortlists_path(model_dir);
        let shortlists = CentroidShortlists::load(&shortlists_path).with_context(|| {
            format!(
                "failed to load Cinder centroid shortlists {}",
                shortlists_path.display()
            )
        })?;

        let alignment = load_doc_id_alignment(model_dir)?;

        Ok(Self {
            table,
            mixer,
            shortlists,
            alignment,
        })
    }

    /// The loaded centroid shortlists (SXCS). Consumed at build time by the
    /// shortlist-union assigner (see module docs) — the build path clones these
    /// into the [`next_plaid::IdAwareCodeAssigner`] closure — and by tests that
    /// assert the artifact loaded correctly.
    #[must_use]
    pub fn shortlists(&self) -> &CentroidShortlists {
        &self.shortlists
    }

    /// Encode documents into per-token embeddings, one `Array2<f32>` per
    /// document — same output type/shape contract as
    /// [`crate::embedding::colbert::ColbertEmbedder::encode_documents`] and
    /// [`crate::embedding::static_token::StaticTokenEmbedder::encode_documents`].
    /// Every row is L2-normalized (or all-zero when the whole window missed the
    /// table, which MaxSim ignores downstream).
    ///
    /// Documents are encoded in PARALLEL (`rayon::par_iter`): each document's
    /// encoding is fully independent — it only reads the shared, immutable
    /// table/mixer/alignment (`CinderEncoder` is `Sync`) and owns its scratch
    /// buffer — so parallelizing is byte-identical to the serial map. Collecting
    /// an indexed parallel iterator preserves order, so row `i` of the output is
    /// still document `i`.
    pub fn encode_documents(&self, texts: &[String]) -> Result<Vec<TokenEmbeddings>> {
        texts
            .par_iter()
            .map(|text| {
                let ids = build_doc_token_ids(&self.alignment, text)?;
                Ok(self.encode_ids(&ids))
            })
            .collect()
    }

    /// Like [`Self::encode_documents`], but ALSO returns, per document, the
    /// 9-tap window ids used at each token position (one `Vec<u32>` per row, in
    /// the SAME row order as the embeddings). These are exactly the
    /// `(embedding, window)` inputs the shortlist-union argmax consumes at build
    /// time (see [`crate::embedding::shortlists::shortlist_argmax`]), so Cinder's
    /// compiled build path can hand them straight to
    /// [`next_plaid::CompiledIndexWriter::add_document_with_ids`] alongside the
    /// embeddings — no re-tokenization, no drift from the encode path.
    ///
    /// `window_ids.len() == embeddings.nrows()` holds for every returned
    /// document (both are one entry per token position).
    ///
    /// Documents are encoded in PARALLEL (`rayon::par_iter`), the same way (and
    /// for the same reason) as [`Self::encode_documents`]: per-document encoding
    /// is independent and reads only the shared immutable artifacts, so the
    /// result is byte-identical to the serial map with document order preserved.
    /// This is Cinder's build-path hot loop (`build_cinder` feeds the output to
    /// `add_document_with_ids`), so preserving order keeps each `(embedding,
    /// window)` pair matched to its document.
    pub fn encode_documents_with_window_ids(
        &self,
        texts: &[String],
    ) -> Result<Vec<(TokenEmbeddings, Vec<Vec<u32>>)>> {
        texts
            .par_iter()
            .map(|text| {
                let ids = build_doc_token_ids(&self.alignment, text)?;
                Ok(self.encode_ids_with_window_ids(&ids))
            })
            .collect()
    }

    /// Contextualize one document's token-id sequence into an `[n_tokens, dims]`
    /// matrix, replicating `mixer_train`'s STUDENT-input convention EXACTLY:
    /// the 9-tap window (same edge-replication as training via
    /// [`window_ids_at`]), each window row from `table.lookup` (a miss →
    /// all-zero row, matching `gather_student_inputs`), the center row taken
    /// from window slot [`MIXER_CENTER`], all fed through [`MicroMixer::forward`].
    fn encode_ids(&self, ids: &[u32]) -> TokenEmbeddings {
        let dims = self.table.dims;
        let mut out = Array2::<f32>::zeros((ids.len(), dims));
        let mut e = vec![0.0f32; dims];
        for i in 0..ids.len() {
            self.mix_at(ids, i, &mut e);
            out.row_mut(i)
                .assign(&ndarray::ArrayView1::from(e.as_slice()));
        }
        out
    }

    /// Like [`Self::encode_ids`] but also collects, per token position, the
    /// 9-tap window ids [`Self::mix_at`] used. Routes through the SAME `mix_at`
    /// (the single source of truth for Cinder's windowing/gather/mix), so the
    /// embeddings it returns are identical to `encode_ids`' and the window ids
    /// can never drift from the ones the mixing actually consumed.
    fn encode_ids_with_window_ids(&self, ids: &[u32]) -> (TokenEmbeddings, Vec<Vec<u32>>) {
        let dims = self.table.dims;
        let mut out = Array2::<f32>::zeros((ids.len(), dims));
        let mut windows: Vec<Vec<u32>> = Vec::with_capacity(ids.len());
        let mut e = vec![0.0f32; dims];
        for i in 0..ids.len() {
            let window_ids = self.mix_at(ids, i, &mut e);
            out.row_mut(i)
                .assign(&ndarray::ArrayView1::from(e.as_slice()));
            windows.push(window_ids.to_vec());
        }
        (out, windows)
    }

    /// Contextualize token position `i` of `ids` into `e` (must already be
    /// length `self.table.dims`; fully overwritten), returning the 9-tap window
    /// ids used. This is the single source of truth for Cinder's per-position
    /// mixing — [`Self::encode_ids`] and [`Self::agreement_samples`] both route
    /// through it so the encode path and the C4 agreement measurement can never
    /// drift apart.
    fn mix_at(&self, ids: &[u32], i: usize, e: &mut [f32]) -> [u32; MIXER_WINDOW] {
        let dims = self.table.dims;
        let window_ids = window_ids_at::<MIXER_WINDOW>(ids, i);
        // Gather window rows (owned) so `forward` can borrow them as slices;
        // a table miss contributes an all-zero row, exactly as
        // `mixer_train::gather_student_inputs` does.
        let window_rows: Vec<Vec<f32>> = window_ids
            .iter()
            .map(|&id| {
                self.table
                    .lookup(id)
                    .map_or_else(|| vec![0.0f32; dims], <[f32]>::to_vec)
            })
            .collect();
        let window_refs: [&[f32]; MIXER_WINDOW] =
            std::array::from_fn(|k| window_rows[k].as_slice());
        let center_row = window_rows[MIXER_CENTER].as_slice();
        self.mixer.forward(&window_refs, center_row, e);
        window_ids
    }

    /// Reservoir-sample up to `max_samples` `(mixed_embedding, window_ids)`
    /// pairs across EVERY token position of EVERY document in `texts`, for gate
    /// C4's shortlist-agreement measurement
    /// ([`crate::embedding::shortlists::shortlist_agreement`]). Each pair is
    /// `(e_i, window_ids_at(ids, i))` — exactly the `(embedding, window)` inputs
    /// the deferred shortlist-union argmax would consume at build time (see the
    /// module docs' v1-scope note), so the measured agreement reflects real
    /// build-time inputs, not a proxy.
    ///
    /// Uses Algorithm-R reservoir sampling seeded by `seed`, so the result is a
    /// uniform draw over the whole corpus (not just its first N tokens) and is
    /// fully deterministic given the same corpus iteration order and seed.
    pub fn agreement_samples(
        &self,
        texts: impl IntoIterator<Item = String>,
        max_samples: usize,
        seed: u64,
    ) -> Result<Vec<(Vec<f32>, Vec<u32>)>> {
        if max_samples == 0 {
            return Ok(Vec::new());
        }
        let dims = self.table.dims;
        let mut reservoir: Vec<(Vec<f32>, Vec<u32>)> = Vec::new();
        let mut n_seen: u64 = 0;
        let mut rng = SplitMix64::new(seed);
        let mut e = vec![0.0f32; dims];
        for text in texts {
            let ids = build_doc_token_ids(&self.alignment, &text)?;
            for i in 0..ids.len() {
                let window_ids = self.mix_at(&ids, i, &mut e);
                if reservoir.len() < max_samples {
                    reservoir.push((e.clone(), window_ids.to_vec()));
                } else {
                    // Algorithm R: the (n_seen+1)-th item replaces a
                    // uniformly-chosen reservoir slot with probability
                    // max_samples/(n_seen+1).
                    let j = rng.next_below(n_seen + 1) as usize;
                    if j < max_samples {
                        reservoir[j] = (e.clone(), window_ids.to_vec());
                    }
                }
                n_seen += 1;
            }
        }
        Ok(reservoir)
    }
}

/// Minimal splitmix64 PRNG for the deterministic reservoir sample in
/// [`CinderEncoder::agreement_samples`]. Mirrors the same tiny generator
/// already copied into `mixer_train` / `centroid_train` / `shortlists` tests
/// (each private to its own module); no cryptographic randomness is needed —
/// just a reproducible stream for uniform reservoir replacement.
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

    /// Uniform in `0..bound` (bound clamped to ≥1 so the modulo is defined).
    fn next_below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::centroid_train::save_centroids_npy;
    use crate::embedding::colbert::ColbertEmbedder;
    use crate::embedding::static_token::StaticTokenEmbedder;
    use ndarray::Array2;

    /// Locate the local LateOn-Code-edge tokenizer files, if downloaded.
    /// Mirrors `static_token.rs`'s `test_tokenizer_dir` gating: skip (not fail)
    /// the tests that need a real tokenizer so CI without the model still runs
    /// the pure-arithmetic tests.
    fn test_tokenizer_dir() -> Option<std::path::PathBuf> {
        let dir = crate::config::SemantexConfig::default()
            .models_dir()
            .join("LateOn-Code-edge");
        (dir.join("tokenizer.json").exists() && dir.join("onnx_config.json").exists())
            .then_some(dir)
    }

    /// `CinderEncoder` (like `StaticTokenEmbedder`) does not derive `Debug`
    /// (its `DocIdAlignment` field can't), so `Result::unwrap_err` — which needs
    /// the `Ok` side to be `Debug` — can't be used; extract the error by hand.
    fn expect_err(result: Result<CinderEncoder>) -> anyhow::Error {
        match result {
            Ok(_) => panic!("expected an error, got Ok"),
            Err(e) => e,
        }
    }

    // ── (b) constructor error path: a missing artifact is named ──────────────

    #[test]
    fn new_errors_when_table_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let err = expect_err(CinderEncoder::new(tmp.path()));
        assert!(
            err.to_string().contains("static_token_table.bin"),
            "expected a static-token-table error, got: {err}"
        );
    }

    #[test]
    fn new_errors_naming_the_missing_mixer() {
        // Table present + mixer absent → the error must name `cinder_mixer.bin`
        // (this is the exact artifact `cinder_for_build`'s warn surfaces on the
        // flag-on-but-mixer-missing fallback).
        let tmp = tempfile::TempDir::new().unwrap();
        let table = StaticTokenTable::new(4, 2, [0.0, 0.0, 1.0, 0.0, 0.0]);
        table
            .save(&model_manager::static_token_table_path(tmp.path()))
            .unwrap();
        let err = expect_err(CinderEncoder::new(tmp.path()));
        assert!(
            err.to_string().contains("cinder_mixer.bin"),
            "expected a mixer-loading error naming cinder_mixer.bin, got: {err}"
        );
    }

    #[test]
    fn new_errors_naming_the_missing_shortlists() {
        // Table + mixer present, shortlists absent → error names the shortlists.
        let tmp = tempfile::TempDir::new().unwrap();
        let table = StaticTokenTable::new(4, 2, [0.0, 0.0, 1.0, 0.0, 0.0]);
        table
            .save(&model_manager::static_token_table_path(tmp.path()))
            .unwrap();
        MicroMixer::zeros(2)
            .save(&model_manager::cinder_mixer_path(tmp.path()))
            .unwrap();
        let err = expect_err(CinderEncoder::new(tmp.path()));
        assert!(
            err.to_string().contains("cinder_shortlists.bin"),
            "expected a shortlists-loading error naming cinder_shortlists.bin, got: {err}"
        );
    }

    // ── (c) end-to-end equivalence: a ZERO mixer ≡ pure-center lookup ────────

    /// With an all-zero [`MicroMixer`], `e_i = L2norm(center + Wp·GELU(b1) + b2)
    /// = L2norm(center)` — pure normalized center-token lookup. That is exactly
    /// what `StaticTokenEmbedder` produces with `mix_weights = [0,0,1,0,0]`
    /// (only the center weight nonzero). This gated end-to-end test builds both
    /// over the SAME table + real tokenizer and asserts row-for-row equality —
    /// a strong cross-implementation consistency check that Cinder's windowing /
    /// gather / normalization line up with the established Ember Plan A path.
    ///
    /// Every token id in the doc is given a distinct nonzero table row so no
    /// center ever misses (the two encoders' OOV-center fallbacks differ:
    /// Cinder → all-zero, `StaticTokenEmbedder` → neighbor mean), isolating the
    /// present-center equivalence the identity mixer is meant to prove.
    #[test]
    fn zero_mixer_equals_static_center_lookup_end_to_end() {
        let Some(model_dir) = test_tokenizer_dir() else {
            return;
        };
        let vocab_size = ColbertEmbedder::new(&model_dir)
            .unwrap()
            .tokenizer_vocab_size()
            .unwrap();
        let dims = 4usize;

        // Table: mix_weights = [0,0,1,0,0] (center-only) so StaticTokenEmbedder
        // reduces to normalized center lookup; every id gets a distinct nonzero
        // row (last component always 1.0 → never all-zero → never a lookup miss).
        let mut table = StaticTokenTable::new(vocab_size, dims, [0.0, 0.0, 1.0, 0.0, 0.0]);
        for id in 0..vocab_size as u32 {
            let f = id as f32;
            table.set_row(
                id,
                &[(f % 97.0) - 48.0, (f % 13.0) - 6.0, (f % 5.0) - 2.0, 1.0],
            );
        }

        // Hand-build all four artifacts in a temp model dir.
        let tmp = tempfile::TempDir::new().unwrap();
        table
            .save(&model_manager::static_token_table_path(tmp.path()))
            .unwrap();
        // ZERO mixer → identity-plus-normalize (see the doc comment above).
        MicroMixer::zeros(dims)
            .save(&model_manager::cinder_mixer_path(tmp.path()))
            .unwrap();
        // Centroids only needed to DERIVE the shortlists (the encoder loads the
        // shortlists, not the centroids); a tiny synthetic set suffices.
        let centroids =
            Array2::<f32>::from_shape_fn((6, dims), |(i, j)| ((i * 7 + j) as f32 * 0.31).sin());
        save_centroids_npy(
            &model_manager::frozen_centroids_path(tmp.path()),
            &centroids,
        )
        .unwrap();
        CentroidShortlists::derive(&table, &centroids.view(), 3)
            .unwrap()
            .save(&model_manager::cinder_shortlists_path(tmp.path()))
            .unwrap();
        // Tokenizer + config for the id-alignment.
        for f in ["tokenizer.json", "onnx_config.json"] {
            std::fs::copy(model_dir.join(f), tmp.path().join(f)).unwrap();
        }

        let cinder = CinderEncoder::new(tmp.path()).expect("all four artifacts valid");
        // The loaded shortlists artifact is exercised (m clamps to 3 here).
        assert_eq!(cinder.shortlists().m, 3);

        let static_emb = StaticTokenEmbedder::new(tmp.path()).unwrap();

        let texts = vec!["fn main() { let value = compute_sum(1, 2); }".to_string()];
        let cinder_out = cinder.encode_documents(&texts).unwrap();
        let static_out = static_emb.encode_documents(&texts).unwrap();

        assert_eq!(cinder_out.len(), 1);
        assert_eq!(cinder_out[0].shape(), static_out[0].shape());
        assert!(cinder_out[0].nrows() > 0, "doc should produce token rows");
        assert_eq!(cinder_out[0].ncols(), dims);

        for (i, (cr, sr)) in cinder_out[0]
            .rows()
            .into_iter()
            .zip(static_out[0].rows())
            .enumerate()
        {
            // Every row is unit-norm (no center misses by construction).
            let norm: f32 = cr.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "cinder row {i} norm {norm} not ~1.0"
            );
            for (a, b) in cr.iter().zip(sr.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "row {i}: zero-mixer Cinder {a} != static center lookup {b}"
                );
            }
        }
    }

    // ── (Task 6b) genuine end-to-end WIRING correctness ─────────────────────

    /// Concatenate every `{i}.codes.npy` chunk file in `dir` (ascending chunk
    /// order) into one flat `Vec<i64>` of centroid ids — the codes actually
    /// written to disk.
    fn read_all_codes(dir: &std::path::Path) -> Vec<i64> {
        use ndarray_npy::ReadNpyExt;
        let mut all = Vec::new();
        let mut i = 0usize;
        loop {
            let p = dir.join(format!("{i}.codes.npy"));
            if !p.exists() {
                break;
            }
            let arr = ndarray::Array1::<i64>::read_npy(std::fs::File::open(&p).unwrap()).unwrap();
            all.extend(arr.iter().copied());
            i += 1;
        }
        all
    }

    /// The real production wiring, proven end to end WITHOUT the model: build a
    /// tiny index through `CompiledIndexWriter::add_document_with_ids` + the REAL
    /// `shortlist_argmax`/`CentroidShortlists` (Task 3) — the exact closure shape
    /// `build_cinder` installs — then assert the codes on disk EXACTLY equal
    /// calling `shortlist_argmax` directly on the same embeddings/window-ids.
    ///
    /// This is strictly stronger than Task 6's mechanism-gate test (which used a
    /// per-row exhaustive stand-in): it proves the SHORTLIST-UNION function
    /// itself is what reaches disk through the id-aware writer path, row-for-row,
    /// across a chunk boundary, with `m < n_centroids` so the union is a genuine
    /// subset (the interesting case). `add_document`/`with_assigner` byte-identity
    /// is untouched — this exercises only the new id-aware surface.
    #[test]
    fn add_document_with_ids_writes_real_shortlist_codes() {
        use crate::embedding::shortlists::{CentroidShortlists, shortlist_argmax};
        use crate::embedding::static_table::StaticTokenTable;
        use ndarray::{Array1, Array2};
        use next_plaid::{CompiledIndexWriter, IdAwareCodeAssigner, IndexConfig};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let dim = 8usize;
        let n_centroids = 24usize;
        let vocab = 40usize;

        // Deterministic unit-norm centroids + a fully-populated table; derive
        // REAL shortlists with m (6) < n_centroids (24) so each window's union
        // is a strict subset of all centroids.
        let mut centroids = Array2::<f32>::from_shape_fn((n_centroids, dim), |(i, j)| {
            ((i * 13 + j * 5) as f32 * 0.31).cos()
        });
        for mut r in centroids.rows_mut() {
            let n = r.dot(&r).sqrt();
            if n > 0.0 {
                r.mapv_inplace(|v| v / n);
            }
        }
        let mut table = StaticTokenTable::new(vocab, dim, [0.2; 5]);
        for id in 0..vocab as u32 {
            let row: Vec<f32> = (0..dim)
                .map(|j| ((id as usize * 7 + j * 3) as f32 * 0.17).sin())
                .collect();
            table.set_row(id, &row);
        }
        let shortlists = CentroidShortlists::derive(&table, &centroids.view(), 6).unwrap();

        // Synthetic docs: unit-norm embeddings + 1..=3 in-vocab window ids per
        // token. 37 docs at batch_size 16 → multiple flushed chunks.
        let make_doc = |seed: usize, n_tokens: usize| -> (Array2<f32>, Vec<Vec<u32>>) {
            let mut e = Array2::<f32>::from_shape_fn((n_tokens, dim), |(i, j)| {
                ((seed * 31 + i * 7 + j) as f32 * 0.37).sin()
            });
            for mut r in e.rows_mut() {
                let n = r.dot(&r).sqrt();
                if n > 0.0 {
                    r.mapv_inplace(|v| v / n);
                }
            }
            let windows: Vec<Vec<u32>> = (0..n_tokens)
                .map(|i| {
                    let len = 1 + i % 3;
                    (0..len)
                        .map(|k| ((seed + i * 5 + k * 11) % vocab) as u32)
                        .collect()
                })
                .collect();
            (e, windows)
        };
        let docs: Vec<(Array2<f32>, Vec<Vec<u32>>)> =
            (0..37).map(|s| make_doc(s, 2 + s % 5)).collect();

        // Real shortlist assigner — identical in shape to build_cinder's closure
        // (one scratch buffer reused across all rows in a chunk).
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_cl = Arc::clone(&calls);
        let a_shortlists = shortlists.clone();
        let a_centroids = centroids.clone();
        let assigner: IdAwareCodeAssigner =
            Box::new(move |batch: &Array2<f32>, windows: &[Vec<u32>]| {
                calls_cl.fetch_add(1, Ordering::SeqCst);
                let cview = a_centroids.view();
                let mut scratch: Vec<u16> = Vec::new();
                let mut out = Vec::with_capacity(batch.nrows());
                for (r, row) in batch.rows().into_iter().enumerate() {
                    let e = row
                        .as_slice()
                        .expect("standard-layout batch rows are contiguous");
                    out.push(shortlist_argmax(
                        e,
                        &windows[r],
                        &a_shortlists,
                        &cview,
                        &mut scratch,
                    ));
                }
                Array1::from_vec(out)
            });

        let config = IndexConfig {
            nbits: 4,
            batch_size: 16,
            force_cpu: true,
            ..Default::default()
        };
        let tmp = tempfile::TempDir::new().unwrap();
        let out_dir = tmp.path().join("idx");
        let sample: Vec<Array2<f32>> = docs.iter().map(|(e, _)| e.clone()).collect();
        let mut w = CompiledIndexWriter::new(
            out_dir.to_str().unwrap(),
            centroids.clone(),
            &config,
            &sample,
        )
        .unwrap()
        .with_id_aware_assigner(assigner);
        for (e, win) in &docs {
            w.add_document_with_ids(e, win).unwrap();
        }
        w.finalize().unwrap();
        assert!(
            calls.load(Ordering::SeqCst) > 0,
            "the id-aware shortlist assigner was never invoked"
        );

        // Expected: direct shortlist_argmax over every token, in doc/token order.
        let cview = centroids.view();
        let mut scratch: Vec<u16> = Vec::new();
        let mut expected: Vec<i64> = Vec::new();
        for (e, win) in &docs {
            for (r, row) in e.rows().into_iter().enumerate() {
                let es = row.as_slice().unwrap();
                expected
                    .push(shortlist_argmax(es, &win[r], &shortlists, &cview, &mut scratch) as i64);
            }
        }

        let got = read_all_codes(&out_dir);
        assert!(!expected.is_empty(), "no codes written");
        assert_eq!(got.len(), expected.len(), "token counts differ");
        assert_eq!(
            got, expected,
            "on-disk codes must EXACTLY equal direct shortlist_argmax output — \
             the real shortlist-union function must be what the writer persisted"
        );
    }
}
