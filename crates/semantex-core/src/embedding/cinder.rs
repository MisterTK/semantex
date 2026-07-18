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
//! # v1 scope
//!
//! `encode_documents` hands f32 embeddings to [`next_plaid::CompiledIndexWriter`],
//! which assigns/quantizes them — so byte-identity with the reference PLAID
//! path is trivially preserved and the "integer codes end to end" optimization
//! (which would route through the writer's `with_assigner` shortlist hook) is
//! deferred to a later profiling-driven pass. `CentroidShortlists` is therefore
//! LOADED and validated here (so the presence/corruptness contract in
//! [`CinderEncoder::new`] holds and the artifact is ready for that later path)
//! but not yet consumed by `encode_documents`; [`CinderEncoder::shortlists`]
//! exposes it.

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
    /// Per-vocab centroid shortlists (SXCS). Loaded/validated here; consumed by
    /// the deferred codes-end-to-end assigner path (see module docs), exposed
    /// via [`Self::shortlists`].
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

    /// The loaded centroid shortlists (SXCS). Not consumed by v1's
    /// `encode_documents` (see module docs) — exposed for the deferred
    /// codes-end-to-end assigner path and for tests that assert the artifact
    /// loaded correctly.
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
    pub fn encode_documents(&self, texts: &[String]) -> Result<Vec<TokenEmbeddings>> {
        texts
            .iter()
            .map(|text| {
                let ids = build_doc_token_ids(&self.alignment, text)?;
                Ok(self.encode_ids(&ids))
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
            self.mixer.forward(&window_refs, center_row, &mut e);
            out.row_mut(i)
                .assign(&ndarray::ArrayView1::from(e.as_slice()));
        }
        out
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
}
