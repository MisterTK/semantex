//! Tier-0, encoder-free document embedder (Ember Plan A, Task 5).
//!
//! Reconstructs per-token embeddings from a pre-computed [`StaticTokenTable`]
//! (Task 1) via the five-token mixing formula [`crate::embedding::static_distill`]
//! fit `mix_weights` against (Task 3), instead of running the ONNX ColBERT
//! model. Tokenization reuses
//! [`crate::embedding::colbert::build_doc_token_ids`] verbatim — the exact
//! filtered/marked id sequence Task 2's investigation established — so this
//! embedder's input ids can never drift from the contract the fitted weights
//! were calibrated against.
//!
//! See `docs/superpowers/plans/2026-07-17-ember-plan-a-gate1.md` (Task 5).

use crate::embedding::colbert::{
    DocIdAlignment, TokenEmbeddings, build_doc_token_ids, load_doc_id_alignment,
};
use crate::embedding::static_table::StaticTokenTable;
use anyhow::{Context, Result};
use ndarray::Array2;
use std::path::Path;

/// Width of the five-token mixing window (two tokens of left context, the
/// center token, two tokens of right context). Must match
/// `static_distill::WINDOW_LEN` — the fitted `mix_weights` this embedder
/// consumes were calibrated against exactly this window width.
const WINDOW_LEN: usize = 5;
/// Index of the center token within the window. Must match
/// `static_distill::CENTER_OFFSET`.
const CENTER_OFFSET: usize = 2;

/// Tier-0 document embedder: produces per-token embeddings via table lookup
/// and a fixed five-position mixing formula, with no neural network
/// evaluation. Intended as a cheap fallback for hosts where the full ONNX
/// ColBERT model is unavailable or too costly to run (see Task 6's
/// `SEMANTEX_STATIC_DOC_EMBED` switch).
pub struct StaticTokenEmbedder {
    table: StaticTokenTable,
    alignment: DocIdAlignment,
}

impl StaticTokenEmbedder {
    /// Load a [`StaticTokenEmbedder`] from `model_dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if `static_token_table.bin`, `tokenizer.json`, or
    /// `onnx_config.json` is missing or malformed under `model_dir`. The
    /// caller decides the fallback (e.g. to the full
    /// [`crate::embedding::colbert::ColbertEmbedder`]).
    pub fn new(model_dir: &Path) -> Result<Self> {
        let table_path = crate::embedding::model_manager::static_token_table_path(model_dir);
        let table = StaticTokenTable::load(&table_path).with_context(|| {
            format!("failed to load static token table {}", table_path.display())
        })?;
        let alignment = load_doc_id_alignment(model_dir)?;
        Ok(Self { table, alignment })
    }

    /// Encode documents into per-token embeddings, one `Array2<f32>` per
    /// document — same output type/shape contract as
    /// [`crate::embedding::colbert::ColbertEmbedder::encode_documents`].
    pub fn encode_documents(&self, texts: &[String]) -> Result<Vec<TokenEmbeddings>> {
        texts
            .iter()
            .map(|text| {
                let ids = build_doc_token_ids(&self.alignment, text)?;
                Ok(mix_document(&self.table, &ids))
            })
            .collect()
    }
}

/// Build the [`WINDOW_LEN`]-token window centered at position `i` in `ids`.
///
/// Uses the SAME edge convention
/// `static_distill::Accumulator::ingest_document` used to build the
/// reservoir samples the fitted `mix_weights` were calibrated against
/// (`static_distill.rs`, around lines 196-208): a window position that falls
/// outside the document reuses the CENTER token's own id, rather than
/// reading into a neighboring document or padding with an out-of-vocab
/// sentinel. Getting this wrong would apply the fitted weights inconsistently
/// with how they were trained — silently, with no test catching it unless it
/// specifically checks this edge case.
fn window_ids_at(ids: &[u32], i: usize) -> [u32; WINDOW_LEN] {
    let id = ids[i];
    let mut window = [id; WINDOW_LEN];
    for offset in 1..=CENTER_OFFSET {
        if i >= offset {
            window[CENTER_OFFSET - offset] = ids[i - offset];
        }
        if i + offset < ids.len() {
            window[CENTER_OFFSET + offset] = ids[i + offset];
        }
    }
    window
}

/// L2-normalize `v` in place. Leaves `v` as all-zero if its norm is already
/// zero (avoids dividing by zero / producing NaNs).
fn normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Mix one token position's embedding from its five-token window:
/// `v_i = normalize(Σ_k w_k · table[window[k]])`, where a `table.lookup`
/// miss contributes zero to the sum. If the CENTER token itself misses,
/// falls back to the un-mixed mean of whichever neighbor lookups DID hit
/// (ignoring `weights` entirely for that fallback). If every lookup in the
/// window misses, returns an all-zero row (MaxSim ignores it downstream).
fn mix_token(
    table: &StaticTokenTable,
    window: [u32; WINDOW_LEN],
    weights: [f32; WINDOW_LEN],
) -> Vec<f32> {
    let dims = table.dims;
    let rows: [Option<&[f32]>; WINDOW_LEN] = std::array::from_fn(|k| table.lookup(window[k]));

    if rows[CENTER_OFFSET].is_some() {
        let mut v = vec![0.0f32; dims];
        for (k, row) in rows.iter().enumerate() {
            if let Some(row) = row {
                let w = weights[k];
                for (vi, &ri) in v.iter_mut().zip(row.iter()) {
                    *vi += w * ri;
                }
            }
        }
        normalize_in_place(&mut v);
        v
    } else {
        // Center missed: fall back to the un-mixed mean of whichever
        // neighbor lookups DID hit, ignoring `weights` entirely.
        let mut sum = vec![0.0f32; dims];
        let mut hits: u32 = 0;
        for (k, row) in rows.iter().enumerate() {
            if k == CENTER_OFFSET {
                continue;
            }
            if let Some(row) = row {
                hits += 1;
                for (si, &ri) in sum.iter_mut().zip(row.iter()) {
                    *si += ri;
                }
            }
        }
        if hits == 0 {
            return vec![0.0f32; dims]; // everything in the window missed
        }
        let inv = 1.0 / hits as f32;
        for s in &mut sum {
            *s *= inv;
        }
        normalize_in_place(&mut sum);
        sum
    }
}

/// Mix every token position of one document into an `[n_tokens, dims]`
/// matrix.
fn mix_document(table: &StaticTokenTable, ids: &[u32]) -> TokenEmbeddings {
    let dims = table.dims;
    let mut out = Array2::<f32>::zeros((ids.len(), dims));
    for i in 0..ids.len() {
        let window = window_ids_at(ids, i);
        let row = mix_token(table, window, table.mix_weights);
        out.row_mut(i).assign(&ndarray::ArrayView1::from(&row));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-built 6-token table (ids 0..=5) with known, easy-to-hand-check
    /// rows. Token id 0 is deliberately left unset (all-zero row), so
    /// `table.lookup(0)` reports `None` — an OOV/unseen stand-in.
    fn hand_built_table() -> StaticTokenTable {
        let mut t = StaticTokenTable::new(6, 3, [0.1, 0.2, 0.4, 0.2, 0.1]);
        t.set_row(1, &[1.0, 0.0, 0.0]);
        t.set_row(2, &[0.0, 1.0, 0.0]);
        t.set_row(3, &[0.0, 0.0, 1.0]);
        t.set_row(4, &[1.0, 1.0, 0.0]);
        t.set_row(5, &[0.0, 1.0, 1.0]);
        t
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    fn assert_close(got: &[f32], want: &[f32], msg: &str) {
        assert_eq!(got.len(), want.len(), "{msg}: length mismatch");
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "{msg}: got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn interior_position_matches_hand_computed_weighted_mix() {
        let table = hand_built_table();
        let ids = vec![1u32, 2, 3, 4, 5];
        let doc = mix_document(&table, &ids);

        // Center position i=2 (id=3): no edge reuse needed, all five window
        // positions are distinct, present neighbors [1,2,3,4,5].
        let w = table.mix_weights;
        let rows: Vec<&[f32]> = vec![
            table.lookup(1).unwrap(),
            table.lookup(2).unwrap(),
            table.lookup(3).unwrap(),
            table.lookup(4).unwrap(),
            table.lookup(5).unwrap(),
        ];
        let mut expected = vec![0.0f32; 3];
        for (k, row) in rows.iter().enumerate() {
            for (e, &r) in expected.iter_mut().zip(row.iter()) {
                *e += w[k] * r;
            }
        }
        let expected = normalize(&expected);

        assert_close(
            doc.row(2).as_slice().unwrap(),
            &expected,
            "interior mixed row",
        );
    }

    #[test]
    fn rows_are_l2_normalized() {
        let table = hand_built_table();
        let ids = vec![1u32, 2, 3, 4, 5];
        let doc = mix_document(&table, &ids);
        for i in 0..ids.len() {
            let row = doc.row(i);
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "row {i} norm {norm} is not ~1.0: {row:?}"
            );
        }
    }

    /// Verifies the edge convention independently confirmed in
    /// `static_distill.rs` (`Accumulator::ingest_document`, ~lines 196-208):
    /// window positions past either edge of the document reuse the CENTER
    /// token's own id, not an out-of-vocab sentinel and not a wraparound
    /// into a neighboring document.
    #[test]
    fn edge_position_reuses_center_id_per_task3_convention() {
        let table = hand_built_table();
        let ids = vec![1u32, 2, 3, 4, 5];
        let doc = mix_document(&table, &ids);

        // Position i=0 (id=1): offsets -1 and -2 fall outside the document,
        // so both window slots reuse id 1 (the center), alongside the two
        // genuine right-neighbors at offsets +1 (id 2) and +2 (id 3).
        let w = table.mix_weights;
        let row1 = table.lookup(1).unwrap(); // reused for k=0,1,2
        let row2 = table.lookup(2).unwrap(); // k=3 (offset +1)
        let row3 = table.lookup(3).unwrap(); // k=4 (offset +2)
        let mut expected = vec![0.0f32; 3];
        for (i, &v) in row1.iter().enumerate() {
            expected[i] += (w[0] + w[1] + w[2]) * v;
        }
        for (i, &v) in row2.iter().enumerate() {
            expected[i] += w[3] * v;
        }
        for (i, &v) in row3.iter().enumerate() {
            expected[i] += w[4] * v;
        }
        let expected = normalize(&expected);

        assert_close(
            doc.row(0).as_slice().unwrap(),
            &expected,
            "edge-position mixed row (i=0)",
        );
    }

    #[test]
    fn oov_center_falls_back_to_unmixed_mean_of_neighbor_hits() {
        let table = hand_built_table();
        // Center token at i=2 has id=0 (OOV, unset row). Neighbors 1, 2, 4,
        // 5 are all present in the table (no edge reuse at this position).
        let ids = vec![1u32, 2, 0, 4, 5];
        let doc = mix_document(&table, &ids);

        let neighbors = [
            table.lookup(1).unwrap(),
            table.lookup(2).unwrap(),
            table.lookup(4).unwrap(),
            table.lookup(5).unwrap(),
        ];
        let mut mean = vec![0.0f32; 3];
        for row in &neighbors {
            for (m, &v) in mean.iter_mut().zip(row.iter()) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= neighbors.len() as f32;
        }
        let expected = normalize(&mean);

        assert_close(
            doc.row(2).as_slice().unwrap(),
            &expected,
            "OOV-center fallback row",
        );
    }

    #[test]
    fn everything_missing_emits_all_zero_row() {
        let table = hand_built_table();
        // Single-token document, id=0 (OOV). The edge convention reuses the
        // center id for every window position, so the whole window is
        // [0,0,0,0,0] — every lookup misses, and the row must be all-zero.
        let ids = vec![0u32];
        let doc = mix_document(&table, &ids);
        let row = doc.row(0);
        assert!(
            row.iter().all(|&v| v == 0.0),
            "expected an all-zero row when every window lookup misses, got {row:?}"
        );
    }

    /// `StaticTokenEmbedder` intentionally does not derive `Debug`
    /// (`DocIdAlignment`'s `Tokenizer` field doesn't either), so
    /// `Result::unwrap_err` (which requires the `Ok` side to be `Debug`)
    /// can't be used here — extract the error by hand instead.
    fn expect_err(result: Result<StaticTokenEmbedder>) -> anyhow::Error {
        match result {
            Ok(_) => panic!("expected an error, got Ok"),
            Err(e) => e,
        }
    }

    #[test]
    fn new_errors_when_static_token_table_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let err = expect_err(StaticTokenEmbedder::new(tmp.path()));
        assert!(
            err.to_string().contains("static_token_table.bin"),
            "expected a static-token-table error, got: {err}"
        );
    }

    #[test]
    fn new_errors_when_tokenizer_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Table present, but tokenizer.json/onnx_config.json are not.
        let table = StaticTokenTable::new(4, 2, [0.0, 0.0, 1.0, 0.0, 0.0]);
        table
            .save(&tmp.path().join("static_token_table.bin"))
            .unwrap();
        let err = expect_err(StaticTokenEmbedder::new(tmp.path()));
        assert!(
            err.to_string().to_lowercase().contains("tokenizer"),
            "expected a tokenizer-loading error, got: {err}"
        );
    }

    /// Locate the local LateOn-Code-edge tokenizer files used by the
    /// end-to-end gated test below. Mirrors Task 2's `test_model_dir`
    /// pattern in `colbert.rs`: skip, don't fail, when the model hasn't
    /// been downloaded (CI, fresh clones). This embedder never touches the
    /// ONNX model itself, so gating only requires the tokenizer + config,
    /// not `model_int8.onnx`.
    fn test_tokenizer_dir() -> Option<std::path::PathBuf> {
        let dir = crate::config::SemantexConfig::default()
            .models_dir()
            .join("LateOn-Code-edge");
        (dir.join("tokenizer.json").exists() && dir.join("onnx_config.json").exists())
            .then_some(dir)
    }

    /// End-to-end integration: the real tokenizer plus a hand-built table,
    /// proving `StaticTokenEmbedder::new`/`encode_documents` plumbing (file
    /// loading, `build_doc_token_ids` reuse, row-count shape) works
    /// together. The mixing arithmetic itself is covered by the pure unit
    /// tests above, which need no model files and always run in CI.
    #[test]
    fn end_to_end_with_real_tokenizer_and_hand_built_table() {
        let Some(model_dir) = test_tokenizer_dir() else {
            return;
        };

        let vocab_size = crate::embedding::colbert::ColbertEmbedder::new(&model_dir)
            .unwrap()
            .tokenizer_vocab_size()
            .unwrap();

        // Only token id 0 has a real row; a real document is very unlikely
        // to tokenize down to id 0, so most lookups will miss. That's fine
        // — this test's job is to prove the plumbing works end-to-end and
        // produces the right shape, not to re-verify mixing arithmetic.
        let mut table = StaticTokenTable::new(vocab_size, 4, [0.1, 0.2, 0.4, 0.2, 0.1]);
        table.set_row(0, &[1.0, 0.0, 0.0, 0.0]);

        let tmp = tempfile::TempDir::new().unwrap();
        table
            .save(&tmp.path().join("static_token_table.bin"))
            .unwrap();
        std::fs::copy(
            model_dir.join("tokenizer.json"),
            tmp.path().join("tokenizer.json"),
        )
        .unwrap();
        std::fs::copy(
            model_dir.join("onnx_config.json"),
            tmp.path().join("onnx_config.json"),
        )
        .unwrap();

        let embedder = StaticTokenEmbedder::new(tmp.path()).unwrap();
        let texts = vec!["fn main() { println!(\"hi\"); }".to_string()];
        let out = embedder.encode_documents(&texts).unwrap();
        assert_eq!(out.len(), 1);
        assert!(
            out[0].nrows() > 0,
            "document should produce at least one token row"
        );
        assert_eq!(out[0].ncols(), 4, "row width must match the table's dims");
    }
}
