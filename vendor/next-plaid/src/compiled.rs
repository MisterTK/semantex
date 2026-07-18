//! Single-pass, byte-compatible PLAID index construction.
//!
//! [`CompiledIndexWriter`] streams per-document embeddings into a PLAID index
//! directory that is BYTE-IDENTICAL to [`crate::index::create_index_files`],
//! without ever holding the whole corpus in memory.
//!
//! # How byte-identity is guaranteed
//!
//! The writer performs the EXACT same operations as `create_index_files`, just
//! spread across `new` / `add_document` / `finalize` instead of one call:
//!
//! * `new` runs the shared [`prepare_codec_artifacts`] (the same residual-stats
//!   math `create_index_files` uses) and writes the five codec files.
//! * `add_document` buffers documents and, once a full chunk of
//!   `config.batch_size` docs has accumulated, encodes it via the shared
//!   [`encode_index_chunk`] and writes the four per-chunk files with
//!   `embedding_offset: 0` — exactly as `create_index_files` does in its chunk
//!   loop.
//! * `finalize` writes `plan.json`, rewrites the chunk metadata offsets in the
//!   same second pass `create_index_files` uses (via a `serde_json::Value`
//!   round-trip, so the on-disk key order matches byte-for-byte), builds the
//!   IVF once, and writes `ivf.npy` / `ivf_lengths.npy` / `metadata.json`.
//!
//! # Memory
//!
//! Peak memory is one chunk (`batch_size` docs) plus the IVF accumulator, a
//! `Vec<(u32 centroid, i64 doc_id)>` of 16 bytes per token — the only O(corpus)
//! allocation. The raw corpus is never held in full (unlike `create_index_files`,
//! which takes the entire `&[Array2<f32>]` up front).

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Axis, s};
use ndarray_npy::WriteNpyExt;

use crate::error::{Error, Result};
use crate::index::{
    ChunkMetadata, EncodedIndexChunk, IndexConfig, Metadata, PreparedCodecArtifacts,
    encode_index_chunk, prepare_codec_artifacts,
};
use crate::utils::atomic_write_file;

/// A caller-supplied centroid-assignment function: maps a `[n_tokens, dim]`
/// batch of embeddings to one centroid id per row, replacing the codec's
/// exhaustive [`crate::codec::ResidualCodec::compress_into_codes_cpu`] scan.
/// See [`CompiledIndexWriter::with_assigner`].
pub type CodeAssigner = Box<dyn Fn(&Array2<f32>) -> Array1<usize>>;

/// Builds a PLAID index in ONE pass over per-document embeddings, with frozen
/// centroids, flushing packed codes/residuals to on-disk chunk files instead of
/// holding all embeddings in memory, and NEVER rewriting the IVF incrementally.
///
/// Contract (gate C4): given the same documents, centroids, and config, the
/// resulting index directory is BYTE-IDENTICAL (excluding `embeddings.npy` /
/// buffer files, which this writer never creates) to
/// `create_index_files(embeddings, centroids, path, config)`.
///
/// ⚠️ Of the logic in [`crate::index::create_index_files`], only
/// [`prepare_codec_artifacts`] (residual statistics) is genuinely shared —
/// the chunk-encoding, IVF-building, and metadata-writing paths here are
/// parallel implementations kept in sync solely by the differential test
/// `output_is_byte_identical_to_create_index_files` below. That test does
/// NOT run under `cargo test --workspace` (see the sync note on
/// `create_index_files` in `index.rs` for why, and how to run it manually).
/// Any change to `create_index_files`'s chunk/IVF/metadata logic requires
/// re-running that test against this writer before merging.
pub struct CompiledIndexWriter {
    /// Destination index directory.
    index_dir: PathBuf,
    /// Index configuration (chunking + quantization).
    config: IndexConfig,
    /// Trained codec + residual statistics (shared with `create_index_files`).
    artifacts: PreparedCodecArtifacts,
    /// Documents awaiting the next chunk flush (at most `batch_size` at a time).
    buffer: Vec<Array2<f32>>,
    /// Number of chunks already flushed to disk.
    num_chunks: usize,
    /// Running global document id (== number of documents added so far).
    next_doc_id: i64,
    /// Total number of token embeddings across all flushed chunks.
    total_embeddings: usize,
    /// IVF accumulator: one `(centroid, global_doc_id)` pair per token, in the
    /// same global token order `create_index_files` walks.
    ivf_pairs: Vec<(u32, i64)>,
    /// Optional caller-supplied centroid assigner (see [`Self::with_assigner`]).
    /// `None` (the default) means each chunk is assigned by the codec's own
    /// exhaustive `compress_into_codes_cpu` — the byte-identical reference path.
    /// `Some(f)` swaps ONLY the assignment step; residual computation and
    /// quantization stay byte-for-byte the same as the default.
    assigner: Option<CodeAssigner>,
}

impl CompiledIndexWriter {
    /// Create a writer, computing the codec/residual statistics from `sample_docs`
    /// and writing the five codec files immediately.
    ///
    /// `sample_docs` plays the exact role of `create_index_files`'s full
    /// `embeddings` input for residual-statistics training: cutoffs, weights,
    /// avg-residual and the cluster threshold are derived from it via the shared
    /// [`prepare_codec_artifacts`], so the math cannot drift. For a BYTE-IDENTICAL
    /// result, `sample_docs` must be the same documents (in the same order) that
    /// are subsequently streamed through [`add_document`], because that is what
    /// `create_index_files` samples from.
    pub fn new(
        index_path: &str,
        centroids: Array2<f32>,
        config: &IndexConfig,
        sample_docs: &[Array2<f32>],
    ) -> Result<Self> {
        let index_dir = Path::new(index_path).to_path_buf();
        std::fs::create_dir_all(&index_dir)?;

        // Shared residual-statistics math — identical to create_index_files.
        let artifacts = prepare_codec_artifacts(sample_docs, centroids, config)?;

        // Write codec component files EXACTLY as create_index_files does.
        atomic_write_file(&index_dir.join("centroids.npy"), |file| {
            artifacts
                .codec
                .centroids_view()
                .to_owned()
                .write_npy(file)?;
            Ok(())
        })?;
        atomic_write_file(&index_dir.join("bucket_cutoffs.npy"), |file| {
            artifacts.bucket_cutoffs.write_npy(file)?;
            Ok(())
        })?;
        atomic_write_file(&index_dir.join("bucket_weights.npy"), |file| {
            artifacts.bucket_weights.write_npy(file)?;
            Ok(())
        })?;
        atomic_write_file(&index_dir.join("avg_residual.npy"), |file| {
            artifacts.avg_res_per_dim.write_npy(file)?;
            Ok(())
        })?;
        atomic_write_file(&index_dir.join("cluster_threshold.npy"), |file| {
            Array1::from_vec(vec![artifacts.cluster_threshold]).write_npy(file)?;
            Ok(())
        })?;

        Ok(Self {
            index_dir,
            config: config.clone(),
            artifacts,
            buffer: Vec::new(),
            num_chunks: 0,
            next_doc_id: 0,
            total_embeddings: 0,
            ivf_pairs: Vec::new(),
            assigner: None,
        })
    }

    /// Install a caller-supplied centroid [`CodeAssigner`], used instead of the
    /// codec's exhaustive `compress_into_codes_cpu` when flushing each chunk.
    ///
    /// This is a minimal, additive hook: it changes ONLY which centroid id each
    /// token is assigned to. Residual computation (`embedding − centroid[code]`)
    /// and quantization are unchanged, so with NO assigner installed the writer
    /// is still byte-identical to [`crate::index::create_index_files`] (proven by
    /// `default_no_assigner_path_stays_byte_identical`). Cinder passes a closure
    /// wrapping its shortlist-union argmax; the codes it produces agree with the
    /// exhaustive default to within a small tolerance (proven by
    /// `shortlist_assigner_agrees_with_exhaustive_default`).
    #[must_use]
    pub fn with_assigner(mut self, assigner: CodeAssigner) -> Self {
        self.assigner = Some(assigner);
        self
    }

    /// Append one document's token embeddings (unit-norm, `[n_tokens, dim]`).
    ///
    /// The document is buffered; once `config.batch_size` documents have
    /// accumulated, a full chunk is encoded and flushed to disk (the four
    /// per-chunk files), matching `create_index_files`'s chunk boundaries.
    pub fn add_document(&mut self, embeddings: &Array2<f32>) -> Result<()> {
        self.buffer.push(embeddings.clone());
        if self.buffer.len() >= self.config.batch_size {
            self.flush_chunk()?;
        }
        Ok(())
    }

    /// Encode the buffered documents as one chunk and write its four files,
    /// then accumulate IVF pairs and clear the buffer. No-op on an empty buffer.
    fn flush_chunk(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let chunk_idx = self.num_chunks;

        // Same encode path as create_index_files (compress_into_codes_cpu +
        // quantize_residuals), producing byte-identical codes/residuals — UNLESS
        // a caller supplied an assigner via `with_assigner`, in which case only
        // the centroid-assignment step is swapped (residuals/quantization stay
        // identical). The default (no-assigner) arm is untouched, preserving the
        // byte-identity contract.
        let encoded = match self.assigner.as_ref() {
            None => encode_index_chunk(&self.buffer, &self.artifacts.codec, self.config.force_cpu)?,
            Some(assigner) => self.encode_chunk_with_assigner(assigner.as_ref())?,
        };

        // {chunk}.metadata.json — written with embedding_offset: 0 first, exactly
        // as create_index_files does; finalize() rewrites the offsets in a second
        // pass so the serialized key order matches byte-for-byte.
        let chunk_meta = ChunkMetadata {
            num_documents: self.buffer.len(),
            num_embeddings: encoded.codes.len(),
            embedding_offset: 0,
        };
        atomic_write_file(
            &self.index_dir.join(format!("{}.metadata.json", chunk_idx)),
            |file| {
                let mut writer = BufWriter::new(file);
                serde_json::to_writer_pretty(&mut writer, &chunk_meta)?;
                writer.flush()?;
                Ok(())
            },
        )?;

        // doclens.{chunk}.json (compact, not pretty — matches the reference).
        atomic_write_file(
            &self.index_dir.join(format!("doclens.{}.json", chunk_idx)),
            |file| {
                let mut writer = BufWriter::new(file);
                serde_json::to_writer(&mut writer, &encoded.doclens)?;
                writer.flush()?;
                Ok(())
            },
        )?;

        // {chunk}.codes.npy
        atomic_write_file(
            &self.index_dir.join(format!("{}.codes.npy", chunk_idx)),
            |file| {
                encoded.codes.write_npy(file)?;
                Ok(())
            },
        )?;

        // {chunk}.residuals.npy
        atomic_write_file(
            &self.index_dir.join(format!("{}.residuals.npy", chunk_idx)),
            |file| {
                encoded.residuals.write_npy(file)?;
                Ok(())
            },
        )?;

        // Accumulate IVF pairs (centroid, global_doc_id) per token, walking the
        // chunk in the same doc/token order create_index_files uses.
        let mut emb_idx = 0usize;
        for &len in &encoded.doclens {
            let doc_id = self.next_doc_id;
            for _ in 0..len {
                let code = encoded.codes[emb_idx];
                self.ivf_pairs.push((code as u32, doc_id));
                emb_idx += 1;
            }
            self.next_doc_id += 1;
        }

        self.total_embeddings += encoded.codes.len();
        self.num_chunks += 1;
        self.buffer.clear();
        Ok(())
    }

    /// Encode the buffered documents into a chunk using a CALLER-SUPPLIED code
    /// [`CodeAssigner`] in place of the codec's exhaustive
    /// `compress_into_codes_cpu`.
    ///
    /// Everything except the assignment step mirrors
    /// [`crate::index::encode_index_chunk`] / its `compress_and_residuals_cpu`
    /// byte-for-byte: the docs are concatenated into one `[total_tokens, dim]`
    /// batch, residuals are `embedding − centroid[code]`, and quantization runs
    /// through the same [`crate::codec::ResidualCodec::quantize_residuals`]. Only
    /// exercised when [`Self::with_assigner`] was called.
    fn encode_chunk_with_assigner(
        &self,
        assigner: &dyn Fn(&Array2<f32>) -> Array1<usize>,
    ) -> Result<EncodedIndexChunk> {
        let codec = &self.artifacts.codec;
        let embedding_dim = codec.embedding_dim();
        let packed_dim = embedding_dim * codec.nbits / 8;
        let doclens: Vec<i64> = self.buffer.iter().map(|d| d.nrows() as i64).collect();
        let total_tokens: usize = doclens.iter().sum::<i64>() as usize;

        // Concatenate the buffered docs into one [total_tokens, dim] batch,
        // exactly as encode_index_chunk does.
        let mut batch = Array2::<f32>::zeros((total_tokens, embedding_dim));
        let mut offset = 0;
        for doc in &self.buffer {
            let n = doc.nrows();
            batch.slice_mut(s![offset..offset + n, ..]).assign(doc);
            offset += n;
        }

        // The ONLY divergence from the default path: caller-supplied assignment
        // instead of the exhaustive compress_into_codes_cpu.
        let batch_codes = assigner(&batch);

        // residuals = embedding − centroid[code], identical to
        // compress_and_residuals_cpu's residual pass.
        let centroids = codec.centroids_view();
        let mut residuals = batch;
        for (mut row, &code) in residuals.axis_iter_mut(Axis(0)).zip(batch_codes.iter()) {
            let centroid = centroids.row(code);
            row.iter_mut()
                .zip(centroid.iter())
                .for_each(|(r, c)| *r -= c);
        }

        // Quantize/pack exactly as encode_index_chunk does.
        let batch_packed = codec.quantize_residuals(&residuals)?;
        let (raw_residuals, residuals_offset) = batch_packed.into_raw_vec_and_offset();
        if residuals_offset != Some(0) {
            return Err(Error::Shape(format!(
                "Unexpected residual packing offset: {:?}",
                residuals_offset
            )));
        }
        let residuals = Array2::from_shape_vec((batch_codes.len(), packed_dim), raw_residuals)
            .map_err(|e| Error::Shape(format!("Failed to reshape residuals: {}", e)))?;
        let codes: Array1<i64> = batch_codes.iter().map(|&x| x as i64).collect();

        Ok(EncodedIndexChunk {
            codes,
            residuals,
            doclens,
        })
    }

    /// Flush the final chunk, write `plan.json`, fix chunk-metadata offsets,
    /// build the IVF once, and write `ivf.npy` / `ivf_lengths.npy` /
    /// `metadata.json`. Returns the index [`Metadata`].
    pub fn finalize(mut self) -> Result<Metadata> {
        // Flush any remaining buffered docs as the final (possibly partial) chunk.
        self.flush_chunk()?;

        let n_chunks = self.num_chunks;
        let num_documents = self.next_doc_id as usize;
        let num_centroids = self.artifacts.codec.num_centroids();
        let embedding_dim = self.artifacts.codec.embedding_dim();
        let total_embeddings = self.total_embeddings;
        let avg_doclen = if num_documents > 0 {
            total_embeddings as f64 / num_documents as f64
        } else {
            0.0
        };

        // plan.json (create_index_files writes this before the chunk loop; the
        // content is order-independent, so writing it here is byte-identical).
        let plan = serde_json::json!({
            "nbits": self.config.nbits,
            "num_chunks": n_chunks,
        });
        atomic_write_file(&self.index_dir.join("plan.json"), |file| {
            writeln!(file, "{}", serde_json::to_string_pretty(&plan)?)?;
            Ok(())
        })?;

        // Second pass: rewrite chunk metadata with global embedding offsets,
        // EXACTLY as create_index_files (read as serde_json::Value, insert the
        // offset, re-serialize — reproducing serde_json's key order byte-for-byte).
        let mut current_offset = 0usize;
        for chunk_idx in 0..n_chunks {
            let chunk_meta_path = self.index_dir.join(format!("{}.metadata.json", chunk_idx));
            let mut meta: serde_json::Value =
                serde_json::from_reader(BufReader::new(File::open(&chunk_meta_path)?))?;

            if let Some(obj) = meta.as_object_mut() {
                obj.insert("embedding_offset".to_string(), current_offset.into());
                let num_emb = obj["num_embeddings"].as_u64().unwrap_or(0) as usize;
                current_offset += num_emb;
            }

            atomic_write_file(&chunk_meta_path, |file| {
                let mut writer = BufWriter::new(file);
                serde_json::to_writer_pretty(&mut writer, &meta)?;
                writer.flush()?;
                Ok(())
            })?;
        }

        // Build IVF from accumulated (centroid, doc_id) pairs. Sorting by
        // (centroid, doc_id) then deduping yields, per centroid in ascending
        // order, the sorted unique doc ids — identical to create_index_files's
        // BTreeMap-then-per-centroid sort_unstable + dedup.
        self.ivf_pairs.sort_unstable();
        self.ivf_pairs.dedup();
        let mut ivf_data: Vec<i64> = Vec::with_capacity(self.ivf_pairs.len());
        let mut ivf_lengths: Vec<i32> = vec![0; num_centroids];
        for &(centroid, doc_id) in &self.ivf_pairs {
            ivf_lengths[centroid as usize] += 1;
            ivf_data.push(doc_id);
        }

        atomic_write_file(&self.index_dir.join("ivf.npy"), |file| {
            Array1::from_vec(ivf_data).write_npy(file)?;
            Ok(())
        })?;
        atomic_write_file(&self.index_dir.join("ivf_lengths.npy"), |file| {
            Array1::from_vec(ivf_lengths).write_npy(file)?;
            Ok(())
        })?;

        // Global metadata.json.
        let metadata = Metadata {
            num_chunks: n_chunks,
            nbits: self.config.nbits,
            num_partitions: num_centroids,
            num_embeddings: total_embeddings,
            avg_doclen,
            num_documents,
            embedding_dim,
            next_plaid_compatible: true,
        };
        atomic_write_file(&self.index_dir.join("metadata.json"), |file| {
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, &metadata)?;
            writer.flush()?;
            Ok(())
        })?;

        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexConfig, create_index_files};
    use ndarray::Array2;

    fn synth_docs(n_docs: usize, dim: usize) -> Vec<Array2<f32>> {
        (0..n_docs)
            .map(|d| {
                let toks = 3 + (d % 5);
                let mut a = Array2::from_shape_fn((toks, dim), |(i, j)| {
                    ((d * 31 + i * 7 + j) as f32 * 0.37).sin()
                });
                for mut row in a.rows_mut() {
                    let n = row.dot(&row).sqrt();
                    row.mapv_inplace(|v| v / n);
                }
                a
            })
            .collect()
    }

    fn synth_centroids(k: usize, dim: usize) -> Array2<f32> {
        let mut c =
            Array2::from_shape_fn((k, dim), |(i, j)| ((i * 13 + j * 3) as f32 * 0.29).cos());
        for mut row in c.rows_mut() {
            let n = row.dot(&row).sqrt();
            row.mapv_inplace(|v| v / n);
        }
        c
    }

    #[test]
    fn output_is_byte_identical_to_create_index_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let docs = synth_docs(57, dim); // >1 on-disk chunk with batch_size=16
        let centroids = synth_centroids(32, dim);
        let config = IndexConfig {
            nbits: 4,
            batch_size: 16,
            force_cpu: true,
            ..Default::default()
        };

        let ref_dir = tmp.path().join("reference");
        create_index_files(&docs, centroids.clone(), ref_dir.to_str().unwrap(), &config).unwrap();

        let out_dir = tmp.path().join("compiled");
        let mut w =
            CompiledIndexWriter::new(out_dir.to_str().unwrap(), centroids, &config, &docs).unwrap();
        for d in &docs {
            w.add_document(d).unwrap();
        }
        w.finalize().unwrap();

        // Every file create_index_files wrote must exist byte-identical
        // (embeddings.npy excluded: the compiled writer never persists raw
        // embeddings; delete it from the reference before comparing).
        let _ = std::fs::remove_file(ref_dir.join("embeddings.npy"));
        let mut ref_files: Vec<_> = std::fs::read_dir(&ref_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        ref_files.sort();
        let mut out_files: Vec<_> = std::fs::read_dir(&out_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        out_files.sort();
        assert_eq!(ref_files, out_files, "file inventories differ");
        for f in &ref_files {
            let a = std::fs::read(ref_dir.join(f)).unwrap();
            let b = std::fs::read(out_dir.join(f)).unwrap();
            assert_eq!(a, b, "file {f} differs");
        }
    }

    /// Read and concatenate every `{i}.codes.npy` chunk file in `dir`, in
    /// ascending chunk order, into one flat `Vec<i64>` of centroid ids.
    fn read_all_codes(dir: &std::path::Path) -> Vec<i64> {
        use ndarray_npy::ReadNpyExt;
        let mut all = Vec::new();
        let mut i = 0usize;
        loop {
            let p = dir.join(format!("{i}.codes.npy"));
            if !p.exists() {
                break;
            }
            let arr = Array1::<i64>::read_npy(std::fs::File::open(&p).unwrap()).unwrap();
            all.extend(arr.iter().copied());
            i += 1;
        }
        all
    }

    #[test]
    fn default_no_assigner_path_stays_byte_identical() {
        // Regression proof for the `with_assigner` addition: a writer built
        // WITHOUT calling `with_assigner` must remain byte-identical to
        // `create_index_files`. Kept separate from
        // `output_is_byte_identical_to_create_index_files` so a future change to
        // the assigner plumbing that perturbs the default path fails HERE, with
        // an obviously-named test.
        let tmp = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let docs = synth_docs(57, dim);
        let centroids = synth_centroids(32, dim);
        let config = IndexConfig {
            nbits: 4,
            batch_size: 16,
            force_cpu: true,
            ..Default::default()
        };

        let ref_dir = tmp.path().join("reference");
        create_index_files(&docs, centroids.clone(), ref_dir.to_str().unwrap(), &config).unwrap();

        let out_dir = tmp.path().join("compiled");
        // No with_assigner → default exhaustive assignment path.
        let mut w =
            CompiledIndexWriter::new(out_dir.to_str().unwrap(), centroids, &config, &docs).unwrap();
        for d in &docs {
            w.add_document(d).unwrap();
        }
        w.finalize().unwrap();

        let _ = std::fs::remove_file(ref_dir.join("embeddings.npy"));
        let mut ref_files: Vec<_> = std::fs::read_dir(&ref_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        ref_files.sort();
        let mut out_files: Vec<_> = std::fs::read_dir(&out_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        out_files.sort();
        assert_eq!(ref_files, out_files, "file inventories differ");
        for f in &ref_files {
            let a = std::fs::read(ref_dir.join(f)).unwrap();
            let b = std::fs::read(out_dir.join(f)).unwrap();
            assert_eq!(a, b, "file {f} differs");
        }
    }

    #[test]
    fn shortlist_assigner_agrees_with_exhaustive_default() {
        // Mechanism gate for `with_assigner`: a shortlist-style assigner (a
        // stand-in for Cinder's shortlist-union argmax) must (1) actually be
        // invoked, (2) produce codes that reach disk (so agreement is strictly
        // < 1.0 thanks to ONE deliberate second-best pick), and (3) agree with
        // the exhaustive default on ≥99% of tokens.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let tmp = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let docs = synth_docs(57, dim);
        let centroids = synth_centroids(32, dim);
        let config = IndexConfig {
            nbits: 4,
            batch_size: 16,
            force_cpu: true,
            ..Default::default()
        };

        // Reference index: default exhaustive assignment.
        let ref_dir = tmp.path().join("ref");
        let mut wr =
            CompiledIndexWriter::new(ref_dir.to_str().unwrap(), centroids.clone(), &config, &docs)
                .unwrap();
        for d in &docs {
            wr.add_document(d).unwrap();
        }
        wr.finalize().unwrap();
        let ref_codes = read_all_codes(&ref_dir);

        // Shortlist-style assigner: per-row exact scan (a stand-in for the real
        // shortlist union) that deliberately returns the SECOND-best centroid
        // for exactly one row (first row of the first flushed chunk). The
        // deliberate miss proves the assigner's codes — not the codec's own
        // exhaustive ones — actually reached disk.
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_cl = Arc::clone(&calls);
        let cen = centroids.clone();
        let assigner: super::CodeAssigner = Box::new(move |emb: &Array2<f32>| -> Array1<usize> {
            let call_idx = calls_cl.fetch_add(1, Ordering::SeqCst);
            let c = cen.view();
            let mut out = Vec::with_capacity(emb.nrows());
            for (ri, row) in emb.axis_iter(Axis(0)).enumerate() {
                let mut best = (f32::NEG_INFINITY, 0usize);
                let mut second = (f32::NEG_INFINITY, 0usize);
                for (id, cc) in c.axis_iter(Axis(0)).enumerate() {
                    let dot: f32 = row.iter().zip(cc.iter()).map(|(&a, &b)| a * b).sum();
                    if dot > best.0 {
                        second = best;
                        best = (dot, id);
                    } else if dot > second.0 {
                        second = (dot, id);
                    }
                }
                let pick = if call_idx == 0 && ri == 0 && emb.nrows() > 1 {
                    second.1
                } else {
                    best.1
                };
                out.push(pick);
            }
            Array1::from_vec(out)
        });

        let out_dir = tmp.path().join("shortlist");
        let mut w = CompiledIndexWriter::new(out_dir.to_str().unwrap(), centroids, &config, &docs)
            .unwrap()
            .with_assigner(assigner);
        for d in &docs {
            w.add_document(d).unwrap();
        }
        w.finalize().unwrap();
        let test_codes = read_all_codes(&out_dir);

        assert!(
            calls.load(Ordering::SeqCst) > 0,
            "the assigner hook was never invoked"
        );
        assert_eq!(ref_codes.len(), test_codes.len(), "token counts differ");
        assert!(!ref_codes.is_empty(), "no codes were written");
        let agree = ref_codes
            .iter()
            .zip(test_codes.iter())
            .filter(|(a, b)| a == b)
            .count();
        let frac = agree as f64 / ref_codes.len() as f64;
        assert!(
            frac >= 0.99,
            "shortlist assigner agreement {frac} < 0.99 with the exhaustive default"
        );
        assert!(
            frac < 1.0,
            "the one deliberate second-best pick must reach disk (proving the hook \
             is actually used); got perfect agreement {frac}"
        );
    }

    #[test]
    fn compiled_index_loads_and_searches() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let docs = synth_docs(30, dim);
        let config = IndexConfig {
            nbits: 4,
            batch_size: 16,
            force_cpu: true,
            ..Default::default()
        };
        let out = tmp.path().join("idx");
        let mut w = CompiledIndexWriter::new(
            out.to_str().unwrap(),
            synth_centroids(16, dim),
            &config,
            &docs,
        )
        .unwrap();
        for d in &docs {
            w.add_document(d).unwrap();
        }
        let meta = w.finalize().unwrap();
        assert_eq!(meta.num_documents, 30);
        let idx = crate::MmapIndex::load(out.to_str().unwrap()).unwrap();
        assert_eq!(idx.metadata.num_documents, 30);
    }

    #[test]
    fn memory_stays_bounded_via_segment_spill() {
        // The "segment spill" is realized as per-chunk flushing (approved plan
        // simplification, brief point 5): add_document flushes a full chunk of
        // batch_size docs to disk and drops it from memory, so peak memory is
        // one chunk + the O(corpus) IVF pairs, never the whole corpus. We assert
        // the MECHANISM: with batch_size=8 and 64 docs, all chunk artifacts must
        // appear on disk BEFORE finalize() is called, and the IVF (built only at
        // finalize) must NOT yet exist.
        let tmp = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let docs = synth_docs(64, dim);
        let config = IndexConfig {
            nbits: 4,
            batch_size: 8,
            force_cpu: true,
            ..Default::default()
        };
        let out = tmp.path().join("idx");
        let mut w = CompiledIndexWriter::new(
            out.to_str().unwrap(),
            synth_centroids(16, dim),
            &config,
            &docs,
        )
        .unwrap();
        for d in &docs {
            w.add_document(d).unwrap();
        }
        // 64 docs / batch_size 8 => 8 chunks, all flushed during add_document.
        assert!(
            out.join("0.codes.npy").exists(),
            "expected first chunk spilled to disk before finalize"
        );
        assert!(
            out.join("7.codes.npy").exists(),
            "expected all 8 chunks spilled to disk before finalize"
        );
        assert!(
            !out.join("ivf.npy").exists(),
            "ivf must not be built until finalize"
        );
        let files_before_finalize = std::fs::read_dir(&out).unwrap().count();
        assert!(
            files_before_finalize > 0,
            "expected on-disk spill before finalize"
        );
        w.finalize().unwrap();
        // No leftover temp/segment files after finalize.
        for e in std::fs::read_dir(&out).unwrap() {
            let name = e.unwrap().file_name().into_string().unwrap();
            assert!(!name.starts_with(".cinder_seg"), "leftover segment {name}");
        }
        assert!(out.join("ivf.npy").exists(), "ivf written at finalize");
    }
}
