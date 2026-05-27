use crate::search::code_tokenizer::CodeTokenizer;
use crate::types::ScoredChunkId;
use anyhow::{Context, Result};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    FAST, Field, INDEXED, IndexRecordOption, STORED, Schema, TantivyDocument, TextFieldIndexing,
    TextOptions, Value,
};
use tantivy::tokenizer::{Stemmer, TextAnalyzer};
use tantivy::{Index, IndexReader, IndexSettings, IndexWriter, ReloadPolicy};

/// BM25 sparse search using tantivy
pub struct SparseIndex {
    index: Index,
    reader: IndexReader,
    chunk_id_field: Field,
    content_field: Field,
    path_field: Field,
}

/// Register the code-aware tokenizer on an index.
///
/// `use_stemmer` toggles the English Snowball stemmer filter (v0.4 Item 18).
/// `CodeTokenizer` already emits lowercased sub-tokens, so no `LowerCaser`
/// filter is needed.
///
/// Tantivy 0.26 `TextAnalyzerBuilder::filter` returns a builder whose
/// generic parameter encodes the filter chain at the type level, which
/// makes a conditional `if`-branch produce two incompatible types. We use
/// `filter_dynamic` to box the filter and keep the builder type stable
/// across both branches. The performance cost is negligible at index-build
/// rates (analyzer is invoked once per document, not per query).
fn register_code_tokenizer(index: &Index, use_stemmer: bool) {
    let mut analyzer_builder = TextAnalyzer::builder(CodeTokenizer).dynamic();
    if use_stemmer {
        analyzer_builder = analyzer_builder.filter_dynamic(Stemmer::default());
    }
    let code_analyzer = analyzer_builder.build();
    index.tokenizers().register("code", code_analyzer);
}

/// Build the schema with the code-aware tokenizer for content and file_path fields.
fn build_schema() -> (Schema, Field, Field, Field) {
    let mut schema_builder = Schema::builder();
    let chunk_id_field = schema_builder.add_u64_field("chunk_id", INDEXED | STORED | FAST);
    let text_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("code")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    let content_field = schema_builder.add_text_field("content", text_options.clone());
    let path_field = schema_builder.add_text_field("file_path", text_options);
    let schema = schema_builder.build();
    (schema, chunk_id_field, content_field, path_field)
}

impl SparseIndex {
    /// Create a new tantivy index at the given path.
    ///
    /// `use_stemmer` controls whether the English Snowball stemmer filter is
    /// applied to indexed tokens (v0.4 Item 18). The same value MUST be passed
    /// to `open` on subsequent loads; mismatched stemmer state between build
    /// and search produces query/index token mismatches (silent recall loss).
    /// See `docs/CONFIGURATION.md` (`use_bm25_stemmer`) for the limitation.
    pub fn create(index_path: &Path, use_stemmer: bool) -> Result<Self> {
        std::fs::create_dir_all(index_path)?;

        let (schema, chunk_id_field, content_field, path_field) = build_schema();

        let dir = tantivy::directory::MmapDirectory::open(index_path)?;
        let index = Index::create(dir, schema, IndexSettings::default())?;
        register_code_tokenizer(&index, use_stemmer);
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            chunk_id_field,
            content_field,
            path_field,
        })
    }

    /// Open an existing tantivy index.
    ///
    /// `use_stemmer` MUST match the value passed to `create` when the index
    /// was built. See `create` for the rationale.
    pub fn open(index_path: &Path, use_stemmer: bool) -> Result<Self> {
        let (_, chunk_id_field, content_field, path_field) = build_schema();

        let dir = tantivy::directory::MmapDirectory::open(index_path)?;
        let index = Index::open(dir)?;
        register_code_tokenizer(&index, use_stemmer);
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            chunk_id_field,
            content_field,
            path_field,
        })
    }

    /// Get a writer for batch operations
    pub fn writer(&self) -> Result<SparseIndexWriter> {
        let writer = self.index.writer(50_000_000)?; // 50MB buffer
        Ok(SparseIndexWriter {
            writer,
            chunk_id_field: self.chunk_id_field,
            content_field: self.content_field,
            path_field: self.path_field,
        })
    }

    /// Search the index using BM25
    #[tracing::instrument(skip(self, query), fields(query_len = query.len(), top_k))]
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<ScoredChunkId>> {
        let start = std::time::Instant::now();
        let searcher = self.reader.searcher();
        let mut query_parser =
            QueryParser::for_index(&self.index, vec![self.content_field, self.path_field]);
        query_parser.set_field_boost(self.path_field, 2.0);
        let query = query_parser
            .parse_query(query)
            .with_context(|| "Failed to parse query".to_string())?;

        // tantivy 0.26: `TopDocs::with_limit` returns a `TopDocs`, not a `Collector`.
        // Call `.order_by_score()` to obtain a Collector emitting `(Score, DocAddress)`
        // pairs (the same fruit type the previous `with_limit`-as-Collector produced).
        let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k).order_by_score())?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(chunk_id) = doc.get_first(self.chunk_id_field)
                && let Some(id) = chunk_id.as_u64()
            {
                results.push(ScoredChunkId::new(id, score));
            }
        }

        let duration = start.elapsed();
        tracing::debug!(
            top_k,
            results_count = results.len(),
            duration_ms = duration.as_millis() as u64,
            "BM25 sparse search complete"
        );

        Ok(results)
    }

    /// Reload the reader to see committed changes
    pub fn reload(&self) -> Result<()> {
        self.reader.reload()?;
        Ok(())
    }
}

/// Writer handle for batch document insertion
pub struct SparseIndexWriter {
    writer: IndexWriter,
    chunk_id_field: Field,
    content_field: Field,
    path_field: Field,
}

impl SparseIndexWriter {
    /// Add a document to the index
    pub fn add_document(&mut self, chunk_id: u64, content: &str, file_path: &str) -> Result<()> {
        let mut doc = TantivyDocument::new();
        doc.add_u64(self.chunk_id_field, chunk_id);
        doc.add_text(self.content_field, content);
        doc.add_text(self.path_field, file_path);
        self.writer.add_document(doc)?;
        Ok(())
    }

    /// Delete documents by chunk IDs
    pub fn delete_documents(&mut self, chunk_ids: &[u64]) -> Result<()> {
        for &id in chunk_ids {
            let term = tantivy::Term::from_field_u64(self.chunk_id_field, id);
            self.writer.delete_term(term);
        }
        Ok(())
    }

    /// Commit pending changes
    pub fn commit(mut self) -> Result<()> {
        self.writer.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Helper: build a fresh index with three documents (`retry`, `retries`,
    /// `unrelated`), commit, reload, and return the index ready to query.
    fn build_index_with_retry_corpus(index_path: &Path, use_stemmer: bool) -> SparseIndex {
        let index = SparseIndex::create(index_path, use_stemmer).expect("create");
        {
            let mut writer = index.writer().expect("writer");
            writer
                .add_document(1, "function retry handler implementation", "a.rs")
                .expect("add 1");
            writer
                .add_document(2, "the retries counter is incremented here", "b.rs")
                .expect("add 2");
            writer
                .add_document(3, "wholly unrelated content about cats", "c.rs")
                .expect("add 3");
            writer.commit().expect("commit");
        }
        index.reload().expect("reload");
        index
    }

    /// v0.4 Item 18 acceptance criterion (spec §9.2.4):
    /// "Setting `use_bm25_stemmer: false` MUST produce a tantivy index in
    ///  which `retry` and `retries` are NOT conflated."
    ///
    /// With stemmer on: query `retries` matches both doc 1 (`retry`, stemmed
    /// to `retri`) and doc 2 (`retries`, stemmed to `retri`).
    /// With stemmer off: query `retries` matches ONLY doc 2 (`retries`
    /// literal), and a query for `retry` matches ONLY doc 1.
    #[test]
    fn stemmer_flag_controls_retry_retries_conflation() {
        let dir_on = tempdir().expect("tempdir on");
        let dir_off = tempdir().expect("tempdir off");

        let index_on = build_index_with_retry_corpus(dir_on.path(), true);
        let index_off = build_index_with_retry_corpus(dir_off.path(), false);

        let hits_on_retries = index_on.search("retries", 10).expect("search on");
        let hits_off_retries = index_off.search("retries", 10).expect("search off");

        let ids_on: std::collections::BTreeSet<u64> =
            hits_on_retries.iter().map(|s| s.chunk_id).collect();
        let ids_off: std::collections::BTreeSet<u64> =
            hits_off_retries.iter().map(|s| s.chunk_id).collect();

        // Stemmer ON: query 'retries' conflates with 'retry' -> both docs hit.
        assert!(
            ids_on.contains(&1) && ids_on.contains(&2),
            "stemmer on: expected 'retries' query to match docs 1 (retry) and 2 (retries), got {:?}",
            ids_on
        );

        // Stemmer OFF: query 'retries' must NOT match doc 1 ('retry' literal).
        assert!(
            !ids_off.contains(&1),
            "stemmer off: 'retries' must NOT conflate with 'retry' (doc 1), got {:?}",
            ids_off
        );
        assert!(
            ids_off.contains(&2),
            "stemmer off: 'retries' must still match its own literal in doc 2, got {:?}",
            ids_off
        );

        // The two result sets MUST differ — this is the core acceptance signal.
        assert_ne!(
            ids_on, ids_off,
            "stemmer flag must produce different result sets (on={:?}, off={:?})",
            ids_on, ids_off
        );
    }

    /// Symmetric check: with stemmer OFF, the literal `retry` must not pull
    /// in the `retries` document either.
    #[test]
    fn stemmer_off_keeps_retry_and_retries_disjoint() {
        let dir = tempdir().expect("tempdir");
        let index = build_index_with_retry_corpus(dir.path(), false);

        let hits_retry: std::collections::BTreeSet<u64> = index
            .search("retry", 10)
            .expect("search retry")
            .iter()
            .map(|s| s.chunk_id)
            .collect();

        assert!(
            hits_retry.contains(&1),
            "'retry' must match doc 1 (literal 'retry'), got {:?}",
            hits_retry
        );
        assert!(
            !hits_retry.contains(&2),
            "stemmer off: 'retry' must NOT match doc 2 ('retries'), got {:?}",
            hits_retry
        );
    }
}
