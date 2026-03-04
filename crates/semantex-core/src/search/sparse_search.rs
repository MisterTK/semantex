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
fn register_code_tokenizer(index: &Index) {
    let code_analyzer = TextAnalyzer::builder(CodeTokenizer)
        .filter(Stemmer::default()) // English Snowball stemmer
        .build();
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
    /// Create a new tantivy index at the given path
    pub fn create(index_path: &Path) -> Result<Self> {
        std::fs::create_dir_all(index_path)?;

        let (schema, chunk_id_field, content_field, path_field) = build_schema();

        let dir = tantivy::directory::MmapDirectory::open(index_path)?;
        let index = Index::create(dir, schema, IndexSettings::default())?;
        register_code_tokenizer(&index);
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

    /// Open an existing tantivy index
    pub fn open(index_path: &Path) -> Result<Self> {
        let (_, chunk_id_field, content_field, path_field) = build_schema();

        let dir = tantivy::directory::MmapDirectory::open(index_path)?;
        let index = Index::open(dir)?;
        register_code_tokenizer(&index);
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

        let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k))?;

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
