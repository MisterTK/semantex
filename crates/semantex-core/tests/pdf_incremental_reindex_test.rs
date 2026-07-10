//! Regression test for a latent bug found during a post-merge review sweep:
//! `IndexBuilder`'s per-file loop calls `pdf_chunker.chunk(file_path, "")`
//! with the chunker's absolute filesystem path (needed so the chunker can
//! open the file itself), and the PDF chunker stamps that same absolute path
//! onto every `Chunk::file_path` it emits. Every other file type is chunked
//! from pre-read `content` and stamped with the project-relative `rel_path`
//! instead. Since the chunk store's hash lookup / deletion / BM25 doc id all
//! key on `rel_path`, a PDF's chunks were stored under a path nothing else
//! ever looks up by — so `delete_chunks_for_file(rel_path)` found zero rows
//! on the next reindex and stale PDF chunks were never cleaned up.
//!
//! `#[ignore]` like this crate's other full-index-build integration tests
//! (see `semantic_cache_reindex_test.rs`) — building a real index requires
//! the ONNX runtime / dense embedder. Run explicitly with
//! `cargo test --test pdf_incremental_reindex_test -- --ignored`.

use lopdf::content::{Content, Operation};
use lopdf::{Document, Object, Stream, dictionary};
use rusqlite::Connection;
use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use std::path::Path;

fn write_pdf(path: &Path, text: &str) {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Courier",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! { "F1" => font_id },
    });
    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![50.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal(text)]),
            Operation::new("ET", vec![]),
        ],
    };
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
    });
    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![page_id.into()],
        "Count" => 1,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    doc.save(path).unwrap();
}

fn chunk_rows(chunks_db: &Path) -> Vec<(String, String)> {
    let conn = Connection::open(chunks_db).unwrap();
    let mut stmt = conn
        .prepare("SELECT file_path, content FROM chunks")
        .unwrap();
    stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })
    .unwrap()
    .collect::<Result<Vec<_>, _>>()
    .unwrap()
}

#[test]
#[ignore = "requires ONNX runtime + builds full index; run explicitly with ORT_DYLIB_PATH set"]
fn pdf_chunks_are_stamped_with_relative_path_and_cleaned_up_on_reindex() {
    let tmp = tempfile::TempDir::new().unwrap();
    let project_dir = tmp.path();
    let config = SemantexConfig::default();
    let chunks_db = project_dir.join(".semantex").join("chunks.db");

    write_pdf(&project_dir.join("doc.pdf"), "AlphaOriginalContentMarker");
    IndexBuilder::new(&config)
        .unwrap()
        .build(project_dir)
        .unwrap();

    let rows_v1 = chunk_rows(&chunks_db);
    assert!(!rows_v1.is_empty(), "PDF must produce at least one chunk");
    assert!(
        rows_v1.iter().all(|(path, _)| path == "doc.pdf"),
        "chunk.file_path must be the project-relative path (\"doc.pdf\"), not the \
         chunker's absolute filesystem path; got: {rows_v1:?}"
    );

    // Overwrite the PDF with different content and reindex.
    write_pdf(&project_dir.join("doc.pdf"), "BetaReplacementContentMarker");
    IndexBuilder::new(&config)
        .unwrap()
        .build(project_dir)
        .unwrap();

    let rows_v2 = chunk_rows(&chunks_db);
    assert!(
        rows_v2
            .iter()
            .all(|(_, c)| !c.contains("AlphaOriginalContentMarker")),
        "stale v1 PDF chunk must be deleted on reindex, found: {rows_v2:?}"
    );
    assert!(
        rows_v2
            .iter()
            .any(|(_, c)| c.contains("BetaReplacementContentMarker")),
        "v2 PDF content must be present after reindex, found: {rows_v2:?}"
    );
}
