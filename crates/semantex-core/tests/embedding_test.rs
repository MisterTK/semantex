use anyhow::Result;
use semantex_core::embedding::colbert::ColbertEmbedder;
use semantex_core::embedding::model_manager;
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    semantex_core::config::SemantexConfig::default().models_dir()
}

#[test]
fn test_colbert_query_encoding() -> Result<()> {
    let model_dir = model_manager::ensure_colbert_model(&models_dir())?;
    let embedder = ColbertEmbedder::new(&model_dir)?;

    let code_snippet = "fn add(a: i32, b: i32) -> i32 { a + b }";
    let query_emb = embedder.encode_query(code_snippet)?;

    // ColBERT produces [N_tokens, 48] — at least 1 token, 48d per token
    assert!(
        query_emb.nrows() >= 1,
        "Should produce at least 1 token embedding, got {}",
        query_emb.nrows()
    );
    assert_eq!(
        query_emb.ncols(),
        48,
        "Embedding dimension should be 48, got {}",
        query_emb.ncols()
    );

    println!(
        "✓ ColBERT query encoding: {} tokens × {}d",
        query_emb.nrows(),
        query_emb.ncols()
    );
    Ok(())
}

#[test]
fn test_colbert_document_encoding() -> Result<()> {
    let model_dir = model_manager::ensure_colbert_model(&models_dir())?;
    let embedder = ColbertEmbedder::new(&model_dir)?;

    let code_snippets = vec![
        "fn add(a: i32, b: i32) -> i32 { a + b }".to_string(),
        "def multiply(x, y): return x * y".to_string(),
        "function subtract(a, b) { return a - b; }".to_string(),
    ];

    let embeddings = embedder.encode_documents(&code_snippets)?;

    assert_eq!(embeddings.len(), 3, "Should return 3 document embeddings");
    for (i, emb) in embeddings.iter().enumerate() {
        assert!(emb.nrows() >= 1, "Doc {i} should have at least 1 token");
        assert_eq!(emb.ncols(), 48, "Doc {i} should have 48d per token");
    }

    println!(
        "✓ ColBERT document encoding: {} documents",
        embeddings.len()
    );
    Ok(())
}

#[test]
fn test_colbert_global_singleton() -> Result<()> {
    let model_dir = model_manager::ensure_colbert_model(&models_dir())?;

    // First call initializes
    let emb1 = ColbertEmbedder::global(&model_dir)?;
    // Second call returns same instance
    let emb2 = ColbertEmbedder::global(&model_dir)?;

    // Should be the same pointer
    assert!(
        std::ptr::eq(emb1, emb2),
        "global() should return the same singleton"
    );

    println!("✓ ColBERT global singleton test passed");
    Ok(())
}
