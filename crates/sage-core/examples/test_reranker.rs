use fastembed::RerankerModel;
use sage_core::search::fastembed_reranker::FastembedReranker;
use sage_core::types::{Chunk, ChunkType};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("Testing fastembed reranker integration...\n");

    // Create a reranker with JINA model
    println!("Loading JINA Reranker V2 model...");
    let mut reranker = FastembedReranker::new(RerankerModel::JINARerankerV2BaseMultiligual, true)?;
    println!("Model loaded successfully!\n");

    // Create test chunks with varying relevance
    let test_query = "How to implement binary search in Rust?";
    let test_chunks = [
        Chunk {
            id: 1,
            file_path: PathBuf::from("test.rs"),
            content: "The giant panda is a bear species endemic to China.".to_string(),
            start_line: 1,
            end_line: 1,
            chunk_type: ChunkType::TextWindow { window_index: 0 },
        },
        Chunk {
            id: 2,
            file_path: PathBuf::from("test.rs"),
            content: "Binary search is an efficient algorithm for finding an item in a sorted array by repeatedly dividing the search interval in half.".to_string(),
            start_line: 2,
            end_line: 2,
            chunk_type: ChunkType::TextWindow { window_index: 1 },
        },
        Chunk {
            id: 3,
            file_path: PathBuf::from("test.rs"),
            content: "Rust provides the binary_search method on slices for efficient searching in sorted data.".to_string(),
            start_line: 3,
            end_line: 3,
            chunk_type: ChunkType::TextWindow { window_index: 2 },
        },
        Chunk {
            id: 4,
            file_path: PathBuf::from("test.rs"),
            content: "Pizza is a popular Italian dish consisting of a usually round, flat base of leavened wheat-based dough.".to_string(),
            start_line: 4,
            end_line: 4,
            chunk_type: ChunkType::TextWindow { window_index: 3 },
        },
        Chunk {
            id: 5,
            file_path: PathBuf::from("test.rs"),
            content: "To implement binary search in Rust, you define a function that takes a sorted slice and a target value.".to_string(),
            start_line: 5,
            end_line: 5,
            chunk_type: ChunkType::TextWindow { window_index: 4 },
        },
    ];

    println!("Query: \"{test_query}\"");
    println!("\nTest chunks:");
    for (i, chunk) in test_chunks.iter().enumerate() {
        println!(
            "  {}. {}",
            i + 1,
            &chunk.content[..chunk.content.len().min(80)]
        );
    }

    // Rerank
    println!("\nReranking...");
    let docs: Vec<&str> = test_chunks.iter().map(|c| c.content.as_str()).collect();
    let results = reranker.rerank(test_query, &docs, 3)?;

    println!("\nTop 3 results after reranking:");
    for (rank, (idx, score)) in results.iter().enumerate() {
        println!(
            "  {}. [Score: {:.4}] {}",
            rank + 1,
            score,
            &test_chunks[*idx].content[..test_chunks[*idx].content.len().min(80)]
        );
    }

    // Verify that relevant chunks are ranked higher
    assert!(
        results[0].0 == 4 || results[0].0 == 2 || results[0].0 == 1,
        "Top result should be one of the relevant chunks about binary search"
    );

    println!("\n✓ Reranking test passed! Relevant results are ranked higher.");
    Ok(())
}
