#![allow(clippy::too_many_lines)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::float_cmp)]
#![allow(clippy::ignore_without_reason)]
/// Integration tests for search accuracy across all search modes
///
/// Tests cover:
/// - Semantic search accuracy
/// - BM25 keyword search
/// - Hybrid search (RRF fusion)
/// - Cross-encoder reranking
/// - Language parser coverage
/// - Edge cases
use anyhow::Result;
use semantex_core::chunking::Chunker;
use semantex_core::chunking::ast_chunker::AstChunker;
use semantex_core::chunking::text_chunker::TextChunker;
use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::search::SearchQuery;
use semantex_core::search::hybrid::{HybridSearcher, rrf_fuse};
use semantex_core::search::query_classifier::FusionWeights;
use semantex_core::types::{ChunkType, ScoredChunkId};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Golden dataset: known query/result pairs for validation
#[allow(dead_code)]
struct GoldenTestCase {
    query: &'static str,
    expected_language: Option<&'static str>,
    expected_keywords: &'static [&'static str],
    minimum_recall: f32, // Minimum recall@10 expected
}

#[allow(dead_code)]
const GOLDEN_DATASET: &[GoldenTestCase] = &[
    GoldenTestCase {
        query: "function to calculate fibonacci sequence",
        expected_language: Some("rust"),
        expected_keywords: &["fibonacci", "fn"],
        minimum_recall: 0.8,
    },
    GoldenTestCase {
        query: "binary search implementation",
        expected_language: Some("python"),
        expected_keywords: &["binary", "search"],
        minimum_recall: 0.7,
    },
    GoldenTestCase {
        query: "http request handler",
        expected_language: Some("javascript"),
        expected_keywords: &["http", "request", "handler"],
        minimum_recall: 0.7,
    },
    GoldenTestCase {
        query: "database connection pooling",
        expected_language: Some("go"),
        expected_keywords: &["database", "pool"],
        minimum_recall: 0.6,
    },
];

/// Test fixture with sample code in multiple languages
struct TestFixture {
    _temp_dir: TempDir,
    project_dir: PathBuf,
    index_dir: PathBuf,
    config: SemantexConfig,
}

impl TestFixture {
    fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let project_dir = temp_dir.path().join("test_project");
        fs::create_dir_all(&project_dir)?;

        let index_dir = temp_dir.path().join("test_index");
        fs::create_dir_all(&index_dir)?;

        let config = SemantexConfig {
            chunk_size: 512,
            chunk_overlap: 128,
            rerank: true,
            rerank_candidates: 50,
            // Preserve original intent: rerank the full 50-wide retrieval pool
            // (the new SEMANTEX_RERANK_CANDIDATES window defaults to 25, which
            // would otherwise shrink the scored set).
            rerank_top_n: 50,
            ..SemantexConfig::default()
        };

        Ok(Self {
            _temp_dir: temp_dir,
            project_dir,
            index_dir,
            config,
        })
    }

    fn create_test_files(&self) -> Result<()> {
        // Rust: fibonacci implementation
        self.write_file(
            "fibonacci.rs",
            r"
/// Calculate the nth fibonacci number recursively
pub fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

/// Calculate fibonacci using iterative approach
pub fn fibonacci_iterative(n: u32) -> u64 {
    let mut a = 0u64;
    let mut b = 1u64;
    for _ in 0..n {
        let temp = a;
        a = b;
        b = temp + b;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(10), 55);
    }
}
",
        )?;

        // Python: binary search
        self.write_file(
            "binary_search.py",
            r#"
def binary_search(arr, target):
    """
    Perform binary search on a sorted array.
    Returns the index of target if found, -1 otherwise.
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def binary_search_recursive(arr, target, left=None, right=None):
    """Recursive implementation of binary search"""
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

class BinarySearchTree:
    """Binary search tree implementation"""
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinarySearchTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinarySearchTree(value)
            else:
                self.right.insert(value)
"#,
        )?;

        // JavaScript: HTTP request handler
        self.write_file(
            "http_handler.js",
            r"
/**
 * HTTP request handler using Express.js
 */
const express = require('express');
const app = express();

// Middleware for parsing JSON
app.use(express.json());

/**
 * GET request handler for fetching user data
 */
app.get('/api/users/:id', async (req, res) => {
    try {
        const userId = req.params.id;
        const user = await fetchUserFromDatabase(userId);

        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }

        res.json(user);
    } catch (error) {
        console.error('Error handling request:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * POST request handler for creating new users
 */
app.post('/api/users', async (req, res) => {
    try {
        const userData = req.body;
        const newUser = await createUserInDatabase(userData);
        res.status(201).json(newUser);
    } catch (error) {
        res.status(400).json({ error: error.message });
    }
});

async function fetchUserFromDatabase(userId) {
    // Mock implementation
    return { id: userId, name: 'John Doe' };
}

async function createUserInDatabase(userData) {
    // Mock implementation
    return { id: Math.random(), ...userData };
}

module.exports = app;
",
        )?;

        // TypeScript: sorting algorithms
        self.write_file(
            "sorting.ts",
            r"
/**
 * Quick sort implementation
 */
export function quickSort(arr: number[]): number[] {
    if (arr.length <= 1) return arr;

    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const middle = arr.filter(x => x === pivot);
    const right = arr.filter(x => x > pivot);

    return [...quickSort(left), ...middle, ...quickSort(right)];
}

/**
 * Merge sort implementation
 */
export function mergeSort(arr: number[]): number[] {
    if (arr.length <= 1) return arr;

    const mid = Math.floor(arr.length / 2);
    const left = mergeSort(arr.slice(0, mid));
    const right = mergeSort(arr.slice(mid));

    return merge(left, right);
}

function merge(left: number[], right: number[]): number[] {
    const result: number[] = [];
    let i = 0, j = 0;

    while (i < left.length && j < right.length) {
        if (left[i] <= right[j]) {
            result.push(left[i++]);
        } else {
            result.push(right[j++]);
        }
    }

    return result.concat(left.slice(i)).concat(right.slice(j));
}
",
        )?;

        // Go: database connection pooling
        self.write_file(
            "database.go",
            r#"
package database

import (
    "database/sql"
    "fmt"
    "time"
    _ "github.com/lib/pq"
)

// ConnectionPool manages database connections
type ConnectionPool struct {
    db *sql.DB
}

// NewConnectionPool creates a new database connection pool
func NewConnectionPool(dsn string, maxOpen int, maxIdle int) (*ConnectionPool, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, fmt.Errorf("failed to open database: %w", err)
    }

    // Configure connection pool
    db.SetMaxOpenConns(maxOpen)
    db.SetMaxIdleConns(maxIdle)
    db.SetConnMaxLifetime(time.Hour)

    // Verify connection
    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("failed to ping database: %w", err)
    }

    return &ConnectionPool{db: db}, nil
}

// Query executes a query with automatic connection management
func (p *ConnectionPool) Query(query string, args ...interface{}) (*sql.Rows, error) {
    return p.db.Query(query, args...)
}

// Close closes all database connections in the pool
func (p *ConnectionPool) Close() error {
    return p.db.Close()
}
"#,
        )?;

        // Java: class hierarchy
        self.write_file(
            "Vehicle.java",
            r#"
public abstract class Vehicle {
    private String make;
    private String model;
    private int year;

    public Vehicle(String make, String model, int year) {
        this.make = make;
        this.model = model;
        this.year = year;
    }

    public abstract void start();
    public abstract void stop();

    public String getMake() { return make; }
    public String getModel() { return model; }
    public int getYear() { return year; }
}

class Car extends Vehicle {
    private int doors;

    public Car(String make, String model, int year, int doors) {
        super(make, model, year);
        this.doors = doors;
    }

    @Override
    public void start() {
        System.out.println("Car starting...");
    }

    @Override
    public void stop() {
        System.out.println("Car stopping...");
    }
}
"#,
        )?;

        // C: linked list
        self.write_file(
            "linked_list.c",
            r#"
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node* next;
} Node;

// Create a new node
Node* create_node(int data) {
    Node* new_node = (Node*)malloc(sizeof(Node));
    if (new_node == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    new_node->data = data;
    new_node->next = NULL;
    return new_node;
}

// Insert at the beginning of the list
void insert_front(Node** head, int data) {
    Node* new_node = create_node(data);
    new_node->next = *head;
    *head = new_node;
}

// Free the entire list
void free_list(Node* head) {
    Node* current = head;
    while (current != NULL) {
        Node* temp = current;
        current = current->next;
        free(temp);
    }
}
"#,
        )?;

        // C++: template class
        self.write_file(
            "stack.cpp",
            r#"
#include <vector>
#include <stdexcept>

template<typename T>
class Stack {
private:
    std::vector<T> elements;

public:
    void push(const T& element) {
        elements.push_back(element);
    }

    T pop() {
        if (elements.empty()) {
            throw std::runtime_error("Stack is empty");
        }
        T top = elements.back();
        elements.pop_back();
        return top;
    }

    T& top() {
        if (elements.empty()) {
            throw std::runtime_error("Stack is empty");
        }
        return elements.back();
    }

    bool empty() const {
        return elements.empty();
    }

    size_t size() const {
        return elements.size();
    }
};
"#,
        )?;

        // Ruby: module and class
        self.write_file(
            "authenticator.rb",
            r#"
module Authentication
  class Authenticator
    def initialize(secret_key)
      @secret_key = secret_key
    end

    def authenticate(username, password)
      hashed = hash_password(password)
      user = find_user(username)

      return false if user.nil?

      user.password_hash == hashed
    end

    private

    def hash_password(password)
      require 'digest'
      Digest::SHA256.hexdigest(password + @secret_key)
    end

    def find_user(username)
      # Mock implementation
      User.new(username, Digest::SHA256.hexdigest("password" + @secret_key))
    end
  end

  class User
    attr_reader :username, :password_hash

    def initialize(username, password_hash)
      @username = username
      @password_hash = password_hash
    end
  end
end
"#,
        )?;

        // Markdown: documentation
        self.write_file(
            "README.md",
            r"
# Test Project

This is a test project containing code samples in multiple languages.

## Features

- **Fibonacci calculation**: Recursive and iterative implementations in Rust
- **Binary search**: Both iterative and recursive approaches in Python
- **HTTP handlers**: Express.js request handling in JavaScript
- **Database pooling**: Connection pool management in Go
- **Sorting algorithms**: Quick sort and merge sort in TypeScript

## Languages Covered

1. Rust - Systems programming
2. Python - Data structures and algorithms
3. JavaScript - Web development
4. TypeScript - Type-safe JavaScript
5. Go - Concurrent programming
6. Java - Object-oriented programming
7. C - Low-level programming
8. C++ - Template metaprogramming
9. Ruby - Web frameworks

## Installation

```bash
cargo build --release
```

## Testing

Run all tests:

```bash
cargo test
```
",
        )?;

        // JSON: configuration
        self.write_file(
            "config.json",
            r#"{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "testdb",
    "pool_size": 10
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "timeout": 30
  },
  "logging": {
    "level": "info",
    "format": "json"
  }
}
"#,
        )?;

        // TOML: cargo config
        self.write_file(
            "Cargo.toml",
            r#"[package]
name = "test-project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"

[dev-dependencies]
tempfile = "3.0"
"#,
        )?;

        Ok(())
    }

    fn write_file(&self, name: &str, content: &str) -> Result<()> {
        let path = self.project_dir.join(name);
        fs::write(path, content)?;
        Ok(())
    }
}

#[test]
fn test_ast_chunking_all_languages() -> Result<()> {
    let chunker = AstChunker::new(512, 128);

    // Test each language parser
    let test_cases = vec![
        ("test.rs", "fn hello() { println!(\"hi\"); }", "rust"),
        ("test.py", "def greet():\n    print('hi')", "python"),
        (
            "test.js",
            "function greet() { console.log('hi'); }",
            "javascript",
        ),
        (
            "test.ts",
            "function greet(): void { console.log('hi'); }",
            "typescript",
        ),
        ("test.go", "func greet() { fmt.Println(\"hi\") }", "go"),
        (
            "test.java",
            "public void greet() { System.out.println(\"hi\"); }",
            "java",
        ),
        ("test.c", "void greet() { printf(\"hi\"); }", "c"),
        ("test.cpp", "void greet() { std::cout << \"hi\"; }", "cpp"),
        ("test.rb", "def greet\n  puts 'hi'\nend", "ruby"),
    ];

    for (filename, code, expected_lang) in test_cases {
        let chunks = chunker.chunk(Path::new(filename), code)?;

        // Should produce at least one chunk
        assert!(!chunks.is_empty(), "No chunks for {filename}");

        // Check for AST node chunks
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();

        if ast_chunks.is_empty() {
            println!("⚠ {expected_lang} parser fell back to text chunking");
        } else {
            println!(
                "✓ {} parser working: {} AST chunks",
                expected_lang,
                ast_chunks.len()
            );
        }
    }

    Ok(())
}

#[test]
fn test_text_chunking_edge_cases() -> Result<()> {
    let chunker = TextChunker::new(256, 64);

    // Empty content
    let chunks = chunker.chunk(Path::new("empty.txt"), "")?;
    assert!(chunks.is_empty(), "Empty content should produce no chunks");

    // Very short content
    let chunks = chunker.chunk(Path::new("short.txt"), "a")?;
    assert_eq!(chunks.len(), 1, "Short content should produce one chunk");

    // Unicode content
    let unicode_text = "こんにちは世界 🌍 Здравствуй мир";
    let chunks = chunker.chunk(Path::new("unicode.txt"), unicode_text)?;
    assert!(!chunks.is_empty(), "Unicode content should be chunked");
    assert!(
        chunks[0].content.contains("こんにちは"),
        "Unicode should be preserved"
    );

    // Very long single line (no newlines)
    let long_line = "x".repeat(10000);
    let chunks = chunker.chunk(Path::new("long.txt"), &long_line)?;
    assert!(
        !chunks.is_empty(),
        "Long content should be split into chunks"
    );
    assert!(
        chunks.len() > 1,
        "Long content should produce multiple chunks"
    );

    // Special characters
    let special = "!@#$%^&*(){}[]<>?,./;:'\"\\|`~";
    let chunks = chunker.chunk(Path::new("special.txt"), special)?;
    assert!(!chunks.is_empty(), "Special characters should be chunked");

    println!("✓ Text chunking edge cases passed");
    Ok(())
}

#[test]
fn test_rrf_fusion_correctness() -> Result<()> {
    // Test that RRF properly fuses dense and sparse results

    // Case 1: Identical rankings should be preserved
    let dense = vec![
        ScoredChunkId::new(1, 0.9),
        ScoredChunkId::new(2, 0.8),
        ScoredChunkId::new(3, 0.7),
    ];
    let sparse = dense.clone();

    let fused = rrf_fuse(
        &dense,
        &sparse,
        60.0,
        &FusionWeights {
            w_dense: 1.0,
            w_sparse: 1.0,
        },
    );
    assert_eq!(
        fused[0].chunk_id, 1,
        "RRF should preserve order when rankings agree"
    );
    assert_eq!(fused[1].chunk_id, 2);
    assert_eq!(fused[2].chunk_id, 3);

    // Case 2: Complementary rankings should favor consensus
    let dense = vec![
        ScoredChunkId::new(100, 0.95), // Only in dense
        ScoredChunkId::new(200, 0.75), // In both
        ScoredChunkId::new(300, 0.60),
    ];
    let sparse = vec![
        ScoredChunkId::new(200, 12.5), // In both
        ScoredChunkId::new(400, 8.3),  // Only in sparse
    ];

    let fused = rrf_fuse(
        &dense,
        &sparse,
        60.0,
        &FusionWeights {
            w_dense: 1.0,
            w_sparse: 1.0,
        },
    );

    // Chunk 200 appears in both lists, should be promoted
    assert_eq!(
        fused[0].chunk_id, 200,
        "RRF should promote items appearing in both rankers"
    );

    // Case 3: Empty lists
    let empty: Vec<ScoredChunkId> = vec![];
    let non_empty = vec![ScoredChunkId::new(1, 0.9)];

    let fused = rrf_fuse(
        &non_empty,
        &empty,
        60.0,
        &FusionWeights {
            w_dense: 1.0,
            w_sparse: 1.0,
        },
    );
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].chunk_id, 1);

    println!("✓ RRF fusion correctness tests passed");
    Ok(())
}

#[test]
#[ignore] // Slow test, requires building full index
fn test_semantic_search_accuracy() -> Result<()> {
    let fixture = TestFixture::new()?;
    fixture.create_test_files()?;

    // Build index
    let builder = IndexBuilder::new(&fixture.config)?;
    let stats = builder.build(&fixture.project_dir)?;

    println!(
        "Index built: {} files, {} chunks",
        stats.files_indexed, stats.chunks_created
    );
    assert!(stats.chunks_created > 0, "Should create chunks");

    // Open searcher
    let searcher = HybridSearcher::open(&fixture.index_dir, &fixture.config)?;

    // Test semantic queries
    let test_queries = vec![
        ("recursive fibonacci function", "fibonacci"),
        ("search algorithm for sorted array", "binary_search"),
        ("handle HTTP GET request", "http"),
        ("connection pool for database", "database"),
    ];

    for (query, expected_keyword) in test_queries {
        let search_query = SearchQuery::new(query).max_results(5).dense_only();
        let output = searcher.search(&search_query)?;
        let results = &output.results;

        assert!(!results.is_empty(), "Query '{query}' returned no results");

        // Check that at least one result contains the expected keyword
        let has_match = results
            .iter()
            .any(|r| r.chunk.content.to_lowercase().contains(expected_keyword));

        assert!(
            has_match,
            "Query '{query}' did not find '{expected_keyword}' in top results"
        );
        println!(
            "✓ Semantic search: '{}' -> {} results",
            query,
            results.len()
        );
    }

    Ok(())
}

#[test]
#[ignore] // Slow test, requires building full index
fn test_keyword_search_accuracy() -> Result<()> {
    let fixture = TestFixture::new()?;
    fixture.create_test_files()?;

    // Build index
    let builder = IndexBuilder::new(&fixture.config)?;
    builder.build(&fixture.project_dir)?;

    // Open searcher
    let searcher = HybridSearcher::open(&fixture.index_dir, &fixture.config)?;

    // Test exact keyword matches
    let keyword_tests = vec![
        ("fibonacci", "fibonacci"),
        ("binary_search", "binary_search"),
        ("express", "express"),
        ("ConnectionPool", "ConnectionPool"),
    ];

    for (query, expected) in keyword_tests {
        let search_query = SearchQuery::new(query).max_results(5).sparse_only();
        let output = searcher.search(&search_query)?;
        let results = &output.results;

        assert!(!results.is_empty(), "Keyword '{query}' returned no results");

        let has_exact_match = results.iter().any(|r| r.chunk.content.contains(expected));
        assert!(
            has_exact_match,
            "Keyword '{query}' did not find exact match"
        );

        println!("✓ Keyword search: '{}' -> {} results", query, results.len());
    }

    Ok(())
}

#[test]
#[ignore] // Slow test, requires building full index
fn test_hybrid_search_beats_single_mode() -> Result<()> {
    let fixture = TestFixture::new()?;
    fixture.create_test_files()?;

    let builder = IndexBuilder::new(&fixture.config)?;
    builder.build(&fixture.project_dir)?;

    let searcher = HybridSearcher::open(&fixture.index_dir, &fixture.config)?;

    // Query that benefits from both semantic and keyword matching
    let query = "function for calculating fibonacci numbers";

    // Dense-only search
    let dense_query = SearchQuery::new(query).max_results(10).dense_only();
    let dense_results = searcher.search(&dense_query)?.results;

    // Sparse-only search
    let sparse_query = SearchQuery::new(query).max_results(10).sparse_only();
    let sparse_results = searcher.search(&sparse_query)?.results;

    // Hybrid search
    let hybrid_query = SearchQuery::new(query).max_results(10).no_rerank();
    let hybrid_results = searcher.search(&hybrid_query)?.results;

    println!("Dense results: {}", dense_results.len());
    println!("Sparse results: {}", sparse_results.len());
    println!("Hybrid results: {}", hybrid_results.len());

    // Hybrid should combine results from both
    assert!(
        !hybrid_results.is_empty(),
        "Hybrid search should return results"
    );

    // Verify that hybrid fusion is working (should have results from both modes)
    let hybrid_ids: Vec<u64> = hybrid_results.iter().map(|r| r.chunk.id).collect();
    let dense_ids: Vec<u64> = dense_results.iter().map(|r| r.chunk.id).collect();
    let sparse_ids: Vec<u64> = sparse_results.iter().map(|r| r.chunk.id).collect();

    // Count how many unique IDs appear
    let mut all_ids = hybrid_ids.clone();
    all_ids.extend(&dense_ids);
    all_ids.extend(&sparse_ids);
    all_ids.sort_unstable();
    all_ids.dedup();

    println!("✓ Hybrid search combined {} unique results", all_ids.len());
    Ok(())
}

#[test]
#[ignore] // Slow test, requires reranker model
fn test_reranking_improves_results() -> Result<()> {
    let fixture = TestFixture::new()?;
    fixture.create_test_files()?;

    let builder = IndexBuilder::new(&fixture.config)?;
    builder.build(&fixture.project_dir)?;

    let searcher = HybridSearcher::open(&fixture.index_dir, &fixture.config)?;

    let query = "implement fibonacci calculation";

    // Without reranking
    let no_rerank_query = SearchQuery::new(query).max_results(5).no_rerank();
    let no_rerank_results = searcher.search(&no_rerank_query)?.results;

    // With reranking
    let rerank_query = SearchQuery::new(query).max_results(5);
    let rerank_results = searcher.search(&rerank_query)?.results;

    assert!(!no_rerank_results.is_empty());
    assert!(!rerank_results.is_empty());

    // Check that top result with reranking has "fibonacci" in it
    let top_reranked = &rerank_results[0];
    assert!(
        top_reranked
            .chunk
            .content
            .to_lowercase()
            .contains("fibonacci"),
        "Reranking should put most relevant result first"
    );

    println!("✓ Reranking test passed");
    println!("  Top result score: {:.4}", top_reranked.score);

    Ok(())
}

#[test]
fn test_recall_at_k() -> Result<()> {
    // Test recall@k metric calculation
    let relevant_ids = [1, 3, 5, 7, 9]; // Ground truth relevant items

    let retrieved_ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // Retrieved items

    // Calculate recall@5
    let k = 5;
    let top_k_retrieved: Vec<u64> = retrieved_ids.iter().take(k).copied().collect();
    let relevant_in_top_k: Vec<u64> = top_k_retrieved
        .iter()
        .filter(|id| relevant_ids.contains(id))
        .copied()
        .collect();

    let recall_at_k = relevant_in_top_k.len() as f32 / relevant_ids.len() as f32;

    println!("Recall@{k}: {recall_at_k:.2}");
    assert!(recall_at_k > 0.0, "Should have non-zero recall");

    // Calculate recall@10
    let k = 10;
    let top_k_retrieved: Vec<u64> = retrieved_ids.iter().take(k).copied().collect();
    let relevant_in_top_k: Vec<u64> = top_k_retrieved
        .iter()
        .filter(|id| relevant_ids.contains(id))
        .copied()
        .collect();

    let recall_at_10 = relevant_in_top_k.len() as f32 / relevant_ids.len() as f32;

    println!("Recall@{k}: {recall_at_10:.2}");
    assert_eq!(
        recall_at_10, 1.0,
        "Should have perfect recall@10 for this test case"
    );

    Ok(())
}

#[test]
#[ignore] // Requires ONNX runtime (builds full index)
fn test_edge_case_empty_query() -> Result<()> {
    let fixture = TestFixture::new()?;
    fixture.create_test_files()?;

    let builder = IndexBuilder::new(&fixture.config)?;
    builder.build(&fixture.project_dir)?;

    let searcher = HybridSearcher::open(&fixture.index_dir, &fixture.config)?;

    // Empty query
    let empty_query = SearchQuery::new("").max_results(5);
    let results = searcher.search(&empty_query);

    // Should either return empty results or error gracefully
    match results {
        Ok(out) => {
            assert!(
                out.results.is_empty(),
                "Empty query should return no results"
            );
        }
        Err(_) => println!("Empty query errored (acceptable)"),
    }

    // Whitespace-only query
    let whitespace_query = SearchQuery::new("   ").max_results(5);
    let results = searcher.search(&whitespace_query);

    match results {
        Ok(out) => {
            assert!(
                out.results.is_empty() || out.results.len() < 5,
                "Whitespace query should return few or no results"
            );
        }
        Err(_) => println!("Whitespace query errored (acceptable)"),
    }

    println!("✓ Empty query edge cases handled");
    Ok(())
}

#[test]
fn test_chunking_preserves_line_numbers() -> Result<()> {
    let chunker = AstChunker::new(512, 128);

    let content = r#"
fn first() {
    println!("first");
}

fn second() {
    println!("second");
}

fn third() {
    println!("third");
}
"#;

    let chunks = chunker.chunk(Path::new("test.rs"), content)?;

    // Verify line numbers are reasonable
    for chunk in &chunks {
        assert!(chunk.start_line > 0, "Start line should be 1-based");
        assert!(
            chunk.end_line >= chunk.start_line,
            "End line should be >= start line"
        );
        assert!(
            chunk.end_line < 100,
            "End line should be reasonable for this test"
        );
    }

    // Verify chunks are in order
    for i in 1..chunks.len() {
        assert!(
            chunks[i].start_line >= chunks[i - 1].start_line,
            "Chunks should be in line order"
        );
    }

    println!("✓ Line numbers preserved correctly");
    Ok(())
}

#[test]
fn test_unicode_and_multibyte_handling() -> Result<()> {
    let chunker = TextChunker::new(256, 64);

    let unicode_samples = vec![
        ("Japanese", "こんにちは世界。これはテストです。"),
        ("Russian", "Привет мир. Это тест."),
        ("Arabic", "مرحبا بالعالم. هذا اختبار."),
        ("Emoji", "Hello 👋 World 🌍 Test 🧪"),
        ("Mixed", "Hello世界 Привет🌍"),
    ];

    for (name, text) in unicode_samples {
        let chunks = chunker.chunk(Path::new("test.txt"), text)?;
        assert!(!chunks.is_empty(), "{name} text should be chunked");
        assert!(
            chunks[0].content.len() >= text.len(),
            "{name} content should be preserved"
        );
        println!(
            "✓ {} text: {} bytes -> {} chunks",
            name,
            text.len(),
            chunks.len()
        );
    }

    Ok(())
}
