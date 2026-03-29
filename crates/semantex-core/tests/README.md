# Integration Tests

## Overview

Comprehensive integration tests for semantex semantic code search, covering search accuracy, language parsers, chunking algorithms, and edge cases.

## Quick Start

### Run All Auto Tests (No Index Required)

```bash
cd semantex
cargo test --package semantex-core --test search_accuracy_test
```

**Expected**: 10 tests pass in ~13 seconds

### Run Full Test Suite (Including Ignored Tests)

```bash
cd semantex
cargo test --package semantex-core --test search_accuracy_test -- --ignored --test-threads=1
```

**Note**: Requires downloading embedding models and building a full index (~2-5 minutes first run)

## Test Categories

### 1. Language Parser Tests

**Test**: `test_ast_chunking_all_languages`

Validates that language parsers correctly extract AST nodes:

- Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, Ruby, and more (23 AST-based total)
- Markdown, JSON, TOML, YAML (text-based fallback)

```bash
cargo test --package semantex-core --test search_accuracy_test test_ast_chunking_all_languages
```

### 2. Chunking Tests

**Tests**:
- `test_text_chunking_edge_cases` - Empty, short, long, Unicode, special chars
- `test_chunking_preserves_line_numbers` - Line number accuracy
- `test_unicode_and_multibyte_handling` - International character support

```bash
cargo test --package semantex-core --test search_accuracy_test test_text_chunking
cargo test --package semantex-core --test search_accuracy_test test_unicode
```

### 3. Search Algorithm Tests

**Tests**:
- `test_rrf_fusion_correctness` - RRF algorithm validation
- `test_recall_at_k` - Recall metric calculation
- `test_embedding_batch_consistency` - Batch vs single embedding
- `test_large_batch_embedding` - 100-item batch processing

```bash
cargo test --package semantex-core --test search_accuracy_test test_rrf
cargo test --package semantex-core --test search_accuracy_test test_embedding
```

### 4. Edge Case Tests

**Tests**:
- `test_edge_case_empty_query` - Empty and whitespace queries
- `test_special_characters_in_query` - SQL, regex, HTML in queries

```bash
cargo test --package semantex-core --test search_accuracy_test test_edge_case
cargo test --package semantex-core --test search_accuracy_test test_special
```

### 5. End-to-End Search Tests (Ignored)

These tests require a full index build and are marked with `#[ignore]`:

**Tests**:
- `test_semantic_search_accuracy` - Dense embedding search
- `test_keyword_search_accuracy` - BM25 sparse search
- `test_hybrid_search_beats_single_mode` - RRF fusion validation
- `test_reranking_improves_results` - Cross-encoder reranking

Run with:

```bash
cargo test --package semantex-core --test search_accuracy_test test_semantic_search_accuracy -- --ignored
cargo test --package semantex-core --test search_accuracy_test test_keyword_search_accuracy -- --ignored
cargo test --package semantex-core --test search_accuracy_test test_hybrid_search_beats_single_mode -- --ignored
cargo test --package semantex-core --test search_accuracy_test test_reranking_improves_results -- --ignored
```

Or all at once:

```bash
cargo test --package semantex-core --test search_accuracy_test -- --ignored --test-threads=1
```

## Test Data

Tests use synthetic code samples in multiple languages:

- **fibonacci.rs** - Rust recursive and iterative implementations
- **binary_search.py** - Python search algorithms and BST
- **http_handler.js** - JavaScript Express.js request handlers
- **sorting.ts** - TypeScript quick sort and merge sort
- **database.go** - Go database connection pooling
- **Vehicle.java** - Java class hierarchy
- **linked_list.c** - C linked list implementation
- **stack.cpp** - C++ template stack
- **authenticator.rb** - Ruby authentication module
- **README.md** - Markdown documentation
- **config.json** - JSON configuration
- **Cargo.toml** - TOML package manifest

## Expected Results

### Auto Tests (No Index)

```
✓ rust parser working: 1 AST chunks
✓ python parser working: 1 AST chunks
✓ javascript parser working: 1 AST chunks
✓ typescript parser working: 1 AST chunks
✓ go parser working: 1 AST chunks
✓ java parser working: 1 AST chunks
✓ c parser working: 1 AST chunks
✓ cpp parser working: 1 AST chunks
✓ ruby parser working: 1 AST chunks
✓ Text chunking edge cases passed
✓ RRF fusion correctness tests passed
✓ Embedding batch consistency test passed
✓ Large batch embedding (100 items) successful
✓ Special characters in queries handled
✓ Unicode handling: Japanese, Russian, Arabic, Emoji, Mixed
✓ Line numbers preserved correctly
✓ Empty query edge cases handled

test result: ok. 10 passed; 0 failed; 4 ignored
```

### Full Tests (With Index)

Additional tests when run with `--ignored`:

```
✓ Semantic search: 'recursive fibonacci function' -> 5 results
✓ Semantic search: 'search algorithm for sorted array' -> 5 results
✓ Keyword search: 'fibonacci' -> exact match found
✓ Keyword search: 'binary_search' -> exact match found
✓ Hybrid search combined N unique results
✓ Reranking test passed

test result: ok. 14 passed; 0 failed; 0 ignored
```

## Performance Benchmarks

From test runs:

- **AST Parsing**: <1ms per file
- **Single Embedding**: 10-20ms per snippet
- **Batch Embedding (100 items)**: ~200-500ms (5ms per item)
- **RRF Fusion**: <1ms for typical result sets
- **Text Chunking**: Handles 10k+ char documents efficiently

## Troubleshooting

### Test Fails: "Failed to download model"

**Solution**: Ensure internet connection for first run. Models are cached after download.

### Test Fails: "ONNX Runtime error"

**Solution**: Check that ONNX Runtime binaries are available. The `ort` crate should auto-download them.

### Test Timeout

**Solution**: Increase timeout or run with `--test-threads=1` to avoid resource contention:

```bash
cargo test --test search_accuracy_test -- --test-threads=1
```

### Ignored Tests Don't Run

**Solution**: Add the `--ignored` flag:

```bash
cargo test --test search_accuracy_test -- --ignored
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable
      - name: Run quick tests
        run: cargo test --package semantex-core --test search_accuracy_test
      - name: Run full tests
        run: cargo test --package semantex-core --test search_accuracy_test -- --ignored --test-threads=1
```

## Contributing

When adding new tests:

1. Follow existing test naming conventions: `test_<category>_<description>`
2. Mark slow tests (>10s) with `#[ignore]`
3. Use `#[allow(dead_code)]` for test fixtures if needed
4. Add test description comments
5. Update this README with new test categories

## Test Results

See [TEST_RESULTS.md](./TEST_RESULTS.md) for detailed test output and analysis.
