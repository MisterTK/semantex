#!/bin/bash
# Basic sage usage examples

# 1. Index a project
echo "=== Indexing a project ==="
sage index /path/to/your/project

# 2. Basic search
echo "=== Basic semantic search ==="
sage "authentication logic" /path/to/your/project

# 3. Search with content snippets
echo "=== Search with content ==="
sage --content "database connection pool" /path/to/your/project

# 4. Search with context lines
echo "=== Search with context ==="
sage --content --context 3 "error handling" /path/to/your/project

# 5. Limit number of results
echo "=== Limit results ==="
sage --max-count 5 "API endpoint" /path/to/your/project

# 6. Dense-only search (semantic only)
echo "=== Dense-only search ==="
sage --dense-only "user verification" /path/to/your/project

# 7. Sparse-only search (keyword only)
echo "=== Sparse-only search ==="
sage --sparse-only "authenticate" /path/to/your/project

# 8. Grep mode (exact + BM25, exhaustive)
echo "=== Grep mode ==="
sage -G "ConnectionFactory" /path/to/your/project

# 9. JSON output for scripting
echo "=== JSON output ==="
sage --json "middleware" /path/to/your/project | jq '.[0].path'

# 10. Check index status
echo "=== Check status ==="
sage status /path/to/your/project

# 11. Auto-reindex on file changes
echo "=== Watch mode ==="
sage watch /path/to/your/project
