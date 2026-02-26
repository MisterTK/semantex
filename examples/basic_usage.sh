#!/bin/bash
# Basic semantex usage examples

# 1. Index a project
echo "=== Indexing a project ==="
semantex index /path/to/your/project

# 2. Basic search
echo "=== Basic semantic search ==="
semantex "authentication logic" /path/to/your/project

# 3. Search with content snippets
echo "=== Search with content ==="
semantex --content "database connection pool" /path/to/your/project

# 4. Search with context lines
echo "=== Search with context ==="
semantex --content --context 3 "error handling" /path/to/your/project

# 5. Limit number of results
echo "=== Limit results ==="
semantex --max-count 5 "API endpoint" /path/to/your/project

# 6. Dense-only search (semantic only)
echo "=== Dense-only search ==="
semantex --dense-only "user verification" /path/to/your/project

# 7. Sparse-only search (keyword only)
echo "=== Sparse-only search ==="
semantex --sparse-only "authenticate" /path/to/your/project

# 8. Grep mode (exact + BM25, exhaustive)
echo "=== Grep mode ==="
semantex -G "ConnectionFactory" /path/to/your/project

# 9. JSON output for scripting
echo "=== JSON output ==="
semantex --json "middleware" /path/to/your/project | jq '.[0].path'

# 10. Check index status
echo "=== Check status ==="
semantex status /path/to/your/project

# 11. Auto-reindex on file changes
echo "=== Watch mode ==="
semantex watch /path/to/your/project
