# Contributing to sage

Thanks for your interest in contributing to sage!

## Getting Started

```bash
git clone https://github.com/MisterTK/semantex.git
cd sage
cargo build --release
cargo test --all
```

### Requirements

- Rust 1.91+ (edition 2024)
- ~200MB disk for ColBERT model (downloaded on first `semantex index`)

## Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes
4. Run checks:
   ```bash
   cargo fmt --all
   cargo clippy --all
   cargo test --all
   ```
5. Commit and push
6. Open a pull request

## Code Guidelines

- Run `cargo fmt` and `cargo clippy` before submitting
- Add tests for new features
- Follow existing code style and patterns
- Keep changes focused — one feature or fix per PR

### OSS Quality Rules

sage is used across thousands of diverse codebases. All code in `crates/` must be **repo-agnostic**:

- **No hardcoded paths** — never embed absolute paths in production code
- **No test-repo metadata** — don't add synonyms, heuristics, or boosting rules tailored to specific repositories
- **Synonym table must be universal** — every entry in `query_expander.rs` should help any arbitrary codebase
- **Avoid overly generic tokens** — words like `query`, `create`, `hash` match too broadly

See [CLAUDE.md](CLAUDE.md) for the full checklist.

## Reporting Issues

- Search existing issues before opening a new one
- Include reproduction steps, expected behavior, and actual behavior
- For search quality issues, include the query, expected files, and actual results

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
