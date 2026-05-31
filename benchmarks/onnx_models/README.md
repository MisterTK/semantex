# ONNX model prep (S2 embedder / S3 reranker)

One-time ops tool that turns the "host the ONNX models" prerequisite
(`docs/superpowers/plans/2026-05-31-integration-and-cutover.md` §5) into a single
command. It **downloads an existing permissively-licensed community ONNX export →
verifies it → int8-quantizes it → re-hosts it** into a project-controlled HF repo
in the layout semantex's downloader (`embedding/model_manager.rs`) expects
(`model_int8.onnx` + `tokenizer.json` + `config.json`).

We re-host (rather than point production at third-party personal repos) so the
artifact is stable, audited, and can't vanish or change under us.

## Models (verified on HF 2026-05-31)

| Role | Upstream | License | Default source ONNX | int8? |
|------|----------|---------|---------------------|-------|
| embedder (S2) | `nomic-ai/CodeRankEmbed` | **MIT** | `sirasagi62/code-rank-embed-onnx` (fp32) | we quantize |
| embedder alt | — | MIT | `mrsladoje/CodeRankEmbed-onnx-int8` | already int8 → `--no-quantize` |
| reranker (S3) | `Qwen/Qwen3-Reranker-0.6B` | **Apache-2.0** | `shawnw3i/Qwen3-Reranker-0.6B-ONNX` (float, yes/no-logit) | we quantize |
| reranker alt | — | Apache-2.0 | `zhiqing/Qwen3-Reranker-0.6B-ONNX` | we quantize |
| reranker alt (classifier head) | — | Apache-2.0 | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` (PyTorch → export) | simpler `ClassifierLogit` path |

## Setup

```bash
cd benchmarks/onnx_models
python -m venv .venv && source .venv/bin/activate
pip install "huggingface_hub>=0.23" onnx "onnxruntime>=1.17" "tokenizers>=0.15" numpy click
huggingface-cli login        # only needed for --upload
```

## Run

```bash
# Dry run (download → verify → int8 → stage locally; no upload):
python prepare_models.py embedder
python prepare_models.py reranker

# Publish to project-owned repos:
python prepare_models.py embedder --target-repo MisterTK/CodeRankEmbed-onnx-int8 --upload
python prepare_models.py reranker --target-repo MisterTK/Qwen3-Reranker-0.6B-onnx-int8 --upload

# Use the already-int8 embedder export instead of quantizing the fp32 one:
python prepare_models.py embedder --source-repo mrsladoje/CodeRankEmbed-onnx-int8 --no-quantize
```

After upload, set the corresponding S8 `ModelSpec.source = "hf:<target-repo>"` and
record the URL in `docs/superpowers/plans/2026-05-31-research-notes.md`.

## What the smoke verify checks

A CPU `onnxruntime` run on a hand-built (query, relevant-code, irrelevant-text)
triple, asserting the relevant item scores higher — catching a broken/mis-quantized
graph before publishing. For rigorous parity against the PyTorch reference, run the
upstream model (`transformers`/`sentence-transformers`, `trust_remote_code=True` for
CodeRankEmbed) and compare embeddings/scores; that's the gate before flipping any
default.

## License note

Both upstreams are permissive (MIT / Apache-2.0), so re-hosting is allowed. The
generated model card retains the original license and attribution; keep it.
