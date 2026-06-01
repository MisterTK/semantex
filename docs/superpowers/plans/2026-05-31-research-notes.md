# SOTA Overhaul — Spike Research Notes

> Append-only, shared by the stream spikes. The embedder (S2) and reranker (S3) ONNX
> artifacts were exported/selected, verified, and **HOSTED ahead of the build on 2026-06-01**
> via `benchmarks/onnx_models/prepare_models.py`. The S2/S3 spike tasks reduce to "verify the
> recorded facts below + wire the source," not "produce them."

## S2 — CodeRankEmbed embedder (DONE: hosted + verified)

- **ModelSpec.source:** `hf:MisterTK/CodeRankEmbed-onnx-int8` (MIT)
- **Upstream:** `nomic-ai/CodeRankEmbed` (MIT), base `Snowflake/snowflake-arctic-embed-m-long`; arch `nomic_bert` (custom_code, but baked into the ONNX graph — Rust `ort` needs no Python).
- **Precision/files:** int8 dynamic quant, **ONNX external-data format** → the download file list MUST be `["model_int8.onnx", "model_int8.onnx.data", "tokenizer.json", "config.json"]` (the `.onnx` graph is ~1.2 MB; the `.onnx.data` weights are ~137 MB — both must be co-located for `ort` to load).
- **Embedding:** dim **768**; pooling **mean** (mask-weighted) → **L2-normalize**; query prefix **`Represent this query for searching relevant code: `** (documents/code get NO prefix); max context 8192. **No Matryoshka** (fixed 768-dim).
- **ONNX I/O:** inputs `input_ids`, `attention_mask` (int64) [+ `token_type_ids`→zeros if the graph declares it]; output `last_hidden_state [batch, seq, hidden]` → mean-pool (if a future export already pools to `[batch, hidden]`, use directly).
- **Verified (CPU smoke):** sim(query, relevant code) = **0.656** vs sim(query, unrelated) = **0.104**.

## S3 — Qwen3-Reranker-0.6B reranker (DONE: hosted + verified)

- **ModelSpec.source:** `hf:MisterTK/Qwen3-Reranker-0.6B-onnx` (Apache-2.0)
- **Upstream:** `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0); re-hosted as-is from community export `shawnw3i/Qwen3-Reranker-0.6B-ONNX`.
- **Precision/files:** **fp16, NOT int8** (the source is already quantized; re-quantizing produced an invalid graph — fp16 scales). Files: `["model.onnx", "tokenizer.json", "config.json"]` — **filename is `model.onnx`, not `model_int8.onnx`**; adjust S3's download sentinel/file list. ~1.1 GB. A true int8 build needs a fresh fp32 export (future optimization).
- **ScoreStrategy = YesNoLogit:** chat-format the prompt, run, take `logits[:, -1, :]`, softmax over the `yes`/`no` token logits → P(yes) is the relevance score.
- **Token ids (Qwen tokenizer):** `yes` = **9693**, `no` = **2152**.
- **ONNX I/O:** inputs `input_ids`, `attention_mask`, **`position_ids`** (all int64); output `logits [batch, seq, vocab]`.
- **Prompt template (verified):**
  ```
  <|im_start|>system
  Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
  <|im_start|>user
  <Instruct>: {instruction}
  <Query>: {query}
  <Document>: {doc}<|im_end|>
  <|im_start|>assistant
  <think>

  </think>

  ```
- **Verified (CPU smoke):** P(yes | relevant) = **0.990** vs P(yes | irrelevant) = **0.000**.
- Reranker stays **off by default** (D8); this is the opt-in code-capable option, bge-reranker-v2-m3 remains the permissive fallback.
