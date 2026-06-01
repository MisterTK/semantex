#!/usr/bin/env python3
"""Prepare semantex's ONNX models: download an existing permissive ONNX export,
verify it, int8-quantize it, and re-host it into a project-controlled HF repo.

This turns the S2/S3 "ops prerequisite" (integration-and-cutover doc §5) into one
command. It does NOT train or fundamentally convert models — it starts from
already-published, permissively-licensed community ONNX exports (verified
2026-05-31) and produces the stable, project-owned artifacts that semantex's
downloader (`embedding/model_manager.rs`) points at.

  embedder (S2): CodeRankEmbed (MIT)          — single-vector, mean-pool, ~768-dim
  reranker (S3): Qwen3-Reranker-0.6B (Apache) — yes/no-logit cross-encoder

Output layout (matches the model_manager.rs download contract):
  <work>/<role>-out/
    model_int8.onnx
    tokenizer.json
    config.json
    README.md            (license + attribution to the upstream authors)
  └─ uploaded to --target-repo when --upload is passed.

Usage:
  python prepare_models.py embedder --target-repo myorg/CodeRankEmbed-onnx-int8 --upload
  python prepare_models.py reranker --target-repo myorg/Qwen3-Reranker-0.6B-onnx-int8 --upload
  python prepare_models.py embedder --source-repo mrsladoje/CodeRankEmbed-onnx-int8 --no-quantize  # already int8

Deps:   pip install "huggingface_hub>=0.23" onnx "onnxruntime>=1.17" "tokenizers>=0.15" numpy click
        (--full-verify additionally needs: transformers torch sentence-transformers)
Auth:   huggingface-cli login   (or export HF_TOKEN=...)  — only needed for --upload
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Model configs — the documented facts of the community exports (verified
# 2026-05-31). These are DATA; nothing model-specific is hardcoded in the steps.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    role: str
    default_source: str          # community ONNX repo to start from
    upstream: str                # original (PyTorch) model, for attribution
    upstream_author: str
    license: str                 # SPDX id — must stay permissive
    source_is_int8: bool         # True if the default source is already int8
    # verification prompts (smoke test): a query, a relevant doc, an irrelevant doc
    query: str = "parse a json configuration file from disk"
    relevant: str = "def load_config(path):\n    with open(path) as f:\n        return json.load(f)"
    irrelevant: str = "the quick brown fox jumps over the lazy dog"
    # embedder-only
    query_prefix: str = ""
    # reranker-only (yes/no-logit contract; lifted from the community card)
    prompt_template: str = ""
    extra_notes: tuple[str, ...] = field(default_factory=tuple)


EMBEDDER = ModelConfig(
    role="embedder",
    default_source="sirasagi62/code-rank-embed-onnx",   # fp32, most-vetted; we quantize
    upstream="nomic-ai/CodeRankEmbed",
    upstream_author="Nomic AI",
    license="mit",
    source_is_int8=False,
    query_prefix="Represent this query for searching relevant code: ",
    extra_notes=(
        "Single-vector bi-encoder (nomic_bert). Documents embed RAW code (no prefix); "
        "queries get the query_prefix above. Pool = mean over tokens, then L2-normalize. "
        "Alt source already int8: mrsladoje/CodeRankEmbed-onnx-int8 (--source-repo + --no-quantize).",
    ),
)

RERANKER = ModelConfig(
    role="reranker",
    default_source="shawnw3i/Qwen3-Reranker-0.6B-ONNX",  # float; we quantize to int8
    upstream="Qwen/Qwen3-Reranker-0.6B",
    upstream_author="Qwen (Alibaba)",
    license="apache-2.0",
    source_is_int8=False,
    prompt_template=(
        '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
        'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        '<|im_start|>user\n<Instruct>: Given a code-search query, judge whether the document is relevant.\n'
        "<Query>: {query}\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    extra_notes=(
        "Generative yes/no-logit reranker: take logits[:, -1, :], read the 'yes' and 'no' "
        "token logits, softmax → P(yes) is the relevance score. ONNX inputs: input_ids, "
        "attention_mask, position_ids (all int64). Output: logits [batch, seq, vocab]. "
        "Simpler CPU alt: export tomaarsen/Qwen3-Reranker-0.6B-seq-cls (classifier head, one logit).",
    ),
)

CONFIGS = {"embedder": EMBEDDER, "reranker": RERANKER}


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _download(source_repo: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    click.echo(f"  downloading {source_repo} → {dest}")
    snapshot_download(repo_id=source_repo, local_dir=str(dest))
    return dest


def _find_onnx(snap: Path, *, prefer_int8: bool) -> Path:
    onnx_files = sorted(snap.rglob("*.onnx"))
    if not onnx_files:
        raise click.ClickException(f"no .onnx file found under {snap}")

    def is_int8(p: Path) -> bool:
        return any(t in p.name.lower() for t in ("int8", "quant", "uint8"))

    if prefer_int8:
        for p in onnx_files:
            if is_int8(p):
                return p
    # else prefer the non-quantized base to quantize ourselves
    for p in onnx_files:
        if not is_int8(p):
            return p
    return onnx_files[0]


def _quantize_int8(src_onnx: Path, out_onnx: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    click.echo(f"  int8-quantizing {src_onnx.name} → {out_onnx.name}")
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(src_onnx),
        model_output=str(out_onnx),
        weight_type=QuantType.QInt8,
        use_external_data_format=True,  # 0.6B reranker exceeds the 2 GB protobuf limit
    )
    return out_onnx


def _load_tokenizer(snap: Path):
    from tokenizers import Tokenizer

    tjson = snap / "tokenizer.json"
    if not tjson.exists():
        hits = list(snap.rglob("tokenizer.json"))
        if not hits:
            raise click.ClickException(f"no tokenizer.json under {snap}")
        tjson = hits[0]
    return Tokenizer.from_file(str(tjson)), tjson


def _verify_embedder(onnx_path: Path, snap: Path, cfg: ModelConfig) -> None:
    import numpy as np
    import onnxruntime as ort

    tok, _ = _load_tokenizer(snap)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names = {i.name for i in sess.get_inputs()}

    def embed(text: str) -> np.ndarray:
        enc = tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in in_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = sess.run(None, feed)[0]
        if out.ndim == 3:  # [1, seq, hidden] → mask-weighted mean pool
            m = mask[0][:, None]
            vec = (out[0] * m).sum(0) / max(float(mask[0].sum()), 1.0)
        else:  # [1, hidden] already pooled
            vec = out[0]
        return vec / (np.linalg.norm(vec) + 1e-9)

    q = embed(cfg.query_prefix + cfg.query)
    pos, neg = embed(cfg.relevant), embed(cfg.irrelevant)
    s_pos, s_neg = float(q @ pos), float(q @ neg)
    click.echo(f"  embedder smoke: dim={len(q)}  sim(q,code)={s_pos:.3f}  sim(q,unrelated)={s_neg:.3f}")
    if not s_pos > s_neg:
        raise click.ClickException("embedder ONNX failed smoke: code not ranked above unrelated text")


def _verify_reranker(onnx_path: Path, snap: Path, cfg: ModelConfig) -> None:
    import numpy as np
    import onnxruntime as ort

    tok, _ = _load_tokenizer(snap)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names = {i.name for i in sess.get_inputs()}
    yes_id = tok.encode("yes", add_special_tokens=False).ids[-1]
    no_id = tok.encode("no", add_special_tokens=False).ids[-1]

    def score(query: str, doc: str) -> float:
        enc = tok.encode(cfg.prompt_template.format(query=query, doc=doc))
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "position_ids" in in_names:
            feed["position_ids"] = np.arange(ids.shape[1], dtype=np.int64)[None, :]
        logits = sess.run(None, feed)[0]
        last = logits[0, -1, :].astype(np.float64)
        pair = np.array([last[no_id], last[yes_id]])
        e = np.exp(pair - pair.max())
        return float(e[1] / e.sum())

    s_pos = score(cfg.query, cfg.relevant)
    s_neg = score(cfg.query, cfg.irrelevant)
    click.echo(f"  reranker smoke: yes_id={yes_id} no_id={no_id}  P(yes|rel)={s_pos:.3f}  P(yes|irrel)={s_neg:.3f}")
    if not s_pos > s_neg:
        raise click.ClickException("reranker ONNX failed smoke: relevant doc not scored above irrelevant")


def _write_card(out_dir: Path, cfg: ModelConfig, source_repo: str, target_repo: str) -> None:
    notes = "\n".join(f"- {n}" for n in cfg.extra_notes)
    card = f"""---
license: {cfg.license}
tags: [onnx, int8, semantex, {cfg.role}]
base_model: {cfg.upstream}
---

# {target_repo.split('/')[-1]}

int8 ONNX build of [`{cfg.upstream}`](https://huggingface.co/{cfg.upstream})
({cfg.upstream_author}), prepared for [semantex](https://github.com/MisterTK/semantex)'s
local CPU `{cfg.role}` path.

- **License:** {cfg.license.upper()} — inherited from the upstream model; original
  copyright and attribution to **{cfg.upstream_author}** are retained.
- **Derived from ONNX export:** `{source_repo}`.
- **Files:** `model_int8.onnx`, `tokenizer.json`, `config.json` (the layout semantex's
  downloader expects).

## Notes
{notes}

Produced by `benchmarks/onnx_models/prepare_models.py`. Do not edit by hand.
"""
    (out_dir / "README.md").write_text(card)


def _stage_outputs(snap: Path, model_int8: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copy2(model_int8, out_dir / "model_int8.onnx")
    # external-data sidecar, if quantize produced one
    for extra in model_int8.parent.glob(model_int8.name + "*"):
        if extra.name != model_int8.name:
            shutil.copy2(extra, out_dir / extra.name.replace(model_int8.stem, "model_int8"))
    for fname in ("tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        hits = list(snap.rglob(fname))
        if hits:
            shutil.copy2(hits[0], out_dir / fname)


def _upload(out_dir: Path, target_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    click.echo(f"  creating + uploading → https://huggingface.co/{target_repo}")
    api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=target_repo, repo_type="model")
    click.echo(f"  done. Set the semantex ModelSpec source to hf:{target_repo}")


def _run(cfg: ModelConfig, source_repo: str, target_repo: str | None,
         work_dir: Path, quantize: bool, verify: bool, upload: bool) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    snap = _download(source_repo, work_dir / f"{cfg.role}-src")
    base_onnx = _find_onnx(snap, prefer_int8=cfg.source_is_int8 and not quantize)

    if quantize and not cfg.source_is_int8:
        model_int8 = _quantize_int8(base_onnx, work_dir / f"{cfg.role}-int8" / "model_int8.onnx")
    else:
        click.echo(f"  using source ONNX as-is (int8): {base_onnx.name}")
        model_int8 = base_onnx

    if verify:
        click.echo("  verifying (CPU smoke test)…")
        (_verify_embedder if cfg.role == "embedder" else _verify_reranker)(model_int8, snap, cfg)

    out_dir = work_dir / f"{cfg.role}-out"
    _stage_outputs(snap, model_int8, out_dir)
    _write_card(out_dir, cfg, source_repo, target_repo or f"<org>/{cfg.role}")
    click.echo(f"  staged artifacts in {out_dir}")

    if upload:
        if not target_repo:
            raise click.ClickException("--upload requires --target-repo")
        _upload(out_dir, target_repo)
    else:
        click.echo("  (skipped upload; pass --upload --target-repo <org>/<name> to publish)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _command(role: str):
    cfg = CONFIGS[role]

    @click.command(name=role, help=f"Prepare the {role} ONNX ({cfg.upstream}, {cfg.license.upper()}).")
    @click.option("--source-repo", default=cfg.default_source, show_default=True,
                  help="Community ONNX repo to start from.")
    @click.option("--target-repo", default=None, help="HF repo to re-host into (required with --upload).")
    @click.option("--work-dir", default=f"/tmp/semantex-onnx/{role}", show_default=True, type=Path)
    @click.option("--quantize/--no-quantize", default=not cfg.source_is_int8, show_default=True,
                  help="int8-quantize the source ONNX (skip if the source is already int8).")
    @click.option("--verify/--no-verify", default=True, show_default=True, help="CPU smoke-test the ONNX.")
    @click.option("--upload/--no-upload", default=False, show_default=True, help="Publish to --target-repo.")
    def cmd(source_repo, target_repo, work_dir, quantize, verify, upload):
        click.echo(f"== {role}: {cfg.upstream} ({cfg.license.upper()}) ==")
        _run(cfg, source_repo, target_repo, Path(work_dir), quantize, verify, upload)

    return cmd


@click.group(help=__doc__)
def cli() -> None:
    pass


cli.add_command(_command("embedder"))
cli.add_command(_command("reranker"))


if __name__ == "__main__":
    try:
        cli()
    except click.ClickException:
        raise
    except Exception as exc:  # surface the real cause; never silently "succeed"
        click.echo(f"ERROR: {type(exc).__name__}: {exc}", err=True)
        sys.exit(1)
