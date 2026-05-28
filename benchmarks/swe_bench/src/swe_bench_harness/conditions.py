"""Condition definitions for the three benchmark arms."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Condition:
    id: str
    label: str
    agent_model: str
    semantex_enabled: bool
    semantex_features: str
    semantex_llm_features: bool
    semantex_llm_provider: str = ""
    semantex_llm_model: str = ""
    semantex_llm_fallback_provider: str = ""
    semantex_llm_fallback_model: str = ""


def load_conditions(path: Path) -> dict[str, Condition]:
    raw = yaml.safe_load(Path(path).read_text())
    return {
        key: Condition(
            id=val["id"],
            label=val["label"],
            agent_model=val["agent_model"],
            semantex_enabled=val["semantex_enabled"],
            semantex_features=val.get("semantex_features", ""),
            semantex_llm_features=val["semantex_llm_features"],
            semantex_llm_provider=val.get("semantex_llm_provider", ""),
            semantex_llm_model=val.get("semantex_llm_model", ""),
            semantex_llm_fallback_provider=val.get("semantex_llm_fallback_provider", ""),
            semantex_llm_fallback_model=val.get("semantex_llm_fallback_model", ""),
        )
        for key, val in raw.items()
    }
