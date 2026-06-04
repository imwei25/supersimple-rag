# rag/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    llm: dict
    embedding: dict
    vectorstore: dict
    retrieval: dict
    split: dict
    server: dict
    paths: dict

    def persist_dir(self) -> Path:
        return Path(self.vectorstore["persist_dir"]).resolve()

    def docs_dir(self) -> Path:
        return Path(self.paths["docs_dir"]).resolve()


def load_config(path: Path = Path("config.yaml")) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
