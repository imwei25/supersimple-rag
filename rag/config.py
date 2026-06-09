# rag/config.py
from __future__ import annotations
from dataclasses import dataclass, field
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
    # 入库阶段配置(语义增强等);旧 config.yaml 无此段时默认空,行为不变。
    ingest: dict = field(default_factory=dict)

    def persist_dir(self) -> Path:
        return Path(self.vectorstore["persist_dir"]).resolve()

    def docs_dir(self) -> Path:
        return Path(self.paths["docs_dir"]).resolve()


def load_config(path: Path = Path("config.yaml")) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
