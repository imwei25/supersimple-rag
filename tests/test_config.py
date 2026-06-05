# tests/test_config.py
from pathlib import Path
from rag.config import load_config

SAMPLE = """\
llm:
  provider: ollama
  model: test-model
embedding:
  model: ./m
vectorstore:
  persist_dir: ./chroma_db
  collection: docs
retrieval:
  top_k: 4
split:
  chunk_size: 500
  chunk_overlap: 80
server:
  api_port: 8001
paths:
  docs_dir: ./docs_kb
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(SAMPLE, encoding="utf-8")
    return p


def test_load_config_reads_yaml(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.llm["model"] == "test-model"
    assert cfg.retrieval["top_k"] == 4
    assert cfg.split["chunk_size"] == 500
    assert cfg.paths["docs_dir"] == "./docs_kb"


def test_config_persist_dir_is_path(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.persist_dir().name == "chroma_db"


def test_real_config_loads():
    """线上 config.yaml 至少能正常解析并含必需字段(不绑定具体可调值)。"""
    cfg = load_config(Path("config.yaml"))
    assert isinstance(cfg.split["chunk_size"], int)
    assert "model" in cfg.llm
    assert "top_k" in cfg.retrieval
