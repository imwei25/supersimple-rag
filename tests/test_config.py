# tests/test_config.py
from pathlib import Path
from rag.config import load_config

def test_load_config_reads_yaml():
    cfg = load_config(Path("config.yaml"))
    assert cfg.llm["model"] == "minicpm5-1b"
    assert cfg.retrieval["top_k"] == 4
    assert cfg.split["chunk_size"] == 500
    assert cfg.paths["docs_dir"] == "./docs_kb"

def test_config_persist_dir_is_path():
    cfg = load_config(Path("config.yaml"))
    assert cfg.persist_dir().name == "chroma_db"
