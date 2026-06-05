# ingest.py
from __future__ import annotations
import rag.proxyfix  # noqa: F401  必须在导入网络库前清理代理
import logging

from rag.config import load_config
from rag.rag import RagEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ingest")


def main() -> None:
    cfg = load_config()
    engine = RagEngine(cfg)
    stats = engine.rebuild_index()
    log.info("%s", stats["message"])
    log.info("当前向量数: %d", engine.store.count())


if __name__ == "__main__":
    main()
