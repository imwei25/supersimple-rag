# ingest.py
from __future__ import annotations
import rag.proxyfix  # noqa: F401  必须在导入网络库前清理代理
import logging

from rag.config import load_config
from rag.loader import load_documents
from rag.splitter import split_text
from rag.embedder import Embedder
from rag.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ingest")


def main() -> None:
    cfg = load_config()
    docs = load_documents(cfg.docs_dir())
    if not docs:
        log.warning("未在 %s 找到 PDF/Word 文档", cfg.docs_dir())
        return

    embedder = Embedder(
        cfg.embedding["model"],
        device=cfg.embedding.get("device", "cuda"),
        batch_size=cfg.embedding.get("batch_size", 32),
    )
    store = VectorStore(cfg.persist_dir(), cfg.vectorstore["collection"])

    ids, texts, sources = [], [], []
    for text, source in docs:
        chunks = split_text(text, cfg.split["chunk_size"], cfg.split["chunk_overlap"])
        for idx, chunk in enumerate(chunks):
            ids.append(f"{source}::{idx}")
            texts.append(chunk)
            sources.append(source)
    log.info("共 %d 个文档,切出 %d 个 chunk,开始编码...", len(docs), len(texts))

    vectors = embedder.encode(texts)
    store.add(ids, texts, vectors, sources)
    store.build_bm25()
    log.info("入库完成。当前向量数: %d", store.count())


if __name__ == "__main__":
    main()
