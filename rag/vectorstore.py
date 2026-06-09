# rag/vectorstore.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import pickle

import chromadb
import jieba
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


class VectorStore:
    def __init__(self, persist_dir: Path, collection: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self.bm25_path = self.persist_dir / "bm25.pkl"

    def reset(self) -> None:
        """清空知识库:删除并重建 collection,同时删除 BM25 索引文件。"""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        if self.bm25_path.exists():
            self.bm25_path.unlink()

    def add(self, ids: List[str], texts: List[str], vectors: List[List[float]],
            sources: List[str], chapters: List[str] | None = None,
            sections: List[str] | None = None) -> None:
        chapters = chapters or [None] * len(ids)
        sections = sections or [None] * len(ids)
        metadatas = []
        for s, ch, se in zip(sources, chapters, sections):
            m: Dict[str, Any] = {"source": s}
            if ch:                         # Chroma 不接受 None,空值直接省略
                m["chapter"] = ch
            if se:
                m["section"] = se
            metadatas.append(m)
        self.collection.add(
            ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas,
        )

    def build_bm25(self) -> None:
        """从 collection 全量取文档,构建 BM25 并落盘。"""
        data = self.collection.get(include=["documents", "metadatas"])
        docs = data["documents"]
        ids = data["ids"]
        metas = data["metadatas"]
        sources = [m["source"] for m in metas]
        tokenized = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized) if tokenized else None
        with open(self.bm25_path, "wb") as f:
            pickle.dump(
                {"bm25": bm25, "ids": ids, "docs": docs,
                 "sources": sources, "metas": metas},
                f,
            )

    def load_bm25(self) -> Dict[str, Any] | None:
        if not self.bm25_path.exists():
            return None
        with open(self.bm25_path, "rb") as f:
            return pickle.load(f)

    def count(self) -> int:
        return self.collection.count()

    def query_vector(self, query_vec: List[float], k: int) -> List[Dict[str, Any]]:
        res = self.collection.query(
            query_embeddings=[query_vec], n_results=k,
            include=["documents", "metadatas"],
        )
        out = []
        for i, doc in enumerate(res["documents"][0]):
            m = res["metadatas"][0][i]
            out.append({
                "id": res["ids"][0][i],
                "text": doc,
                "source": m["source"],
                "chapter": m.get("chapter"),
                "section": m.get("section"),
            })
        return out

    def get_window(self, source: str, idx: int, radius: int) -> Dict[int, Dict[str, Any]]:
        """取同一来源中 idx 两侧 radius 个相邻片段(含自身),返回 {idx: 记录}。
        用于召回后对短片段做上下文扩充。缺失的 id 自动跳过。"""
        wanted = [f"{source}::{i}" for i in range(max(0, idx - radius), idx + radius + 1)]
        data = self.collection.get(ids=wanted, include=["documents", "metadatas"])
        out: Dict[int, Dict[str, Any]] = {}
        for _id, doc, m in zip(data["ids"], data["documents"], data["metadatas"]):
            try:
                i = int(_id.rsplit("::", 1)[1])
            except (ValueError, IndexError):
                continue
            out[i] = {"id": _id, "text": doc, "source": m.get("source", source),
                      "chapter": m.get("chapter"), "section": m.get("section")}
        return out
