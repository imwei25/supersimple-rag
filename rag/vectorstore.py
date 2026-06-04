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

    def add(self, ids: List[str], texts: List[str], vectors: List[List[float]],
            sources: List[str]) -> None:
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=[{"source": s} for s in sources],
        )

    def build_bm25(self) -> None:
        """从 collection 全量取文档,构建 BM25 并落盘。"""
        data = self.collection.get(include=["documents", "metadatas"])
        docs = data["documents"]
        ids = data["ids"]
        sources = [m["source"] for m in data["metadatas"]]
        tokenized = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized) if tokenized else None
        with open(self.bm25_path, "wb") as f:
            pickle.dump(
                {"bm25": bm25, "ids": ids, "docs": docs, "sources": sources},
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
            out.append({
                "id": res["ids"][0][i],
                "text": doc,
                "source": res["metadatas"][0][i]["source"],
            })
        return out
