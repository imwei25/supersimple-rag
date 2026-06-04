# rag/retriever.py
from __future__ import annotations
from typing import List, Dict, Any

from rag.embedder import Embedder
from rag.vectorstore import VectorStore, _tokenize


def rrf_fuse(vector_ids: List[str], bm25_ids: List[str], rrf_k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for rank, _id in enumerate(vector_ids):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, _id in enumerate(bm25_ids):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, cfg: dict):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg

    def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        data = self.store.load_bm25()
        if not data or data["bm25"] is None:
            return []
        scores = data["bm25"].get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{
            "id": data["ids"][i],
            "text": data["docs"][i],
            "source": data["sources"][i],
        } for i in ranked]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        c = self.cfg
        qvec = self.embedder.encode([query])[0]
        vec_hits = self.store.query_vector(qvec, c["vector_k"])
        if not c.get("hybrid", True):
            return vec_hits[: c["top_k"]]
        bm25_hits = self._bm25_search(query, c["bm25_k"])
        by_id = {h["id"]: h for h in (vec_hits + bm25_hits)}
        fused_ids = rrf_fuse(
            [h["id"] for h in vec_hits],
            [h["id"] for h in bm25_hits],
            c.get("rrf_k", 60),
        )
        return [by_id[i] for i in fused_ids[: c["top_k"]]]
