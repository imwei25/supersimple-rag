# rag/embedder.py
from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 32,
                 query_prefix: str = ""):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        # BGE-zh 系列要求查询(query)侧加指令前缀,文档侧不加。
        self.query_prefix = query_prefix

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, query: str) -> List[float]:
        """编码检索查询:自动加上 query_prefix(仅查询侧),文档入库仍用 encode。"""
        return self.encode([self.query_prefix + query])[0]
