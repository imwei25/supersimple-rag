# rag/reranker.py
from __future__ import annotations
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-Encoder 重排器:对 (query, chunk) 直接打分,比向量召回更精确。
    加载失败时降级为 None(由调用方处理),不影响主检索流程。"""

    def __init__(self, model_name: str, device: str = "cuda"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not hits:
            return hits
        pairs = [[query, h["text"]] for h in hits]
        scores = self.model.predict(pairs)
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order[:top_k]:
            h = dict(hits[i])
            h["rerank_score"] = float(scores[i])
            out.append(h)
        return out


def make_reranker(cfg: dict):
    """按配置构建重排器;未启用或加载失败返回 None(优雅降级)。"""
    if not cfg.get("rerank", False):
        return None
    try:
        return Reranker(
            cfg.get("reranker_model", "./models/bge-reranker-base"),
            device=cfg.get("reranker_device", cfg.get("device", "cuda")),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Reranker 加载失败,降级为不重排: %s", e)
        return None
