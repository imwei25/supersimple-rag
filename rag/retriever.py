# rag/retriever.py
from __future__ import annotations
from typing import List, Dict, Any

from rag.embedder import Embedder
from rag.vectorstore import VectorStore, _tokenize


def _parse_id(_id: str):
    """'source::idx' → (source, idx);解析失败返回 (None, None)。"""
    try:
        src, idx = _id.rsplit("::", 1)
        return src, int(idx)
    except (ValueError, AttributeError):
        return None, None


def _stitch(prev: str, nxt: str, max_overlap: int = 200) -> str:
    """拼接相邻片段并去掉衔接处的重复(相邻片段本就有句级 overlap)。
    取 prev 的最长后缀使其等于 nxt 的前缀,去重后拼接。"""
    limit = min(len(prev), len(nxt), max_overlap)
    for k in range(limit, 5, -1):
        if prev[-k:] == nxt[:k]:
            return prev + nxt[k:]
    return prev + "\n\n" + nxt


def expand_short_hit(hit: Dict[str, Any], window: Dict[int, Dict[str, Any]],
                     min_chars: int, target: int, max_neighbors: int) -> Dict[str, Any]:
    """短片段上下文扩充:向同章相邻片段两端扩展,直到达到 target 或邻居用尽。
    方向:无标题开头(节中延续)优先向上找回主题;否则优先向下补正文;随后两端交替。
    硬约束:只在同一章(chapter 相同)内扩;metadata 简单合并。"""
    if len(hit["text"]) >= min_chars:
        return hit
    src, idx = _parse_id(hit["id"])
    if idx is None or idx not in window:
        return hit

    base = window[idx]
    chapter = base.get("chapter")

    def same_chapter(i: int) -> bool:
        return i in window and window[i].get("chapter") == chapter

    # 方向偏好:片段不以标题/面包屑开头 → 是节中延续 → 先向上
    stripped = base["text"].lstrip()
    prefer_up = not stripped.startswith(("#", "【"))
    order = ["up", "down"] if prefer_up else ["down", "up"]

    left = right = idx
    total = len(base["text"])
    while total < target and (right - left) < 2 * max_neighbors:
        progressed = False
        for side in order:
            if side == "up" and same_chapter(left - 1) and (idx - (left - 1)) <= max_neighbors:
                left -= 1
                total += len(window[left]["text"])
                progressed = True
            elif side == "down" and same_chapter(right + 1) and ((right + 1) - idx) <= max_neighbors:
                right += 1
                total += len(window[right]["text"])
                progressed = True
            if total >= target:
                break
        if not progressed:
            break

    if left == right:
        return hit
    text = window[left]["text"]
    for i in range(left + 1, right + 1):
        text = _stitch(text, window[i]["text"])
    return {**hit, "text": text, "source": src,
            "chapter": chapter, "section": base.get("section"),
            "expanded": True, "expanded_span": [left, right]}


def rrf_fuse(vector_ids: List[str], bm25_ids: List[str], rrf_k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for rank, _id in enumerate(vector_ids):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, _id in enumerate(bm25_ids):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, cfg: dict, reranker=None):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self.reranker = reranker

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
        top_k = c["top_k"]
        # 启用重排时,先扩大候选池,再由重排器收窄到 top_k
        cand_k = c.get("rerank_candidates", top_k) if self.reranker else top_k

        qvec = self.embedder.encode_query(query)
        vec_hits = self.store.query_vector(qvec, c["vector_k"])
        if not c.get("hybrid", True):
            candidates = vec_hits[:cand_k]
        else:
            bm25_hits = self._bm25_search(query, c["bm25_k"])
            by_id = {h["id"]: h for h in (vec_hits + bm25_hits)}
            fused_ids = rrf_fuse(
                [h["id"] for h in vec_hits],
                [h["id"] for h in bm25_hits],
                c.get("rrf_k", 60),
            )
            candidates = [by_id[i] for i in fused_ids[:cand_k]]

        if self.reranker:
            hits = self.reranker.rerank(query, candidates, top_k)
        else:
            hits = candidates[:top_k]
        return self._expand_short(hits)

    def _expand_short(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对召回结果中过短的片段做上下文扩充(配置开启时)。"""
        c = self.cfg
        if not c.get("expand_short", False):
            return hits
        min_chars = c.get("expand_min_chars", 200)
        target = c.get("expand_target_chars", min_chars * 2)
        max_nb = c.get("expand_max_neighbors", 4)
        out = []
        for h in hits:
            if len(h["text"]) >= min_chars:
                out.append(h)
                continue
            src, idx = _parse_id(h["id"])
            if idx is None:
                out.append(h)
                continue
            window = self.store.get_window(src, idx, max_nb)
            out.append(expand_short_hit(h, window, min_chars, target, max_nb))
        return out
