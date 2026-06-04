# rag/rag.py
from __future__ import annotations
from typing import Iterator, List, Dict, Any

from rag.config import Config
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever
from rag.providers.factory import make_provider

PROMPT_TEMPLATE = """你是一个严谨的中文问答助手。请仅根据下面的【已知信息】回答【问题】。
如果已知信息中没有答案,请直接说"知识库中没有相关内容",不要编造。

【已知信息】
{context}

【问题】
{question}

请用简洁的中文回答:"""


class RagEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = Embedder(
            cfg.embedding["model"],
            device=cfg.embedding.get("device", "cuda"),
            batch_size=cfg.embedding.get("batch_size", 32),
        )
        self.store = VectorStore(cfg.persist_dir(), cfg.vectorstore["collection"])
        self.retriever = Retriever(self.store, self.embedder, cfg.retrieval)
        self.provider = make_provider(cfg.llm)

    def _build_prompt(self, question: str, hits: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(
            f"[来源:{h['source']}] {h['text']}" for h in hits
        ) or "(无)"
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def answer_stream(self, question: str) -> Iterator[Dict[str, Any]]:
        """yield {'type':'token','data':str} 多次,最后 yield {'type':'sources','data':[...]}。"""
        if self.store.count() == 0:
            yield {"type": "token",
                   "data": "知识库为空,请先运行 `python ingest.py` 建立知识库。"}
            yield {"type": "sources", "data": []}
            return
        hits = self.retriever.retrieve(question)
        prompt = self._build_prompt(question, hits)
        for token in self.provider.stream(prompt):
            yield {"type": "token", "data": token}
        seen, sources = set(), []
        for h in hits:
            if h["source"] not in seen:
                seen.add(h["source"])
                sources.append(h["source"])
        yield {"type": "sources", "data": sources}

    def health(self) -> Dict[str, Any]:
        return {"llm": self.provider.health(), "doc_count": self.store.count()}
