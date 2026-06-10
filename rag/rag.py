# rag/rag.py
from __future__ import annotations
from typing import Iterator, List, Dict, Any

import pickle
import logging

from rag.config import Config
from rag.embedder import Embedder, text_key as embed_text_key
from rag.vectorstore import VectorStore
from rag.retriever import Retriever
from rag.reranker import make_reranker
from rag.loader import load_documents
from rag.splitter import split_records
from rag.enrich import enrich_texts, enrich_texts_iter
from rag.tables import augment_chunk_tables

logger = logging.getLogger(__name__)
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
            query_prefix=cfg.embedding.get("query_prefix", ""),
        )
        self.store = VectorStore(cfg.persist_dir(), cfg.vectorstore["collection"])
        # 懒加载:重排模型与 LLM 仅在查询时才加载。建库(ingest)只需向量模型,
        # 不再白白把重排(~1G)和 LLM(~1-2G)读进内存——显著降低建库内存,避免低配机 swap。
        self.retriever = Retriever(self.store, self.embedder, cfg.retrieval,
                                   reranker_factory=lambda: make_reranker(cfg.retrieval))
        self._provider = None

    @property
    def provider(self):
        if self._provider is None:
            self._provider = make_provider(self.cfg.llm)
        return self._provider

    @staticmethod
    def _load_embed_cache(path, model_name: str) -> Dict[str, Any]:
        """读取向量缓存;模型不一致或读取失败则返回空(自动全量重编)。"""
        if path.exists():
            try:
                data = pickle.loads(path.read_bytes())
                if data.get("model") == model_name:
                    return data.get("vectors", {})
            except Exception as e:  # noqa: BLE001
                logger.warning("向量缓存读取失败,忽略: %s", e)
        return {}

    @staticmethod
    def _save_embed_cache(path, model_name: str, cache: Dict[str, Any],
                          texts: List[str]) -> None:
        """落盘向量缓存;按本次出现的片段裁剪,避免无限增长。失败不阻断建库。"""
        keep = {embed_text_key(t) for t in texts}
        pruned = {k: v for k, v in cache.items() if k in keep}
        try:
            path.write_bytes(pickle.dumps({"model": model_name, "vectors": pruned}))
        except Exception as e:  # noqa: BLE001
            logger.warning("向量缓存写入失败: %s", e)

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
        # 先把召回的片段抛出,便于前端/调用方查看检索结果(含章节定位与扩充标记)
        yield {"type": "chunks",
               "data": [{"source": h["source"], "text": h["text"],
                         "chapter": h.get("chapter"), "section": h.get("section"),
                         "expanded": h.get("expanded", False)} for h in hits]}
        prompt = self._build_prompt(question, hits)
        for token in self.provider.stream(prompt):
            yield {"type": "token", "data": token}
        seen, sources = set(), []
        for h in hits:
            if h["source"] not in seen:
                seen.add(h["source"])
                sources.append(h["source"])
        yield {"type": "sources", "data": sources}

    def rebuild_index_iter(self) -> Iterator[Dict[str, Any]]:
        """流式重建:逐阶段 yield {'type':'status','msg':...} 进度,最后 yield
        {'type':'done','stats':{...}}。关键:reset() 推迟到写入前一刻执行,
        重建期间旧索引保持可查,避免长时间"知识库为空"。"""
        yield {"type": "status", "msg": "📖 正在解析文档(分栏/表格识别)…"}
        docs = load_documents(self.cfg.docs_dir())
        if not docs:
            self.store.reset()
            stats = {"doc_files": 0, "chunks": 0,
                     "message": f"未在 {self.cfg.docs_dir()} 找到 PDF/Word 文档"}
            yield {"type": "done", "stats": stats}
            return

        ids, texts, sources, chapters, sections = [], [], [], [], []
        linearize = self.cfg.ingest.get("table_linearize", False)
        for text, source in docs:
            recs = split_records(
                text, self.cfg.split["chunk_size"], self.cfg.split["chunk_overlap"]
            )
            for idx, rec in enumerate(recs):
                ids.append(f"{source}::{idx}")
                rec_text = augment_chunk_tables(rec["text"]) if linearize else rec["text"]
                texts.append(rec_text)
                sources.append(source)
                chapters.append(rec.get("chapter"))
                sections.append(rec.get("section"))
        yield {"type": "status",
               "msg": f"✂️ 切分完成:{len(docs)} 文档 → {len(texts)} 片段"}

        # P1 语义增强(流式进度);带哈希缓存,未改动片段复用不调用 LLM
        enrich_cache = self.cfg.persist_dir() / "enrich_cache.json"
        if self.cfg.ingest.get("enrich", False):
            gen = enrich_texts_iter(self.provider, texts, self.cfg.ingest, enrich_cache)
            while True:
                try:
                    done, total, reused, newly = next(gen)
                except StopIteration as e:
                    texts = e.value
                    break
                if done % 5 == 0 or done == total:
                    yield {"type": "status",
                           "msg": f"🤖 生成关键词/假设问题 {done}/{total} "
                                  f"(复用缓存 {reused},新生成 {newly})"}

        # 向量缓存:文本未变的片段复用已编码向量,避免每次重建全量重算。
        # 按 embedding 模型隔离(换模型则缓存失效,自动全量重编)。
        embed_cache_path = self.cfg.persist_dir() / "embed_cache.pkl"
        model_name = self.cfg.embedding["model"]
        cache = self._load_embed_cache(embed_cache_path, model_name)
        to_encode = sum(1 for t in texts if embed_text_key(t) not in cache)
        yield {"type": "status",
               "msg": f"🧮 正在编码向量 0/{to_encode}(复用缓存 {len(texts) - to_encode})…"}
        enc = self.embedder.encode_cached_iter(texts, cache)
        while True:
            try:
                done, total = next(enc)
            except StopIteration as e:
                vectors = e.value
                break
            yield {"type": "status",
                   "msg": f"🧮 正在编码向量 {done}/{total}(复用缓存 {len(texts) - total})…"}
        self._save_embed_cache(embed_cache_path, model_name, cache, texts)

        # 全部就绪后再清空旧库并写入,空窗仅在此一瞬
        yield {"type": "status", "msg": "💾 正在写入索引…"}
        self.store.reset()
        self.store.add(ids, texts, vectors, sources, chapters, sections)
        self.store.build_bm25()
        stats = {"doc_files": len(docs), "chunks": len(texts),
                 "message": f"重建完成:{len(docs)} 个文档,{len(texts)} 个片段"}
        yield {"type": "done", "stats": stats}

    def rebuild_index(self) -> Dict[str, Any]:
        """阻塞版重建(供 ingest.py / 测试使用),内部复用流式实现。"""
        stats: Dict[str, Any] = {}
        for ev in self.rebuild_index_iter():
            if ev["type"] == "done":
                stats = ev["stats"]
        return stats

    def health(self) -> Dict[str, Any]:
        return {"llm": self.provider.health(), "doc_count": self.store.count()}
