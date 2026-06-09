# scripts/ab_tables.py
"""Layer A 表格线性化的检索 A/B:对"查询词只出现在表格单元格里"的口语化问题,
对比 chunk【原文】vs【原文+线性化】两种编码下,黄金片段的向量召回排名。
全程内存计算(用 bge-large 对全库编码两次),不动 chroma,隔离出 Layer A 的纯效果。
"""
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import load_config
from rag.vectorstore import VectorStore
from rag.tables import augment_chunk_tables

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 查询词只藏在表格单元格里(行头或值),正文不直接共现 → 最能体现线性化价值
PROBES = [
    {"q": "奥美拉唑是被哪种CYP酶代谢的？",
     "gold": "实验诊断学（第2版） (尚红) (Z-Library).txt::1312"},
    {"q": "华法林由哪个肝药酶代谢？",
     "gold": "实验诊断学（第2版） (尚红) (Z-Library).txt::1312"},
    {"q": "丙米嗪经过哪些CYP酶代谢？",
     "gold": "实验诊断学（第2版） (尚红) (Z-Library).txt::1312"},
    {"q": "蛋白质C缺乏的易栓症怎么分型？",
     "gold": "实验诊断学（第2版） (尚红) (Z-Library).txt::193"},
]


def ranks(model, docs, ids, gold, q):
    qv = model.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
    sims = np.asarray(docs) @ qv
    order = np.argsort(-sims)
    for r, i in enumerate(order):
        if ids[i] == gold:
            return r + 1
    return None


def main():
    cfg = load_config()
    store = VectorStore(cfg.persist_dir(), cfg.vectorstore["collection"])
    data = store.collection.get(include=["documents"])
    ids, raw = data["ids"], data["documents"]

    model = SentenceTransformer(cfg.embedding["model"], device="cuda")
    print("编码全库(原文)…")
    emb_orig = model.encode(raw, batch_size=64, normalize_embeddings=True,
                            show_progress_bar=False)
    print("编码全库(原文+表格线性化)…")
    aug = [augment_chunk_tables(t) for t in raw]
    emb_aug = model.encode(aug, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False)

    print(f"\n{'query':<28}{'原文':>8}{'线性化':>10}")
    print("-" * 48)
    for p in PROBES:
        ro = ranks(model, emb_orig, ids, p["gold"], p["q"])
        ra = ranks(model, emb_aug, ids, p["gold"], p["q"])
        print(f"{p['q'][:26]:<28}{str(ro):>8}{str(ra):>10}")


if __name__ == "__main__":
    main()
