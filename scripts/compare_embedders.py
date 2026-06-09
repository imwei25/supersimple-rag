# scripts/compare_embedders.py
"""Embedding 模型对比:在同一测试集上,把多个 embedding 模型逐一接入检索链路,
对比黄金片段的 Hit@k / MRR。BM25 与 reranker 与 embedding 无关,全程复用,
因此差异纯粹来自 embedding 模型本身。

向量召回用内存计算(对全库片段编码后算余弦),不动 chroma 索引,
所以无需为每个模型重建知识库。

用法:
    python -m scripts.compare_embedders
    python -m scripts.compare_embedders --models ./models/bge-small-zh-v1.5 ./models/bge-large-zh-v1.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import load_config
from rag.vectorstore import VectorStore, _tokenize
from rag.retriever import rrf_fuse
from rag.reranker import make_reranker

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def encode(model: SentenceTransformer, texts: List[str], bs: int = 64) -> np.ndarray:
    return model.encode(texts, batch_size=bs, normalize_embeddings=True,
                        show_progress_bar=False)


def rank_of(gold: str, ids: List[str]) -> int | None:
    for r, _id in enumerate(ids):
        if _id == gold:
            return r + 1
    return None


def eval_model(model_path: str, ids: List[str], docs: List[str],
               testset: List[Dict], bm25, reranker, depth: int,
               rrf_k: int) -> Dict[str, List]:
    print(f"\n>>> 编码全库({len(docs)} 片段)用 {model_path} …")
    model = SentenceTransformer(model_path, device="cuda")
    doc_mat = encode(model, docs)                     # (N, d) 已归一化
    ranks = {"vector": [], "hybrid": [], "reranked": []}

    for item in testset:
        q, gold = item["question"], item["gold_id"]
        qvec = encode(model, [QUERY_PREFIX + q])[0]    # (d,)
        sims = doc_mat @ qvec                          # 余弦(均已归一化)
        order = np.argsort(-sims)[:depth]
        vec_ids = [ids[i] for i in order]

        if bm25 and bm25["bm25"] is not None:
            bscores = bm25["bm25"].get_scores(_tokenize(q))
            border = sorted(range(len(bscores)), key=lambda j: bscores[j],
                            reverse=True)[:depth]
            bm25_ids = [bm25["ids"][j] for j in border]
        else:
            bm25_ids = []
        fused = rrf_fuse(vec_ids, bm25_ids, rrf_k)

        if reranker:
            text_by_id = dict(zip(ids, docs))
            cand = [{"id": i, "text": text_by_id.get(i, "")} for i in fused[:depth]]
            rer = reranker.rerank(q, cand, len(cand))
            rer_ids = [h["id"] for h in rer]
        else:
            rer_ids = fused

        ranks["vector"].append(rank_of(gold, vec_ids))
        ranks["hybrid"].append(rank_of(gold, fused))
        ranks["reranked"].append(rank_of(gold, rer_ids))

    del model
    import torch
    torch.cuda.empty_cache()
    return ranks


def summarize(name: str, ranks: Dict[str, List], k: int) -> None:
    print(f"\n模型:{name}")
    print(f"{'stage':<10} {'Hit@1':>7} {'Hit@'+str(k):>7} {'MRR':>7} {'命中/总':>9}")
    print("-" * 46)
    for s in ("vector", "hybrid", "reranked"):
        rs = ranks[s]
        n = len(rs)
        hit1 = sum(1 for r in rs if r == 1) / n
        hitk = sum(1 for r in rs if r and r <= k) / n
        mrr = sum(1 / r for r in rs if r) / n
        found = sum(1 for r in rs if r is not None)
        print(f"{s:<10} {hit1:>7.2f} {hitk:>7.2f} {mrr:>7.3f} {found:>4}/{n:<4}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["./models/bge-small-zh-v1.5",
                             "./models/bge-large-zh-v1.5"])
    ap.add_argument("--testset", default="eval_testset.json")
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config()
    store = VectorStore(cfg.persist_dir(), cfg.vectorstore["collection"])
    data = store.collection.get(include=["documents"])
    ids, docs = data["ids"], data["documents"]
    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    bm25 = store.load_bm25()
    reranker = make_reranker(cfg.retrieval)
    rrf_k = cfg.retrieval.get("rrf_k", 60)

    all_ranks = {}
    for m in args.models:
        all_ranks[m] = eval_model(m, ids, docs, testset, bm25, reranker,
                                  args.depth, rrf_k)

    print("\n" + "=" * 50)
    for m in args.models:
        summarize(Path(m).name, all_ranks[m], args.k)

    # 逐条 vector-only 排名对比(最能体现 embedding 差异)
    print("\n逐条 vector-only 黄金片段排名对比:")
    hdr = "".join(f"{Path(m).name[:14]:>16}" for m in args.models)
    print(f"{'#':>2}{hdr}  问题")
    for n, item in enumerate(testset, 1):
        cells = "".join(f"{str(all_ranks[m]['vector'][n-1]):>16}"
                        for m in args.models)
        print(f"{n:>2}{cells}  {item['question'][:24]}")


if __name__ == "__main__":
    main()
