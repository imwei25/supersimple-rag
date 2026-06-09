# scripts/eval_retrieval.py
"""检索质量评测:从知识库随机抽 N 个片段 → LLM 为每个片段生成一个自然提问
和参考答案(测试集)→ 用该提问检索,看「黄金片段(问题的来源)」在各阶段的排名。

三个阶段对比,逐层定位质量瓶颈:
  1. vector   : 仅向量召回(embedding 模型质量)
  2. hybrid   : 向量 + BM25 经 RRF 融合(混合检索参数)
  3. reranked : 融合候选再过 Cross-Encoder 重排(reranker 表现)

指标:Hit@1 / Hit@k / MRR(对黄金片段)。重排阶段额外报告排名变化。

用法:
    python -m scripts.eval_retrieval                  # 默认抽 10 条
    python -m scripts.eval_retrieval -n 15 --seed 7 --pool 80
    python -m scripts.eval_retrieval --testset eval_testset.json   # 复用已生成测试集
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

from rag.config import load_config
from rag.embedder import Embedder
from rag.vectorstore import VectorStore, _tokenize
from rag.retriever import rrf_fuse
from rag.reranker import make_reranker
from rag.providers.factory import make_provider

GEN_PROMPT = """你是医学考试出题老师。下面是一段教材内容。请基于这段内容,设计【一个】
病人/医学生最可能用口语提出的问题,问题的答案必须能在这段内容里找到。
再给出简洁的参考答案。严格按如下格式输出,不要多余内容:
问题:<一句话问题>
答案:<简洁答案>

【教材内容】
{passage}
"""


def _collect(provider, prompt: str) -> str:
    return "".join(provider.stream(prompt)).strip()


def _parse_qa(text: str) -> Dict[str, str] | None:
    q = re.search(r"问题[::]\s*(.+)", text)
    a = re.search(r"答案[::]\s*([\s\S]+)", text)
    if not q:
        return None
    return {"question": q.group(1).strip(),
            "answer": (a.group(1).strip() if a else "")}


def build_testset(cfg, store: VectorStore, n: int, seed: int,
                  pool: int, min_len: int) -> List[Dict]:
    """随机抽 n 个足够长的片段,LLM 各生成一个 Q&A,记录黄金片段 id。"""
    data = store.collection.get(include=["documents", "metadatas"])
    ids, docs = data["ids"], data["documents"]
    idxs = [i for i in range(len(ids)) if len(docs[i]) >= min_len]
    random.Random(seed).shuffle(idxs)

    provider = make_provider(cfg.llm)
    out: List[Dict] = []
    for i in idxs:
        if len(out) >= n:
            break
        passage = docs[i][:1200]
        qa = _parse_qa(_collect(provider, GEN_PROMPT.format(passage=passage)))
        if not qa or len(qa["question"]) < 4:
            continue
        qa["gold_id"] = ids[i]
        qa["gold_source"] = data["metadatas"][i].get("source", "")
        out.append(qa)
        print(f"  [{len(out)}/{n}] {qa['question']}")
    return out


def _rank_of(gold_id: str, ranked_ids: List[str]) -> int | None:
    """1-based 排名;未命中返回 None。"""
    for r, _id in enumerate(ranked_ids):
        if _id == gold_id:
            return r + 1
    return None


def evaluate(cfg, store: VectorStore, testset: List[Dict],
             k: int, depth: int) -> None:
    embedder = Embedder(cfg.embedding["model"],
                        device=cfg.embedding.get("device", "cuda"),
                        batch_size=cfg.embedding.get("batch_size", 32),
                        query_prefix=cfg.embedding.get("query_prefix", ""))
    reranker = make_reranker(cfg.retrieval)
    bm25 = store.load_bm25()
    rrf_k = cfg.retrieval.get("rrf_k", 60)

    stages = ["vector", "hybrid", "reranked"]
    ranks: Dict[str, List] = {s: [] for s in stages}

    print(f"\n评测 {len(testset)} 条 | 候选深度 depth={depth} | Hit@k 的 k={k}\n")
    for item in testset:
        gold = item["gold_id"]
        q = item["question"]

        qvec = embedder.encode_query(q)
        vec_hits = store.query_vector(qvec, depth)
        vec_ids = [h["id"] for h in vec_hits]

        if bm25 and bm25["bm25"] is not None:
            scores = bm25["bm25"].get_scores(_tokenize(q))
            order = sorted(range(len(scores)), key=lambda j: scores[j],
                           reverse=True)[:depth]
            bm25_ids = [bm25["ids"][j] for j in order]
        else:
            bm25_ids = []
        fused_ids = rrf_fuse(vec_ids, bm25_ids, rrf_k)

        # reranker 作用于融合后的候选池(取前 depth 个)
        if reranker:
            by_id = {h["id"]: h for h in vec_hits}
            for j in order[:depth]:
                _id = bm25["ids"][j]
                by_id.setdefault(_id, {"id": _id, "text": bm25["docs"][j]})
            cand = [by_id[i] for i in fused_ids[:depth] if i in by_id]
            reranked = reranker.rerank(q, cand, len(cand))
            rer_ids = [h["id"] for h in reranked]
        else:
            rer_ids = fused_ids

        for stage, ids_ in (("vector", vec_ids), ("hybrid", fused_ids),
                            ("reranked", rer_ids)):
            ranks[stage].append(_rank_of(gold, ids_))

    print(f"{'stage':<10} {'Hit@1':>7} {'Hit@'+str(k):>7} {'MRR':>7} "
          f"{'命中/总':>9}")
    print("-" * 46)
    for s in stages:
        rs = ranks[s]
        hit1 = sum(1 for r in rs if r == 1) / len(rs)
        hitk = sum(1 for r in rs if r and r <= k) / len(rs)
        mrr = sum(1 / r for r in rs if r) / len(rs)
        found = sum(1 for r in rs if r is not None)
        print(f"{s:<10} {hit1:>7.2f} {hitk:>7.2f} {mrr:>7.3f} "
              f"{found:>4}/{len(rs):<4}")

    # 逐条排名表,直观看 rerank 把谁救回/带偏
    print(f"\n逐条黄金片段排名(None=候选 depth={depth} 内未召回):")
    print(f"{'#':>2} {'vector':>7} {'hybrid':>7} {'reranked':>9}  问题")
    for n, item in enumerate(testset, 1):
        v, h, r = (ranks['vector'][n-1], ranks['hybrid'][n-1],
                   ranks['reranked'][n-1])
        print(f"{n:>2} {str(v):>7} {str(h):>7} {str(r):>9}  "
              f"{item['question'][:30]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10, help="抽样片段数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", type=int, default=60, help="(保留)候选池上限")
    ap.add_argument("--min-len", type=int, default=200, help="片段最短字符数")
    ap.add_argument("--depth", type=int, default=20, help="每路召回候选深度")
    ap.add_argument("-k", type=int, default=4, help="Hit@k 的 k")
    ap.add_argument("--testset", type=str, default="eval_testset.json")
    ap.add_argument("--regen", action="store_true", help="强制重新生成测试集")
    args = ap.parse_args()

    cfg = load_config()
    store = VectorStore(cfg.persist_dir(), cfg.vectorstore["collection"])
    ts_path = Path(args.testset)

    if ts_path.exists() and not args.regen:
        testset = json.loads(ts_path.read_text(encoding="utf-8"))
        print(f"复用测试集 {ts_path}({len(testset)} 条)")
    else:
        print("生成测试集(LLM 出题中)…")
        testset = build_testset(cfg, store, args.n, args.seed,
                                args.pool, args.min_len)
        ts_path.write_text(json.dumps(testset, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        print(f"已保存测试集 → {ts_path}")

    evaluate(cfg, store, testset, args.k, args.depth)


if __name__ == "__main__":
    main()
