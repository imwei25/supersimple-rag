# scripts/eval_answers.py
"""端到端问答评测:对测试集每个问题跑完整 RAG(检索→LLM 生成),
输出 问题 / 参考答案 / 模型答案 / 黄金片段是否被召回,供人工判分。

用法:
    python -m scripts.eval_answers
    python -m scripts.eval_answers -o answers.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import load_config
from rag.rag import RagEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="eval_testset.json")
    ap.add_argument("-o", "--out", default="eval_answers.json")
    args = ap.parse_args()

    cfg = load_config()
    engine = RagEngine(cfg)
    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))

    results = []
    for n, item in enumerate(testset, 1):
        q, gold_id = item["question"], item["gold_id"]
        chunks, answer = [], []
        for ev in engine.answer_stream(q):
            if ev["type"] == "chunks":
                chunks = ev["data"]
            elif ev["type"] == "token":
                answer.append(ev["data"])
        ans = "".join(answer).strip()
        hit = any(
            f"{c['source']}::" in gold_id and
            gold_id.split("::")[0] == c["source"]
            for c in chunks
        )
        # 精确判断黄金片段是否在召回里:比对 id 前缀 + 文本包含
        gold_src = gold_id.split("::")[0]
        retrieved_srcs = [c["source"] for c in chunks]
        rec = {
            "n": n, "question": q,
            "reference": item.get("answer", ""),
            "answer": ans,
            "gold_source": gold_src,
            "retrieved_sources": retrieved_srcs,
        }
        results.append(rec)
        print(f"\n{'='*70}\n[{n}] {q}")
        print(f"参考:{rec['reference']}")
        print(f"模型:{ans}")

    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 → {args.out}")


if __name__ == "__main__":
    main()
