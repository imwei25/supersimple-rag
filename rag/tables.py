# rag/tables.py
"""Layer A:表格线性化(确定性,不调 LLM)。

OCR 出来的 markdown 表格里,行头与单元格值被 `|` 拆在不同列,字面上不连续,
向量/BM25 难以把"奥美拉唑"和它所在行的"CYP2C19"当成强共现 → 召回差,
小模型也常读错横竖对齐。本模块把每一行"拍平"成一句自然语言
(行头:值1、值2…),让实体在同一句里共现,既提升检索命中,也给 LLM
一份干净可读的表格。纯规则、忠实度 100%(只重排原文,不编造)。

产物以【表格说明】块追加在 chunk 末尾,原始表格保留(答题时仍可取精确数值)。
"""
from __future__ import annotations
from typing import List

from rag.splitter import _split_blocks, _merge_captions, _CAPTION_RE


def _is_separator(cells: List[str]) -> bool:
    """markdown 分隔行(| --- | --- |):每格只含 - 或空。"""
    return all(set(c) <= {"-", ""} for c in cells)


def _cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def linearize_table(block: str) -> str:
    """把单个表格块拍平成逐行自然语言。无可用数据行时返回 ""。"""
    caption: List[str] = []
    rows: List[List[str]] = []
    for raw in block.splitlines():
        s = raw.strip()
        if not s or s.startswith("[表格]"):
            continue
        if s.startswith("|"):
            rows.append(_cells(s))
        else:
            caption.append(s)                 # 表标题/说明等非管道行
    rows = [r for r in rows if not _is_separator(r)]
    if not rows:
        return ""

    sents: List[str] = []
    cap = " ".join(caption).strip()
    if cap:
        sents.append(cap)
    # 首行多半是列头;数据行用「行头:其余值」串成一句,使实体同句共现
    for r in rows[1:] if len(rows) > 1 else rows:
        nonempty = [c for c in r if c]
        if not nonempty:
            continue
        head, rest = nonempty[0], nonempty[1:]
        sents.append(f"{head}:{'、'.join(rest)}" if rest else head)
    return "\n".join(sents)


def augment_chunk_tables(text: str) -> str:
    """若 chunk 内含表格,追加【表格说明】线性化块;否则原样返回。"""
    blocks = _merge_captions(_split_blocks(text))
    lin = [linearize_table(b) for kind, b in blocks if kind == "table"]
    lin = [s for s in lin if s]
    if not lin:
        return text
    return text + "\n\n【表格说明】\n" + "\n".join(lin)
