#!/usr/bin/env python
"""一次性裁剪:针对《实验诊断学(第2版)》PP-StructureV3 结构化输出。

PP-StructureV3 已在版面层面自动剔除逐页页眉/页脚,并正确排序双栏、还原表格,
因此这里只需按页码裁掉对 RAG 无用的整页内容:

  1. 删前置内容:第 1~27 页(封面/版权/编委/简介/序/前言/目录)
  2. 删书尾索引:第 561 页起的「中英文名词对照索引」
  3. 删各章末「## 参考文献」块(文献条目对临床问答检索是噪声):从参考文献
     标题行起,删到下一个 Markdown 标题(下一章节)或文末为止

去掉 ===== 第 N 页 ===== 分隔标记,正文按页拼接写出,覆盖 loader 读取的 .txt。
原结构化原文保留在 *.structure.raw.txt,可随时重裁。
用法: python scripts/trim_shiyan_structure.py
"""
from __future__ import annotations

import re
from pathlib import Path

RAW = Path("docs_kb/_scanned_source/实验诊断学（第2版） (尚红) (Z-Library).structure.raw.txt")
OUT = Path("docs_kb/实验诊断学（第2版） (尚红) (Z-Library).txt")

BODY_START_PAGE = 28
INDEX_START_PAGE = 561
PAGE_RE = re.compile(r"^=====\s*第\s*(\d+)\s*页\s*=====\s*$")
REF_HEADING_RE = re.compile(r"^#{1,6}\s*参考文献\s*$")
HEADING_RE = re.compile(r"^#{1,6}\s")


def strip_references(text: str) -> tuple[str, int]:
    """删除「## 参考文献」块:自参考文献标题行起,丢弃直到下一个 Markdown 标题
    (下一章节)或文末。返回 (清理后文本, 删除的块数)。"""
    out, in_ref, removed = [], False, 0
    for line in text.split("\n"):
        if REF_HEADING_RE.match(line):
            in_ref, removed = True, removed + 1
            continue
        if in_ref:
            if HEADING_RE.match(line):   # 下一个标题 → 文献块结束,保留该行
                in_ref = False
                out.append(line)
            continue                     # 块内文献条目 → 丢弃
        out.append(line)
    return "\n".join(out), removed


def main() -> int:
    cur_no, cur_lines, kept = None, [], []
    dropped_front = dropped_index = 0

    def flush():
        nonlocal dropped_front, dropped_index
        if cur_no is None:
            return
        if cur_no < BODY_START_PAGE:
            dropped_front += 1
            return
        if cur_no >= INDEX_START_PAGE:
            dropped_index += 1
            return
        body = "\n".join(cur_lines).strip()
        if body:
            kept.append(body)

    for line in RAW.read_text(encoding="utf-8").splitlines():
        m = PAGE_RE.match(line)
        if m:
            flush()
            cur_no, cur_lines = int(m.group(1)), []
        else:
            cur_lines.append(line)
    flush()

    out_text = "\n\n".join(kept) + "\n"
    out_text, ref_blocks = strip_references(out_text)
    out_text = re.sub(r"\n{3,}", "\n\n", out_text).strip() + "\n"
    OUT.write_text(out_text, encoding="utf-8")
    print(f"删除前置内容(<{BODY_START_PAGE}页): {dropped_front} 页")
    print(f"删除书尾索引(>={INDEX_START_PAGE}页): {dropped_index} 页")
    print(f"删除参考文献块: {ref_blocks} 处")
    print(f"保留正文页: {len(kept)} 页,共 {len(out_text)} 字 → {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
