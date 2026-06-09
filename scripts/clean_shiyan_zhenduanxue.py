#!/usr/bin/env python
"""一次性数据清洗:针对《实验诊断学(第2版)》OCR 文本。

仅适用于这一本书的 OCR 产物(规则按其版式硬编码,不追求泛化)。把对 RAG
无用的内容剔除,提升入库与检索质量:

  1. 删前置内容:第 1~27 页(封面/版权/编委/简介/序/前言/目录)
  2. 删书尾索引:第 561 页起的「中英文名词对照索引」(纯术语-页码表)
  3. 删页眉:每页首个非空行的 running header(第X章 + 章名 + 页码的各种 OCR 变体)
  4. 删页脚:每页末个非空行若为纯页码数字
  5. 删页码分隔标记 ===== 第 N 页 =====(OCR 注入物,检索噪声)

原始 OCR 文本另存为 *.raw.bak,清洗结果覆盖原 .txt 供 loader 读取。
用法: python scripts/clean_shiyan_zhenduanxue.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

SRC = Path("docs_kb/实验诊断学（第2版） (尚红) (Z-Library).txt")

BODY_START_PAGE = 28    # 正文起始页(第一章)
INDEX_START_PAGE = 561  # 书尾「中英文名词对照索引」起始页,自此到末页全删

PAGE_RE = re.compile(r"^=====\s*第\s*(\d+)\s*页\s*=====\s*$")
# 页眉:首行形如 "第X章...""4第X章...""第X章...5""一章...""38一第二章" 等
HEADER_PREFIX_RE = re.compile(r"^\d{0,3}\s*第?\s*[一二三四五六七八九十百]+\s*章")
HEADER_CONTAIN_RE = re.compile(r"第[一二三四五六七八九十百]+章")
NUM_ONLY_RE = re.compile(r"^\d{1,3}$")
# 全局漏网页眉:整行就是「(页码)第X章(章名)(页码)」且无句读标点 → 是 running
# header 而非正文句子(正文引用会带"见""如"等前缀或标点,不会整行裸列)。
STANDALONE_HEADER_RE = re.compile(
    r"^\d{0,3}\s*第[一二三四五六七八九十百]+章[^。,，、:：;；?？!！()（）]{0,18}\d{0,3}$")
# 章名型 running header:整行就是「(页码)XXX(的)实验诊断与应用(页码)」。本书各章
# 页眉均为章标题,几乎都以「实验诊断与应用」结尾,逐页重复;真正章标题仅章首出现
# 一次,留存价值低于去噪收益,一并清除。短行+无句读 → 必为页眉而非正文。
TITLE_HEADER_RE = re.compile(
    r"^\d{0,3}[一-鿿、]{0,24}实验诊断与应用\d{0,3}$")


def _is_header(line: str) -> bool:
    """判断页首行是否为 running header(章标题/页码),长度<30 防误伤正文长句。"""
    s = line.strip()
    if not s or len(s) > 30:
        return False
    if NUM_ONLY_RE.match(s):
        return True
    if HEADER_PREFIX_RE.match(s):
        return True
    if HEADER_CONTAIN_RE.search(s):
        return True
    return False


def _split_pages(text: str) -> list[tuple[int, list[str]]]:
    """按页码标记切成 [(页号, 该页行列表)]。"""
    pages: list[tuple[int, list[str]]] = []
    cur_no, cur_lines = None, []
    for line in text.splitlines():
        m = PAGE_RE.match(line)
        if m:
            if cur_no is not None:
                pages.append((cur_no, cur_lines))
            cur_no, cur_lines = int(m.group(1)), []
        else:
            cur_lines.append(line)
    if cur_no is not None:
        pages.append((cur_no, cur_lines))
    return pages


def _clean_page(lines: list[str]) -> list[str]:
    """去掉本页的页眉(首个非空行)与页脚页码(末个非空行若纯数字)。"""
    # 找首个非空行 → 若是页眉则丢弃
    first = next((i for i, l in enumerate(lines) if l.strip()), None)
    if first is not None and _is_header(lines[first]):
        lines = lines[:first] + lines[first + 1:]
    # 找末个非空行 → 若是纯页码则丢弃
    last = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)
    if last is not None and NUM_ONLY_RE.match(lines[last].strip()):
        lines = lines[:last] + lines[last + 1:]
    return lines


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    pages = _split_pages(text)
    total = len(pages)

    kept: list[str] = []
    dropped_front = dropped_index = headers_removed = 0
    for no, lines in pages:
        if no < BODY_START_PAGE:
            dropped_front += 1
            continue
        if no >= INDEX_START_PAGE:
            dropped_index += 1
            continue
        before = "\n".join(lines)
        cleaned = _clean_page(lines)
        if "\n".join(cleaned) != before:
            headers_removed += 1
        # 全局再扫一遍:剔除被排版打乱、未落在页首页脚的漏网章标题独立行
        cleaned = [l for l in cleaned
                   if not STANDALONE_HEADER_RE.match(l.strip())
                   and not TITLE_HEADER_RE.match(l.strip())]
        body = "\n".join(cleaned).strip()
        if body:
            kept.append(body)

    out_text = "\n\n".join(kept) + "\n"

    bak = SRC.with_suffix(".raw.bak")
    if not bak.exists():
        shutil.copy2(SRC, bak)
    SRC.write_text(out_text, encoding="utf-8")

    print(f"总页数: {total}")
    print(f"删除前置内容(<{BODY_START_PAGE}页): {dropped_front} 页")
    print(f"删除书尾索引(>={INDEX_START_PAGE}页): {dropped_index} 页")
    print(f"清理过页眉/页脚的正文页: {headers_removed} 页")
    print(f"原始备份: {bak.name}")
    print(f"清洗后字数: {len(out_text)}  → {SRC.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
