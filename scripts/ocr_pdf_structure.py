#!/usr/bin/env python
"""扫描版(图片)PDF → 结构化文本 离线转换工具(一次性,版面感知)。

相比 ocr_pdf.py 的纯 OCR,本脚本用 PaddleOCR 的 PP-StructureV3 版面分析流水线:
  - 自动区分正文/表格/图/页眉页脚,按阅读顺序输出(正确处理双栏,不再交织)
  - 逐页重复的页眉/页脚被版面模型识别并自动剔除
  - 表格识别为结构,本脚本再转成 Markdown 表格(| 列 | 列 |),保留行列对齐
  - 章节标题输出为 Markdown 标题(## ...)

每页以 ===== 第 N 页 ===== 分隔写出,便于后续按页码做前置/索引裁剪。
OCR 仅离线跑一次,不进入建库热路径。

用法:
    python scripts/ocr_pdf_structure.py "docs_kb/_scanned_source/某书.pdf" \
        -o "docs_kb/某书.txt" --dpi 200
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from pdf2image import pdfinfo_from_path, convert_from_path

# 复用 loader 里成熟的二维表→Markdown 实现,保持与其他文档表格风格一致
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.loader import _table_to_markdown  # noqa: E402

_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.S | re.I)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
# 只剥离真正的 HTML 结构标签;不动正文里的 < / >(如 <120、数学式),避免误删
_STRUCT_TAG_RE = re.compile(r"</?(div|html|body|span|p|img|br|table|tr|td|th)[^>]*>", re.I)


def _html_table_to_md(table_html: str) -> str:
    rows = []
    for tr in _ROW_RE.findall(table_html):
        cells = [re.sub(r"<[^>]+>", " ", c).strip() for c in _CELL_RE.findall(tr)]
        if cells:
            rows.append(cells)
    md = _table_to_markdown(rows)
    return "\n[表格]\n" + md if md else ""


def _md_to_text(md: str) -> str:
    """把 PP-StructureV3 的 markdown 输出规整为入库文本:HTML 表格转 Markdown,
    剥离居中 div 等结构标签。"""
    md = _TABLE_RE.sub(lambda m: _html_table_to_md(m.group(0)), md)
    md = _STRUCT_TAG_RE.sub("", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _page_markdown(res) -> str:
    md = res.markdown
    if isinstance(md, dict):
        md = md.get("markdown_texts", "")
    return md or ""


def main() -> int:
    ap = argparse.ArgumentParser(description="扫描版 PDF 版面感知 OCR → 结构化文本")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--batch", type=int, default=10, help="每批栅格化页数")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()

    pdf = args.pdf
    if not pdf.exists():
        print(f"找不到文件: {pdf}", file=sys.stderr)
        return 1
    out = args.out or (Path("docs_kb") / (pdf.stem + ".txt"))
    out.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from paddleocr import PPStructureV3

    total = pdfinfo_from_path(str(pdf))["Pages"]
    end = args.end or total
    print(f"PDF: {pdf.name} 共 {total} 页,处理 {args.start}~{end},DPI={args.dpi}")

    pipe = PPStructureV3()
    t0, done = time.time(), 0
    with open(out, "w", encoding="utf-8") as f:
        for first in range(args.start, end + 1, args.batch):
            last = min(first + args.batch - 1, end)
            images = convert_from_path(str(pdf), dpi=args.dpi,
                                       first_page=first, last_page=last)
            for offset, image in enumerate(images):
                pageno = first + offset
                res = list(pipe.predict(np.array(image)))
                text = _md_to_text("\n".join(_page_markdown(r) for r in res))
                f.write(f"\n\n===== 第 {pageno} 页 =====\n{text}\n")
                done += 1
                if done % 5 == 0 or pageno == end:
                    rate = done / (time.time() - t0)
                    eta = (end - pageno) / rate / 60 if rate else 0
                    print(f"  已完成 {pageno}/{end}  ({rate:.2f} 页/秒, 剩约 {eta:.1f} 分)",
                          flush=True)
            f.flush()

    print(f"完成:{done} 页,用时 {(time.time()-t0)/60:.1f} 分钟 → {out} "
          f"({out.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
