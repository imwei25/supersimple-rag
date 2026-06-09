#!/usr/bin/env python
"""扫描版(图片)PDF → 文本 离线转换工具(一次性)。

逐页把 PDF 栅格化为图片,用 PaddleOCR(GPU)识别中文正文,结果写成同名 .txt
放进知识库目录,供 rag.loader 直接读取。OCR 仅离线跑一次,不进入建库热路径。

用法:
    python scripts/ocr_pdf.py "docs_kb/某扫描书.pdf"
    python scripts/ocr_pdf.py "input.pdf" -o docs_kb/output.txt --dpi 200 --lang ch

识别完成后,原始大体积扫描 PDF 会被移到 docs_kb/_scanned_source/ 子目录,
避免 loader 每次建库都去解析它(glob 不递归子目录,自动跳过)。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pdf2image import pdfinfo_from_path, convert_from_path
from paddleocr import PaddleOCR


def _ocr_page(ocr: PaddleOCR, image) -> str:
    """对单页 PIL 图片做 OCR,按版面顺序返回文本行拼接。"""
    import numpy as np

    result = ocr.predict(np.array(image))
    lines: list[str] = []
    for page in result:  # predict 返回每张图一个结果
        texts = page.get("rec_texts", [])
        lines.extend(t for t in texts if t and t.strip())
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="扫描版 PDF OCR → 文本")
    ap.add_argument("pdf", type=Path, help="输入 PDF 路径")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="输出 .txt 路径(默认:同名 .txt 放 docs_kb)")
    ap.add_argument("--dpi", type=int, default=200, help="栅格化 DPI(默认 200)")
    ap.add_argument("--lang", default="ch", help="PaddleOCR 语言(默认 ch=中英混合)")
    ap.add_argument("--batch", type=int, default=20,
                    help="每批栅格化页数,控制内存(默认 20)")
    ap.add_argument("--start", type=int, default=1, help="起始页(1-based,含)")
    ap.add_argument("--end", type=int, default=0, help="结束页(0=到末页)")
    ap.add_argument("--keep-source", action="store_true",
                    help="完成后不移动原始 PDF")
    args = ap.parse_args()

    pdf = args.pdf
    if not pdf.exists():
        print(f"找不到文件: {pdf}", file=sys.stderr)
        return 1

    out = args.out or (Path("docs_kb") / (pdf.stem + ".txt"))
    out.parent.mkdir(parents=True, exist_ok=True)

    total = pdfinfo_from_path(str(pdf))["Pages"]
    end = args.end or total
    print(f"PDF: {pdf.name}  共 {total} 页,处理 {args.start}~{end},DPI={args.dpi},lang={args.lang}")

    ocr = PaddleOCR(lang=args.lang, use_textline_orientation=True)

    t0 = time.time()
    done = 0
    # 边 OCR 边落盘,中途崩溃也不丢前面成果。
    with open(out, "w", encoding="utf-8") as f:
        for first in range(args.start, end + 1, args.batch):
            last = min(first + args.batch - 1, end)
            images = convert_from_path(str(pdf), dpi=args.dpi,
                                       first_page=first, last_page=last)
            for offset, image in enumerate(images):
                pageno = first + offset
                text = _ocr_page(ocr, image)
                f.write(f"\n\n===== 第 {pageno} 页 =====\n{text}\n")
                done += 1
                if done % 5 == 0 or pageno == end:
                    rate = done / (time.time() - t0)
                    print(f"  已完成 {pageno}/{end}  ({rate:.2f} 页/秒)", flush=True)
            f.flush()

    dt = time.time() - t0
    print(f"完成:{done} 页,用时 {dt/60:.1f} 分钟 → {out}  ({out.stat().st_size//1024} KB)")

    if not args.keep_source and pdf.resolve().parent.name == "docs_kb":
        dest_dir = pdf.parent / "_scanned_source"
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / pdf.name
        pdf.rename(dest)
        print(f"原始扫描 PDF 已移至: {dest}(loader 不再解析)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
