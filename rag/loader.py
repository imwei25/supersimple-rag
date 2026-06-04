# rag/loader.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import logging

from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_documents(docs_dir: Path) -> List[Tuple[str, str]]:
    """返回 [(全文, 文件名)],解析失败的文件跳过并记日志。"""
    results: List[Tuple[str, str]] = []
    docs_dir = Path(docs_dir)
    for path in sorted(docs_dir.glob("*")):
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                text = _read_pdf(path)
            elif ext == ".docx":
                text = _read_docx(path)
            else:
                continue
        except Exception as e:  # noqa: BLE001
            logger.warning("解析失败,跳过 %s: %s", path.name, e)
            continue
        if text.strip():
            results.append((text, path.name))
    return results
