# rag/splitter.py
from __future__ import annotations
from typing import List


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks
