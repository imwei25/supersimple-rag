from __future__ import annotations
from typing import Iterator, Iterable, Dict, Any
from pathlib import Path
import logging

from rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)


def resolve_model_path(model: str, models_dir: str = "./models") -> Path:
    """绝对路径或已存在的路径原样返回;否则在 models_dir 下解析为文件名。"""
    p = Path(model).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return Path(models_dir) / model


def _iter_content(chunks: Iterable[Dict[str, Any]]) -> Iterator[str]:
    """从 llama.cpp chat-completion 流式分块里抽取增量文本。"""
    for chunk in chunks:
        piece = chunk["choices"][0].get("delta", {}).get("content")
        if piece:
            yield piece
