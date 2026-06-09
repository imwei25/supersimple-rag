# rag/runtime.py
from __future__ import annotations
import sys
from pathlib import Path


def base_dir() -> Path:
    """运行基准目录:PyInstaller 冻结态用 exe 所在目录,否则用当前工作目录。
    用于让 config.yaml / models / docs_kb / chroma_db 等相对路径在打包后仍可用。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()
