# rag/runtime.py
from __future__ import annotations
import sys
from pathlib import Path


def base_dir() -> Path:
    """运行基准目录:PyInstaller 冻结态用 exe 所在目录,否则用当前工作目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def app_root(marker: str = "config.yaml") -> Path:
    """数据根目录:从基准目录起向上查找含 marker(默认 config.yaml)的目录并返回。
    这样可把 exe + _internal 放进子文件夹(如 程序\\),把 config.yaml / models /
    chroma_db / docs_kb 放在外层数据目录——重打包只替换子文件夹,数据目录打包工具
    永远碰不到。找不到 marker 时回退到基准目录(行为同旧的同级布局,向后兼容)。"""
    start = base_dir()
    for d in (start, *start.parents):
        if (d / marker).exists():
            return d
    return start
