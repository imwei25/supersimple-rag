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


class LlamaCppProvider(LLMProvider):
    """进程内 llama.cpp(GGUF)生成后端。CPU 可跑、完全离线。
    用 chat completion 让 GGUF 自带对话模板生效,instruct 模型更稳。
    加载失败抛异常,由 factory/调用方决定降级。"""

    def __init__(self, cfg: dict):
        self.model_path = resolve_model_path(
            cfg["model"], cfg.get("models_dir", "./models"))
        self.temperature = cfg.get("temperature", 0.2)
        self.max_tokens = cfg.get("max_tokens", 2096)
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF 模型不存在: {self.model_path}")
        from llama_cpp import Llama          # 延迟导入,避免无该依赖时整体不可用
        n_threads = cfg.get("n_threads") or None   # falsy(0/None/缺省)→ None,由 llama.cpp 自动选核数
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=cfg.get("n_ctx", 4096),
            n_threads=n_threads,
            n_gpu_layers=cfg.get("n_gpu_layers", 0),
            verbose=False,
        )

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            chunks = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            yield from _iter_content(chunks)
        except Exception as e:  # noqa: BLE001
            yield f"\n[错误] llama.cpp 生成失败: {e}"

    def health(self) -> bool:
        return self.model_path.exists() and getattr(self, "llm", None) is not None
