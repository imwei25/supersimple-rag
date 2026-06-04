# rag/providers/transformers_provider.py
from __future__ import annotations
from typing import Iterator

from rag.providers.base import LLMProvider


class TransformersProvider(LLMProvider):
    """预留:后续用 HuggingFace transformers 本地加载模型。
    TODO: 实现 AutoModelForCausalLM + TextIteratorStreamer 流式生成。"""

    def __init__(self, cfg: dict):
        raise NotImplementedError(
            "transformers provider 尚未实现,请在 config.yaml 中使用 provider: ollama"
        )

    def stream(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError

    def health(self) -> bool:
        raise NotImplementedError
