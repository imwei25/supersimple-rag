# rag/providers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """逐 token / 逐块 yield 文本。"""
        raise NotImplementedError

    @abstractmethod
    def health(self) -> bool:
        """后端是否可用。"""
        raise NotImplementedError
