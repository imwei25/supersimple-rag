# rag/providers/factory.py
from __future__ import annotations

from rag.providers.base import LLMProvider
from rag.providers.ollama_provider import OllamaProvider
from rag.providers.transformers_provider import TransformersProvider


def make_provider(llm_cfg: dict) -> LLMProvider:
    provider = llm_cfg.get("provider", "ollama")
    if provider == "ollama":
        return OllamaProvider(llm_cfg)
    if provider == "transformers":
        return TransformersProvider(llm_cfg)
    raise ValueError(f"未知 provider: {provider}")
