# rag/providers/ollama_provider.py
from __future__ import annotations
from typing import Iterator
import json
import requests

from rag.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, cfg: dict):
        self.base_url = cfg["ollama_base_url"].rstrip("/")
        self.model = cfg["model"]
        self.temperature = cfg.get("temperature", 0.3)
        self.max_tokens = cfg.get("max_tokens", 1024)

    def stream(self, prompt: str) -> Iterator[str]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            with requests.post(url, json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    if data.get("response"):
                        yield data["response"]
                    if data.get("done"):
                        break
        except requests.exceptions.RequestException as e:
            yield f"\n[错误] 无法连接 Ollama,请确认 `ollama serve` 正在运行: {e}"

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False
