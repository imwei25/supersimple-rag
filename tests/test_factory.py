import pytest
import rag.providers.factory as factory


def test_dispatch_llama_cpp(monkeypatch):
    captured = {}

    class FakeLlama:
        def __init__(self, cfg):
            captured["cfg"] = cfg

    monkeypatch.setattr(factory, "LlamaCppProvider", FakeLlama)
    prov = factory.make_provider({"provider": "llama_cpp", "model": "m.gguf"})
    assert isinstance(prov, FakeLlama)
    assert captured["cfg"]["model"] == "m.gguf"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        factory.make_provider({"provider": "nope"})
