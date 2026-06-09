import pytest
from pathlib import Path
from rag.providers.llamacpp_provider import resolve_model_path, _iter_content, LlamaCppProvider


def test_init_raises_when_model_missing():
    with pytest.raises(FileNotFoundError):
        LlamaCppProvider({"model": "/no/such/model.gguf"})


def test_absolute_path_used_as_is():
    p = resolve_model_path("/abs/dir/m.gguf", models_dir="./models")
    assert p == Path("/abs/dir/m.gguf")


def test_bare_filename_resolved_under_models_dir():
    p = resolve_model_path("m.gguf", models_dir="./models")
    assert p == Path("./models/m.gguf")


def test_existing_relative_path_kept(tmp_path):
    f = tmp_path / "here.gguf"
    f.write_text("x")
    p = resolve_model_path(str(f), models_dir="./models")
    assert p == Path(str(f))


def _fake_chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def test_iter_content_skips_empty():
    chunks = [_fake_chunk("你"), {"choices": [{"delta": {}}]}, _fake_chunk("好")]
    assert "".join(_iter_content(chunks)) == "你好"


class _FakeLlama:
    def create_chat_completion(self, messages, stream, temperature, max_tokens):
        assert stream is True
        return [_fake_chunk("答"), _fake_chunk("案")]


def test_provider_stream_with_injected_llm(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_text("x")
    prov = LlamaCppProvider.__new__(LlamaCppProvider)   # 跳过 __init__ 的真实加载
    prov.model_path = f
    prov.temperature = 0.2
    prov.max_tokens = 16
    prov.llm = _FakeLlama()
    assert "".join(prov.stream("问题")) == "答案"
    assert prov.health() is True


def test_health_false_when_model_missing():
    prov = LlamaCppProvider.__new__(LlamaCppProvider)
    prov.model_path = Path("/no/such.gguf")
    prov.llm = object()
    assert prov.health() is False
