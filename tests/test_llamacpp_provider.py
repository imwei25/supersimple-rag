from pathlib import Path
from rag.providers.llamacpp_provider import resolve_model_path


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
