import sys
from pathlib import Path
from rag.runtime import base_dir


def test_base_dir_is_cwd_when_not_frozen(monkeypatch, tmp_path):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.chdir(tmp_path)
    assert base_dir() == tmp_path


def test_base_dir_is_exe_dir_when_frozen(monkeypatch, tmp_path):
    exe = tmp_path / "app.exe"
    exe.write_text("x")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe), raising=False)
    assert base_dir() == tmp_path
