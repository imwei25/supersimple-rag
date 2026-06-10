import sys
from pathlib import Path
from rag.runtime import base_dir, app_root


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


def test_app_root_walks_up_to_config(monkeypatch, tmp_path):
    # 布局:<root>/config.yaml,exe 在 <root>/程序/app.exe
    (tmp_path / "config.yaml").write_text("x")
    prog = tmp_path / "程序"
    prog.mkdir()
    exe = prog / "app.exe"
    exe.write_text("x")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe), raising=False)
    assert app_root() == tmp_path          # 向上找到含 config.yaml 的数据根


def test_app_root_falls_back_to_base_dir(monkeypatch, tmp_path):
    exe = tmp_path / "app.exe"
    exe.write_text("x")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe), raising=False)
    # 没有 config.yaml → 回退到 exe 目录(同级布局,向后兼容)
    assert app_root() == tmp_path
