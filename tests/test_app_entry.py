import app as appmod


def test_parse_mode_default_both():
    assert appmod.parse_mode([]) == "both"


def test_parse_mode_explicit():
    assert appmod.parse_mode(["--mode", "api"]) == "api"
    assert appmod.parse_mode(["--mode", "ingest"]) == "ingest"


def test_run_dispatches_ingest(monkeypatch):
    calls = []
    monkeypatch.setattr(appmod, "_run_ingest", lambda: calls.append("ingest"))
    monkeypatch.setattr(appmod, "_run_api", lambda eng: calls.append("api"))
    monkeypatch.setattr(appmod, "_run_webui", lambda eng: calls.append("webui"))
    monkeypatch.setattr(appmod, "_run_both", lambda eng: calls.append("both"))
    monkeypatch.setattr(appmod, "_make_engine", lambda: object())
    appmod.run("ingest")
    assert calls == ["ingest"]


def test_run_dispatches_both(monkeypatch):
    calls = []
    monkeypatch.setattr(appmod, "_run_both", lambda eng: calls.append("both"))
    monkeypatch.setattr(appmod, "_make_engine", lambda: object())
    appmod.run("both")
    assert calls == ["both"]


def test_main_reraises_and_does_not_pause_when_not_frozen(monkeypatch):
    import sys
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(appmod, "parse_mode", lambda argv: "both")
    monkeypatch.setattr(appmod.os, "chdir", lambda p: None)

    def boom(_mode):
        raise RuntimeError("启动失败")
    monkeypatch.setattr(appmod, "run", boom)

    def _no_input(*a, **k):
        raise AssertionError("非冻结态不应调用 input()")
    monkeypatch.setattr("builtins.input", _no_input)

    import pytest
    with pytest.raises(RuntimeError):
        appmod.main()
