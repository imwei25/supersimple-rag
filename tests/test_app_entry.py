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
