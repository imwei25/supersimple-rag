import types


def _fake_engine():
    e = types.SimpleNamespace()
    e.answer_stream = lambda q: iter([{"type": "token", "data": "x"}])
    e.health = lambda: {"llm": True, "doc_count": 0}
    e.store = types.SimpleNamespace(count=lambda: 0)
    e.rebuild_index_iter = lambda: iter([])
    return e


def test_create_api_app_returns_fastapi():
    from fastapi import FastAPI
    from api import create_app
    app = create_app(_fake_engine())
    assert isinstance(app, FastAPI)
    paths = {r.path for r in app.routes}
    assert "/chat" in paths and "/health" in paths


def test_create_webui_demo_builds():
    from webui import create_demo
    demo = create_demo(_fake_engine())
    assert demo is not None
