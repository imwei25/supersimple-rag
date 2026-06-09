# api.py
from __future__ import annotations
import rag.proxyfix  # noqa: F401  必须在导入网络库前清理代理
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.rag import RagEngine


class ChatRequest(BaseModel):
    question: str


def create_app(engine: RagEngine) -> FastAPI:
    app = FastAPI(title="本地 RAG 问答")

    @app.get("/health")
    def health():
        return engine.health()

    @app.post("/chat")
    def chat(req: ChatRequest):
        def gen():
            for ev in engine.answer_stream(req.question):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


if __name__ == "__main__":
    import uvicorn
    from rag.config import load_config
    cfg = load_config()
    eng = RagEngine(cfg)
    uvicorn.run(create_app(eng), host=cfg.server["api_host"],
                port=cfg.server["api_port"])
