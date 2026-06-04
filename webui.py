# webui.py
from __future__ import annotations
import rag.proxyfix  # noqa: F401  必须在导入 gradio/网络库前清理代理
import gradio as gr

from rag.config import load_config
from rag.rag import RagEngine

cfg = load_config()
engine = RagEngine(cfg)


def respond(message, history):
    answer = ""
    sources = []
    for ev in engine.answer_stream(message):
        if ev["type"] == "token":
            answer += ev["data"]
            yield answer
        else:
            sources = ev["data"]
    if sources:
        yield answer + "\n\n---\n📚 来源: " + ", ".join(sources)


demo = gr.ChatInterface(
    fn=respond,
    title="本地 RAG 问答 (Ollama minicpm5-1b)",
    description="基于本地知识库的中文问答。先运行 python ingest.py 建库。",
)

if __name__ == "__main__":
    demo.launch(server_name=cfg.server["web_host"], server_port=cfg.server["web_port"])
