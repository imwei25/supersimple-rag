# webui.py
from __future__ import annotations
import rag.proxyfix  # noqa: F401  必须在导入 gradio/网络库前清理代理
import gradio as gr

from rag.config import load_config
from rag.rag import RagEngine

cfg = load_config()
engine = RagEngine(cfg)


def format_think(text: str) -> str:
    """把 <think>...</think> 推理内容渲染为可折叠区块,最终答案正常显示。
    流式过程中 think 未闭合时默认展开,闭合后折叠。"""
    if "<think>" not in text:
        return text
    after = text.split("<think>", 1)[1]
    if "</think>" in after:
        think, answer = after.split("</think>", 1)
        block = (
            f"<details><summary>💭 思考过程(点击展开)</summary>\n\n"
            f"{think.strip()}\n\n</details>\n\n"
        )
        return block + answer.strip()
    # 仍在思考中
    return (
        f"<details open><summary>💭 思考中…</summary>\n\n"
        f"{after.strip()}\n\n</details>"
    )


def respond(message, history):
    answer = ""
    sources = []
    for ev in engine.answer_stream(message):
        if ev["type"] == "token":
            answer += ev["data"]
            yield format_think(answer)
        else:
            sources = ev["data"]
    final = format_think(answer)
    if sources:
        final += "\n\n---\n📚 来源: " + ", ".join(sources)
    yield final


demo = gr.ChatInterface(
    fn=respond,
    title="本地 RAG 问答 (Ollama minicpm5-1b)",
    description="基于本地知识库的中文问答。先运行 python ingest.py 建库。",
)

if __name__ == "__main__":
    demo.launch(server_name=cfg.server["web_host"], server_port=cfg.server["web_port"])
