# webui.py
from __future__ import annotations
import rag.proxyfix  # noqa: F401  必须在导入 gradio/网络库前清理代理
import gradio as gr

from rag.rag import RagEngine


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
    return (
        f"<details open><summary>💭 思考中…</summary>\n\n"
        f"{after.strip()}\n\n</details>"
    )


def format_chunks(chunks: list) -> str:
    """把召回的片段渲染为可折叠区块。"""
    if not chunks:
        return ""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"**[{i}] 来源:{c['source']}**\n\n{c['text']}")
    body = "\n\n---\n\n".join(parts)
    return (
        f"\n\n<details><summary>🔍 召回片段({len(chunks)} 条,点击展开)</summary>\n\n"
        f"{body}\n\n</details>"
    )


def create_demo(engine: RagEngine):
    def respond(message, history):
        answer = ""
        sources = []
        chunks = []
        for ev in engine.answer_stream(message):
            if ev["type"] == "chunks":
                chunks = ev["data"]
            elif ev["type"] == "token":
                answer += ev["data"]
                yield format_think(answer)
            else:
                sources = ev["data"]
        final = format_think(answer)
        if sources:
            final += "\n\n---\n📚 来源: " + ", ".join(sources)
        final += format_chunks(chunks)
        yield final

    def rebuild_kb():
        """流式重建:实时回显各阶段进度。重建期间旧索引仍可查询。"""
        yield "⏳ 开始重建,请稍候(首次或文档有改动时较耗时)…"
        try:
            for ev in engine.rebuild_index_iter():
                if ev["type"] == "status":
                    yield ev["msg"]
                elif ev["type"] == "done":
                    yield f"✅ {ev['stats']['message']}(当前向量数 {engine.store.count()})"
        except Exception as e:  # noqa: BLE001
            yield f"❌ 重建失败:{e}"

    with gr.Blocks(title="本地 RAG 问答") as demo:
        gr.Markdown("# 本地 RAG 问答\n"
                    "基于本地知识库的中文问答。把 PDF/Word 放入 `docs_kb/` 后点「重建知识库」。")
        with gr.Row():
            rebuild_btn = gr.Button("🔄 重建知识库", variant="secondary", scale=0)
            kb_status = gr.Markdown("")
        rebuild_btn.click(fn=rebuild_kb, inputs=None, outputs=kb_status)
        gr.ChatInterface(fn=respond)
    return demo


if __name__ == "__main__":
    from rag.config import load_config
    cfg = load_config()
    demo = create_demo(RagEngine(cfg))
    demo.queue()
    demo.launch(server_name=cfg.server["web_host"],
                server_port=cfg.server["web_port"])
