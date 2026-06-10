# app.py
from __future__ import annotations
import argparse
import os
import sys
import threading
from typing import List

import rag.proxyfix  # noqa: F401
from rag.runtime import app_root


def parse_mode(argv: List[str]) -> str:
    ap = argparse.ArgumentParser(prog="知识库问答")
    ap.add_argument("--mode", choices=["both", "api", "webui", "ingest"],
                    default="both")
    return ap.parse_args(argv).mode


def _make_engine():
    from rag.config import load_config
    from rag.rag import RagEngine
    return RagEngine(load_config())


def _cfg():
    from rag.config import load_config
    return load_config()


def _run_ingest() -> None:
    from rag.config import load_config
    from rag.rag import RagEngine
    engine = RagEngine(load_config())
    # 逐阶段打印进度(含“编码向量 N/总数”),CPU 上较慢时让控制台有反馈
    for ev in engine.rebuild_index_iter():
        if ev["type"] == "status":
            print(ev["msg"], flush=True)
        elif ev["type"] == "done":
            print(ev["stats"].get("message", ev["stats"]), flush=True)


def _run_api(engine) -> None:
    import uvicorn
    from api import create_app
    cfg = _cfg()
    uvicorn.run(create_app(engine), host=cfg.server["api_host"],
                port=cfg.server["api_port"])


def _run_webui(engine) -> None:
    from webui import create_demo
    cfg = _cfg()
    demo = create_demo(engine)
    demo.queue()
    demo.launch(server_name=cfg.server["web_host"],
                server_port=cfg.server["web_port"], inbrowser=True)


def _run_both(engine) -> None:
    """API 在后台线程,WebUI 占主线程(Gradio 需在主线程 launch)。"""
    import uvicorn
    from api import create_app
    cfg = _cfg()
    api_app = create_app(engine)
    t = threading.Thread(
        target=lambda: uvicorn.run(
            api_app, host=cfg.server["api_host"], port=cfg.server["api_port"],
            log_level="warning"),
        daemon=True)
    t.start()
    _run_webui(engine)


def run(mode: str) -> None:
    if mode == "ingest":
        _run_ingest()
        return
    engine = _make_engine()
    {"api": _run_api, "webui": _run_webui, "both": _run_both}[mode](engine)


def main() -> None:
    os.chdir(app_root())          # 切到数据根目录(向上找 config.yaml),支持程序/数据分家
    try:
        run(parse_mode(sys.argv[1:]))
    except Exception:             # noqa: BLE001
        # 打包后双击运行时,异常会让控制台瞬间关闭。冻结态下打印完整堆栈并停住,
        # 方便直接看到报错(开发态不拦截,交给正常 Traceback)。
        import traceback
        traceback.print_exc()
        if getattr(sys, "frozen", False):
            print("\n程序启动失败(见上方错误)。常见原因:config.yaml 的 device 未改为 cpu、"
                  "llm.model 未指向 models 下的 GGUF 文件名,或 models 目录缺文件。")
            input("\n按回车键关闭…")
        raise


if __name__ == "__main__":
    main()
