# llama.cpp 生成后端 + Windows CPU 打包 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增进程内 llama.cpp(GGUF)生成后端,并提供统一入口 `app.py` 与 PyInstaller onedir 打包方案,使项目在 Windows CPU 上免装 Ollama 即可运行。

**Architecture:** 在现有 `LLMProvider` 抽象下新增 `LlamaCppProvider`(用 `llama-cpp-python` 的 chat completion 流式生成);把 `api.py`/`webui.py` 重构成接收共享 `engine` 的 builder 函数;新增 `app.py` 以 `--mode`(默认 both)启动 API+WebUI 或建库;新增冻结态路径处理(启动即 `chdir` 到 exe 所在目录),配合 `app.spec` 打成 onedir,模型/配置/知识库外置。

**Tech Stack:** Python 3.10、llama-cpp-python(CPU)、sentence-transformers、chromadb、FastAPI/uvicorn、Gradio、PyInstaller、pytest。

参考设计:`docs/superpowers/specs/2026-06-09-llamacpp-provider-and-windows-packaging-design.md`

---

### Task 1: llama.cpp 模型路径解析(纯函数)

**Files:**
- Create: `rag/providers/llamacpp_provider.py`
- Test: `tests/test_llamacpp_provider.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_llamacpp_provider.py
from pathlib import Path
from rag.providers.llamacpp_provider import resolve_model_path


def test_absolute_path_used_as_is():
    p = resolve_model_path("/abs/dir/m.gguf", models_dir="./models")
    assert p == Path("/abs/dir/m.gguf")


def test_bare_filename_resolved_under_models_dir():
    p = resolve_model_path("m.gguf", models_dir="./models")
    assert p == Path("./models/m.gguf")


def test_existing_relative_path_kept(tmp_path):
    f = tmp_path / "here.gguf"
    f.write_text("x")
    p = resolve_model_path(str(f), models_dir="./models")
    assert p == Path(str(f))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llamacpp_provider.py -v`
Expected: FAIL,`ModuleNotFoundError: rag.providers.llamacpp_provider`

- [ ] **Step 3: 写最小实现(仅路径解析 + chunk 适配器)**

```python
# rag/providers/llamacpp_provider.py
from __future__ import annotations
from typing import Iterator, Iterable, Dict, Any
from pathlib import Path
import logging

from rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)


def resolve_model_path(model: str, models_dir: str = "./models") -> Path:
    """绝对路径或已存在的路径原样返回;否则在 models_dir 下解析为文件名。"""
    p = Path(model).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return Path(models_dir) / model


def _iter_content(chunks: Iterable[Dict[str, Any]]) -> Iterator[str]:
    """从 llama.cpp chat-completion 流式分块里抽取增量文本。"""
    for chunk in chunks:
        piece = chunk["choices"][0].get("delta", {}).get("content")
        if piece:
            yield piece
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llamacpp_provider.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add rag/providers/llamacpp_provider.py tests/test_llamacpp_provider.py
git commit -m "feat: llama.cpp provider 模型路径解析与流式分块适配器"
```

---

### Task 2: chunk 适配器测试 + `LlamaCppProvider` 类

**Files:**
- Modify: `rag/providers/llamacpp_provider.py`
- Test: `tests/test_llamacpp_provider.py`

- [ ] **Step 1: 写失败测试(适配器 + 类的 stream/health,注入假 llm 不加载真模型)**

```python
# 追加到 tests/test_llamacpp_provider.py
from rag.providers.llamacpp_provider import _iter_content, LlamaCppProvider


def _fake_chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def test_iter_content_skips_empty():
    chunks = [_fake_chunk("你"), {"choices": [{"delta": {}}]}, _fake_chunk("好")]
    assert "".join(_iter_content(chunks)) == "你好"


class _FakeLlama:
    def create_chat_completion(self, messages, stream, temperature, max_tokens):
        assert stream is True
        return [_fake_chunk("答"), _fake_chunk("案")]


def test_provider_stream_with_injected_llm(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_text("x")
    prov = LlamaCppProvider.__new__(LlamaCppProvider)   # 跳过 __init__ 的真实加载
    prov.model_path = f
    prov.temperature = 0.2
    prov.max_tokens = 16
    prov.llm = _FakeLlama()
    assert "".join(prov.stream("问题")) == "答案"
    assert prov.health() is True


def test_health_false_when_model_missing():
    prov = LlamaCppProvider.__new__(LlamaCppProvider)
    prov.model_path = Path("/no/such.gguf")
    prov.llm = object()
    assert prov.health() is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llamacpp_provider.py -v`
Expected: FAIL,`ImportError: cannot import name 'LlamaCppProvider'`

- [ ] **Step 3: 实现 `LlamaCppProvider` 类(追加到同文件)**

```python
class LlamaCppProvider(LLMProvider):
    """进程内 llama.cpp(GGUF)生成后端。CPU 可跑、完全离线。
    用 chat completion 让 GGUF 自带对话模板生效,instruct 模型更稳。
    加载失败抛异常,由 factory/调用方决定降级。"""

    def __init__(self, cfg: dict):
        self.model_path = resolve_model_path(
            cfg["model"], cfg.get("models_dir", "./models"))
        self.temperature = cfg.get("temperature", 0.2)
        self.max_tokens = cfg.get("max_tokens", 2096)
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF 模型不存在: {self.model_path}")
        from llama_cpp import Llama          # 延迟导入,避免无该依赖时整体不可用
        n_threads = cfg.get("n_threads") or None   # 0/None -> 自动按核数
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=cfg.get("n_ctx", 4096),
            n_threads=n_threads,
            n_gpu_layers=cfg.get("n_gpu_layers", 0),
            verbose=False,
        )

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            chunks = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            yield from _iter_content(chunks)
        except Exception as e:  # noqa: BLE001
            yield f"\n[错误] llama.cpp 生成失败: {e}"

    def health(self) -> bool:
        return self.model_path.exists() and getattr(self, "llm", None) is not None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llamacpp_provider.py -v`
Expected: PASS(6 passed)

- [ ] **Step 5: 提交**

```bash
git add rag/providers/llamacpp_provider.py tests/test_llamacpp_provider.py
git commit -m "feat: LlamaCppProvider 流式生成与 health(注入式单测)"
```

---

### Task 3: factory 接入 `llama_cpp`

**Files:**
- Modify: `rag/providers/factory.py`
- Test: `tests/test_factory.py`

- [ ] **Step 1: 写失败测试(monkeypatch 掉真实构造,只验证分发)**

```python
# tests/test_factory.py
import pytest
import rag.providers.factory as factory


def test_dispatch_llama_cpp(monkeypatch):
    captured = {}

    class FakeLlama:
        def __init__(self, cfg):
            captured["cfg"] = cfg

    monkeypatch.setattr(factory, "LlamaCppProvider", FakeLlama)
    prov = factory.make_provider({"provider": "llama_cpp", "model": "m.gguf"})
    assert isinstance(prov, FakeLlama)
    assert captured["cfg"]["model"] == "m.gguf"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        factory.make_provider({"provider": "nope"})
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_factory.py -v`
Expected: FAIL,`AttributeError: ... has no attribute 'LlamaCppProvider'`

- [ ] **Step 3: 修改 factory**

```python
# rag/providers/factory.py 顶部 import 区追加:
from rag.providers.llamacpp_provider import LlamaCppProvider

# 在 make_provider 内,return OllamaProvider 之后、TransformersProvider 之前追加:
    if provider == "llama_cpp":
        return LlamaCppProvider(llm_cfg)
```

完整 `make_provider` 应为:

```python
def make_provider(llm_cfg: dict) -> LLMProvider:
    provider = llm_cfg.get("provider", "ollama")
    if provider == "ollama":
        return OllamaProvider(llm_cfg)
    if provider == "llama_cpp":
        return LlamaCppProvider(llm_cfg)
    if provider == "transformers":
        return TransformersProvider(llm_cfg)
    raise ValueError(f"未知 provider: {provider}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_factory.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add rag/providers/factory.py tests/test_factory.py
git commit -m "feat: factory 支持 llama_cpp provider"
```

---

### Task 4: 冻结态运行目录(`rag/runtime.py`)

**Files:**
- Create: `rag/runtime.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_runtime.py
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_runtime.py -v`
Expected: FAIL,`ModuleNotFoundError: rag.runtime`

- [ ] **Step 3: 实现**

```python
# rag/runtime.py
from __future__ import annotations
import sys
from pathlib import Path


def base_dir() -> Path:
    """运行基准目录:PyInstaller 冻结态用 exe 所在目录,否则用当前工作目录。
    用于让 config.yaml / models / docs_kb / chroma_db 等相对路径在打包后仍可用。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_runtime.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add rag/runtime.py tests/test_runtime.py
git commit -m "feat: 冻结态运行基准目录 base_dir"
```

---

### Task 5: 重构 api.py / webui.py 为 builder 函数

**Files:**
- Modify: `api.py`
- Modify: `webui.py`
- Test: `tests/test_app_builders.py`

说明:把"创建 engine"与"组装 app/demo"分开,使 `app.py both` 模式能共享同一个
`engine`(只加载一次 embedder/reranker/GGUF)。保留各自 `__main__` 行为不变。

- [ ] **Step 1: 写失败测试(builder 接收 engine、返回对象、不启动服务)**

```python
# tests/test_app_builders.py
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_app_builders.py -v`
Expected: FAIL,`ImportError: cannot import name 'create_app'`

- [ ] **Step 3: 重构 api.py**

把模块级 `cfg/engine/app` 改为 `create_app(engine)` 工厂,`__main__` 内自建 engine:

```python
# api.py(整体替换)
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
```

- [ ] **Step 4: 重构 webui.py**

把建 UI 的部分包进 `create_demo(engine)`,`respond`/`rebuild_kb` 改为闭包引用传入的
`engine`。`__main__` 内自建 engine 并 launch。保留 `format_think`/`format_chunks` 等纯函数在模块级。

```python
# webui.py 末尾:把原 `with gr.Blocks(...) as demo:` 整段包成函数
def create_demo(engine):
    import gradio as gr

    def respond(message, history):
        # ……原 respond 逻辑,把对全局 engine 的引用改为这里的 engine……
        ...

    def rebuild_kb():
        # ……原 rebuild_kb 逻辑,引用传入的 engine……
        ...

    with gr.Blocks(title="本地 RAG 问答") as demo:
        gr.Markdown("# 本地 RAG 问答\n基于本地知识库的中文问答。"
                    "把 PDF/Word 放入 `docs_kb/` 后点「重建知识库」。")
        with gr.Row():
            rebuild_btn = gr.Button("🔄 重建知识库", variant="secondary", scale=0)
            kb_status = gr.Markdown("")
        rebuild_btn.click(fn=rebuild_kb, inputs=None, outputs=kb_status)
        gr.ChatInterface(fn=respond)
    return demo


if __name__ == "__main__":
    from rag.config import load_config
    from rag.rag import RagEngine
    cfg = load_config()
    demo = create_demo(RagEngine(cfg))
    demo.queue()
    demo.launch(server_name=cfg.server["web_host"],
                server_port=cfg.server["web_port"])
```

注意:`respond`/`rebuild_kb` 的内部实现整体搬入 `create_demo`,凡引用旧全局 `engine`
处改为函数参数 `engine`;`format_think`、`format_chunks` 等保持模块级不变,供 `respond` 调用。

- [ ] **Step 5: 运行测试确认通过 + 回归既有测试**

Run: `pytest tests/test_app_builders.py -v && pytest -q`
Expected: 新测试 PASS;既有用例全部通过(无回归)。

- [ ] **Step 6: 提交**

```bash
git add api.py webui.py tests/test_app_builders.py
git commit -m "refactor: api/webui 改为接收共享 engine 的 builder 函数"
```

---

### Task 6: 统一入口 `app.py`(--mode,默认 both)

**Files:**
- Create: `app.py`
- Test: `tests/test_app_entry.py`

- [ ] **Step 1: 写失败测试(参数解析 + 各 mode 分发,注入假回调)**

```python
# tests/test_app_entry.py
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_app_entry.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: 实现 app.py**

```python
# app.py
from __future__ import annotations
import argparse
import os
import sys
import threading
import webbrowser
from typing import List

import rag.proxyfix  # noqa: F401
from rag.runtime import base_dir


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
    stats = RagEngine(load_config()).rebuild_index()
    print(stats.get("message", stats))


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
                port=cfg.server["web_port"], inbrowser=True)


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
    os.chdir(base_dir())          # 让 ./config.yaml ./models 等相对路径在打包后可用
    run(parse_mode(sys.argv[1:]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_app_entry.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
git add app.py tests/test_app_entry.py
git commit -m "feat: 统一入口 app.py(--mode,默认 both)"
```

---

### Task 7: config.yaml 切到 llama_cpp(首测用 MiniCPM5 GGUF)

**Files:**
- Modify: `config.yaml`
- Test: `tests/test_config.py`(回归,确认仍可加载)

- [ ] **Step 1: 改 config.yaml 的 llm 段**

```yaml
llm:
  provider: llama_cpp        # ollama / llama_cpp
  model: /media/wei/samsung/model/MiniCPM5-1B-F16.gguf
  models_dir: ./models       # 当 model 为裸文件名时在此解析
  n_ctx: 4096
  n_threads: 0               # 0 = 自动按核数
  n_gpu_layers: 0            # CPU 全 0
  temperature: 0.2
  max_tokens: 2096
  # 仍兼容 ollama:把 provider 改回 ollama 并提供 ollama_base_url 即可
  ollama_base_url: http://localhost:11434
```

- [ ] **Step 2: 运行 config 回归测试**

Run: `pytest tests/test_config.py -v`
Expected: PASS(配置仍能正常加载)

- [ ] **Step 3: 真实 smoke(本机有该 GGUF 时)— 验证 provider 能出字**

Run:
```bash
python -c "from rag.config import load_config; from rag.providers.factory import make_provider; \
p=make_provider(load_config().llm); print('health', p.health()); \
print(''.join(list(p.stream('用一句话说明什么是高血压'))[:50]))"
```
Expected: 打印 `health True` 且随后有中文输出(若未装 `llama-cpp-python` 则先 `pip install -r requirements-llamacpp.txt`)。

- [ ] **Step 4: 提交**

```bash
git add config.yaml
git commit -m "config: 默认生成后端切到 llama_cpp(MiniCPM5 GGUF 首测)"
```

---

### Task 8: 打包依赖与 spec

**Files:**
- Create: `requirements-llamacpp.txt`
- Create: `app.spec`
- Create: `build_windows.ps1`

- [ ] **Step 1: 写 `requirements-llamacpp.txt`**

```text
# 进程内 llama.cpp 生成后端(CPU wheel)。Windows CPU 打包必装。
llama-cpp-python==0.3.2
# Windows 打包工具
pyinstaller==6.11.1
```

- [ ] **Step 2: 写 `app.spec`**

```python
# app.spec —— PyInstaller onedir 配置(在 Windows 上运行 pyinstaller app.spec)
# 模型/config.yaml/docs_kb/chroma_db 均不打入归档,保持 exe 同级外置。
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = [], [], []
for pkg in ["gradio", "gradio_client", "safehttpx", "groovy",
            "chromadb", "sentence_transformers", "transformers",
            "tokenizers", "jieba", "llama_cpp"]:
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

binaries += collect_dynamic_libs("llama_cpp")   # llama.dll / ggml*.dll
hiddenimports += ["onnxruntime", "hnswlib", "tiktoken_ext",
                  "uvicorn.logging", "uvicorn.protocols",
                  "uvicorn.protocols.http.auto", "uvicorn.lifespan.on"]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["nvidia", "triton", "tensorboard"],   # CPU 瘦身:排除 CUDA 相关
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name="知识库问答", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="知识库问答")
```

- [ ] **Step 3: 写 `build_windows.ps1`**

```powershell
# build_windows.ps1 —— 在 Windows 上构建 onedir 包
# 前置:已装 Python 3.10、已建好 .venv 并装 requirements.txt + requirements-llamacpp.txt
# 重要:torch 用 CPU-only wheel,否则包会膨胀数 GB
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-llamacpp.txt

pyinstaller --noconfirm app.spec

# 产物在 dist/知识库问答/。把外置资源放进去:
New-Item -ItemType Directory -Force -Path "dist/知识库问答/models" | Out-Null
New-Item -ItemType Directory -Force -Path "dist/知识库问答/docs_kb" | Out-Null
Copy-Item config.yaml "dist/知识库问答/config.yaml" -Force
# 然后手动拷入 models/(GGUF + bge-large-zh-v1.5/ + bge-reranker-base/)
Write-Host "构建完成。请把 GGUF 与 bge/reranker 模型放入 dist/知识库问答/models/,"
Write-Host "并把 config.yaml 的 device 改为 cpu、model 改为 models 下的 GGUF 文件名。"
```

- [ ] **Step 4: 提交(本步无自动化测试,产物需在 Windows 验证)**

```bash
git add requirements-llamacpp.txt app.spec build_windows.ps1
git commit -m "build: PyInstaller onedir 打包(spec + Windows 构建脚本 + llamacpp 依赖)"
```

---

### Task 9: 分发用 .bat 与 README

**Files:**
- Create: `dist_assets/① 建库.bat`
- Create: `dist_assets/② 启动.bat`
- Modify: `README.md`

- [ ] **Step 1: 写两个 .bat(随包分发,放到 exe 同级)**

```bat
REM dist_assets/① 建库.bat
@echo off
chcp 65001 >nul
"%~dp0知识库问答.exe" --mode ingest
pause
```

```bat
REM dist_assets/② 启动.bat
@echo off
chcp 65001 >nul
"%~dp0知识库问答.exe" --mode both
```

- [ ] **Step 2: README 增补 Windows 打包/运行章节**

在 `README.md` 末尾追加:

```markdown
## Windows CPU 打包与运行

本项目支持两种生成后端:`ollama`(默认开发用)与 `llama_cpp`(进程内 GGUF,
Windows CPU 免装 Ollama)。切换只改 `config.yaml` 的 `llm.provider`。

### 在 Windows 上构建 exe
PyInstaller 不能跨平台编译,需在 Windows 机器上构建:
```powershell
.\build_windows.ps1
```
产物在 `dist\知识库问答\`。把以下资源放入该目录:
- `models\<生成模型>.gguf`、`models\bge-large-zh-v1.5\`、`models\bge-reranker-base\`
- `config.yaml`(`embedding.device: cpu`、`llm.provider: llama_cpp`、`llm.model` 指向上面的 GGUF 文件名)
- 把 `dist_assets\` 里的两个 .bat 拷到 exe 同级

### 终端用户使用
1. 把 PDF/Word 放进 `docs_kb\`
2. 双击「① 建库.bat」建立知识库
3. 双击「② 启动.bat」→ 浏览器自动打开问答页(API 同时在 8001)
```

- [ ] **Step 3: 提交**

```bash
git add "dist_assets/① 建库.bat" "dist_assets/② 启动.bat" README.md
git commit -m "docs: Windows 分发 .bat 与打包运行说明"
```

---

## 自检(Self-Review)

- **Spec 覆盖**:Section 1(provider/factory/config)→ Task 1-3、7;Section 2
  (app.py/布局/spec/构建/CPU)→ Task 6、8、9;冻结态路径 → Task 4;
  builder 重构(both 共享 engine)→ Task 5。错误处理(加载失败降级)在 Task 2
  `stream` try/except 与 `health`。测试散落各 Task。无遗漏。
- **占位符**:无 TBD/TODO;每个代码步骤含完整代码。
- **类型/命名一致**:`resolve_model_path`、`_iter_content`、`LlamaCppProvider`、
  `create_app`、`create_demo`、`parse_mode`、`run`、`base_dir` 在定义与引用处一致。
- **注意**:Task 5 webui 重构需把 `respond`/`rebuild_kb` 内对旧全局 `engine` 的
  引用全部改为闭包参数;`format_think`/`format_chunks` 保持模块级。
