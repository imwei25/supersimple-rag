# 设计:llama.cpp 生成后端 + Windows CPU 打包

日期:2026-06-09
状态:已与用户确认,待评审

## 背景与目标

当前 RAG 系统的生成环节只支持 Ollama(`OllamaProvider`,走 `http://localhost:11434`)。
为了让项目能**轻松迁移到 Windows CPU 环境、双击即用、无需安装 Ollama**,需要:

1. 新增第二种 LLM 接入方式:**进程内 llama.cpp(GGUF)**,纯 CPU 可跑、完全离线;
2. 设计一套 **PyInstaller 打包方案**,在 Windows 上构建出一个 onedir 程序,
   拷贝过去即可运行(依赖随包,知识库首次运行自建)。

非目标:不替换 embedder/reranker(继续走 sentence-transformers CPU);
不做 GPU 优化;不打包知识库本身。

## 决策记录(用户确认)

- 新后端:**llama.cpp(GGUF)进程内**(`llama-cpp-python`)。
- 打包边界:**含依赖、含模型,不含知识库**(`docs_kb`/`chroma_db` 首次运行自建)。
- 打包入口:**API + WebUI 都要**,统一为一个 `app.py`,**无参默认同时起两者**。
- 打包方式:**PyInstaller `--onedir` + 模型/配置/知识库外置于 exe 同级**。
- 默认 embedder:**bge-large-zh-v1.5**(CPU 编码较慢、包较大,换取召回质量)。
- 首测 GGUF:`/media/wei/samsung/model/MiniCPM5-1B-F16.gguf`(2.1GB,先打通链路);
  正式打包再换量化的 Qwen 3B(更小更强)。

## Section 1 — LlamaCppProvider

### 新文件 `rag/providers/llamacpp_provider.py`

实现 `LLMProvider` 接口(与 `OllamaProvider` 对等):

- 依赖 `llama-cpp-python`(CPU wheel)。`__init__` 一次性加载 GGUF(常驻)。
- **模型路径解析**:若 `cfg["model"]` 是绝对路径或当前能直接找到,直接用;
  否则在模型目录(默认 `./models/`)下解析为 `models/<model>`。
- `stream(prompt)`:用 `create_chat_completion(messages=[{"role":"user","content":prompt}],
  stream=True, temperature, max_tokens)`,逐块 `yield choices[0]["delta"].get("content","")`。
  **用 chat completion 而非 raw completion**:让 GGUF 自带对话模板自动套用,
  instruct 模型更稳(减少 `<think>` 失控/不收尾)。
- `health()`:模型文件存在且已成功加载返回 True;加载失败时优雅降级
  (try/except,给出可读错误信息),不抛断主流程。

### `factory.py`

增加分支:`provider == "llama_cpp"` → `LlamaCppProvider`。
`transformers` stub 保留不动。

### `config.yaml`

复用 `model` 字段当 GGUF 文件名/路径:

```yaml
llm:
  provider: llama_cpp        # ollama / llama_cpp
  model: /media/wei/samsung/model/MiniCPM5-1B-F16.gguf   # 绝对路径或 models/ 下文件名
  n_ctx: 4096
  n_threads: 0               # 0 = 自动按核数
  n_gpu_layers: 0            # CPU 全 0
  temperature: 0.2
  max_tokens: 2096
```

`embedder`/`reranker` 配置不变;三种 provider 共用同一 `Config` 与 `RagEngine`。

### requirements

新增可选依赖文件 `requirements-llamacpp.txt`(含 `llama-cpp-python`),
避免强制所有用户安装 C 扩展。

## Section 2 — Windows 打包

### 统一入口 `app.py`(新增)

```
app.py             # 无参 = both
app.py --mode both # 同进程起 API(8001) + WebUI(7860)
app.py --mode webui
app.py --mode api
app.py --mode ingest
```

`both` 模式:在后台线程起 uvicorn(api),主线程起 gradio;或反之。
复用现有 `api.py` / `webui.py` 的 app 对象,不重写业务。

### 分发目录布局

```
知识库问答/
├─ 知识库问答.exe           ← PyInstaller onedir 入口(app.py)
├─ _internal/               ← Python + 所有依赖(打包产物)
├─ config.yaml              ← 外置可改(provider: llama_cpp, device: cpu)
├─ models/                  ← 外置可换
│   ├─ <生成模型>.gguf
│   ├─ bge-large-zh-v1.5/
│   └─ bge-reranker-base/
├─ docs_kb/                 ← 用户放 PDF/Word
├─ chroma_db/               ← 首次建库生成
├─ ① 建库.bat              → exe --mode ingest
└─ ② 启动.bat              → exe --mode both(并打开浏览器)
```

模型/配置/知识库均在 exe 同级,**不进 PyInstaller 归档**。

### PyInstaller spec 要点(`app.spec`)

- `--collect-all`:`gradio`、`gradio_client`、`safehttpx`、`groovy`、
  `chromadb`、`sentence_transformers`、`transformers`、`tokenizers`、
  `jieba`、`llama_cpp`。
- `llama_cpp`:`collect_dynamic_libs("llama_cpp")` 带上 `llama.dll`/`ggml*.dll`。
- **CPU 瘦身**:构建环境装 CPU-only torch(`--index-url .../cpu`)与 CPU 版
  `llama-cpp-python`;spec `excludes` 掉 `nvidia*` 与 CUDA 相关,避免膨胀数 GB。
- chromadb 相关 hidden-imports:`onnxruntime`、`hnswlib`、`tiktoken_ext`。
- `datas`:**不**打入 `config.yaml`/`models`/`docs_kb`(保持外置)。
- 路径解析:代码读取 `config.yaml`/`models` 时以 **exe 所在目录**为基准
  (冻结态用 `sys.executable` 的目录,而非 `__file__`),需在 `app.py` 统一处理。

### 构建脚本 `build_windows.ps1`

`pyinstaller app.spec`。

⚠️ **限制**:PyInstaller 不跨平台编译。**必须在 Windows 上构建**才能得到
Windows `.exe`。spec 与脚本在本仓库提供,出 exe 这一步在 Windows 机器执行。

## 错误处理与降级

- llama.cpp 加载失败 → `health()` 返回 False,WebUI/API 显示可读错误;不崩。
- 冻结态找不到 `models/` 或 GGUF → 启动时明确提示"请把模型放入 models/"。
- 知识库为空 → 复用现有"请先建库"提示。

## 测试

- 单测:`LlamaCppProvider` 路径解析、`stream` 产出非空、`health` 真值
  (用 `MiniCPM5-1B-F16.gguf` 做一次真实 smoke;CI 无模型时跳过/标记)。
- 端到端:`app.py --mode ingest` 建库 → `--mode both` 问答冒烟。
- 冻结后:Windows 上跑一次 建库.bat → 启动.bat 全链路冒烟。

## 影响面

- 新增:`rag/providers/llamacpp_provider.py`、`app.py`、`app.spec`、
  `build_windows.ps1`、`requirements-llamacpp.txt`、测试。
- 修改:`rag/providers/factory.py`(加分支)、`config.yaml`(provider 段)、
  `README.md`(Windows 打包/运行说明)。
- 不动:`RagEngine`、检索链路、embedder/reranker、`api.py`/`webui.py` 业务逻辑。
