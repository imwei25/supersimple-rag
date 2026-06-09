# 本地 RAG 问答系统

接入本地 Ollama `minicpm5-1b`,中文 PDF/Word 知识库,混合检索 + 流式输出,提供 API 与 WebUI。

## 依赖
- Python 3.10+
- Ollama(已 `ollama pull minicpm5-1b`),运行 `ollama serve`

## 安装

### Linux / macOS
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> Windows 无 GPU 时,把 config.yaml 的 `embedding.device` 改为 `cpu`。

## 使用
1. 把 PDF / Word 放入 `docs_kb/`
2. 建库: `python ingest.py`
3. 启 API: `python api.py`  → `http://<IP>:8001/chat`
4. 启 WebUI: `python webui.py` → `http://<IP>:7860`

## 配置
所有参数在 `config.yaml`。切换模型/供应商只需改 `llm.provider` 与 `llm.model`
(目前支持 `ollama`,`transformers` 为预留接口)。

## API
- `GET /health` → `{"llm": bool, "doc_count": int}`
- `POST /chat` body `{"question": "..."}` → SSE 流,每行 `data: {"type":"token"|"sources","data":...}`

## 局域网访问
host 已设为 `0.0.0.0`。用 `ip addr`(Linux)/`ipconfig`(Windows)查本机 IP,
确保防火墙放行 8001 / 7860 端口,同网段设备即可访问。

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
