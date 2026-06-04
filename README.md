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
3. 启 API: `python api.py`  → `http://<IP>:8000/chat`
4. 启 WebUI: `python webui.py` → `http://<IP>:7860`

## 配置
所有参数在 `config.yaml`。切换模型/供应商只需改 `llm.provider` 与 `llm.model`
(目前支持 `ollama`,`transformers` 为预留接口)。

## API
- `GET /health` → `{"llm": bool, "doc_count": int}`
- `POST /chat` body `{"question": "..."}` → SSE 流,每行 `data: {"type":"token"|"sources","data":...}`

## 局域网访问
host 已设为 `0.0.0.0`。用 `ip addr`(Linux)/`ipconfig`(Windows)查本机 IP,
确保防火墙放行 8000 / 7860 端口,同网段设备即可访问。
