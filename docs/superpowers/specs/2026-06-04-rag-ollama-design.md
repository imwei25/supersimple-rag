# 本地 RAG 问答系统 — 设计文档

日期: 2026-06-04
状态: 已确认设计,待实现

## 1. 目标

构建一个本地化的 RAG(检索增强生成)问答系统:

- 接入本地 Ollama `minicpm5-1b` 模型做问答
- 提供 **API 接口**(FastAPI)和 **WebUI 入口**(Gradio)
- 部署到本地局域网(监听 `0.0.0.0`)
- 支持中文 PDF / Word 文档作为知识库
- **流式输出**、**混合检索**、**配置驱动的模型/供应商切换**
- 当前运行于 Linux,需能简单迁移到 Windows

## 2. 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| LLM | Ollama `minicpm5-1b` | 已安装,HTTP `:11434` |
| Embedding | `BAAI/bge-small-zh-v1.5` | HuggingFace 自动下载 ~100MB,GPU(RTX 4090)推理 |
| 向量库 | Chroma | 本地持久化 `./chroma_db`,零配置 |
| 关键词检索 | `rank_bm25` | 内存 BM25,与向量结果做 RRF 融合 |
| PDF 解析 | `pypdf` | |
| Word 解析 | `python-docx` | |
| API | FastAPI + Uvicorn | `0.0.0.0:8000`,SSE 流式 |
| WebUI | Gradio | `0.0.0.0:7860`,聊天界面 |
| 配置 | PyYAML | `config.yaml` 集中配置 |

## 3. 架构

```
   浏览器(局域网)──── Gradio WebUI  :7860
   curl/其他应用 ──── FastAPI       :8000  /chat (SSE) /health
                              │
                      RAG 核心引擎 (rag.py)
                  1. 混合检索 (向量 + BM25 → RRF)
                  2. 拼中文 prompt(上下文 + 问题)
                  3. 流式调 LLM provider
                ┌──────────┴──────────┐
          Chroma 向量库          LLM Provider(可插拔)
          + BM25 索引            ├ ollama_provider (现)
                                 └ transformers_provider (预留)

   离线: ingest.py  docs/ → 解析 → 切分 → embedding → Chroma + BM25 落盘
```

## 4. 模块划分(单一职责)

| 文件 | 职责 | 依赖 |
|------|------|------|
| `config.yaml` | 全部可调参数 | — |
| `config.py` | 加载/校验 yaml,提供配置对象 | PyYAML |
| `loader.py` | `docs/` 下 PDF/Word → `(文本, 来源)` 列表 | pypdf, python-docx |
| `splitter.py` | 按字数+重叠切分文本 | — |
| `providers/base.py` | `LLMProvider` 接口: `stream(prompt)->Iterator[str]` | — |
| `providers/ollama_provider.py` | 调 Ollama `/api/generate` 流式 | requests |
| `providers/transformers_provider.py` | 预留占位,抛 NotImplementedError + TODO | (后续) |
| `providers/factory.py` | 按 config 返回对应 provider | — |
| `embedder.py` | 封装 bge-small-zh,`encode(texts)->vectors` | sentence-transformers |
| `vectorstore.py` | Chroma 增/查 + BM25 索引构建与持久化 | chromadb, rank_bm25 |
| `retriever.py` | 混合检索:向量 top-k + BM25 top-k → RRF 融合 | — |
| `ingest.py` | 离线脚本:loader→splitter→embedder→vectorstore | 上述 |
| `rag.py` | 检索 + 拼 prompt + 流式生成,返回答案+来源 | retriever, factory |
| `api.py` | FastAPI: `POST /chat`(SSE 流)、`GET /health` | fastapi, uvicorn |
| `webui.py` | Gradio 聊天界面,流式展示,调 rag.py | gradio |
| `requirements.txt` | 固定版本依赖 | — |
| `README.md` | Linux + Windows 启动说明 | — |

## 5. config.yaml 结构(草案)

```yaml
llm:
  provider: ollama            # ollama | transformers(后续)
  model: minicpm5-1b
  ollama_base_url: http://localhost:11434
  temperature: 0.3
  max_tokens: 1024

embedding:
  provider: sentence-transformers
  model: BAAI/bge-small-zh-v1.5
  device: cuda                # cuda | cpu
  batch_size: 32

vectorstore:
  persist_dir: ./chroma_db
  collection: docs

retrieval:
  top_k: 4                    # 最终返回片段数
  vector_k: 10                # 向量召回数
  bm25_k: 10                  # BM25 召回数
  rrf_k: 60                   # RRF 常数
  hybrid: true                # false 则仅向量

split:
  chunk_size: 500
  chunk_overlap: 80

server:
  api_host: 0.0.0.0
  api_port: 8000
  web_host: 0.0.0.0
  web_port: 7860

paths:
  docs_dir: ./docs
```

## 6. 数据流

**离线建库** (`python ingest.py`):
docs/ 扫描 → 解析 PDF/Word → 切分(500字/重叠80)→ bge 编码 → 写入 Chroma;同时把所有 chunk 文本构建 BM25 索引并持久化(pickle 到 persist_dir)。

**在线问答**:
1. 问题 → bge 编码 → Chroma 取 `vector_k`
2. 问题分词 → BM25 取 `bm25_k`
3. RRF 融合两路结果 → 取 `top_k`
4. 拼中文 prompt:`已知信息:{片段}\n\n问题:{q}\n\n请仅根据已知信息用中文回答,无法回答时说明。`
5. provider 流式生成 → 逐 token 返回
6. 返回答案 + 引用来源(文件名 + 片段)

## 7. 流式实现

- `LLMProvider.stream()` 返回 token 迭代器
- `rag.answer_stream()` 先检索,再 yield token,最后 yield 来源
- `api.py`:`POST /chat` 用 `StreamingResponse` 发 SSE(`data: {token}\n\n`)
- `webui.py`:Gradio `ChatInterface` 的 generator 模式,逐步刷新

## 8. 错误处理

- Ollama 未启动 / 连接失败:返回明确提示"请确认 ollama serve 正在运行"
- 向量库为空:提示"请先运行 python ingest.py 建立知识库"
- 单个文档解析失败:跳过该文件并记日志,不中断整体
- 检索为空:直接让模型回答并标注"知识库中无相关内容"

## 9. 跨平台(Linux → Windows)

- 所有路径用 `pathlib.Path`,不拼字符串
- 文件读写显式 `encoding="utf-8"`
- 依赖固定版本,纯 pip 可装(Chroma/bge/gradio 均跨平台)
- `device` 由 config 控制(Windows 无 GPU 时改 `cpu`)
- README 提供两套启动命令;不依赖任何 shell 脚本,统一用 `python xxx.py`

## 10. 验证

1. 放一个示例中文 PDF 到 `docs/`
2. `python ingest.py` → 确认 chroma_db 生成、有 chunk 数输出
3. 启 `python api.py`,`curl -N -X POST :8000/chat -d '{"question":"..."}'` → 流式返回基于文档的答案
4. 启 `python webui.py`,浏览器 `http://<本机IP>:7860` 问同一问题 → 一致作答
5. `GET /health` 返回 ollama + 向量库状态
```
```
