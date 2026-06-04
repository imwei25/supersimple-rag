# 本地 RAG 问答系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个本地 RAG 问答系统,接入 Ollama `minicpm5-1b`,支持中文 PDF/Word 知识库、混合检索、流式输出,提供 FastAPI 接口和 Gradio WebUI,部署到局域网,并可简单迁移到 Windows。

**Architecture:** 配置驱动(`config.yaml`)。离线 `ingest.py` 把文档解析→切分→bge embedding→写入 Chroma 并构建 BM25 索引。在线 `rag.py` 做向量+BM25 混合检索(RRF 融合)→拼中文 prompt→经可插拔 LLM provider 流式生成。`api.py`(FastAPI/SSE)与 `webui.py`(Gradio)两个入口共用 `rag.py`。

**Tech Stack:** Python 3.10, Ollama, sentence-transformers (bge-small-zh-v1.5), chromadb, rank_bm25, pypdf, python-docx, FastAPI/uvicorn, Gradio, PyYAML, pytest。

---

## File Structure

```
doctor-t/
├── config.yaml                    # 全部可调参数
├── requirements.txt
├── README.md
├── docs_kb/                       # 知识库源文档(PDF/Word)放这里
├── chroma_db/                     # 运行时生成(向量库 + bm25.pkl)
├── rag/
│   ├── __init__.py
│   ├── config.py                  # 加载 yaml → Config 对象
│   ├── loader.py                  # PDF/Word → [(text, source)]
│   ├── splitter.py                # 文本切分
│   ├── embedder.py                # bge 封装
│   ├── vectorstore.py             # Chroma + BM25 持久化
│   ├── retriever.py               # 混合检索 + RRF
│   ├── rag.py                     # 检索+prompt+流式生成
│   └── providers/
│       ├── __init__.py
│       ├── base.py                # LLMProvider 接口
│       ├── ollama_provider.py
│       ├── transformers_provider.py  # 预留占位
│       └── factory.py
├── ingest.py                      # 离线建库脚本
├── api.py                         # FastAPI 入口
├── webui.py                       # Gradio 入口
└── tests/
    ├── test_config.py
    ├── test_splitter.py
    ├── test_retriever.py
    └── test_loader.py
```

注意:知识库目录用 `docs_kb/`(避免与 `docs/` spec 目录冲突)。

---

## Task 1: 项目骨架与依赖

**Files:**
- Create: `requirements.txt`, `config.yaml`, `rag/__init__.py`, `rag/providers/__init__.py`, `.gitignore`

- [ ] **Step 1: 写 requirements.txt**

```
pyyaml==6.0.2
pypdf==5.1.0
python-docx==1.1.2
sentence-transformers==3.3.1
chromadb==0.5.23
rank-bm25==0.2.2
fastapi==0.115.6
uvicorn==0.34.0
gradio==5.9.1
requests==2.32.3
pytest==8.3.4
jieba==0.42.1
```

- [ ] **Step 2: 写 config.yaml**

```yaml
llm:
  provider: ollama
  model: minicpm5-1b
  ollama_base_url: http://localhost:11434
  temperature: 0.3
  max_tokens: 1024

embedding:
  provider: sentence-transformers
  model: BAAI/bge-small-zh-v1.5
  device: cuda
  batch_size: 32

vectorstore:
  persist_dir: ./chroma_db
  collection: docs

retrieval:
  top_k: 4
  vector_k: 10
  bm25_k: 10
  rrf_k: 60
  hybrid: true

split:
  chunk_size: 500
  chunk_overlap: 80

server:
  api_host: 0.0.0.0
  api_port: 8000
  web_host: 0.0.0.0
  web_port: 7860

paths:
  docs_dir: ./docs_kb
```

- [ ] **Step 3: 写 .gitignore 和空 __init__.py**

`.gitignore`:
```
__pycache__/
*.pyc
chroma_db/
.venv/
venv/
*.egg-info/
```
创建空文件 `rag/__init__.py`、`rag/providers/__init__.py`。

- [ ] **Step 4: 创建虚拟环境并安装依赖**

Run: `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
Expected: 全部安装成功(首次较慢)。

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config.yaml .gitignore rag/__init__.py rag/providers/__init__.py
git commit -m "chore: 项目骨架与依赖"
```

---

## Task 2: 配置加载 (config.py)

**Files:**
- Create: `rag/config.py`, `tests/test_config.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py
from pathlib import Path
from rag.config import load_config

def test_load_config_reads_yaml():
    cfg = load_config(Path("config.yaml"))
    assert cfg.llm["model"] == "minicpm5-1b"
    assert cfg.retrieval["top_k"] == 4
    assert cfg.split["chunk_size"] == 500
    assert cfg.paths["docs_dir"] == "./docs_kb"

def test_config_persist_dir_is_path():
    cfg = load_config(Path("config.yaml"))
    assert cfg.persist_dir().name == "chroma_db"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `. .venv/bin/activate && pytest tests/test_config.py -v`
Expected: FAIL,`ModuleNotFoundError: rag.config`

- [ ] **Step 3: 实现 config.py**

```python
# rag/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    llm: dict
    embedding: dict
    vectorstore: dict
    retrieval: dict
    split: dict
    server: dict
    paths: dict

    def persist_dir(self) -> Path:
        return Path(self.vectorstore["persist_dir"]).resolve()

    def docs_dir(self) -> Path:
        return Path(self.paths["docs_dir"]).resolve()


def load_config(path: Path = Path("config.yaml")) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add rag/config.py tests/test_config.py
git commit -m "feat: 配置加载"
```

---

## Task 3: 文本切分 (splitter.py)

**Files:**
- Create: `rag/splitter.py`, `tests/test_splitter.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_splitter.py
from rag.splitter import split_text

def test_split_respects_chunk_size():
    text = "句" * 1200
    chunks = split_text(text, chunk_size=500, overlap=80)
    assert all(len(c) <= 500 for c in chunks)
    assert len(chunks) >= 3

def test_split_has_overlap():
    text = "abcdefghij" * 100  # 1000 chars
    chunks = split_text(text, chunk_size=500, overlap=80)
    # 相邻 chunk 结尾与下一块开头有重叠
    assert chunks[0][-80:] == chunks[1][:80]

def test_split_short_text_single_chunk():
    chunks = split_text("短文本", chunk_size=500, overlap=80)
    assert chunks == ["短文本"]
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_splitter.py -v`
Expected: FAIL,`ModuleNotFoundError`

- [ ] **Step 3: 实现 splitter.py**

```python
# rag/splitter.py
from __future__ import annotations
from typing import List


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_splitter.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add rag/splitter.py tests/test_splitter.py
git commit -m "feat: 文本切分"
```

---

## Task 4: 文档加载 (loader.py)

**Files:**
- Create: `rag/loader.py`, `tests/test_loader.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_loader.py
from pathlib import Path
from docx import Document
from rag.loader import load_documents

def test_load_docx(tmp_path):
    p = tmp_path / "a.docx"
    doc = Document()
    doc.add_paragraph("这是第一段中文。")
    doc.add_paragraph("这是第二段。")
    doc.save(p)
    results = load_documents(tmp_path)
    assert len(results) == 1
    text, source = results[0]
    assert "第一段" in text
    assert source == "a.docx"

def test_skips_unknown_extension(tmp_path):
    (tmp_path / "note.txt").write_text("ignored", encoding="utf-8")
    assert load_documents(tmp_path) == []
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL,`ModuleNotFoundError`

- [ ] **Step 3: 实现 loader.py**

```python
# rag/loader.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import logging

from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_documents(docs_dir: Path) -> List[Tuple[str, str]]:
    """返回 [(全文, 文件名)],解析失败的文件跳过并记日志。"""
    results: List[Tuple[str, str]] = []
    docs_dir = Path(docs_dir)
    for path in sorted(docs_dir.glob("*")):
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                text = _read_pdf(path)
            elif ext == ".docx":
                text = _read_docx(path)
            else:
                continue
        except Exception as e:  # noqa: BLE001
            logger.warning("解析失败,跳过 %s: %s", path.name, e)
            continue
        if text.strip():
            results.append((text, path.name))
    return results
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_loader.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add rag/loader.py tests/test_loader.py
git commit -m "feat: PDF/Word 文档加载"
```

---

## Task 5: Embedding 封装 (embedder.py)

**Files:**
- Create: `rag/embedder.py`

无单元测试(依赖大模型下载,放到 Task 9 集成验证)。

- [ ] **Step 1: 实现 embedder.py**

```python
# rag/embedder.py
from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 32):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
```

- [ ] **Step 2: 冒烟验证(CPU 回退保险)**

Run:
```bash
. .venv/bin/activate && python -c "from rag.embedder import Embedder; e=Embedder('BAAI/bge-small-zh-v1.5', device='cuda'); v=e.encode(['你好']); print(len(v), len(v[0]))"
```
Expected: 打印 `1 512`(bge-small-zh 维度 512)。若 CUDA 报错,改 `device='cpu'` 再试一次。

- [ ] **Step 3: Commit**

```bash
git add rag/embedder.py
git commit -m "feat: bge embedding 封装"
```

---

## Task 6: 向量库 + BM25 (vectorstore.py)

**Files:**
- Create: `rag/vectorstore.py`

- [ ] **Step 1: 实现 vectorstore.py**

```python
# rag/vectorstore.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import pickle

import chromadb
import jieba
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


class VectorStore:
    def __init__(self, persist_dir: Path, collection: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self.bm25_path = self.persist_dir / "bm25.pkl"

    def add(self, ids: List[str], texts: List[str], vectors: List[List[float]],
            sources: List[str]) -> None:
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=[{"source": s} for s in sources],
        )

    def build_bm25(self) -> None:
        """从 collection 全量取文档,构建 BM25 并落盘。"""
        data = self.collection.get(include=["documents", "metadatas"])
        docs = data["documents"]
        ids = data["ids"]
        sources = [m["source"] for m in data["metadatas"]]
        tokenized = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized) if tokenized else None
        with open(self.bm25_path, "wb") as f:
            pickle.dump(
                {"bm25": bm25, "ids": ids, "docs": docs, "sources": sources},
                f,
            )

    def load_bm25(self) -> Dict[str, Any] | None:
        if not self.bm25_path.exists():
            return None
        with open(self.bm25_path, "rb") as f:
            return pickle.load(f)

    def count(self) -> int:
        return self.collection.count()

    def query_vector(self, query_vec: List[float], k: int) -> List[Dict[str, Any]]:
        res = self.collection.query(
            query_embeddings=[query_vec], n_results=k,
            include=["documents", "metadatas"],
        )
        out = []
        for i, doc in enumerate(res["documents"][0]):
            out.append({
                "id": res["ids"][0][i],
                "text": doc,
                "source": res["metadatas"][0][i]["source"],
            })
        return out
```

- [ ] **Step 2: 冒烟验证**

Run:
```bash
python -c "
from rag.vectorstore import VectorStore, _tokenize
print(_tokenize('中文分词测试'))
"
```
Expected: 打印分词列表如 `['中文', '分词', '测试']`。

- [ ] **Step 3: Commit**

```bash
git add rag/vectorstore.py
git commit -m "feat: Chroma 向量库 + BM25 索引"
```

---

## Task 7: 混合检索 (retriever.py)

**Files:**
- Create: `rag/retriever.py`, `tests/test_retriever.py`

- [ ] **Step 1: 写失败测试(RRF 纯函数,可单测)**

```python
# tests/test_retriever.py
from rag.retriever import rrf_fuse

def test_rrf_fuse_merges_and_ranks():
    vector_ids = ["a", "b", "c"]
    bm25_ids = ["b", "d", "a"]
    fused = rrf_fuse(vector_ids, bm25_ids, rrf_k=60)
    # b 在两路都靠前,应排第一
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}

def test_rrf_fuse_single_list():
    assert rrf_fuse(["x", "y"], [], rrf_k=60) == ["x", "y"]
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL,`ModuleNotFoundError`

- [ ] **Step 3: 实现 retriever.py**

```python
# rag/retriever.py
from __future__ import annotations
from typing import List, Dict, Any

from rag.embedder import Embedder
from rag.vectorstore import VectorStore, _tokenize


def rrf_fuse(vector_ids: List[str], bm25_ids: List[str], rrf_k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for rank, _id in enumerate(vector_ids):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, _id in enumerate(bm25_ids):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, cfg: dict):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg

    def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        data = self.store.load_bm25()
        if not data or data["bm25"] is None:
            return []
        scores = data["bm25"].get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{
            "id": data["ids"][i],
            "text": data["docs"][i],
            "source": data["sources"][i],
        } for i in ranked]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        c = self.cfg
        qvec = self.embedder.encode([query])[0]
        vec_hits = self.store.query_vector(qvec, c["vector_k"])
        if not c.get("hybrid", True):
            return vec_hits[: c["top_k"]]
        bm25_hits = self._bm25_search(query, c["bm25_k"])
        by_id = {h["id"]: h for h in (vec_hits + bm25_hits)}
        fused_ids = rrf_fuse(
            [h["id"] for h in vec_hits],
            [h["id"] for h in bm25_hits],
            c.get("rrf_k", 60),
        )
        return [by_id[i] for i in fused_ids[: c["top_k"]]]
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add rag/retriever.py tests/test_retriever.py
git commit -m "feat: 混合检索 + RRF 融合"
```

---

## Task 8: LLM Provider 抽象 (providers/)

**Files:**
- Create: `rag/providers/base.py`, `rag/providers/ollama_provider.py`, `rag/providers/transformers_provider.py`, `rag/providers/factory.py`

- [ ] **Step 1: 实现 base.py**

```python
# rag/providers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """逐 token / 逐块 yield 文本。"""
        raise NotImplementedError

    @abstractmethod
    def health(self) -> bool:
        """后端是否可用。"""
        raise NotImplementedError
```

- [ ] **Step 2: 实现 ollama_provider.py**

```python
# rag/providers/ollama_provider.py
from __future__ import annotations
from typing import Iterator
import json
import requests

from rag.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, cfg: dict):
        self.base_url = cfg["ollama_base_url"].rstrip("/")
        self.model = cfg["model"]
        self.temperature = cfg.get("temperature", 0.3)
        self.max_tokens = cfg.get("max_tokens", 1024)

    def stream(self, prompt: str) -> Iterator[str]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            with requests.post(url, json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    if data.get("response"):
                        yield data["response"]
                    if data.get("done"):
                        break
        except requests.exceptions.RequestException as e:
            yield f"\n[错误] 无法连接 Ollama,请确认 `ollama serve` 正在运行: {e}"

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False
```

- [ ] **Step 3: 实现 transformers_provider.py(预留占位)**

```python
# rag/providers/transformers_provider.py
from __future__ import annotations
from typing import Iterator

from rag.providers.base import LLMProvider


class TransformersProvider(LLMProvider):
    """预留:后续用 HuggingFace transformers 本地加载模型。
    TODO: 实现 AutoModelForCausalLM + TextIteratorStreamer 流式生成。"""

    def __init__(self, cfg: dict):
        raise NotImplementedError(
            "transformers provider 尚未实现,请在 config.yaml 中使用 provider: ollama"
        )

    def stream(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError

    def health(self) -> bool:
        raise NotImplementedError
```

- [ ] **Step 4: 实现 factory.py**

```python
# rag/providers/factory.py
from __future__ import annotations

from rag.providers.base import LLMProvider
from rag.providers.ollama_provider import OllamaProvider
from rag.providers.transformers_provider import TransformersProvider


def make_provider(llm_cfg: dict) -> LLMProvider:
    provider = llm_cfg.get("provider", "ollama")
    if provider == "ollama":
        return OllamaProvider(llm_cfg)
    if provider == "transformers":
        return TransformersProvider(llm_cfg)
    raise ValueError(f"未知 provider: {provider}")
```

- [ ] **Step 5: 冒烟验证(需 ollama 在跑)**

Run:
```bash
python -c "
from rag.config import load_config
from rag.providers.factory import make_provider
p = make_provider(load_config().llm)
print('health:', p.health())
print(''.join(list(p.stream('用一句话说你好'))))
"
```
Expected: `health: True`,随后打印一句中文回答。

- [ ] **Step 6: Commit**

```bash
git add rag/providers/
git commit -m "feat: LLM provider 抽象(ollama + transformers 占位)"
```

---

## Task 9: RAG 核心 (rag.py) + ingest.py

**Files:**
- Create: `rag/rag.py`, `ingest.py`

- [ ] **Step 1: 实现 rag/rag.py**

```python
# rag/rag.py
from __future__ import annotations
from typing import Iterator, List, Dict, Any

from rag.config import Config
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever
from rag.providers.factory import make_provider

PROMPT_TEMPLATE = """你是一个严谨的中文问答助手。请仅根据下面的【已知信息】回答【问题】。
如果已知信息中没有答案,请直接说"知识库中没有相关内容",不要编造。

【已知信息】
{context}

【问题】
{question}

请用简洁的中文回答:"""


class RagEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = Embedder(
            cfg.embedding["model"],
            device=cfg.embedding.get("device", "cuda"),
            batch_size=cfg.embedding.get("batch_size", 32),
        )
        self.store = VectorStore(cfg.persist_dir(), cfg.vectorstore["collection"])
        self.retriever = Retriever(self.store, self.embedder, cfg.retrieval)
        self.provider = make_provider(cfg.llm)

    def _build_prompt(self, question: str, hits: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(
            f"[来源:{h['source']}] {h['text']}" for h in hits
        ) or "(无)"
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def answer_stream(self, question: str) -> Iterator[Dict[str, Any]]:
        """yield {'type':'token','data':str} 多次,最后 yield {'type':'sources','data':[...]}。"""
        if self.store.count() == 0:
            yield {"type": "token",
                   "data": "知识库为空,请先运行 `python ingest.py` 建立知识库。"}
            yield {"type": "sources", "data": []}
            return
        hits = self.retriever.retrieve(question)
        prompt = self._build_prompt(question, hits)
        for token in self.provider.stream(prompt):
            yield {"type": "token", "data": token}
        seen, sources = set(), []
        for h in hits:
            if h["source"] not in seen:
                seen.add(h["source"])
                sources.append(h["source"])
        yield {"type": "sources", "data": sources}

    def health(self) -> Dict[str, Any]:
        return {"llm": self.provider.health(), "doc_count": self.store.count()}
```

- [ ] **Step 2: 实现 ingest.py**

```python
# ingest.py
from __future__ import annotations
import logging

from rag.config import load_config
from rag.loader import load_documents
from rag.splitter import split_text
from rag.embedder import Embedder
from rag.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ingest")


def main() -> None:
    cfg = load_config()
    docs = load_documents(cfg.docs_dir())
    if not docs:
        log.warning("未在 %s 找到 PDF/Word 文档", cfg.docs_dir())
        return

    embedder = Embedder(
        cfg.embedding["model"],
        device=cfg.embedding.get("device", "cuda"),
        batch_size=cfg.embedding.get("batch_size", 32),
    )
    store = VectorStore(cfg.persist_dir(), cfg.vectorstore["collection"])

    ids, texts, sources = [], [], []
    for text, source in docs:
        chunks = split_text(text, cfg.split["chunk_size"], cfg.split["chunk_overlap"])
        for idx, chunk in enumerate(chunks):
            ids.append(f"{source}::{idx}")
            texts.append(chunk)
            sources.append(source)
    log.info("共 %d 个文档,切出 %d 个 chunk,开始编码...", len(docs), len(texts))

    vectors = embedder.encode(texts)
    store.add(ids, texts, vectors, sources)
    store.build_bm25()
    log.info("入库完成。当前向量数: %d", store.count())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 端到端建库验证**

准备示例文档(若无真实文档,生成一个中文 docx):
```bash
python -c "
from docx import Document
import os
os.makedirs('docs_kb', exist_ok=True)
d=Document()
d.add_paragraph('公司年假政策:入职满一年的员工每年享有10天带薪年假。')
d.add_paragraph('报销流程:员工需在消费后30天内提交发票至财务系统。')
d.save('docs_kb/policy.docx')
print('sample created')
"
python ingest.py
```
Expected: 日志显示切出若干 chunk,`入库完成。当前向量数: N`(N≥2)。

- [ ] **Step 4: 端到端问答验证**

```bash
python -c "
from rag.config import load_config
from rag.rag import RagEngine
e=RagEngine(load_config())
for ev in e.answer_stream('年假有几天?'):
    if ev['type']=='token': print(ev['data'], end='')
    else: print('\n来源:', ev['data'])
"
```
Expected: 流式输出包含"10天"的回答,并打印来源 `['policy.docx']`。

- [ ] **Step 5: Commit**

```bash
git add rag/rag.py ingest.py
git commit -m "feat: RAG 核心引擎 + 离线建库脚本"
```

---

## Task 10: FastAPI 接口 (api.py)

**Files:**
- Create: `api.py`

- [ ] **Step 1: 实现 api.py**

```python
# api.py
from __future__ import annotations
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from rag.config import load_config
from rag.rag import RagEngine

cfg = load_config()
engine = RagEngine(cfg)
app = FastAPI(title="本地 RAG 问答")


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return engine.health()


@app.post("/chat")
def chat(req: ChatRequest):
    def gen():
        for ev in engine.answer_stream(req.question):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=cfg.server["api_host"], port=cfg.server["api_port"])
```

- [ ] **Step 2: 启动并验证(需先建好库 + ollama 在跑)**

后台启动: `python api.py &` (等待约 10 秒加载 embedding 模型)
Run:
```bash
curl -s http://localhost:8000/health
curl -N -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question":"报销发票多少天内提交?"}'
```
Expected: health 返回 `{"llm":true,"doc_count":N}`;chat 以 `data: {...}` 形式流式返回含"30天"的答案与来源。验证后 `kill %1` 关闭。

- [ ] **Step 3: Commit**

```bash
git add api.py
git commit -m "feat: FastAPI 接口(SSE 流式)"
```

---

## Task 11: Gradio WebUI (webui.py)

**Files:**
- Create: `webui.py`

- [ ] **Step 1: 实现 webui.py**

```python
# webui.py
from __future__ import annotations
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
```

- [ ] **Step 2: 启动并验证**

Run: `python webui.py`(等待加载)
Expected: 终端打印 `Running on local URL: http://0.0.0.0:7860`。浏览器打开 `http://<本机局域网IP>:7860`,输入"年假有几天?"看到流式回答 + 来源。验证后 Ctrl-C 关闭。

- [ ] **Step 3: Commit**

```bash
git add webui.py
git commit -m "feat: Gradio WebUI(流式聊天)"
```

---

## Task 12: README 与跨平台说明

**Files:**
- Create: `README.md`

- [ ] **Step 1: 写 README.md**

```markdown
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
```

- [ ] **Step 2: 跑全部单测确认绿**

Run: `pytest -v`
Expected: 所有测试 PASS。

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README 与跨平台说明"
```

---

## Self-Review 结论

- **Spec 覆盖**:LLM(T8/9)、bge embedding 下载(T5)、Chroma(T6)、混合检索 RRF(T7)、流式(T8/9/10/11)、config 切换供应商(T2/8)、PDF+Word(T4)、API(T10)、WebUI(T11)、跨平台(T12)、局域网(config host + T12)。全部对应到任务。
- **占位符**:无 TBD/TODO(transformers_provider 的占位是有意预留并明确抛错,属设计而非计划缺口)。
- **类型一致**:`Embedder.encode`、`VectorStore`(add/build_bm25/load_bm25/query_vector/count)、`Retriever.retrieve`、`rrf_fuse`、`LLMProvider.stream/health`、`RagEngine.answer_stream` 的签名跨任务一致。
- **依赖**:`_tokenize` 在 vectorstore 定义,retriever 复用;jieba 已入 requirements。
```
