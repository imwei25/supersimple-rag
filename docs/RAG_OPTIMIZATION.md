# RAG 召回优化进度

> 目标:在**不更换 embedding 模型**(`bge-small-zh-v1.5`)的前提下,提升对现有知识库的召回准确率。
> 参考开源项目 RAGFlow 的优化思路逐步落地。

## 背景:对照 RAGFlow 的差距分析

| 维度 | RAGFlow | 本项目优化前 | 状态 |
|---|---|---|---|
| 文档解析 | DeepDoc OCR + 版面/表格识别 | pypdf/python-docx 纯文本 | ✅ 已做表格感知 |
| 切分 | 模板 + 句子/结构感知 | 字符滑窗硬切 | ✅ 已做句子感知 |
| 语义增强 | Auto-Keyword / Auto-Question | 无 | ✅ 已做 |
| 查询侧 | 词权重 + 改写 | 裸编码 | ✅ 已加 BGE 前缀 |
| 混合检索 | 向量+关键词+元数据 | 向量+BM25+RRF | ✅ 已具备 |
| 重排 | Cross-Encoder | 无 | ✅ 已加 reranker |
| RAPTOR/TreeRAG | ✅ | 无 | ⬜ 未做 |
| GraphRAG | ✅ | 无 | ⬜ 未做 |
| 元数据过滤/引用定位 | ✅ | 仅存 source | ⬜ 未做 |

---

## 已完成

### 1. BGE 查询指令前缀(P0)
- **原理**:BGE-zh 训练时查询侧带固定指令前缀,推理时对齐才不掉精度。纯字符串拼接,非 LLM 改写。
- **前缀**(官方标准,逐字一致):`为这个句子生成表示以用于检索相关文章：`
- **实现**:`rag/embedder.py` 新增 `encode_query()`(仅查询加前缀,文档入库不加);`config.yaml` → `embedding.query_prefix`。
- **重建索引**:不需要(只影响查询编码)。

### 2. Cross-Encoder 重排(P0)
- **模型**:`bge-reranker-base`(modelscope 下载至 `./models/bge-reranker-base`)。
- **流程**:混合检索召回放大到 `rerank_candidates=20` → reranker 打分 → 收窄到 `top_k=4`。
- **实现**:新增 `rag/reranker.py`(`make_reranker` 工厂,加载失败优雅降级为 None);`rag/retriever.py` 接入;`rag/rag.py` 装配。
- **配置**:`config.yaml` → `retrieval.rerank/rerank_candidates/reranker_model`,`vector_k/bm25_k` 提到 20。

### 3. 语义切分(切分语义化)
- **做法**:先按中英文句末标点/换行分句,贪心打包到 `chunk_size`,块间以句子为单位保留 `overlap` 重叠;超长单句退化为字符硬切。
- **实现**:重写 `rag/splitter.py`。
- **参数**:`chunk_size` 1000→500,`chunk_overlap` 180→100(配合重排,小块更精确)。
- **重建索引**:**需要**。

### 4. 表格感知 + 栏感知解析(P0,对照 RAGFlow DeepDoc)
- **背景**:实测发现两篇诊疗指南 PDF 均为**双栏排版**,旧解析(含初版 pdfplumber)跨栏
  逐行抽取,把左右栏中文搅成乱码 —— 喂给向量/BM25 的几乎全是无意义文本,严重拖累召回。
- **做法**:
  - **栏检测**:统计跨中线词占比,极少跨栏且左右均有足量文字 → 判定双栏(`_detect_columns`)。
  - **分栏抽取**:双栏页先左后右分别 crop 抽取再拼接,中文句子恢复连贯(`_extract_page_text`)。
  - **表格**:`pdfplumber` 抽出的表格转 **Markdown 表格**附在页尾;DOCX 同时抽 `doc.tables`。
  - 失败回退 pypdf 纯文本。
- **实现**:`rag/loader.py`(`_detect_columns`/`_extract_page_text`/`_read_pdf`/`_read_docx`/`_table_to_markdown`)。
- **验证**:修复前正文为中英数字交错乱码;修复后中文段落连贯可读。
- **重建索引**:**需要**(这是当前召回质量影响最大的一项)。

### 5. 语义增强 Auto-Keyword/Question(P1,对照 RAGFlow)
- **做法**:入库时用 LLM 为每个 chunk 生成「关键词 + 最多 3 个假设问题」,以 `【检索增强】` 块附加到被索引文本(向量与 BM25 同时受益)。成本前置到入库,查询零额外开销。
- **思维链处理**:自动剥离 `<think>...</think>`(minicpm 等思考模型),未闭合则视为无效跳过。
- **降级**:单个片段生成失败用原文,不阻断建库;`ingest.enrich=false` 可关闭恢复原行为。
- **增量缓存**:按 chunk 文本 SHA1 缓存增强结果到 `<persist_dir>/enrich_cache.json`。
  **重建时内容未变的片段直接复用,不再调用 LLM**;只有新增/改动的片段才生成。
  每次重建后按当次出现的哈希裁剪缓存,避免无限膨胀;仅缓存成功结果(失败下次可重试)。
  `reset()` 不会删除该缓存文件,故重建可持续受益。
- **实现**:新增 `rag/enrich.py`;`rag/rag.py:rebuild_index` 在编码前调用(传入缓存路径);`rag/config.py` 新增 `ingest` 字段(带默认,兼容旧配置);`config.yaml` → `ingest` 段。
- **重建索引**:**需要**(但第二次起仅对改动片段调用 LLM,很快)。

---

## ⚠️ 生效前必做

1. 已下载 reranker 模型(完成)。
2. **重建知识库**(切分/解析/增强变更都依赖):WebUI「重建知识库」按钮 或 `python ingest.py`。
   - 注意:开启 `enrich` 后建库会显著变慢(每片段一次 LLM 调用),属一次性成本。

---

## 未完成 / 后续路线

| 优先级 | 项目 | 说明 |
|---|---|---|
| P1 | 结构感知切分增强 | 识别 Markdown 标题/章节强制断块,并把所属标题作为前缀注入子块 |
| P2 | 元数据过滤 + 引用定位 | 存章节/页码,WebUI 召回区显示定位,支持按文件过滤检索 |
| P3(重) | RAPTOR / TreeRAG | 递归聚类+摘要建树,适合"对整个知识库"的全局性/多段汇总提问 |
| P3(重) | GraphRAG | 实体关系图谱,解决多跳问答;可借力项目已有的 `/graphify` 技能 |
| — | 评测集 `eval.py` | 建「问题→期望来源」小评测集,量化每步改动的命中率,驱动 RAPTOR/Graph 决策 |

---

## 涉及文件

- `rag/embedder.py` — 查询前缀
- `rag/reranker.py` — 重排(新增)
- `rag/retriever.py` — 候选放大 + 重排接入
- `rag/splitter.py` — 语义切分
- `rag/loader.py` — 表格感知解析
- `rag/enrich.py` — 语义增强(新增)
- `rag/rag.py` — 装配 reranker / enrich
- `rag/config.py` — 新增 `ingest` 字段
- `config.yaml` — `embedding.query_prefix` / `retrieval.rerank*` / `split` / `ingest`
