# tests/test_splitter.py
from rag.splitter import split_text, split_records


# ---- 既有行为(回归):尺寸、重叠、短文本 ----
def test_split_respects_chunk_size():
    text = "句" * 1200
    chunks = split_text(text, chunk_size=500, overlap=80)
    assert all(len(c) <= 500 for c in chunks)
    assert len(chunks) >= 3

def test_split_has_overlap():
    text = "abcdefghij" * 100  # 1000 chars
    chunks = split_text(text, chunk_size=500, overlap=80)
    assert chunks[0][-80:] == chunks[1][:80]

def test_split_short_text_single_chunk():
    chunks = split_text("短文本", chunk_size=500, overlap=80)
    assert chunks == ["短文本"]


# ---- 新行为:结构感知 ----
def test_heading_attached_to_its_content():
    """小节标题应与其正文同处一个片段,不被甩到上一段尾部。"""
    text = (
        "## 第一节概述\n\n" + "概述内容。" * 5 + "\n\n"
        "## 第二节分类\n\n" + "分类内容。" * 5
    )
    chunks = split_text(text, chunk_size=200, overlap=20)
    # 每个含"分类内容"的片段都应带着它的小节标题
    cls = [c for c in chunks if "分类内容" in c]
    assert cls and all("## 第二节分类" in c for c in cls)

def test_large_sections_not_merged():
    """足够大的相邻小节各自成片段,不被合并(避免语义稀释)。"""
    text = "## A\n\n" + "甲。" * 150 + "\n\n## B\n\n" + "乙。" * 150
    chunks = split_text(text, chunk_size=500, overlap=40)
    assert not any("甲" in c and "乙" in c for c in chunks)

def test_small_sections_merged_keep_headings():
    """相邻的极短小节在不超 chunk_size 时合并为一片段,但各自标题都保留。"""
    text = "## 【参考范围】\n\n阴性。\n\n## 【临床意义】\n\n见于感染。"
    chunks = split_text(text, chunk_size=500, overlap=20)
    assert len(chunks) == 1
    assert "## 【参考范围】" in chunks[0] and "## 【临床意义】" in chunks[0]

def test_merge_stops_at_chapter_boundary():
    """合并不得跨章(# 一级标题)。"""
    text = "## 末节\n\n甲。\n\n# 第二章\n\n乙。"
    chunks = split_text(text, chunk_size=500, overlap=20)
    assert not any("甲" in c and "乙" in c for c in chunks)


# ---- 新行为:表格原子化 + 表头保留 ----
def _table(rows=8):
    head = "表1-1 示例表\n\n[表格]\n| 列A | 列B | 列C |\n| --- | --- | --- |\n"
    body = "".join(f"| 行{i} | v{i} | w{i} |\n" for i in range(rows))
    return head + body

def test_small_table_kept_atomic():
    text = "## 节\n\n前文。\n\n" + _table(4)
    chunks = split_text(text, chunk_size=500, overlap=40)
    tbl = [c for c in chunks if "列A" in c]
    assert len(tbl) == 1                      # 小表不被拆开
    assert "行0" in tbl[0] and "行3" in tbl[0]

def test_large_table_split_repeats_header():
    """超长表格按行拆分,每个片段都要带表头行。"""
    text = "## 节\n\n" + _table(60)
    chunks = split_text(text, chunk_size=300, overlap=20)
    tbl = [c for c in chunks if "行0" in c or "行59" in c]
    assert len(tbl) >= 2
    for c in [c for c in chunks if "| 行" in c]:
        assert "| 列A | 列B | 列C |" in c       # 每个表片段都含表头


# ---- 新行为:保留换行(表格/标题不被压成一行)----
def test_preserves_table_newlines():
    text = "## 节\n\n" + _table(4)
    chunks = split_text(text, chunk_size=500, overlap=40)
    tbl = [c for c in chunks if "列A" in c][0]
    assert "\n| --- | --- | --- |" in tbl      # 表格逐行,未被压平


# ---- 阶段2:章/节元数据 + 面包屑 ----
def test_records_carry_chapter_section():
    text = ("# 第二章 白细胞疾病\n\n本章提要。\n\n"
            "## 第一节 概述\n\n" + "概述。" * 40)
    recs = split_records(text, chunk_size=200, overlap=20)
    overview = [r for r in recs if "概述" in r["text"]]
    assert overview and all(r["chapter"] == "第二章 白细胞疾病" for r in overview)
    assert all(r["section"] == "第一节 概述" for r in overview)

def test_breadcrumb_prepended_for_subsection():
    text = "# 第二章 白细胞疾病\n\n提要。\n\n## 第一节 概述\n\n" + "概述。" * 40
    recs = split_records(text, chunk_size=200, overlap=20)
    sub = [r for r in recs if "概述。" in r["text"]][0]
    assert sub["text"].startswith("【第二章 白细胞疾病】")   # 章名面包屑补回

def test_chapter_intro_no_redundant_breadcrumb():
    text = "# 第二章 白细胞疾病\n\n本章提要内容。"
    recs = split_records(text, chunk_size=500, overlap=20)
    assert recs[0]["text"].startswith("# 第二章")           # 章首不重复加面包屑
