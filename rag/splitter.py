# rag/splitter.py
from __future__ import annotations
from typing import List, Tuple
import re

# 句子切分:以中文/英文句末标点作为边界,保留标点(段内为连续散文,不含换行)
_SENT_RE = re.compile(r"[^。！？!?；;]*[。！？!?；;]|[^。！？!?；;]+$")
# Markdown 标题行
_HEADING_RE = re.compile(r"^#{1,6}\s")
# 表标题(如 "表1-1 ...""表 1-2..."),用于把表标题粘到表格上
_CAPTION_RE = re.compile(r"^表\s*\d")

_Block = Tuple[str, str]  # (kind, text);kind ∈ {heading, table, para}


def _split_sentences(text: str) -> List[str]:
    return [m.group().strip() for m in _SENT_RE.finditer(text) if m.group().strip()]


# ---------- 1) 解析为带类型的块(保留换行结构) ----------
def _classify(block: str) -> str:
    s = block.lstrip()
    if s.startswith("[表格]") or s.startswith("|"):
        return "table"
    if "\n" not in block.strip() and _HEADING_RE.match(block.strip()):
        return "heading"
    return "para"


def _split_blocks(text: str) -> List[_Block]:
    blocks: List[_Block] = []
    for raw in re.split(r"\n\s*\n", text):
        b = raw.strip("\n")
        if b.strip():
            blocks.append((_classify(b), b))
    return blocks


def _merge_captions(blocks: List[_Block]) -> List[_Block]:
    """把紧贴在表格前的短表标题(表X-Y...)并入表格块,避免标题与表分离。"""
    out: List[_Block] = []
    for kind, b in blocks:
        if kind == "table" and out and out[-1][0] == "para":
            prev = out[-1][1].strip()
            if _CAPTION_RE.match(prev) and len(prev) < 40:
                out.pop()
                b = prev + "\n\n" + b
        out.append((kind, b))
    return out


# ---------- 2) 按标题切小节,并维护标题层级,产出 章/节 上下文 ----------
# OCR 对章标题的 markdown 级别不一致(第一章可能是 ##、第二章是 #),故按"标题文字"
# 语义判断层级:匹配「第X章」即视为章级(1),其余标题视为节级(2)。
_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零〇两]+章")


def _heading_title(line: str) -> str:
    return re.sub(r"^#{1,6}\s*", "", line.strip())


def _semantic_level(title: str) -> int:
    return 1 if _CHAPTER_RE.match(title) else 2


def _sections_with_context(blocks: List[_Block]) -> List[dict]:
    """按标题切小节,同时用标题栈追踪层级:level-1(#)=章,最深 level≥2(##…)=节。
    每个小节带 (heading 行, body 块, chapter, section)。无正文的标题只更新栈,
    其章名由面包屑前缀补回,不再内联拼接。"""
    sections: List[dict] = []
    stack: dict[int, str] = {}
    cur_heading, body = "", []

    def emit():
        if not body:
            return
        chapter = stack.get(1)
        section = next((stack[l] for l in sorted(stack, reverse=True) if l >= 2), None)
        sections.append({"heading": cur_heading, "body": list(body),
                         "chapter": chapter, "section": section})

    for kind, b in blocks:
        if kind == "heading":
            emit()
            body.clear()
            title = _heading_title(b)
            level = _semantic_level(title)
            stack = {l: t for l, t in stack.items() if l < level}
            stack[level] = title
            cur_heading = b.strip()
        else:
            body.append((kind, b))
    emit()
    return sections


# ---------- 3) 表格拆分:超长表按行切,每片重复表标题+表头 ----------
def _split_table(block: str, avail: int) -> List[str]:
    if len(block) <= avail:
        return [block]                     # 小表保持原子
    lines = block.split("\n")
    meta, i = [], 0                        # 表标题 / [表格] 等非表行
    while i < len(lines) and not lines[i].lstrip().startswith("|"):
        meta.append(lines[i])
        i += 1
    rows = lines[i:]
    if not rows:
        return [block]
    header = [rows[0]]
    data_start = 1
    # 分隔行(| --- | --- |)并入表头
    if len(rows) > 1 and set(rows[1].replace("|", "").replace("-", "").strip()) <= {""}:
        header.append(rows[1])
        data_start = 2
    hdr = "\n".join(meta + header)
    groups, cur, cur_len = [], [hdr], len(hdr)
    for r in rows[data_start:]:
        if cur_len + len(r) + 1 > avail and len(cur) > len(meta) + len(header):
            groups.append("\n".join(cur))
            cur, cur_len = [hdr, r], len(hdr) + len(r) + 1
        else:
            cur.append(r)
            cur_len += len(r) + 1
    if len(cur) > 1:
        groups.append("\n".join(cur))
    return groups


# ---------- 4) 小节内打包:句子贪心 + 句级重叠,表格作为整体单元 ----------
def _join_units(units: List[Tuple[str, bool]]) -> str:
    """把单元拼回文本:连续散文句直接相接,表格独立成块(空行分隔),保留结构。"""
    parts, buf = [], []
    for text, is_tbl in units:
        if is_tbl:
            if buf:
                parts.append("".join(buf))
                buf = []
            parts.append(text)
        else:
            buf.append(text)
    if buf:
        parts.append("".join(buf))
    return "\n\n".join(parts)


def _pack_section(heading: str, body: List[_Block],
                  chunk_size: int, overlap: int) -> List[str]:
    prefix = (heading + "\n\n") if heading else ""
    avail = max(60, chunk_size - len(prefix))

    # 构造单元序列:表格→整块(超长则按行切);散文→分句;超长句→字符硬切
    units: List[Tuple[str, bool]] = []
    for kind, b in body:
        if kind == "table":
            units += [(p, True) for p in _split_table(b, avail)]
        else:
            for sent in _split_sentences(b):
                if len(sent) <= avail:
                    units.append((sent, False))
                else:
                    units += [(sent[i:i + avail], False)
                              for i in range(0, len(sent), avail)]

    chunks: List[str] = []
    cur: List[Tuple[str, bool]] = []
    cur_len = 0
    for u, is_tbl in units:
        if cur and cur_len + len(u) > avail:
            chunks.append(_join_units(cur))
            # 句级重叠:仅回收尾部连续的散文句(不重复表格)
            tail, tail_len = [], 0
            for x, t in reversed(cur):
                if t or tail_len + len(x) > overlap:
                    break
                tail.insert(0, (x, t))
                tail_len += len(x)
            cur, cur_len = tail, tail_len
        cur.append((u, is_tbl))
        cur_len += len(u)
    if cur:
        chunks.append(_join_units(cur))

    return [prefix + c for c in chunks if c.strip()]


def _breadcrumb(chapter: str | None, heading: str) -> str:
    """章名面包屑:当片段自身标题不含章名时,前缀【章名】补回章级上下文。"""
    if chapter and not _heading_title(heading).startswith(chapter):
        return f"【{chapter}】\n"
    return ""


def _merge_small_records(recs: List[dict], chunk_size: int) -> List[dict]:
    """合并相邻的"碎片"小节(多为同一检验项目的 【参考范围】/【临床意义】 等微小节,
    短于 frag 阈值),把它们并入前一片段,避免一条信息被打成碎片。约束:合并后不超
    chunk_size、同一章、且不触碰章前言。正常大小的小节保持独立(不被并掉,保留结构)。
    metadata:chapter 不变,section 取首节(主节)。"""
    frag = min(250, chunk_size // 3)

    def is_chapter_intro(rec: dict) -> bool:
        # 章前言:有章名、无节名 → 不参与合并,且作为合并的硬边界
        return rec.get("chapter") is not None and rec.get("section") is None

    out: List[dict] = []
    for r in recs:
        if (out and len(r["text"]) < frag
                and not is_chapter_intro(r) and not is_chapter_intro(out[-1])
                and out[-1].get("chapter") == r.get("chapter")
                and len(out[-1]["text"]) + 2 + len(r["text"]) <= chunk_size):
            out[-1]["text"] += "\n\n" + r["text"]   # 保留前一片段(主节)的 section
        else:
            out.append(dict(r))
    return out


def split_records(text: str, chunk_size: int = 500, overlap: int = 80) -> List[dict]:
    """结构感知切分,返回带元数据的片段记录:
    [{"text", "chapter", "section"}]。在结构切分基础上(标题随正文、表格原子化、
    保留换行、小节合并),为每个片段补章名面包屑前缀并标注所属 章/节。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size and "\n" not in text:
        return [{"text": text, "chapter": None, "section": None}]

    blocks = _merge_captions(_split_blocks(text))
    recs: List[dict] = []
    for sec in _sections_with_context(blocks):
        crumb = _breadcrumb(sec["chapter"], sec["heading"])
        pieces = _pack_section(sec["heading"], sec["body"],
                               chunk_size - len(crumb), overlap)
        for p in pieces:
            p = p.strip()
            if p:
                recs.append({"text": crumb + p,
                             "chapter": sec["chapter"], "section": sec["section"]})
    return _merge_small_records(recs, chunk_size)


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """结构感知切分(纯文本视图,向后兼容)。元数据版见 split_records。"""
    return [r["text"] for r in split_records(text, chunk_size, overlap)]
