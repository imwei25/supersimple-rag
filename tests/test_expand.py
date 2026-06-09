# tests/test_expand.py
from rag.retriever import expand_short_hit, _stitch, _parse_id


def _win(items):
    # items: list of (idx, text, chapter) → window dict
    return {i: {"id": f"bk::{i}", "text": t, "source": "bk",
                "chapter": ch, "section": f"s{i}"} for i, t, ch in items}


def test_parse_id():
    assert _parse_id("a.txt::3") == ("a.txt", 3)
    assert _parse_id("bad") == (None, None)


def test_stitch_dedups_overlap():
    a = "前文内容。中间重叠片段。"
    b = "中间重叠片段。后文内容。"
    assert _stitch(a, b) == "前文内容。中间重叠片段。后文内容。"


def test_long_hit_not_expanded():
    hit = {"id": "bk::2", "text": "x" * 500, "source": "bk"}
    win = _win([(2, "x" * 500, "C1")])
    out = expand_short_hit(hit, win, min_chars=300, target=600, max_neighbors=4)
    assert "expanded" not in out


def test_continuation_expands_up_first():
    # 中心片段无标题(节中延续)→ 应优先向上扩
    win = _win([
        (1, "## 标题一\n\n" + "上文。" * 30, "C1"),
        (2, "延续正文。" * 10, "C1"),       # 短,无标题
        (3, "## 标题二\n\n" + "下文。" * 30, "C1"),
    ])
    hit = dict(win[2])
    out = expand_short_hit(hit, win, min_chars=300, target=120, max_neighbors=4)
    assert out["expanded"] and out["expanded_span"][0] == 1   # 含了上文片段1


def test_expansion_stops_at_chapter():
    win = _win([
        (4, "甲。" * 5, "C1"),              # 短
        (5, "# 第二章\n\n乙。" * 20, "C2"), # 不同章
    ])
    win[3] = {"id": "bk::3", "text": "丙。" * 5, "source": "bk", "chapter": "C2", "section": "s3"}
    hit = dict(win[4])
    out = expand_short_hit(hit, win, min_chars=300, target=600, max_neighbors=4)
    # 两侧都是不同章 → 无法扩充,原样返回
    assert "乙" not in out["text"] and "丙" not in out["text"]
