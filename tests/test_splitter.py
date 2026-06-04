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
