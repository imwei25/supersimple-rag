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
