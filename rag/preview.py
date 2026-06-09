# rag/preview.py
"""切分预览:用与入库完全一致的 loader + splitter + config,把知识库每个文档
切成 chunk,导出一个自包含 HTML,便于人工观察切分粒度与数据预处理质量。

不调用 LLM、不读向量库——只复现"文档→片段"这一步(enrich 仅在片段后追加
关键词,不改变切分边界,故此处略去),让你直观看到每段的真实文本。

用法:
    python -m rag.preview                 # 用 config.yaml,输出 chunks_preview.html
    python -m rag.preview -o out.html --open
"""
from __future__ import annotations

import argparse
import html
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rag.config import load_config
from rag.loader import load_documents
from rag.splitter import split_records


@dataclass
class Chunk:
    source: str
    idx: int
    text: str
    chapter: str | None = None
    section: str | None = None

    @property
    def length(self) -> int:
        return len(self.text)


def build_chunks(docs_dir: Path, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for text, source in load_documents(docs_dir):
        for idx, r in enumerate(split_records(text, chunk_size, overlap)):
            chunks.append(Chunk(source, idx, r["text"], r.get("chapter"), r.get("section")))
    return chunks


# 触发"疑似质量问题"标记的启发式:过短/过长,或疑似表格残片(行多且短、数字密集)
def _flags(c: Chunk, chunk_size: int) -> List[str]:
    flags: List[str] = []
    if c.length < 60:
        flags.append("short")
    if c.length > chunk_size * 1.2:
        flags.append("long")
    lines = [l for l in c.text.splitlines() if l.strip()]
    if lines:
        short_lines = sum(1 for l in lines if len(l.strip()) <= 6)
        digits = sum(ch.isdigit() for ch in c.text)
        if len(lines) >= 6 and short_lines / len(lines) > 0.5:
            flags.append("table")           # 多数是极短行 → 多半是被打散的表格
        elif c.length and digits / c.length > 0.25:
            flags.append("table")           # 数字密度高 → 多半是表格/参考值
    return flags


_CSS = """
* { box-sizing: border-box; }
body { margin:0; font:14px/1.7 -apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif; color:#1a1a1a; background:#f4f5f7; }
header { position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid #e3e6ea; padding:12px 20px; box-shadow:0 1px 4px rgba(0,0,0,.04); }
header h1 { margin:0 0 8px; font-size:16px; }
.stats { color:#555; font-size:13px; }
.stats b { color:#2257d6; }
.controls { margin-top:10px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.controls input[type=search]{ flex:1; min-width:220px; padding:7px 10px; border:1px solid #ccc; border-radius:6px; font-size:13px; }
.controls label{ font-size:13px; color:#444; cursor:pointer; user-select:none; }
.layout { display:flex; align-items:flex-start; }
nav { position:sticky; top:128px; width:260px; max-height:calc(100vh - 140px); overflow:auto; padding:14px; flex:0 0 auto; }
nav a { display:flex; justify-content:space-between; gap:8px; padding:5px 8px; border-radius:5px; color:#333; text-decoration:none; font-size:13px; }
nav a:hover { background:#e9eefb; }
nav a .n { color:#888; }
main { flex:1; padding:14px 20px 60px; min-width:0; }
.src-group > h2 { font-size:15px; margin:22px 0 10px; padding:6px 10px; background:#eef1f6; border-left:4px solid #2257d6; border-radius:4px; position:sticky; top:120px; }
.chunk { background:#fff; border:1px solid #e3e6ea; border-radius:8px; margin:10px 0; padding:0; overflow:hidden; }
.chunk .meta { display:flex; gap:10px; align-items:center; padding:6px 12px; background:#fafbfc; border-bottom:1px solid #eef0f3; font-size:12px; color:#666; }
.chunk .meta .id { font-weight:600; color:#2257d6; }
.chunk .meta .crumb { color:#1f7a4d; background:#e7f6ed; padding:1px 7px; border-radius:4px; font-size:11px; }
.chunk .body { padding:10px 14px; white-space:pre-wrap; word-break:break-word; }
.badge { font-size:11px; padding:1px 7px; border-radius:10px; font-weight:600; }
.b-short { background:#fde2e2; color:#b32020; }
.b-long  { background:#fff0d6; color:#a86400; }
.b-table { background:#e2ecff; color:#1f51b8; }
.len { margin-left:auto; color:#999; }
mark { background:#ffe770; padding:0 1px; }
.hidden { display:none !important; }
"""

_JS = """
const q = document.getElementById('q');
const onlyFlagged = document.getElementById('onlyFlagged');
const chunks = Array.from(document.querySelectorAll('.chunk'));
function esc(s){return s.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&');}
function apply(){
  const term = q.value.trim();
  const re = term ? new RegExp(esc(term),'gi') : null;
  const flaggedOnly = onlyFlagged.checked;
  for(const ch of chunks){
    const body = ch.querySelector('.body');
    const raw = body.dataset.raw;
    let visible = true;
    if(flaggedOnly && ch.dataset.flags==='') visible = false;
    if(re){ re.lastIndex=0; if(!re.test(raw)) visible=false; }
    ch.classList.toggle('hidden', !visible);
    if(re && visible){ body.innerHTML = raw.replace(re, m=>'<mark>'+m+'</mark>'); }
    else { body.textContent = raw; }
  }
  // 隐藏没有可见 chunk 的来源分组
  document.querySelectorAll('.src-group').forEach(g=>{
    const any = g.querySelector('.chunk:not(.hidden)');
    g.classList.toggle('hidden', !any);
  });
}
q.addEventListener('input', apply);
onlyFlagged.addEventListener('change', apply);
"""


def render_html(chunks: List[Chunk], cfg, title: str) -> str:
    by_source: dict[str, List[Chunk]] = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c)

    lengths = [c.length for c in chunks] or [0]
    chunk_size = cfg.split["chunk_size"]
    overlap = cfg.split["chunk_overlap"]
    flagged_total = sum(1 for c in chunks if _flags(c, chunk_size))

    # 顶部统计
    stats = (
        f"<span>文档 <b>{len(by_source)}</b></span> · "
        f"<span>片段 <b>{len(chunks)}</b></span> · "
        f"<span>切分参数 chunk_size=<b>{chunk_size}</b> overlap=<b>{overlap}</b></span> · "
        f"<span>片段长度 min <b>{min(lengths)}</b> / 均 <b>{sum(lengths)//len(lengths)}</b> / max <b>{max(lengths)}</b></span> · "
        f"<span>疑似问题片段 <b>{flagged_total}</b></span>"
    )

    # 侧边导航
    nav_links = "\n".join(
        f'<a href="#src-{i}"><span>{html.escape(src)}</span>'
        f'<span class="n">{len(cs)}</span></a>'
        for i, (src, cs) in enumerate(by_source.items())
    )

    # 主体
    groups = []
    for i, (src, cs) in enumerate(by_source.items()):
        cards = []
        for c in cs:
            fl = _flags(c, chunk_size)
            badges = "".join(
                f'<span class="badge b-{f}">{f}</span>' for f in fl
            )
            crumb = " › ".join(x for x in (c.chapter, c.section) if x)
            crumb_html = (f'<span class="crumb">{html.escape(crumb)}</span>'
                          if crumb else "")
            cards.append(
                f'<div class="chunk" data-flags="{" ".join(fl)}">'
                f'<div class="meta"><span class="id">#{c.idx}</span>{badges}'
                f'{crumb_html}<span class="len">{c.length} 字</span></div>'
                f'<div class="body" data-raw="{html.escape(c.text)}">{html.escape(c.text)}</div>'
                f'</div>'
            )
        groups.append(
            f'<section class="src-group" id="src-{i}">'
            f'<h2>{html.escape(src)} <span style="color:#888;font-weight:400">'
            f'· {len(cs)} 片段</span></h2>{"".join(cards)}</section>'
        )

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style></head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="stats">{stats}</div>
  <div class="controls">
    <input id="q" type="search" placeholder="搜索片段内容(支持高亮)…">
    <label><input id="onlyFlagged" type="checkbox"> 只看疑似问题片段(short / long / table)</label>
  </div>
</header>
<div class="layout">
  <nav>{nav_links}</nav>
  <main>{"".join(groups)}</main>
</div>
<script>{_JS}</script>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="导出知识库切分预览 HTML")
    ap.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("-o", "--out", type=Path, default=Path("chunks_preview.html"))
    ap.add_argument("--open", action="store_true", help="生成后用浏览器打开")
    args = ap.parse_args()

    cfg = load_config(args.config)
    chunks = build_chunks(
        cfg.docs_dir(), cfg.split["chunk_size"], cfg.split["chunk_overlap"]
    )
    html_text = render_html(chunks, cfg, title="知识库切分预览")
    args.out.write_text(html_text, encoding="utf-8")
    print(f"已生成 {args.out}  (片段 {len(chunks)} 个)")
    if args.open:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
