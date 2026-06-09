# rag/enrich.py
"""入库时的语义增强(借鉴 RAGFlow 的 Auto-Keyword / Auto-Question):
让 LLM 为每个 chunk 生成关键词与"假设问题",拼接进被索引的文本,
扩大向量与 BM25 的命中面,显著提升口语化提问的召回。成本前置到入库,
查询时零额外开销。失败时优雅降级:返回原文,不阻断建库。"""
from __future__ import annotations
from typing import List, Optional
from pathlib import Path
import logging
import re
import json
import hashlib

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """去掉思维链模型的 <think>...</think> 块;若只剩未闭合的 <think>,视为无有效输出。"""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:  # 思考未结束就被截断,无可用内容
        return ""
    return text.strip()

_PROMPT = """下面是一段资料。请严格按格式输出,不要任何解释或多余内容:
1) 提取 3-6 个最能代表该资料的关键词,用、分隔;
2) 列出最多 {n} 个该资料可以直接回答的问题。

资料:
{chunk}

输出格式:
关键词:<...>
问题:
- <...>"""


def _complete(provider, prompt: str, char_limit: int = 400) -> str:
    """复用 provider 的流式接口拼成一次性补全。思维链模型先吐 <think>,
    故用更大的硬上限容纳思考,再剥离 think、把净内容截断到 char_limit。"""
    hard_cap = max(char_limit * 6, 1800)
    buf: List[str] = []
    total = 0
    for tok in provider.stream(prompt):
        buf.append(tok)
        total += len(tok)
        if total >= hard_cap:
            break
    cleaned = _strip_think("".join(buf))
    return cleaned[:char_limit].strip()


def _chunk_key(chunk: str) -> str:
    return hashlib.sha1(chunk.encode("utf-8")).hexdigest()


def _load_cache(path: Optional[Path]) -> dict:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("增强缓存读取失败,忽略: %s", e)
    return {}


def enrich_texts_iter(provider, texts: List[str], cfg: dict,
                      cache_path: Optional[Path] = None):
    """enrich_texts 的流式版本:每处理一个片段 yield (done, total, reused, newly),
    全部完成后通过 return 返回增强后的文本列表(用 `yield from` 可取到返回值)。"""
    if not cfg.get("enrich", False) or not texts:
        return texts
    n = cfg.get("enrich_questions", 3)
    char_limit = cfg.get("enrich_char_limit", 400)

    cache = _load_cache(cache_path)
    new_cache: dict = {}
    out: List[str] = []
    ok = reused = newly = 0
    for i, chunk in enumerate(texts):
        key = _chunk_key(chunk)
        gen = cache.get(key)
        if gen is not None:                 # 命中缓存:复用,跳过 LLM
            reused += 1
        else:
            try:
                gen = _complete(provider, _PROMPT.format(n=n, chunk=chunk), char_limit)
                newly += 1
            except Exception as e:          # noqa: BLE001
                logger.warning("第 %d 个片段增强失败,使用原文: %s", i, e)
                gen = ""
        if gen:
            new_cache[key] = gen            # 仅缓存成功结果,失败下次可重试
            out.append(chunk + "\n\n【检索增强】\n" + gen)
            ok += 1
        else:
            out.append(chunk)
        yield (i + 1, len(texts), reused, newly)

    if cache_path is not None:
        try:
            cache_path.write_text(
                json.dumps(new_cache, ensure_ascii=False), encoding="utf-8")
        except Exception as e:              # noqa: BLE001
            logger.warning("增强缓存写入失败: %s", e)
    logger.info("语义增强完成:%d/%d 富化(复用 %d,新生成 %d)", ok, len(texts), reused, newly)
    return out


def enrich_texts(provider, texts: List[str], cfg: dict,
                 cache_path: Optional[Path] = None) -> List[str]:
    """对每个 chunk 生成增强块并附加到原文后返回(阻塞版,供 ingest.py / 测试使用)。

    带内容哈希缓存:chunk 文本未变(SHA1 命中)则复用上次结果,不再调用 LLM;
    只有新增/改动的片段才重新生成。重建后按本次出现的哈希裁剪缓存。"""
    gen = enrich_texts_iter(provider, texts, cfg, cache_path)
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value
