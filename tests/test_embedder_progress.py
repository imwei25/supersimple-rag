import numpy as np
from rag.embedder import Embedder


class _FakeModel:
    def encode(self, texts, **kwargs):
        return np.array([[float(len(t))] for t in texts])


def test_encode_iter_yields_progress_and_returns_all_vectors():
    e = Embedder.__new__(Embedder)          # 跳过真实模型加载
    e.batch_size = 2
    e.model = _FakeModel()
    gen = e.encode_iter(["a", "bb", "ccc", "dddd", "e"], batch=2)
    progress = []
    try:
        while True:
            progress.append(next(gen))
    except StopIteration as stop:
        vectors = stop.value
    assert progress == [(2, 5), (4, 5), (5, 5)]
    assert len(vectors) == 5
    assert vectors[3] == [4.0]              # "dddd" 长度 4


def test_encode_cached_iter_reuses_cache_and_encodes_only_new():
    from rag.embedder import Embedder, text_key
    e = Embedder.__new__(Embedder)
    e.batch_size = 2
    seen = {"n": 0}

    class _CountingModel:
        def encode(self, texts, **kwargs):
            seen["n"] += len(texts)
            return np.array([[float(len(t))] for t in texts])

    e.model = _CountingModel()
    texts = ["aa", "bbb", "cccc"]
    cache = {text_key("bbb"): [99.0]}          # 预置 bbb 的缓存
    gen = e.encode_cached_iter(texts, cache, batch=2)
    prog = []
    try:
        while True:
            prog.append(next(gen))
    except StopIteration as stop:
        vecs = stop.value
    assert vecs[1] == [99.0]                    # 命中缓存,复用
    assert vecs[0] == [2.0] and vecs[2] == [4.0]
    assert seen["n"] == 2                       # 只编码 aa、cccc,未编码 bbb
    assert prog[-1] == (2, 2)
    assert text_key("aa") in cache and text_key("cccc") in cache
