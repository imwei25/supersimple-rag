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
