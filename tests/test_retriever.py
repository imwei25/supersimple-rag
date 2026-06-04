# tests/test_retriever.py
from rag.retriever import rrf_fuse

def test_rrf_fuse_merges_and_ranks():
    vector_ids = ["a", "b", "c"]
    bm25_ids = ["b", "d", "a"]
    fused = rrf_fuse(vector_ids, bm25_ids, rrf_k=60)
    # b 在两路都靠前,应排第一
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}

def test_rrf_fuse_single_list():
    assert rrf_fuse(["x", "y"], [], rrf_k=60) == ["x", "y"]
