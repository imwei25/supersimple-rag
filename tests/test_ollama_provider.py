from rag.providers.ollama_provider import OllamaProvider


def test_sampling_extracted_only_when_present():
    cfg = {"ollama_base_url": "http://x", "model": "m",
           "repeat_penalty": 1.2, "top_k": 40, "foo": 9}
    p = OllamaProvider(cfg)
    assert p.sampling == {"repeat_penalty": 1.2, "top_k": 40}


def test_no_sampling_when_absent():
    p = OllamaProvider({"ollama_base_url": "http://x", "model": "m"})
    assert p.sampling == {}
