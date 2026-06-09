# app.spec —— PyInstaller onedir 配置(在 Windows 上运行 pyinstaller app.spec)
# 模型/config.yaml/docs_kb/chroma_db 均不打入归档,保持 exe 同级外置。
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = [], [], []
for pkg in ["gradio", "gradio_client", "safehttpx", "groovy",
            "chromadb", "sentence_transformers", "transformers",
            "tokenizers", "jieba", "llama_cpp"]:
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

binaries += collect_dynamic_libs("llama_cpp")   # llama.dll / ggml*.dll
hiddenimports += ["onnxruntime", "hnswlib", "tiktoken_ext",
                  "uvicorn.logging", "uvicorn.protocols",
                  "uvicorn.protocols.http.auto", "uvicorn.lifespan.on"]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["nvidia", "triton", "tensorboard"],   # CPU 瘦身:排除 CUDA 相关
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name="知识库问答", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="知识库问答")
