# app.spec —— PyInstaller onedir 配置(在 Windows 上运行 pyinstaller app.spec)
# 模型/config.yaml/docs_kb/chroma_db 均不打入归档,保持 exe 同级外置。
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = [], [], []
# jaraco/backports/pkg_resources:setuptools 的 pkg_resources 运行时会经
# jaraco.text → jaraco.context → backports.tarfile,命名空间包默认收不全,
# 否则报 ModuleNotFoundError: No module named 'backports'。一并 collect_all。
for pkg in ["gradio", "gradio_client", "safehttpx", "groovy",
            "chromadb", "sentence_transformers", "transformers",
            "tokenizers", "jieba", "llama_cpp",
            "jaraco", "backports", "setuptools"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:        # 某些包未安装/无数据时跳过,不阻断打包
        print(f"[app.spec] collect_all({pkg}) 跳过: {e}")

binaries += collect_dynamic_libs("llama_cpp")   # llama.dll / ggml*.dll
hiddenimports += ["onnxruntime", "hnswlib", "tiktoken_ext",
                  "uvicorn.logging", "uvicorn.protocols",
                  "uvicorn.protocols.http.auto", "uvicorn.lifespan.on",
                  "backports", "backports.tarfile",
                  "jaraco.text", "jaraco.context", "jaraco.functools"]

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
