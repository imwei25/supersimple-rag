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
            "pdfplumber", "pdfminer", "pypdfium2",
            "jaraco", "backports", "setuptools"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:        # 某些包未安装/无数据时跳过,不阻断打包
        print(f"[app.spec] collect_all({pkg}) 跳过: {e}")

binaries += collect_dynamic_libs("llama_cpp")   # llama.dll / ggml*.dll
hiddenimports += ["onnxruntime", "hnswlib",
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
    # Gradio 运行时要 inspect 自己的源码,必须以 .py 源码形式落到磁盘
    # (否则报 No such file ... gradio\blocks_events.pyc)。
    module_collection_mode={
        "gradio": "py",
        "gradio_client": "py",
        "safehttpx": "py",
        "groovy": "py",
    },
)
pyz = PYZ(a.pure)
# exe/输出目录用 ASCII 名(doctor-t),避免中文路径在 .bat/.ps1/cmd 各种编码下踩坑。
# 界面/窗口标题仍是中文(在代码里),不影响品牌。
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name="doctor-t", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="doctor-t")
