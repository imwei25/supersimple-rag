# build_windows.ps1 —— 在 Windows 上构建 onedir 包
# 前置:已装 Python 3.10、已建好 .venv 并装 requirements.txt + requirements-llamacpp.txt
# 重要:torch 用 CPU-only wheel,否则包会膨胀数 GB
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-llamacpp.txt

pyinstaller --noconfirm app.spec

# 产物在 dist/知识库问答/。把外置资源放进去:
New-Item -ItemType Directory -Force -Path "dist/知识库问答/models" | Out-Null
New-Item -ItemType Directory -Force -Path "dist/知识库问答/docs_kb" | Out-Null
Copy-Item config.yaml "dist/知识库问答/config.yaml" -Force
# 然后手动拷入 models/(GGUF + bge-large-zh-v1.5/ + bge-reranker-base/)
Write-Host "构建完成。请把 GGUF 与 bge/reranker 模型放入 dist/知识库问答/models/,"
Write-Host "并把 config.yaml 的 device 改为 cpu、model 改为 models 下的 GGUF 文件名。"
