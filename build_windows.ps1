# build_windows.ps1 —— 在 Windows 上构建 onedir 包
# 前置:已装 Python 3.10、已建好 .venv 并装 requirements.txt + requirements-llamacpp.txt
# 重要:torch 用 CPU-only wheel,否则包会膨胀数 GB
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pyinstaller==6.11.1
# setuptools 的 pkg_resources 依赖 backports.tarfile,缺它打包后会报 No module named 'backports'
pip install "setuptools>=70" backports.tarfile
# llama-cpp-python 用预编译 CPU wheel(否则源码编译需 C/C++ 编译器 + CMake)
pip install llama-cpp-python==0.3.28 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 先构建到独立暂存目录,避免 --noconfirm 清空部署目录里你已放好的 models 等
pyinstaller --noconfirm --distpath build_pkg app.spec

$Dest = "dist/知识库问答"
$Src  = "build_pkg/知识库问答"

if (-not (Test-Path $Dest)) { New-Item -ItemType Directory -Force -Path $Dest | Out-Null }

# 只刷新程序文件:exe + _internal(robocopy /MIR 仅镜像 _internal 子目录,不碰其它)
Copy-Item "$Src/知识库问答.exe" "$Dest/知识库问答.exe" -Force
robocopy "$Src/_internal" "$Dest/_internal" /MIR /NFL /NDL /NJH /NJS /NP | Out-Null

# 数据目录:已存在则原样保留(New-Item -Force 对已存在目录是空操作,不清内容)
New-Item -ItemType Directory -Force -Path "$Dest/models"  | Out-Null
New-Item -ItemType Directory -Force -Path "$Dest/docs_kb" | Out-Null

# 分发脚本/说明:覆盖更新以获取最新修复(.bat + 使用说明.txt)
Copy-Item "dist_assets/*" $Dest -Force

# config.yaml:仅首次拷入;已存在则保留你改好的配置,不覆盖
if (-not (Test-Path "$Dest/config.yaml")) { Copy-Item config.yaml "$Dest/config.yaml" -Force }

Write-Host "构建完成 → $Dest(models/docs_kb/chroma_db/config.yaml 均已保留,未覆盖)。"
Write-Host "首次部署:把模型放入 $Dest/models/(GGUF + bge-large-zh-v1.5/ + bge-reranker-base/),"
Write-Host "并把 config.yaml 改为 CPU 配置:embedding.device: cpu、retrieval.reranker_device: cpu、"
Write-Host "llm.provider: llama_cpp、llm.model 改为 models 下的 GGUF 文件名。"
