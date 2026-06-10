# build_windows.ps1 —— 在 Windows 上构建 onedir 程序
# 关键安全设计:本脚本【只】产出到 build_pkg\,绝不触碰你的运行/数据目录,
# 所以无论重打包多少次,都不可能删到 chroma_db / models / docs_kb / config.yaml。
# 前置:已建好 .venv 并激活。CPU-only torch 避免包膨胀数 GB。
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pyinstaller==6.11.1
# setuptools 的 pkg_resources 依赖 backports.tarfile,缺它打包后会报 No module named 'backports'
pip install "setuptools>=70" backports.tarfile
# llama-cpp-python 用预编译 CPU wheel(否则源码编译需 C/C++ 编译器 + CMake)
pip install llama-cpp-python==0.3.28 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 只产出到 build_pkg\,不写 dist\、不写任何运行目录
pyinstaller --noconfirm --distpath build_pkg app.spec

Write-Host ""
Write-Host "============================================================"
Write-Host "构建完成。产物在: build_pkg\doctor-t\"
Write-Host ""
Write-Host "[推荐运行布局 - 程序与数据分家,重打包永不丢数据]"
Write-Host "  自己建一个固定运行目录,例如 D:\知识库问答运行\:"
Write-Host "    D:\知识库问答运行\"
Write-Host "      config.yaml          (从本仓库拷一份,改成 CPU 配置)"
Write-Host "      models\              (GGUF + bge-large-zh-v1.5\ + bge-reranker-base\)"
Write-Host "      docs_kb\             (放 PDF/Word)"
Write-Host "      chroma_db\           (建库.bat 生成,勿手动拷别处的)"
Write-Host "      建库.bat / 启动.bat   (从 dist_assets\ 拷来)"
Write-Host "      bin\                 (只放下面两样;重打包后只替换这俩)"
Write-Host "        doctor-t.exe"
Write-Host "        _internal\"
Write-Host ""
Write-Host "[每次重打包后,只做这一步]"
Write-Host "  把 build_pkg\doctor-t\ 里的 doctor-t.exe 和 _internal\,"
Write-Host "  覆盖拷到 D:\知识库问答运行\bin\ 即可。其它一律别动。"
Write-Host ""
Write-Host "  程序启动会自动向上找到含 config.yaml 的目录作数据根,"
Write-Host "  所以 exe 在 bin\ 子文件夹、数据在外层,完全没问题。"
Write-Host "============================================================"
