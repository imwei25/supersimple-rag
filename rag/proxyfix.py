# rag/proxyfix.py
"""本地局域网应用:bge 与 ollama 均为本地服务,无需走代理。
某些环境设置了 socks:// 代理(httpx 不识别该 scheme,会导致 gradio 等导入崩溃),
故在导入任何网络库之前清除代理环境变量。导入本模块即生效。"""
import os

for _k in (
    "ALL_PROXY", "all_proxy",
    "HTTP_PROXY", "http_proxy",
    "HTTPS_PROXY", "https_proxy",
):
    os.environ.pop(_k, None)
