REM dist_assets/① 建库.bat
@echo off
chcp 65001 >nul
"%~dp0知识库问答.exe" --mode ingest
pause
