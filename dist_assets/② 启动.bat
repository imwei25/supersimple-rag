REM dist_assets/② 启动.bat
@echo off
chcp 65001 >nul
"%~dp0知识库问答.exe" --mode both
REM 程序退出(正常关闭或启动报错)后保留窗口,便于查看信息/错误
pause
