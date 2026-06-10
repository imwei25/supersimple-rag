@echo off
chcp 65001 >nul
"%~dp0bin\doctor-t.exe" --mode both
pause
