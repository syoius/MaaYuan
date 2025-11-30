@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: 定义ANSI颜色（仅启动提示用）
for /f %%a in ('echo prompt $E^| cmd') do set "ESC=%%a"
set "YELLOW=%ESC%[33m"
set "RESET=%ESC%[0m"

:: 检测是否已为管理员
openfiles >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%正在请求管理员权限...%RESET%
    :: 以管理员身份重启自身
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process cmd.exe -ArgumentList '/c ""%~f0""' -Verb RunAs"
    exit /b
)

:: 以管理员身份启动PowerShell执行核心脚本
echo %YELLOW%正在以管理员身份执行安装，请稍候...%RESET%
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0InstallRuntime.ps1"

:: 批处理退出
exit /b

Set-ExecutionPolicy RemoteSigned