@echo off
chcp 65001 >nul

REM 添加所有更改
git add -A

REM 提示用户输入提交信息
set /p commit_message="请输入提交信息: "

REM 提交更改
git commit -m "%commit_message%"

REM 推送到远程仓库
git push