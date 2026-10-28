@echo off
chcp 65001 >nul
cd /d "%~dp0"
"D:\software\Anaconda\envs\pytorch-2.13.0-GPU\python.exe" -X utf8 reach_demo.py random --cycles 2 --playback-speed 0.4
pause
