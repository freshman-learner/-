@echo off
chcp 65001 >nul
cd /d "%~dp0"
"D:\software\Anaconda\envs\pytorch-2.13.0-GPU\python.exe" -m tensorboard.main --logdir ppo_runs
pause
