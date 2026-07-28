@echo off
setlocal
cd /d D:\third_I\si_in\VAD
if "%~1"=="" exit /b 2
if "%~2"=="" exit /b 2
if not exist runs\full_train_logs mkdir runs\full_train_logs
set PYTHONUNBUFFERED=1
set CONFIG=%~1
set LOGPREFIX=%~2
C:\myApps\Miniconda\envs\eis\python.exe -u -m vadbench.cli train --config "%CONFIG%" > "runs\full_train_logs\%LOGPREFIX%.stdout.log" 2> "runs\full_train_logs\%LOGPREFIX%.stderr.log"
echo %ERRORLEVEL% > "runs\full_train_logs\%LOGPREFIX%.exitcode"
exit /b %ERRORLEVEL%
