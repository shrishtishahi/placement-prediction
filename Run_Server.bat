@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Student Placement Prediction

echo.
echo  Student Placement Prediction
echo  ----------------------------
echo.

set "PYCMD="
py -3.12 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.12" && goto :havepy
py -3.11 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.11" && goto :havepy
py -3.13 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.13" && goto :havepy
py -3.10 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.10" && goto :havepy
py -3 -c "import sys" >nul 2>&1 && set "PYCMD=py -3" && goto :havepy
python -c "import sys" >nul 2>&1 && set "PYCMD=python" && goto :havepy

echo Python was not found on this computer.
echo.
echo 1. Install Python 3.11 or newer from https://www.python.org/downloads/
echo 2. On the first setup screen, tick "Add python.exe to PATH"
echo 3. Double-click Run_Server.bat again
echo.
pause
exit /b 1

:havepy
echo Using %PYCMD%
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating a local environment. This happens only once.
    %PYCMD% -m venv .venv
    if errorlevel 1 goto :fail
)

echo Checking required libraries...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
if errorlevel 1 goto :fail

echo.
echo Opening the app in your browser.
echo Keep this window open while you present.
echo Close this window when you are finished.
echo.

set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_HEADLESS=true
start "" http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --browser.gatherUsageStats=false --server.headless=true --server.port=8501
echo.
echo The app has stopped.
pause
exit /b 0

:fail
echo.
echo Setup could not finish.
echo If this is the first time, connect to the internet and try again.
echo.
pause
exit /b 1
