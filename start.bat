@echo off
setlocal
cd /d "%~dp0"
set "FACELAB_PYTHON=python"
python -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
if errorlevel 1 (
  set "FACELAB_PYTHON=py -3"
  py -3 -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
  if errorlevel 1 (
    echo Python 3.10 or newer was not found. Install Python, then run this file again.
    pause
    exit /b 1
  )
)
%FACELAB_PYTHON% -c "import flask,numpy,scipy,PIL,sklearn" >nul 2>&1
if errorlevel 1 (
  echo Installing required packages...
  %FACELAB_PYTHON% -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Installation failed. Please copy the error above for troubleshooting.
    pause
    exit /b 1
  )
)
echo Starting FaceLab...
%FACELAB_PYTHON% main.py
pause
