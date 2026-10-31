@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "VENV_PYW=%~dp0.venv\Scripts\pythonw.exe"
if exist "%VENV_PYW%" (
  start "" "%VENV_PYW%" -m cineblur.app
  exit /b 0
)

where pyw >nul 2>nul
if not errorlevel 1 (
  start "" pyw -3 -m cineblur.app
  exit /b 0
)

where pythonw >nul 2>nul
if not errorlevel 1 (
  start "" pythonw -m cineblur.app
  exit /b 0
)

echo.
echo   [CineBlur] Python environment not found.
echo.
echo   Create the project environment with:
echo     py -3.13 -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo   Or run a packaged build from:  dist\CineBlur\CineBlur.exe
echo.
pause
exit /b 1
