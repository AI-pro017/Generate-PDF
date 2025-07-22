@echo off
REM Windows batch file to build the executable

echo ==========================================
echo PDF Statement Processor - Build Script
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Python found. Installing dependencies...

REM Install requirements with verbose output
echo Installing base requirements...
pip install -r requirements.txt --upgrade

echo Installing build dependencies...
pip install pyinstaller --upgrade

echo.
echo Testing imports before building...
python -c "import tkinter, PyPDF2, reportlab, fitz, pandas; print('All imports successful!')"
if errorlevel 1 (
    echo.
    echo Error: Some dependencies failed to import
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo All dependencies verified. Building executable...
python build_executable.py

if exist "dist\PDF_Statement_Processor.exe" (
    echo.
    echo ==========================================
    echo BUILD SUCCESSFUL!
    echo ==========================================
    echo Executable created: dist\PDF_Statement_Processor.exe
    echo.
    echo Testing the executable...
    echo Starting executable in test mode...
    timeout /t 3 >nul
    start "" "dist\PDF_Statement_Processor.exe"
    echo.
    echo If the application opened successfully, the build is complete!
) else (
    echo.
    echo ==========================================
    echo BUILD FAILED!
    echo ==========================================
    echo Please check the error messages above
)

echo.
pause