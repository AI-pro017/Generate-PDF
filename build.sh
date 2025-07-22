#!/bin/bash
# macOS/Linux shell script to build the executable

echo "=========================================="
echo "PDF Statement Processor - Build Script"
echo "=========================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3 from https://python.org"
    exit 1
fi

echo "Python found. Installing dependencies..."

# Install requirements with verbose output
echo "Installing base requirements..."
pip3 install -r requirements.txt --upgrade

echo "Installing build dependencies..."
pip3 install pyinstaller --upgrade

echo
echo "Testing imports before building..."
python3 -c "import tkinter, PyPDF2, reportlab, fitz, pandas; print('All imports successful!')"
if [ $? -ne 0 ]; then
    echo
    echo "Error: Some dependencies failed to import"
    echo "Please check the error messages above"
    exit 1
fi

echo
echo "All dependencies verified. Building executable..."
python3 build_executable.py

if [ -f "dist/PDF_Statement_Processor" ]; then
    echo
    echo "=========================================="
    echo "BUILD SUCCESSFUL!"
    echo "=========================================="
    echo "Executable created: dist/PDF_Statement_Processor"
    echo
    echo "Making executable file executable..."
    chmod +x "dist/PDF_Statement_Processor"
    echo "Build complete!"
else
    echo
    echo "=========================================="
    echo "BUILD FAILED!"
    echo "=========================================="
    echo "Please check the error messages above"
fi

echo