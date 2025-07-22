# PDF Statement Balance Updater - Maybank Edition

A professional desktop application for automatically updating balance values in Maybank PDF bank statements. This tool recalculates running balances, maintains perfect column alignment, and preserves original PDF formatting and metadata.

## ✨ Features

- 🏦 **Maybank Statement Support**: Specifically designed for Maybank PDF statements
- 🎯 **Perfect Alignment**: Maintains exact column alignment matching the original PDF
- 📊 **Automatic Calculations**: Recalculates running balances based on beginning balance
- 🔒 **Preserves Formatting**: Keeps original PDF layout, fonts, and styling
- 📋 **Metadata Preservation**: Maintains all original PDF metadata (title, author, etc.)
- 💻 **User-Friendly GUI**: Simple, intuitive graphical interface
- 🚀 **No Technical Skills Required**: Just point, click, and process
- 📱 **Cross-Platform**: Works on Windows, Mac, and Linux

## 🚀 Quick Start (Executable Version)

### For End Users (Recommended)

**No Python installation required!** Just download and run the executable:

1. **Download** the executable for your system:
   - Windows: `PDF_Statement_Processor.exe`
   - Mac: `PDF_Statement_Processor`
   - Linux: `PDF_Statement_Processor`

2. **Double-click** the executable to launch

3. **Select your PDF** statement file

4. **Enter the beginning balance** amount

5. **Click "Process PDF Statement"**

6. **Done!** Your updated PDF will be saved automatically

### System Requirements
- **Windows**: Windows 10/11 (64-bit)
- **Mac**: macOS 10.14 or later
- **Linux**: Most modern distributions
- **Disk Space**: At least 100MB free space

## 🛠️ Developer Setup (Python Version)

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python run_gui.py
   ```

### Dependencies
- `PyPDF2>=3.0.0` - PDF manipulation
- `reportlab>=4.0.0` - PDF generation
- `PyMuPDF>=1.23.0` - Advanced PDF processing
- `pandas>=2.0.0` - Data handling

## 🔧 Building Executables

Create standalone executables for distribution to clients who don't have Python installed.

### Windows
```cmd
# Install build dependencies
pip install -r requirements.txt
pip install pyinstaller

# Run build script
build.bat
```

### Mac/Linux
```bash
# Install build dependencies
pip3 install -r requirements.txt
pip3 install pyinstaller

# Make script executable and run
chmod +x build.sh
./build.sh
```

### Build Output
- Executable will be created in the `dist/` folder
- Size: ~50-100MB (includes all dependencies)
- Single file distribution - no additional files needed

## 📖 How to Use

### Step-by-Step Guide

1. **Launch the Application**
   - Double-click the executable OR run `python run_gui.py`

2. **Select Input PDF**
   - Click "Browse" next to "Input PDF"
   - Choose your Maybank statement PDF file

3. **Set Output Location**
   - Click "Browse" next to "Output PDF" (auto-generated if not set)
   - Choose where to save the updated PDF

4. **Enter Beginning Balance**
   - Input the starting balance amount (e.g., 4000.00)
   - Use numbers only, no currency symbols

5. **Process the Statement**
   - Click "🚀 Process PDF Statement"
   - Watch the progress and results in the text area

6. **Review Results**
   - Check the processing log for any issues
   - Open the output folder to view your updated PDF

### Example
```
Input: bank_statement.pdf
Beginning Balance: 2513.01
Output: bank_statement_updated.pdf
``` 