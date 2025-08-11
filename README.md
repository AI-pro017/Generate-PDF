# Multi‑Bank PDF Statement Balance Updater

A desktop app that recalculates and rewrites statement balances in bank PDFs while preserving layout and metadata. The app now supports six banks and implements the new reverse calculation logic.

## ✨ Features

- 🏦 **Multi‑Bank Support**: MBB, PBB, RHB, HLB, OCBC, UOB
- 🔁 **Reverse Calculation**: You enter the closing balance, the app computes all prior running balances bottom‑to‑top
  - Deposits decrease the balance
  - Withdrawals increase the balance
- 🎯 **Perfect Alignment**: Rewrites numbers right‑aligned in the original columns
- 🔒 **Layout Preservation**: Fonts, colors, table rules, and metadata are retained
- 💻 **User‑friendly GUI**
- 📦 **Single Executable** for easy distribution (no Python required)
- 📂 **Output Location**: Updated PDF is saved next to the source PDF

## 🚀 Quick Start (Executable)

### For End Users (Recommended)

**No Python installation required!** Just download and run the executable:

1. **Download** the executable for your system:
   - Windows: `PDF_Statement_Processor.exe`
   - macOS: `PDF_Statement_Processor`

2. **Double-click** the executable to launch

3. **Select your PDF** statement file

4. **Enter the closing balance** amount

5. **Click "Process PDF Statement"**

6. **Done!** Your updated PDF will be saved in the same folder as the source PDF

### System Requirements

- **Windows**: Windows 10/11 (64-bit)
- **Mac**: macOS 10.14 or later
- **Linux**: Most modern distributions
- **Disk Space**: At least 100MB free space

## 🛠️ Developer Setup (Python)

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

- `PyPDF2>=3.0.0` – PDF manipulation
- `reportlab>=4.0.0` – PDF generation
- `PyMuPDF>=1.23.0` – Advanced PDF parsing/rendering
- `pandas>=2.0.0` – Data handling

## 🔧 Build Executables

Create standalone executables for distribution to clients who don't have Python installed.

### Windows

```cmd
# Install build dependencies
pip install -r requirements.txt
pip install pyinstaller

# Run build script
build.bat
```

### macOS / Linux

```bash
# Install build dependencies
pip3 install -r requirements.txt
pip3 install pyinstaller

# Make script executable and run
chmod +x build.sh
./build.sh
```

### Build Output

- Executable is created in `dist/`
- Size: ~100–200 MB (includes all dependencies)
- Single file distribution – no Python required

## 📖 How to Use

### Step-by-Step Guide

1. **Launch the Application**
   - Double‑click the executable OR run `python run_gui.py`

2. **Select Input PDF**
   - Click "Browse" next to "Input PDF"
   - Choose your Maybank statement PDF file

3. **Set Output Location**
   - Click "Browse" next to "Output PDF" (auto-generated if not set)
   - Choose where to save the updated PDF

4. **Enter Closing Balance**
   - Input the statement closing balance (e.g., 4000.00)
   - Numbers only, no currency symbols

5. **Process the Statement**
   - Click "🚀 Process PDF Statement"
   - Watch the progress and results in the text area

6. **Review Results**
   - Check the processing log for any issues
   - Open the output folder to view your updated PDF

### Example

```text
Input: bank_statement.pdf
Closing Balance: 2513.01
Output: bank_statement_MBB_20250101_120000.pdf
```

## ❓ FAQ

- Why do deposits reduce the balance and withdrawals increase it?
  - This project follows a reverse computation flow requested by the client: starting from the final closing balance and walking back through the statement bottom‑to‑top.

- Where is the output saved?
  - Beside the original source PDF (same folder). The GUI shows the path under “Output will be saved to:”.

- Can I still enter an opening/beginning balance?
  - No. The UI was updated to “Closing Balance” to match the reverse calculation logic across all supported banks.
