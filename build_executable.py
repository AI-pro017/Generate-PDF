#!/usr/bin/env python3
"""
Build script to create Windows and Mac executables for PDF Statement Processor
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✅ PyInstaller is already installed")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✅ PyInstaller installed successfully")

def create_spec_file():
    """Create PyInstaller spec file for better control"""
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include any data files if needed
    ],
    hiddenimports=[
        # Core GUI modules
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        
        # PDF processing libraries
        'PyPDF2',
        'PyPDF2.pdf',
        'reportlab',
        'reportlab.pdfgen',
        'reportlab.pdfgen.canvas',
        'reportlab.lib',
        'reportlab.lib.pagesizes',
        'reportlab.lib.units',
        'reportlab.lib.colors',
        'reportlab.platypus',
        'reportlab.lib.styles',
        'fitz',
        
        # Data processing
        'pandas',
        'pandas._libs',
        'pandas._libs.tslibs',
        'pandas.core',
        'pandas.core.dtypes',
        'pandas.io',
        'pandas.io.formats',
        
        # Standard libraries
        'decimal',
        'threading',
        'webbrowser',
        'os',
        'sys',
        're',
        'typing',
        'pathlib',
        'time',
        'traceback',
        
        # Our modules
        'main_processor',
        'pdf_gui'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',
        'numpy.distutils',
        'scipy',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'setuptools',
        'wheel',
        'pip'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PDF_Statement_Processor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for GUI application
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico' if os.path.exists('app_icon.ico') else None,
)
'''
    
    with open('pdf_processor.spec', 'w') as f:
        f.write(spec_content.strip())
    print("✅ Created PyInstaller spec file with all dependencies")

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    required_modules = [
        'tkinter',
        'PyPDF2', 
        'reportlab',
        'fitz',
        'pandas',
        'decimal',
        'threading'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Missing modules: {', '.join(missing)}")
        print("Please install them with:")
        print("pip install PyPDF2 reportlab PyMuPDF pandas")
        return False
    
    print("✅ All imports successful")
    return True

def build_executable():
    """Build the executable using PyInstaller"""
    print("🔨 Building executable...")
    
    # Clean previous builds
    if os.path.exists('build'):
        shutil.rmtree('build')
        print("🗑️ Cleaned build directory")
    
    if os.path.exists('dist'):
        shutil.rmtree('dist')
        print("🗑️ Cleaned dist directory")
    
    # Build using spec file with verbose output
    try:
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            "pdf_processor.spec"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Executable built successfully!")
            
            # Show the output location
            if sys.platform == "win32":
                exe_path = os.path.join("dist", "PDF_Statement_Processor.exe")
            else:
                exe_path = os.path.join("dist", "PDF_Statement_Processor")
            
            if os.path.exists(exe_path):
                print(f"📁 Executable location: {os.path.abspath(exe_path)}")
                print(f"📏 File size: {os.path.getsize(exe_path) / (1024*1024):.1f} MB")
            return True
        else:
            print("❌ Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Build failed with exception: {e}")
        return False

def main():
    """Main build process"""
    print("🚀 PDF Statement Processor - Executable Builder")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('run_gui.py'):
        print("❌ Error: run_gui.py not found. Please run this script from the project directory.")
        return
    
    # Test imports first
    if not test_imports():
        print("❌ Please install missing dependencies before building")
        return
    
    # Install PyInstaller
    install_pyinstaller()
    
    # Create spec file
    create_spec_file()
    
    # Build executable
    if build_executable():
        print("\n🎉 Build completed successfully!")
        print("📁 Check the 'dist' folder for your executable")
        
        # Platform-specific instructions
        if sys.platform == "win32":
            print("💡 Windows users: You can now distribute 'PDF_Statement_Processor.exe'")
        elif sys.platform == "darwin":
            print("💡 Mac users: You can now distribute 'PDF_Statement_Processor'")
        else:
            print("💡 Linux users: You can now distribute 'PDF_Statement_Processor'")
            
        print("⚠️ Note: The executable includes all dependencies and may be large (100-200MB)")
        print("🧪 Test the executable before distribution!")
    else:
        print("❌ Build failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 