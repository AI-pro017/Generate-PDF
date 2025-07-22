#!/usr/bin/env python3
"""
PDF Statement Balance Updater - Main GUI Launcher
Standalone version for executable distribution
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    # Test each dependency individually with detailed error info
    deps_to_check = [
        ('fitz', 'PyMuPDF'),
        ('reportlab.pdfgen', 'reportlab'), 
        ('PyPDF2', 'PyPDF2'),
        ('pandas', 'pandas')
    ]
    
    for import_name, package_name in deps_to_check:
        try:
            __import__(import_name)
        except ImportError as e:
            missing_deps.append(f"{package_name} ({str(e)})")
    
    return missing_deps

def show_dependency_error(missing_deps):
    """Show a detailed dependency error window"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.title("Missing Dependencies")
    
    error_msg = "Missing required libraries:\n\n"
    for dep in missing_deps:
        error_msg += f"• {dep}\n"
    
    error_msg += "\nThis usually happens when:\n"
    error_msg += "1. The executable was not built correctly\n"
    error_msg += "2. Some dependencies were not included during build\n"
    error_msg += "3. Running the Python script without proper installation\n\n"
    error_msg += "Solutions:\n"
    error_msg += "• If using Python: pip install PyPDF2 reportlab PyMuPDF pandas\n"
    error_msg += "• If using executable: Please contact support - the build is incomplete"
    
    messagebox.showerror("Missing Dependencies", error_msg)

def main():
    """Main application entry point"""
    
    # Check dependencies first
    missing_deps = check_dependencies()
    if missing_deps:
        show_dependency_error(missing_deps)
        return 1
    
    try:
        # Import and start the GUI
        from pdf_gui import main as gui_main
        gui_main()
        return 0
        
    except Exception as e:
        # Create error window
        root = tk.Tk()
        root.withdraw()
        
        error_msg = f"An error occurred while starting the application:\n\n{str(e)}\n\n"
        error_msg += "Please ensure all files are present and try again.\n"
        error_msg += "If this is an executable, the build may be incomplete."
        
        messagebox.showerror("Application Error", error_msg)
        
        # Print to console for debugging
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 