#!/usr/bin/env python3
"""
PDF Statement Balance Updater - GUI Version
A user-friendly GUI application to update statement balance values in PDF bank statements.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
from decimal import Decimal
import webbrowser

# Import our PDF processor
from main_processor import PDFStatementProcessor

class PDFProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Statement Balance Updater - Maybank Edition")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.input_file_var = tk.StringVar()
        self.output_file_var = tk.StringVar()
        self.balance_var = tk.StringVar()
        
        # PDF Processor instance
        self.processor = PDFStatementProcessor()
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🏦 PDF Statement Balance Updater",
            font=('Arial', 18, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=15)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # File selection section
        self.create_file_section(main_frame)
        
        # Balance input section
        self.create_balance_section(main_frame)
        
        # Process button
        self.create_process_section(main_frame)
        
        # Progress and results section
        self.create_results_section(main_frame)
        
        # Status bar
        self.create_status_bar()
        
    def create_file_section(self, parent):
        """Create file selection section"""
        
        # Input file frame
        input_frame = tk.LabelFrame(
            parent, 
            text="📄 Select PDF File", 
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        input_frame.pack(fill='x', pady=(0, 15))
        
        # Input file selection
        input_file_frame = tk.Frame(input_frame, bg='#f0f0f0')
        input_file_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            input_file_frame, 
            text="Input PDF:", 
            font=('Arial', 10),
            bg='#f0f0f0'
        ).pack(side='left')
        
        self.input_entry = tk.Entry(
            input_file_frame, 
            textvariable=self.input_file_var,
            font=('Arial', 10),
            width=50
        )
        self.input_entry.pack(side='left', padx=(10, 5), fill='x', expand=True)
        
        tk.Button(
            input_file_frame,
            text="Browse",
            command=self.browse_input_file,
            bg='#3498db',
            fg='white',
            font=('Arial', 10),
            cursor='hand2'
        ).pack(side='right')
        
        # Output file selection
        output_file_frame = tk.Frame(input_frame, bg='#f0f0f0')
        output_file_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        tk.Label(
            output_file_frame, 
            text="Output PDF:", 
            font=('Arial', 10),
            bg='#f0f0f0'
        ).pack(side='left')
        
        self.output_entry = tk.Entry(
            output_file_frame, 
            textvariable=self.output_file_var,
            font=('Arial', 10),
            width=50
        )
        self.output_entry.pack(side='left', padx=(10, 5), fill='x', expand=True)
        
        tk.Button(
            output_file_frame,
            text="Browse",
            command=self.browse_output_file,
            bg='#3498db',
            fg='white',
            font=('Arial', 10),
            cursor='hand2'
        ).pack(side='right')
        
    def create_balance_section(self, parent):
        """Create balance input section"""
        
        balance_frame = tk.LabelFrame(
            parent, 
            text="💰 Beginning Balance", 
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        balance_frame.pack(fill='x', pady=(0, 15))
        
        balance_input_frame = tk.Frame(balance_frame, bg='#f0f0f0')
        balance_input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            balance_input_frame, 
            text="Enter beginning balance (RM):", 
            font=('Arial', 10),
            bg='#f0f0f0'
        ).pack(side='left')
        
        self.balance_entry = tk.Entry(
            balance_input_frame, 
            textvariable=self.balance_var,
            font=('Arial', 12),
            width=20,
            justify='right'
        )
        self.balance_entry.pack(side='left', padx=(10, 0))
        
        tk.Label(
            balance_input_frame, 
            text="(e.g., 4000.00)", 
            font=('Arial', 9),
            fg='gray',
            bg='#f0f0f0'
        ).pack(side='left', padx=(10, 0))
        
    def create_process_section(self, parent):
        """Create process button section"""
        
        process_frame = tk.Frame(parent, bg='#f0f0f0')
        process_frame.pack(fill='x', pady=(0, 15))
        
        self.process_button = tk.Button(
            process_frame,
            text="🚀 Process PDF Statement",
            command=self.process_pdf,
            bg='#27ae60',
            fg='white',
            font=('Arial', 14, 'bold'),
            height=2,
            cursor='hand2'
        )
        self.process_button.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            process_frame, 
            mode='indeterminate',
            length=400
        )
        self.progress.pack(pady=(0, 10))
        
    def create_results_section(self, parent):
        """Create results display section"""
        
        results_frame = tk.LabelFrame(
            parent, 
            text="📊 Processing Results", 
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        results_frame.pack(fill='both', expand=True)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=15,
            font=('Consolas', 9),
            bg='#2c3e50',
            fg='#ecf0f1',
            insertbackground='white'
        )
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Buttons frame
        buttons_frame = tk.Frame(results_frame, bg='#f0f0f0')
        buttons_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.open_output_button = tk.Button(
            buttons_frame,
            text="📁 Open Output Folder",
            command=self.open_output_folder,
            bg='#f39c12',
            fg='white',
            font=('Arial', 10),
            state='disabled',
            cursor='hand2'
        )
        self.open_output_button.pack(side='left', padx=(0, 10))
        
        self.clear_button = tk.Button(
            buttons_frame,
            text="🗑️ Clear Results",
            command=self.clear_results,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10),
            cursor='hand2'
        )
        self.clear_button.pack(side='left')
        
    def create_status_bar(self):
        """Create status bar"""
        
        self.status_bar = tk.Label(
            self.root,
            text="Ready to process PDF statements",
            bd=1,
            relief='sunken',
            anchor='w',
            bg='#34495e',
            fg='white',
            font=('Arial', 9)
        )
        self.status_bar.pack(side='bottom', fill='x')
        
    def browse_input_file(self):
        """Browse for input PDF file"""
        
        filename = filedialog.askopenfilename(
            title="Select PDF Statement",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            self.input_file_var.set(filename)
            # Auto-generate output filename
            if not self.output_file_var.get():
                base_name = os.path.splitext(filename)[0]
                output_name = f"{base_name}_updated.pdf"
                self.output_file_var.set(output_name)
            
            self.log_message(f"📄 Selected input file: {filename}")
            
    def browse_output_file(self):
        """Browse for output PDF file"""
        
        filename = filedialog.asksaveasfilename(
            title="Save Updated PDF As",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            self.output_file_var.set(filename)
            self.log_message(f"💾 Output file set to: {filename}")
            
    def validate_inputs(self):
        """Validate user inputs"""
        
        # Check input file
        if not self.input_file_var.get():
            messagebox.showerror("Error", "Please select an input PDF file")
            return False
            
        if not os.path.exists(self.input_file_var.get()):
            messagebox.showerror("Error", "Input PDF file does not exist")
            return False
            
        # Check output file
        if not self.output_file_var.get():
            messagebox.showerror("Error", "Please specify an output PDF file")
            return False
            
        # Check balance
        try:
            balance = float(self.balance_var.get())
            if balance < 0:
                messagebox.showerror("Error", "Beginning balance cannot be negative")
                return False
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid beginning balance (numbers only)")
            return False
            
        return True
        
    def process_pdf(self):
        """Process the PDF in a separate thread"""
        
        if not self.validate_inputs():
            return
            
        # Disable process button and start progress
        self.process_button.config(state='disabled')
        self.progress.start()
        self.status_bar.config(text="Processing PDF statement...")
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Start processing in separate thread
        thread = threading.Thread(target=self.process_pdf_thread)
        thread.daemon = True
        thread.start()
        
    def process_pdf_thread(self):
        """Process PDF in background thread"""
        
        try:
            input_file = self.input_file_var.get()
            output_file = self.output_file_var.get()
            beginning_balance = float(self.balance_var.get())
            
            # Redirect output to our text widget
            self.log_message("🚀 Starting PDF processing...")
            self.log_message("=" * 60)
            
            # Process the statement
            success = self.processor.process_statement_gui(
                input_file, 
                output_file, 
                beginning_balance,
                self.log_message
            )
            
            # Update UI in main thread
            self.root.after(0, self.process_complete, success, output_file)
            
        except Exception as e:
            error_msg = f"❌ Error during processing: {str(e)}"
            self.root.after(0, self.log_message, error_msg)
            self.root.after(0, self.process_complete, False, None)
            
    def process_complete(self, success, output_file):
        """Called when processing is complete"""
        
        # Stop progress and re-enable button
        self.progress.stop()
        self.process_button.config(state='normal')
        
        if success:
            self.status_bar.config(text=f"✅ Processing completed successfully!")
            self.open_output_button.config(state='normal')
            self.log_message("🎉 Processing completed successfully!")
            self.log_message(f"📁 Updated PDF saved to: {output_file}")
            
            # Show success message
            messagebox.showinfo(
                "Success", 
                f"PDF processing completed successfully!\n\nOutput saved to:\n{output_file}"
            )
        else:
            self.status_bar.config(text="❌ Processing failed")
            self.log_message("❌ Processing failed. Please check the error messages above.")
            messagebox.showerror("Error", "PDF processing failed. Please check the results for details.")
            
    def log_message(self, message):
        """Add message to results text area"""
        
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()
        
    def open_output_folder(self):
        """Open the folder containing the output file"""
        
        output_file = self.output_file_var.get()
        if output_file and os.path.exists(output_file):
            folder = os.path.dirname(output_file)
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                os.system(f"open '{folder}'")
            else:
                os.system(f"xdg-open '{folder}'")
                
    def clear_results(self):
        """Clear the results text area"""
        
        self.results_text.delete(1.0, tk.END)
        self.status_bar.config(text="Results cleared")


def main():
    """Main function to start the GUI"""
    
    # Check if required modules are available
    try:
        import fitz
        from reportlab.pdfgen import canvas
    except ImportError as e:
        messagebox.showerror(
            "Missing Dependencies", 
            f"Required libraries are missing: {e}\n\n"
            "Please install them using:\n"
            "pip install PyPDF2 reportlab PyMuPDF pandas"
        )
        return
    
    root = tk.Tk()
    app = PDFProcessorGUI(root)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main() 