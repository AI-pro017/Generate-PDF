#!/usr/bin/env python3
"""
PDF Statement Balance Updater - Multi-Bank GUI Version
A user-friendly GUI application to update statement balance values in PDF bank statements.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
from decimal import Decimal
import webbrowser
import platform
from datetime import datetime

# Import our multi-bank processor system
try:
    from processors import get_processor, get_supported_banks
    MULTI_BANK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Multi-bank system not available: {e}")
    MULTI_BANK_AVAILABLE = False
    get_processor = None
    get_supported_banks = None

class PDFProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Statement Balance Updater - Multi-Bank Edition")
        self.root.geometry("900x700")
        
        # Detect operating system for platform-specific fixes
        self.is_macos = platform.system() == 'Darwin'
        
        # Configure root window with macOS-compatible settings
        if self.is_macos:
            # Use cross-platform colors on macOS for better compatibility
            self.root.configure(bg='#f5f5f5')  # Light gray that works on macOS
            # Fix for macOS focus issues
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
        else:
            self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.input_file_var = tk.StringVar()
        self.output_file_var = tk.StringVar()
        self.balance_var = tk.StringVar()
        self.bank_var = tk.StringVar()
        
        # Initialize processor system
        self.initialize_processor_system()
        
        # Setup GUI after a small delay on macOS to ensure proper initialization
        if self.is_macos:
            self.root.after(100, self.setup_gui)
        else:
            self.setup_gui()
    
    def initialize_processor_system(self):
        """Initialize the multi-bank processor system"""
        if MULTI_BANK_AVAILABLE and get_processor and get_supported_banks:
            try:
                self.supported_banks = get_supported_banks()
                self.use_multi_bank = True
                # Create default processor (MBB)
                self.processor = get_processor('MBB')
                print(f"✅ Multi-bank system initialized with {len(self.supported_banks)} banks")
            except Exception as e:
                print(f"❌ Error initializing multi-bank system: {e}")
                # Fallback to original system
                self.fallback_to_legacy_system()
        else:
            # Fallback to original system
            self.fallback_to_legacy_system()
    
    def fallback_to_legacy_system(self):
        """Fallback to the original single-bank system"""
        try:
            from main_processor import PDFStatementProcessor
            self.processor = PDFStatementProcessor()
            self.supported_banks = ['MBB']  # Only Maybank supported in original
            self.use_multi_bank = False
            print("✅ Using legacy single-bank system")
        except Exception as e:
            print(f"❌ Error initializing legacy system: {e}")
            self.processor = None
            self.supported_banks = []
            self.use_multi_bank = False
        
    def get_bg_color(self):
        """Get appropriate background color for the platform"""
        if self.is_macos:
            # Use cross-platform colors that work on macOS
            return '#f5f5f5'  # Light gray that looks good on macOS
        else:
            return '#f0f0f0'
    
    def get_title_bg_color(self):
        """Get appropriate title background color for the platform"""
        if self.is_macos:
            # Use cross-platform colors that work on macOS
            return '#2c3e50'  # Dark blue that looks good on macOS
        else:
            return '#2c3e50'
    
    def get_title_fg_color(self):
        """Get appropriate title foreground color for the platform"""
        if self.is_macos:
            # Use cross-platform colors that work on macOS
            return 'white'  # White text on dark background
        else:
            return 'white'
        
    def setup_gui(self):
        """Setup the GUI components"""
        
        # Main title
        title_frame = tk.Frame(self.root, bg=self.get_title_bg_color(), height=60)
        title_frame.pack(fill='x', padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        # Use system fonts on macOS
        title_font = ('System', 18, 'bold') if self.is_macos else ('Arial', 18, 'bold')
        
        title_label = tk.Label(
            title_frame, 
            text="🏦 Multi-Bank PDF Statement Balance Updater",
            font=title_font,
            fg=self.get_title_fg_color(),
            bg=self.get_title_bg_color()
        )
        title_label.pack(pady=15)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg=self.get_bg_color())
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Bank selection section
        self.create_bank_section(main_frame)
        
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
    
    def create_bank_section(self, parent):
        """Create bank selection section"""
        
        label_font = ('System', 12, 'bold') if self.is_macos else ('Arial', 12, 'bold')
        text_font = ('System', 10) if self.is_macos else ('Arial', 10)
        
        # Bank selection frame
        bank_frame = tk.LabelFrame(
            parent, 
            text="🏦 Select Bank", 
            font=label_font,
            bg=self.get_bg_color(),
            fg='#2c3e50'  # Use cross-platform color instead of SystemControlTextColor
        )
        bank_frame.pack(fill='x', pady=(0, 15))
        
        bank_input_frame = tk.Frame(bank_frame, bg=self.get_bg_color())
        bank_input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            bank_input_frame, 
            text="Bank:", 
            font=text_font,
            bg=self.get_bg_color()
        ).pack(side='left')
        
        # Create bank selection dropdown
        self.bank_combo = ttk.Combobox(
            bank_input_frame,
            textvariable=self.bank_var,
            values=self.supported_banks,
            state='readonly',
            font=text_font,
            width=20
        )
        self.bank_combo.pack(side='left', padx=(10, 0))
        
        # Set default bank
        if self.supported_banks:
            self.bank_var.set(self.supported_banks[0])
            self.bank_combo.set(self.supported_banks[0])
        
        # Bind bank selection change
        self.bank_combo.bind('<<ComboboxSelected>>', self.on_bank_selected)
        
        # Bank status label (fully supported with reverse calculation)
        self.bank_status_label = tk.Label(
            bank_input_frame,
            text=f"✅ {self.supported_banks[0]} - Fully supported (reverse calculation)",
            font=('System', 9) if self.is_macos else ('Arial', 9),
            fg='green',  # Use standard green instead of SystemGreenColor
            bg=self.get_bg_color()
        )
        self.bank_status_label.pack(side='right', padx=(10, 0))
        
    def on_bank_selected(self, event=None):
        """Handle bank selection change"""
        selected_bank = self.bank_var.get()
        if not selected_bank:
            return
        
        # Update status label (all banks supported, reverse calculation)
        status_text = f"✅ {selected_bank} - Fully supported (reverse calculation)"
        status_color = 'green'  # Use standard green for all platforms
        
        self.bank_status_label.config(text=status_text, fg=status_color)
        
        # Create processor for selected bank
        if self.use_multi_bank and MULTI_BANK_AVAILABLE:
            try:
                self.processor = get_processor(selected_bank)
                self.log_message(f"🏦 Switched to {selected_bank} processor")
            except Exception as e:
                self.log_message(f"❌ Error creating {selected_bank} processor: {e}")
        
    def create_file_section(self, parent):
        """Create file selection section"""
        
        # Use system fonts on macOS
        label_font = ('System', 12, 'bold') if self.is_macos else ('Arial', 12, 'bold')
        text_font = ('System', 10) if self.is_macos else ('Arial', 10)
        
        # Input file frame
        input_frame = tk.LabelFrame(
            parent, 
            text="📄 Select PDF File", 
            font=label_font,
            bg=self.get_bg_color(),
            fg='#2c3e50'  # Use cross-platform color instead of SystemControlTextColor
        )
        input_frame.pack(fill='x', pady=(0, 15))
        
        # Input file selection
        input_file_frame = tk.Frame(input_frame, bg=self.get_bg_color())
        input_file_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            input_file_frame, 
            text="Input PDF:", 
            font=text_font,
            bg=self.get_bg_color()
        ).pack(side='left')
        
        self.input_entry = tk.Entry(
            input_file_frame, 
            textvariable=self.input_file_var,
            font=text_font,
            width=50
        )
        self.input_entry.pack(side='left', padx=(10, 5), fill='x', expand=True)
        
        # Use system button style on macOS
        browse_btn_config = {
            'text': "Browse",
            'command': self.browse_input_file,
            'font': text_font
        }
        
        if not self.is_macos:
            browse_btn_config.update({
                'bg': '#3498db',
                'fg': 'white',
                'cursor': 'hand2'
            })
        
        tk.Button(input_file_frame, **browse_btn_config).pack(side='right')
        
        # Output folder info
        output_info_frame = tk.Frame(input_frame, bg=self.get_bg_color())
        output_info_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        tk.Label(
            output_info_frame, 
            text="Output will be saved to:", 
            font=text_font,
            bg=self.get_bg_color()
        ).pack(side='left')
        
        # Show output folder path
        output_folder = os.path.join(os.getcwd(), "output")
        self.output_folder_label = tk.Label(
            output_info_frame,
            text=f"📁 {output_folder}",
            font=text_font,
            fg='blue',  # Use standard blue for all platforms
            bg=self.get_bg_color()
        )
        self.output_folder_label.pack(side='left', padx=(10, 0))
        
    def create_balance_section(self, parent):
        """Create balance input section"""
        
        label_font = ('System', 12, 'bold') if self.is_macos else ('Arial', 12, 'bold')
        text_font = ('System', 10) if self.is_macos else ('Arial', 10)
        entry_font = ('System', 12) if self.is_macos else ('Arial', 12)
        
        balance_frame = tk.LabelFrame(
            parent, 
            text="💰 Closing Balance", 
            font=label_font,
            bg=self.get_bg_color(),
            fg='#2c3e50'  # Use cross-platform color instead of SystemControlTextColor
        )
        balance_frame.pack(fill='x', pady=(0, 15))
        
        balance_input_frame = tk.Frame(balance_frame, bg=self.get_bg_color())
        balance_input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            balance_input_frame, 
            text="Enter closing balance (RM):", 
            font=text_font,
            bg=self.get_bg_color()
        ).pack(side='left')
        
        self.balance_entry = tk.Entry(
            balance_input_frame,
            textvariable=self.balance_var,
            font=entry_font,
            width=15
        )
        self.balance_entry.pack(side='left', padx=(10, 0))
        
        tk.Label(
            balance_input_frame, 
            text="(e.g., 4000.00)", 
            font=('System', 9) if self.is_macos else ('Arial', 9),
            fg='gray',  # Use standard gray for all platforms
            bg=self.get_bg_color()
        ).pack(side='left', padx=(10, 0))
        
    def create_process_section(self, parent):
        """Create process button section"""
        
        process_frame = tk.Frame(parent, bg=self.get_bg_color())
        process_frame.pack(fill='x', pady=(0, 15))
        
        # Process button configuration
        button_font = ('System', 14, 'bold') if self.is_macos else ('Arial', 14, 'bold')
        button_config = {
            'text': "🚀 Process PDF Statement",
            'command': self.process_pdf,
            'font': button_font,
            'height': 2
        }
        
        if not self.is_macos:
            button_config.update({
                'bg': '#27ae60',
                'fg': 'white',
                'cursor': 'hand2'
            })
        
        self.process_button = tk.Button(process_frame, **button_config)
        self.process_button.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            process_frame, 
            mode='indeterminate',
            length=400
        )
        self.progress.pack(pady=(0, 10))
        
    def create_results_section(self, parent):
        """Create results section"""
        
        label_font = ('System', 12, 'bold') if self.is_macos else ('Arial', 12, 'bold')
        
        results_frame = tk.LabelFrame(
            parent, 
            text="📊 Processing Results", 
            font=label_font,
            bg=self.get_bg_color(),
            fg='#2c3e50'  # Use cross-platform color instead of SystemControlTextColor
        )
        results_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=15,
            font=('System', 9) if self.is_macos else ('Courier', 9),
            bg='white' if not self.is_macos else 'SystemWindowBackgroundColor',
            fg='black' if not self.is_macos else 'SystemWindowTextColor'
        )
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Buttons frame
        buttons_frame = tk.Frame(results_frame, bg=self.get_bg_color())
        buttons_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        button_font = ('System', 10) if self.is_macos else ('Arial', 10)
        
        # Open output folder button
        open_btn_config = {
            'text': "📁 Open Output Folder",
            'command': self.open_output_folder,
            'font': button_font,
            'state': 'disabled'
        }
        
        if not self.is_macos:
            open_btn_config.update({
                'bg': '#3498db',
                'fg': 'white',
                'cursor': 'hand2'
            })
        
        self.open_output_button = tk.Button(buttons_frame, **open_btn_config)
        self.open_output_button.pack(side='left', padx=(0, 10))
        
        # Clear button
        clear_btn_config = {
            'text': "🗑️ Clear Results",
            'command': self.clear_results,
            'font': button_font
        }
        
        if not self.is_macos:
            clear_btn_config.update({
                'bg': '#e74c3c',
                'fg': 'white',
                'cursor': 'hand2'
            })
        
        self.clear_button = tk.Button(buttons_frame, **clear_btn_config)
        self.clear_button.pack(side='left')
        
    def create_status_bar(self):
        """Create status bar"""
        
        status_bg = '#34495e'  # Use cross-platform color instead of SystemWindowBackgroundColor
        status_fg = 'white'  # Use cross-platform color instead of SystemSecondaryLabelColor
        status_font = ('System', 9) if self.is_macos else ('Arial', 9)
        
        self.status_bar = tk.Label(
            self.root,
            text="Ready to process PDF statements",
            bd=1,
            relief='sunken',
            anchor='w',
            bg=status_bg,
            fg=status_fg,
            font=status_font
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
            self.log_message(f"📄 Selected input file: {filename}")
            # Reflect chosen input folder as the output location in the UI
            try:
                in_dir = os.path.dirname(filename)
                self.output_folder_label.config(text=f"📁 {in_dir}")
                self._last_output_folder = in_dir
            except Exception:
                pass
            
    def generate_output_filename(self, input_file: str, bank_code: str) -> str:
        """Generate output filename next to the input PDF"""
        # Save in the same folder as the source PDF
        output_folder = os.path.dirname(input_file)
        os.makedirs(output_folder, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate output filename
        output_filename = f"{base_name}_{bank_code}_{timestamp}.pdf"
        output_path = os.path.join(output_folder, output_filename)
        # remember for Open Output Folder button
        self._last_output_folder = output_folder
        
        return output_path
        
    def validate_inputs(self):
        """Validate all inputs before processing"""
        
        # Check if processor is available
        if not self.processor:
            messagebox.showerror("Error", "No processor available. Please check system initialization.")
            return False
        
        # Check if bank is selected
        if not self.bank_var.get():
            messagebox.showerror("Error", "Please select a bank.")
            return False
        
        # Check if input file is selected
        if not self.input_file_var.get():
            messagebox.showerror("Error", "Please select an input PDF file.")
            return False
        
        # Check if input file exists
        if not os.path.exists(self.input_file_var.get()):
            messagebox.showerror("Error", "Input file does not exist.")
            return False
        
        # Check if beginning balance is entered
        if not self.balance_var.get():
            messagebox.showerror("Error", "Please enter a beginning balance.")
            return False
        
        # Check if beginning balance is valid
        try:
            float(self.balance_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid beginning balance (numbers only).")
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
            beginning_balance = float(self.balance_var.get())
            selected_bank = self.bank_var.get()
            # Ensure processor matches the selected bank, even if the selection
            # change event was not triggered prior to clicking Process
            try:
                if self.use_multi_bank and MULTI_BANK_AVAILABLE:
                    self.processor = get_processor(selected_bank)
            except Exception as e:
                pass
            
            # Generate output filename
            output_file = self.generate_output_filename(input_file, selected_bank)
            
            # Redirect output to our text widget
            self.log_message("🚀 Starting PDF processing...")
            self.log_message(f"🏦 Selected bank: {selected_bank}")
            self.log_message(f"📁 Output will be saved to: {output_file}")
            self.log_message("=" * 60)
            
            # Process the statement
            success = self.processor.process_statement_gui(
                input_file, 
                output_file, 
                beginning_balance,
                self.log_message
            )
            
            # Update UI in main thread - critical for macOS
            if self.is_macos:
                # Use after_idle for better macOS compatibility
                self.root.after_idle(lambda: self.process_complete(success, output_file))
            else:
                self.root.after(0, self.process_complete, success, output_file)
            
        except Exception as e:
            error_msg = f"❌ Error during processing: {str(e)}"
            if self.is_macos:
                self.root.after_idle(lambda: self.log_message(error_msg))
                self.root.after_idle(lambda: self.process_complete(False, None))
            else:
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
        
        # Use update() instead of update_idletasks() on macOS for better responsiveness
        if self.is_macos:
            self.root.update()
        else:
            self.root.update_idletasks()
            
    def open_output_folder(self):
        """Open the output folder in file explorer"""
        
        output_folder = getattr(self, '_last_output_folder', None)
        if not output_folder and self.input_file_var.get():
            output_folder = os.path.dirname(self.input_file_var.get())
        if not output_folder:
            output_folder = os.getcwd()
        if os.path.exists(output_folder):
            try:
                if platform.system() == "Windows":
                    os.startfile(output_folder)
                elif platform.system() == "Darwin":  # macOS
                    os.system(f"open '{output_folder}'")
                else:  # Linux
                    os.system(f"xdg-open '{output_folder}'")
            except Exception as e:
                self.log_message(f"⚠️ Could not open output folder: {e}")
        else:
            self.log_message("⚠️ Output folder does not exist.")
            
    def clear_results(self):
        """Clear the results text area"""
        
        self.results_text.delete(1.0, tk.END)
        self.log_message("🗑️ Results cleared.")

def main():
    """Main function to start the GUI application"""
    
    root = tk.Tk()
    app = PDFProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 