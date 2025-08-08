#!/usr/bin/env python3
"""
PDF Statement Processor Module - Legacy Compatibility
This module provides backward compatibility with the original single-bank system
"""

import os
import sys
import re
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Tuple, Optional, Callable

# Try to import required PDF libraries
try:
    import PyPDF2
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    import fitz  # PyMuPDF for better text extraction
except ImportError as e:
    print("❌ Missing required libraries. Please install them using:")
    print("pip install PyPDF2 reportlab PyMuPDF pandas")
    raise e

# Try to import the new multi-bank system
try:
    from processors import get_processor, get_supported_banks
    MULTI_BANK_AVAILABLE = True
except ImportError:
    MULTI_BANK_AVAILABLE = False


class PDFStatementProcessor:
    """Legacy compatibility class for processing PDF bank statements"""
    
    def __init__(self):
        self.transactions = []
        self.original_pdf_path = None
        self.output_pdf_path = None
        self.beginning_balance = Decimal('0.00')
        self.balance_replacements = []
        self.balance_column_x = None
        self.balance_column_right = None
        
        # If multi-bank system is available, use it
        if MULTI_BANK_AVAILABLE:
            self.mbb_processor = get_processor('MBB')
        else:
            self.mbb_processor = None
        
    def extract_transactions_from_pdf(self, pdf_path: str, log_func: Callable = print) -> List[Dict]:
        """Extract transaction data from Maybank PDF format"""
        if self.mbb_processor:
            # Use the new multi-bank system
            return self.mbb_processor.extract_transactions_from_pdf(pdf_path, log_func)
        else:
            # Use legacy implementation
            return self._extract_transactions_from_pdf_legacy(pdf_path, log_func)
    
    def _extract_transactions_from_pdf_legacy(self, pdf_path: str, log_func: Callable = print) -> List[Dict]:
        """Legacy implementation for extracting transaction data"""
        # Store the original PDF path for later use
        self.original_pdf_path = pdf_path
        self.balance_replacements = []  # Reset balance replacements
        
        transactions = []
        
        try:
            # Open PDF with PyMuPDF (fitz)
            pdf_document = fitz.open(pdf_path)
            
            log_func(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
            log_func(f"📊 Total pages: {len(pdf_document)}")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Get text with layout information
                text_dict = page.get_text("dict")
                
                log_func(f"📃 Page {page_num + 1} - Looking for transactions and balance values...")
                
                # Extract transactions and find all balance positions
                page_transactions = self._extract_transactions_and_balances(text_dict, page_num + 1, log_func)
                transactions.extend(page_transactions)
            
            pdf_document.close()
            
        except Exception as e:
            log_func(f"❌ Error extracting data from PDF: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        log_func(f"✅ Total extracted transactions: {len(transactions)}")
        log_func(f"✅ Total balance positions found: {len(self.balance_replacements)}")
        return transactions
    
    def _extract_transactions_and_balances(self, text_dict: dict, page_num: int, log_func: Callable) -> List[Dict]:
        """Extract transactions and find all balance values that need to be replaced"""
        transactions = []
        
        try:
            # Collect all text blocks with their positions
            all_blocks = []
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                bbox = span["bbox"]  # [x0, y0, x1, y1]
                                all_blocks.append({
                                    "text": text,
                                    "x": bbox[0],
                                    "y": bbox[1],
                                    "bbox": bbox,
                                    "font_size": span.get("size", 12),
                                    "font": span.get("font", "")
                                })
            
            # Sort blocks by y-coordinate (top to bottom), then x-coordinate (left to right)
            all_blocks.sort(key=lambda b: (b["y"], b["x"]))
            
            # Find the rightmost column position (where balances should be)
            balance_column_x = self._find_balance_column_position(all_blocks, log_func)
            
            # Group blocks into rows
            rows = self._group_blocks_into_rows(all_blocks)
            
            log_func(f"📋 Found {len(rows)} text rows on page {page_num}")
            
            # Process each row
            for row in rows:
                row.sort(key=lambda b: b["x"])  # Sort by x-coordinate
                
                if not row:
                    continue
                
                # Check if this is a transaction row (starts with date)
                if self._is_date(row[0]["text"]):
                    transaction = self._parse_transaction_row_simple(row, page_num, balance_column_x, log_func)
                    if transaction:
                        transactions.append(transaction)
                        # Find balance position for this transaction
                        self._find_balance_position_for_transaction(row, page_num, balance_column_x, log_func)
                
                # Check if this is the beginning balance row
                elif "BEGINNING BALANCE" in " ".join([b["text"] for b in row]):
                    self._find_beginning_balance_position(row, page_num, log_func)
            
        except Exception as e:
            log_func(f"⚠️ Error in extraction: {e}")
            import traceback
            traceback.print_exc()
        
        return transactions
    
    def _find_balance_column_position(self, all_blocks: List[Dict], log_func: Callable) -> float:
        """Find the x-position of the balance column"""
        # Look for "STATEMENT BALANCE" header or similar
        balance_keywords = ["STATEMENT BALANCE", "BAKI PENYATA", "BALANCE"]
        
        for block in all_blocks:
            text_upper = block["text"].upper()
            for keyword in balance_keywords:
                if keyword in text_upper:
                    log_func(f"📍 Found balance column header at x-position: {block['x']}")
                    # Store both left and right boundaries of the balance column
                    self.balance_column_x = block["x"]
                    self.balance_column_right = block["bbox"][2]  # Right edge of the header
                    return block["x"]
        
        # Fallback: find the rightmost column with numeric values
        numeric_blocks = []
        for block in all_blocks:
            text = block["text"].replace(',', '').replace('RM', '').strip()
            if re.match(r'^\d+\.?\d*$', text):
                numeric_blocks.append(block)
        
        if numeric_blocks:
            rightmost_x = max(block["x"] for block in numeric_blocks)
            rightmost_block = max(numeric_blocks, key=lambda b: b["x"])
            self.balance_column_x = rightmost_x
            self.balance_column_right = rightmost_block["bbox"][2]
            log_func(f"📍 Using rightmost numeric column at x-position: {rightmost_x}")
            return rightmost_x
        
        return 0
    
    def _group_blocks_into_rows(self, all_blocks: List[Dict]) -> List[List[Dict]]:
        """Group text blocks into rows based on y-coordinate"""
        rows = []
        current_row = []
        last_y = None
        y_tolerance = 5  # pixels
        
        for block in all_blocks:
            if last_y is None or abs(block["y"] - last_y) <= y_tolerance:
                current_row.append(block)
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [block]
            last_y = block["y"]
        
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _parse_transaction_row_simple(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable) -> Optional[Dict]:
        """Simple transaction parsing focusing on date, description, and amount"""
        try:
            date = row_blocks[0]["text"]
            
            # Find amount (look for patterns like "66.49-" or "100.00+")
            amount = Decimal('0.00')
            for block in row_blocks[1:]:
                text = block["text"].strip()
                amount_match = re.match(r'^([\d,]+\.?\d*)([+-])$', text)
                if amount_match:
                    amount_str = amount_match.group(1)
                    sign = amount_match.group(2)
                    amount = self._clean_amount(amount_str)
                    if sign == '-':
                        amount = -amount
                    break
            
            # Extract description (non-numeric, non-sign text)
            desc_parts = []
            for block in row_blocks[1:]:
                text = block["text"].strip()
                if not re.match(r'^[\d,\.+-]+$', text) and text not in ['+', '-', 'RM', '']:
                    desc_parts.append(text)
            
            description = " ".join(desc_parts).strip()
            
            return {
                'date': date,
                'description': description,
                'amount': amount,
                'original_balance': Decimal('0.00'),
                'new_balance': Decimal('0.00'),
                'page_num': page_num
            }
            
        except Exception as e:
            log_func(f"⚠️ Error parsing transaction: {e}")
            return None
    
    def _find_beginning_balance_position(self, row_blocks: List[Dict], page_num: int, log_func: Callable):
        """Find the beginning balance position"""
        for block in row_blocks:
            text = block["text"].strip()
            if re.match(r'^[\d,]+\.?\d*$', text):
                self.balance_replacements.append({
                    'type': 'beginning_balance',
                    'original_value': self._clean_amount(text),
                    'bbox': block["bbox"],
                    'font_size': block.get("font_size", 12),
                    'font': block.get("font", ""),
                    'page_num': page_num,
                    'y_position': block["bbox"][1]
                })
                log_func(f"📍 Found beginning balance position: {text}")
                break
    
    def _find_balance_position_for_transaction(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable):
        """Find where the balance should be placed for this transaction row"""
        # Look for existing balance values in this row to get exact column position
        balance_block = None
        
        # First, try to find any existing balance value in the rightmost area
        rightmost_numeric = None
        for block in row_blocks:
            text = block["text"].replace(',', '').replace('RM', '').strip()
            # Check if it's numeric and in the rightmost area
            if re.match(r'^\d+\.?\d*$', text):
                if not rightmost_numeric or block["x"] > rightmost_numeric["x"]:
                    rightmost_numeric = block
        
        # If we found a numeric value in the rightmost position, use its location
        if rightmost_numeric:
            balance_block = rightmost_numeric
        else:
            # Otherwise, create a position based on the balance column header
            # Find the rightmost block in this row
            rightmost_block = max(row_blocks, key=lambda b: b["x"])
            
            # Create a position for the balance using the stored column position
            balance_bbox = [
                self.balance_column_x if self.balance_column_x else rightmost_block["x"] + 50,
                rightmost_block["y"],
                self.balance_column_right if self.balance_column_right else (self.balance_column_x + 80) if self.balance_column_x else rightmost_block["x"] + 130,
                rightmost_block["y"] + rightmost_block.get("font_size", 12) + 2
            ]
            
            balance_block = {
                "bbox": balance_bbox,
                "font_size": rightmost_block.get("font_size", 12),
                "font": rightmost_block.get("font", "")
            }
        
        self.balance_replacements.append({
            'type': 'transaction_balance',
            'original_value': Decimal('0.00'),
            'bbox': balance_block["bbox"],
            'font_size': balance_block.get("font_size", 12),
            'font': balance_block.get("font", ""),
            'page_num': page_num,
            'y_position': balance_block["bbox"][1]
        })
    
    def _is_date(self, text: str) -> bool:
        """Check if text looks like a date (DD/MM/YY format)"""
        return bool(re.match(r'^\d{2}/\d{2}/\d{2}$', text.strip()))
    
    def _clean_amount(self, amount_str: str) -> Decimal:
        """Clean and convert amount string to Decimal"""
        if not amount_str or amount_str.lower() in ['none', 'null', '']:
            return Decimal('0.00')
        
        # Remove currency symbols, commas, and spaces
        cleaned = re.sub(r'[RM$,\s]', '', amount_str.strip())
        
        try:
            return Decimal(cleaned).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except:
            return Decimal('0.00')
    
    def recalculate_balances(self, transactions: List[Dict], beginning_balance: Decimal, log_func: Callable = print) -> List[Dict]:
        """Recalculate statement balances based on beginning balance and transaction amounts"""
        
        log_func(f"💰 Beginning balance: RM {beginning_balance:,.2f}")
        log_func("=" * 90)
        log_func(f"{'#':<3} {'Date':<10} {'Description':<40} {'Amount':<12} {'New Balance':<15}")
        log_func("=" * 90)
        
        running_balance = beginning_balance
        
        for i, transaction in enumerate(transactions):
            # Calculate new balance: previous balance + transaction amount
            running_balance += transaction['amount']
            transaction['new_balance'] = running_balance
            
            # Format output
            desc_short = transaction['description'][:37] + "..." if len(transaction['description']) > 37 else transaction['description']
            amount_str = f"RM {transaction['amount']:>8,.2f}"
            balance_str = f"RM {transaction['new_balance']:>10,.2f}"
            
            log_func(f"{i+1:<3} {transaction['date']:<10} {desc_short:<40} {amount_str:<12} {balance_str:<15}")
        
        log_func("=" * 90)
        log_func(f"💰 Final ending balance: RM {running_balance:,.2f}")
        return transactions
    
    def generate_updated_pdf(self, transactions: List[Dict], output_path: str, log_func: Callable = print):
        """Generate updated PDF with new balance values"""
        try:
            log_func("📄 Generating updated PDF...")
            
            # Open the original PDF
            original_pdf = fitz.open(self.original_pdf_path)
            new_pdf = fitz.open()
            
            # Extract and preserve metadata from original PDF
            original_metadata = original_pdf.metadata
            log_func("📋 Extracting metadata from original PDF...")
            
            # Log the original metadata
            for key, value in original_metadata.items():
                if value:
                    log_func(f"   {key}: {value}")
            
            balance_index = 0
            
            # Copy each page and replace balance values
            for page_num in range(len(original_pdf)):
                original_page = original_pdf.load_page(page_num)
                new_page = new_pdf.new_page(width=original_page.rect.width, height=original_page.rect.height)
                
                # Copy the original page content
                new_page.show_pdf_page(new_page.rect, original_pdf, page_num)
                
                # Get balance replacements for this page
                page_replacements = [r for r in self.balance_replacements if r['page_num'] == page_num + 1]
                
                log_func(f"📄 Processing page {page_num + 1} with {len(page_replacements)} balance positions")
                
                # Apply balance replacements
                for replacement in page_replacements:
                    self._apply_balance_replacement(new_page, replacement, log_func)
            
            # Save the updated PDF
            new_pdf.save(output_path)
            new_pdf.close()
            original_pdf.close()
            
            log_func(f"✅ Updated PDF saved to: {output_path}")
            
        except Exception as e:
            log_func(f"❌ Error generating updated PDF: {e}")
            import traceback
            traceback.print_exc()
    
    def _apply_balance_replacement(self, page, replacement: Dict, log_func: Callable):
        """Apply a balance replacement to a page"""
        try:
            # Clear the original value
            cover_rect = fitz.Rect(
                replacement['bbox'][0] - 3,
                replacement['bbox'][1] - 2,
                replacement['bbox'][2] + 3,
                replacement['bbox'][3] + 2
            )
            page.draw_rect(cover_rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
            
            # Calculate the right-aligned position
            font_size = replacement.get('font_size', 12)
            char_width = font_size * 0.5
            text_width = len(str(replacement['new_value'])) * char_width
            
            # Position the new text to end where the original text ended
            text_x = replacement['bbox'][2] - text_width
            text_y = replacement['bbox'][1] + font_size * 0.8
            
            # Insert the new text
            text_point = fitz.Point(text_x, text_y)
            page.insert_text(
                text_point,
                str(replacement['new_value']),
                fontsize=font_size,
                color=(0, 0, 0),
                fontname="helv"
            )
            
        except Exception as e:
            log_func(f"⚠️ Error applying balance replacement: {e}")
    
    def process_statement_gui(self, input_pdf: str, output_pdf: str, beginning_balance: float, log_func: Callable):
        """Main processing function for GUI"""
        
        log_func("🚀 Starting PDF Statement Processing...")
        log_func("=" * 50)
        
        # Validate input file
        if not os.path.exists(input_pdf):
            log_func(f"❌ Input file not found: {input_pdf}")
            return False
        
        # Convert beginning balance to Decimal
        self.beginning_balance = Decimal(str(beginning_balance))
        
        # Extract transactions from PDF
        self.transactions = self.extract_transactions_from_pdf(input_pdf, log_func)
        
        if not self.transactions:
            log_func("❌ No transactions found in PDF.")
            return False
        
        # Recalculate balances
        self.transactions = self.recalculate_balances(self.transactions, self.beginning_balance, log_func)
        
        # Generate updated PDF
        self.generate_updated_pdf(self.transactions, output_pdf, log_func)
        
        log_func("=" * 50)
        log_func("✅ Processing completed successfully!")
        return True