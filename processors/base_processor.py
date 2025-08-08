#!/usr/bin/env python3
"""
Base Processor for PDF Statement Processing
Abstract base class that all bank-specific processors must inherit from
"""

import os
import re
from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Tuple, Optional, Callable

# Try to import required PDF libraries
try:
    import fitz  # PyMuPDF for better text extraction
except ImportError as e:
    print("❌ Missing required libraries. Please install them using:")
    print("pip install PyMuPDF")
    raise e


class BaseProcessor(ABC):
    """Abstract base class for all bank-specific processors"""
    
    def __init__(self):
        self.transactions = []
        self.original_pdf_path = None
        self.output_pdf_path = None
        self.beginning_balance = Decimal('0.00')
        self.balance_replacements = []
        self.balance_column_x = None
        self.balance_column_right = None
        
    @abstractmethod
    def extract_transactions_from_pdf(self, pdf_path: str, log_func: Callable = print) -> List[Dict]:
        """Extract transaction data from PDF - must be implemented by each bank"""
        pass
    
    @abstractmethod
    def _extract_transactions_and_balances(self, text_dict: dict, page_num: int, log_func: Callable) -> List[Dict]:
        """Extract transactions and find balance positions - must be implemented by each bank"""
        pass
    
    @abstractmethod
    def _find_balance_column_position(self, all_blocks: List[Dict], log_func: Callable) -> float:
        """Find the x-position of the balance column - must be implemented by each bank"""
        pass
    
    @abstractmethod
    def _parse_transaction_row(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable) -> Optional[Dict]:
        """Parse a transaction row - must be implemented by each bank"""
        pass
    
    def _group_blocks_into_rows(self, all_blocks: List[Dict]) -> List[List[Dict]]:
        """Group text blocks into rows based on y-coordinate - common implementation"""
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
    
    def _is_date(self, text: str) -> bool:
        """Check if text looks like a date - common implementation"""
        # Common date patterns
        date_patterns = [
            r'^\d{2}/\d{2}/\d{2}$',  # DD/MM/YY
            r'^\d{2}/\d{2}/\d{4}$',  # DD/MM/YYYY
            r'^\d{2}-\d{2}-\d{2}$',  # DD-MM-YY
            r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, text.strip()):
                return True
        return False
    
    def _format_amount(self, value: Decimal) -> str:
        """Format Decimal as string with thousands and 2 decimals"""
        return f"{value:,.2f}"

    def _populate_new_values(self, transactions: List[Dict], log_func: Callable = print):
        """Populate 'new_value' for all balance replacements"""
        try:
            # 1) Beginning balance(s)
            for rep in self.balance_replacements:
                if rep.get('type') == 'beginning_balance':
                    rep['new_value'] = self._format_amount(self.beginning_balance)

            # 2) Map transactions to transaction_balance replacements per page
            from collections import defaultdict
            tx_by_page = defaultdict(list)
            for tx in transactions:
                tx_by_page[tx['page_num']].append(tx)

            reps_by_page = defaultdict(list)
            for rep in self.balance_replacements:
                if rep.get('type') == 'transaction_balance':
                    reps_by_page[rep['page_num']].append(rep)

            # 3) Assign by order (top-to-bottom) on each page
            for page_num, txs in tx_by_page.items():
                reps = sorted(reps_by_page.get(page_num, []), key=lambda r: r.get('y_position', 0))
                n = min(len(txs), len(reps))
                for i in range(n):
                    reps[i]['new_value'] = self._format_amount(txs[i]['new_balance'])

                if len(reps) != len(txs):
                    log_func(f"⚠️ Mismatch on page {page_num}: {len(txs)} transactions vs {len(reps)} balance positions")

            # summary fields (ending balance / totals)
            ending_balance = transactions[-1]["new_balance"] if transactions else self.beginning_balance
            total_credit = sum(t["amount"] for t in transactions if t["amount"] > 0)
            total_debit = sum(-t["amount"] for t in transactions if t["amount"] < 0)

            for rep in self.balance_replacements:
                t = rep.get("type")
                if t == "ending_balance":
                    rep["new_value"] = self._format_amount(ending_balance)
                elif t == "total_credit":
                    rep["new_value"] = self._format_amount(total_credit)
                elif t == "total_debit":
                    rep["new_value"] = self._format_amount(total_debit)

        except Exception as e:
            log_func(f"⚠️ Error populating new values: {e}")
    
    def _clean_amount(self, amount_str: str) -> Decimal:
        """Clean and convert amount string to Decimal - common implementation"""
        if not amount_str or amount_str.lower() in ['none', 'null', '']:
            return Decimal('0.00')
        
        # Remove currency symbols, commas, and spaces
        cleaned = re.sub(r'[RM$,\s]', '', amount_str.strip())
        
        try:
            return Decimal(cleaned).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except:
            return Decimal('0.00')
    
    def recalculate_balances(self, transactions: List[Dict], beginning_balance: Decimal, log_func: Callable = print) -> List[Dict]:
        """Recalculate statement balances based on beginning balance and transaction amounts - common implementation"""
        
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
        """Generate updated PDF with new balance values - common implementation"""
        try:
            log_func("📄 Generating updated PDF...")
            # Ensure replacements have values
            self._populate_new_values(transactions, log_func)
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                log_func(f"📁 Created output directory: {output_dir}")
            
            # Open the original PDF
            original_pdf = fitz.open(self.original_pdf_path)
            new_pdf = fitz.open()
            
            # Copy metadata from the original
            log_func("📋 Extracting metadata from original PDF...")
            for k, v in (original_pdf.metadata or {}).items():
                if v:
                    log_func(f"   {k}: {v}")
            self._copy_metadata(original_pdf, new_pdf, log_func)

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
        """Apply a balance replacement to a page - common implementation"""
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
        """Main processing function for GUI - common implementation"""
        
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

    def _copy_metadata(self, src_doc, dst_doc, log_func: Callable):
        """Copy standard and XMP metadata from source to destination PDF."""
        try:
            meta = src_doc.metadata or {}
            if meta:
                dst_doc.set_metadata(meta)
                log_func("🧾 Copied document Info metadata")

            # Copy XMP metadata if present
            try:
                xmp = src_doc.xmp_metadata  # bytes or None
            except Exception:
                xmp = None
            if xmp:
                try:
                    dst_doc.set_xmp_metadata(xmp)
                    log_func("🧾 Copied XMP metadata")
                except Exception:
                    # some files may have incompatible XMP; safe to ignore
                    pass
        except Exception as e:
            log_func(f"⚠️ Could not copy metadata: {e}") 