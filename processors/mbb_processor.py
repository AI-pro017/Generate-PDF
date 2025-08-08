#!/usr/bin/env python3
"""
Maybank (MBB) PDF Statement Processor
Handles Maybank-specific PDF format and structure
"""

import re
import os
from decimal import Decimal
from typing import List, Dict, Optional, Callable
import fitz

from .base_processor import BaseProcessor


class MBBProcessor(BaseProcessor):
    """Maybank-specific PDF statement processor"""
    
    def __init__(self):
        super().__init__()
        self.bank_name = "Maybank (MBB)"
    
    def extract_transactions_from_pdf(self, pdf_path: str, log_func: Callable = print) -> List[Dict]:
        """Extract transaction data from Maybank PDF format"""
        self.original_pdf_path = pdf_path
        self.balance_replacements = []
        
        transactions = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            log_func(f"📄 Processing {self.bank_name} PDF: {os.path.basename(pdf_path)}")
            log_func(f"📊 Total pages: {len(pdf_document)}")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text_dict = page.get_text("dict")
                
                log_func(f"📃 Page {page_num + 1} - Looking for transactions and balance values...")
                
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
                                bbox = span["bbox"]
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
                row.sort(key=lambda b: b["x"])
                
                if not row:
                    continue
                
                # Check if this is a transaction row (starts with date)
                if self._is_date(row[0]["text"]):
                    transaction = self._parse_transaction_row(row, page_num, balance_column_x, log_func)
                    if transaction:
                        transactions.append(transaction)
                        # Find balance position for this transaction
                        self._find_balance_position_for_transaction(row, page_num, balance_column_x, log_func)
                
                # Check if this is the beginning balance row
                elif "BEGINNING BALANCE" in " ".join([b["text"] for b in row]):
                    self._find_beginning_balance_position(row, page_num, log_func)
            
            self._find_summary_positions(all_blocks, page_num, log_func)
            
        except Exception as e:
            log_func(f"⚠️ Error in extraction: {e}")
            import traceback
            traceback.print_exc()
        
        return transactions

    # Override erase box just for MBB so we don't wipe the right table rule
    def _erase_rect(self, page, bbox):
        """Return a tighter erase rectangle for MBB numbers.

        - Keep height unchanged (do not touch horizontal rules)
        - Shift the right edge slightly left so the vertical border line remains
        - Small left padding to fully cover any light background behind digits
        """
        import fitz
        x0, y0, x1, y1 = bbox
        left_pad = 6.0
        right_inset = 2.0

        rx0 = x0 - left_pad
        rx1 = x1 - right_inset
        if rx1 <= rx0 + 1.0:
            rx1 = rx0 + 1.0

        r = fitz.Rect(rx0, y0, rx1, y1)

        pr = page.rect
        r.x0 = max(pr.x0, r.x0)
        r.y0 = max(pr.y0, r.y0)
        r.x1 = min(pr.x1, r.x1)
        r.y1 = min(pr.y1, r.y1)
        return r
    
    def _find_balance_column_position(self, all_blocks: List[Dict], log_func: Callable) -> float:
        """Find the x-position of the balance column for Maybank"""
        # Look for "STATEMENT BALANCE" header or similar
        balance_keywords = ["STATEMENT BALANCE", "BAKI PENYATA", "BALANCE"]
        
        for block in all_blocks:
            text_upper = block["text"].upper()
            for keyword in balance_keywords:
                if keyword in text_upper:
                    log_func(f"📍 Found balance column header at x-position: {block['x']}")
                    self.balance_column_x = block["x"]
                    self.balance_column_right = block["bbox"][2]
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
    
    def _parse_transaction_row(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable) -> Optional[Dict]:
        """Parse a transaction row for Maybank format"""
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
    
    def _find_balance_position_for_transaction(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable):
        """Find where the balance should be placed for this transaction row"""
        balance_block = None
        
        # First, try to find any existing balance value in the rightmost area
        rightmost_numeric = None
        for block in row_blocks:
            text = block["text"].replace(',', '').replace('RM', '').strip()
            if re.match(r'^\d+\.?\d*$', text):
                if not rightmost_numeric or block["x"] > rightmost_numeric["x"]:
                    rightmost_numeric = block
        
        if rightmost_numeric:
            balance_block = rightmost_numeric
        else:
            rightmost_block = max(row_blocks, key=lambda b: b["x"])
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
    
    def _find_beginning_balance_position(self, row_blocks: List[Dict], page_num: int, log_func: Callable):
        """Find the beginning balance position for Maybank"""
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

    def _find_summary_positions(self, all_blocks: List[Dict], page_num: int, log_func: Callable):
        """Find ENDING BALANCE / TOTAL CREDIT / TOTAL DEBIT numeric positions on the page"""
        label_map = {
            "ENDING BALANCE": "ending_balance",
            "TOTAL CREDIT": "total_credit",
            "TOTAL DEBIT": "total_debit",
        }
        y_tol = 4  # pixels

        for blk in all_blocks:
            key = blk["text"].upper().replace(":", "").strip()
            if key in label_map:
                # find numeric candidates on the same row to the right
                candidates = []
                for b in all_blocks:
                    if b["x"] > blk["x"] and abs(b["y"] - blk["y"]) <= y_tol:
                        t = b["text"].replace(",", "").strip()
                        if re.match(r"^-?[\d]+\.?\d*$", t):
                            candidates.append(b)
                if candidates:
                    target = max(candidates, key=lambda b: b["x"])  # rightmost number
                    txt = target["text"].strip()
                    val = self._clean_amount(txt.replace("+", "").replace("RM", ""))
                    if txt.startswith("-"):
                        val = -val

                    self.balance_replacements.append({
                        "type": label_map[key],
                        "original_value": val,
                        "bbox": target["bbox"],
                        "font_size": target.get("font_size", 12),
                        "font": target.get("font", ""),
                        "page_num": page_num,
                        "y_position": target["bbox"][1],
                    })
                    log_func(f"📍 Found {label_map[key].replace('_',' ')} position: {txt}")