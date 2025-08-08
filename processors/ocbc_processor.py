#!/usr/bin/env python3
"""
OCBC Bank PDF Statement Processor
Parses OCBC statement summary and transactions, and preserves layout by replacing only numbers.
"""

import re
import os
from decimal import Decimal
from typing import List, Dict, Optional, Callable
import fitz

from .base_processor import BaseProcessor


class OCBCProcessor(BaseProcessor):
    """OCBC Bank-specific PDF statement processor"""

    def __init__(self):
        super().__init__()
        self.bank_name = "OCBC Bank"

    # OCBC dates like "02May"
    def _is_date(self, text: str) -> bool:
        text = text.strip()
        return bool(re.match(r'^\d{2}[A-Za-z]{3}$', text)) or \
               bool(re.match(r'^\d{2}\s*[A-Za-z]{3}$', text))

    def extract_transactions_from_pdf(self, pdf_path: str, log_func: Callable = print) -> List[Dict]:
        """Extract transaction data from OCBC PDF format"""
        self.original_pdf_path = pdf_path
        self.balance_replacements = []

        transactions: List[Dict] = []

        try:
            pdf_document = fitz.open(pdf_path)

            log_func(f"📄 Processing {self.bank_name} PDF: {os.path.basename(pdf_path)}")
            log_func(f"📊 Total pages: {len(pdf_document)}")

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text_dict = page.get_text("dict")

                # Parse summary on page 1 (where it exists)
                if page_num == 0:
                    self._find_summary_positions_ocbc(text_dict, page_num + 1, log_func)

                # Parse rows and balance cells
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

    def _clean_num_text(self, text: str) -> Optional[Decimal]:
        t = text.strip().replace(',', '')
        if t in ['', '-', 'RM', 'RM.']:
            return None
        if re.match(r'^-?\d+(\.\d+)?$', t):
            try:
                val = Decimal(t)
                return val
            except:
                return None
        return None

    def _norm(self, s: str) -> str:
        return re.sub(r'[^A-Z]', '', s.upper())  # keep only A–Z

    BAL_BF_ALIASES = ["BALANCEBF", "BALANCEBIF", "BAKIBAWAKEHADAPAN", "BALANCEBFI", "BALANCEB F"]  # tolerant

    LABEL_ALIASES = {
        "OPENINGBALANCE": ["OPENINGBALANCE", "BAKIPEMBUKAAN"],
        "TOTALDEBITS":    ["TOTALDEBITS", "JUMLAHDEBIT"],
        "TOTALCREDITS":   ["TOTALCREDITS", "JUMLAHKREDIT"],
        "CLOSINGBALANCE": ["CLOSINGBALANCE", "BAKIPENUTUPAN"],
    }

    def _find_summary_positions_ocbc(self, text_dict: dict, page_num: int, log_func: Callable):
        """Locate summary values on the first page: OPENING, TOTAL DEBITS, TOTAL CREDITS, CLOSING."""
        try:
            all_blocks = []
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            tx = span["text"].strip()
                            if tx:
                                bbox = span["bbox"]
                                all_blocks.append({
                                    "text": tx,
                                    "x": bbox[0], "y": bbox[1], "bbox": bbox,
                                    "font_size": span.get("size", 12),
                                    "font": span.get("font", "")
                                })

            label_map = {
                "OPENING BALANCE": "beginning_balance",
                "TOTAL DEBITS": "total_debit",
                "TOTAL CREDITS": "total_credit",
                "CLOSING BALANCE": "ending_balance",
            }
            y_tol = 6

            for blk in all_blocks:
                key_norm = self._norm(blk["text"])
                for canon, aliases in self.LABEL_ALIASES.items():
                    if key_norm in aliases:
                        label_name = {
                            "OPENINGBALANCE": "beginning_balance",
                            "TOTALDEBITS": "total_debit",
                            "TOTALCREDITS": "total_credit",
                            "CLOSINGBALANCE": "ending_balance",
                        }[canon]
                        # same-row numeric to the right
                        candidates = []
                        for b in all_blocks:
                            if b["x"] > blk["x"] and abs(b["y"] - blk["y"]) <= y_tol:
                                num = self._clean_num_text(b["text"].replace("+", ""))
                                if num is not None:
                                    candidates.append((b, num))

                        if candidates:
                            # rightmost numeric on that row
                            target, num_val = max(candidates, key=lambda p: p[0]["x"])
                            self.balance_replacements.append({
                                "type": label_name,
                                "original_value": num_val,
                                "bbox": target["bbox"],
                                "font_size": target.get("font_size", 12),
                                "font": target.get("font", ""),
                                "page_num": page_num,
                                "y_position": target["bbox"][1],
                            })
                            log_func(f"📍 Found {label_name.replace('_', ' ')} position: {target['text']}")

        except Exception as e:
            log_func(f"⚠️ Error finding OCBC summary positions: {e}")

    def _extract_transactions_and_balances(self, text_dict: dict, page_num: int, log_func: Callable) -> List[Dict]:
        """Extract transactions (DATE, DESCRIPTION, DEBIT, CREDIT, BALANCE) and record balance bboxes per row."""
        transactions: List[Dict] = []

        try:
            # Flatten spans
            all_blocks = []
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            tx = span["text"].strip()
                            if tx:
                                bbox = span["bbox"]
                                all_blocks.append({
                                    "text": tx,
                                    "x": bbox[0], "y": bbox[1], "bbox": bbox,
                                    "font_size": span.get("size", 12),
                                    "font": span.get("font", "")
                                })

            # Sort by layout
            all_blocks.sort(key=lambda b: (b["y"], b["x"]))

            # Locate column headers
            headers = self._locate_headers(all_blocks)
            if not headers:
                # No table here
                return transactions

            date_x = headers.get("DATE", None)  # not strictly needed
            debit_x = headers["DEBIT"]
            credit_x = headers["CREDIT"]
            balance_x = headers["BALANCE"]

            # Group into rows
            rows = self._group_blocks_into_rows(all_blocks)
            log_func(f"📋 Found {len(rows)} text rows on page {page_num}")

            # Parse rows after header line
            for row in rows:
                row.sort(key=lambda b: b["x"])
                if not row:
                    continue

                # Special-case: Balance b/f row (may not have a clear date/amount spans)
                row_text_norm = self._norm(" ".join(b["text"] for b in row))
                is_balance_bf = any(alias in row_text_norm for alias in self.BAL_BF_ALIASES)

                if is_balance_bf:
                    # find balance cell on same row nearest BALANCE column
                    balance_blk = None
                    for b in row:
                        if self._clean_num_text(b["text"]) is not None:
                            balance_blk = b if ('BALANCE' not in locals() or abs(b["x"] - balance_x) <= abs(balance_blk["x"] - balance_x)) else balance_blk

                    # Synthesize bbox if needed
                    if balance_blk:
                        target_bbox = balance_blk["bbox"]
                        font_size = balance_blk.get("font_size", 12)
                        font_name = balance_blk.get("font", "")
                    else:
                        rightmost_block = max(row, key=lambda b: b["x"])
                        font_size = rightmost_block.get("font_size", 12)
                        font_name = rightmost_block.get("font", "")
                        target_bbox = [balance_x - 40, rightmost_block["y"], balance_x + 40, rightmost_block["y"] + font_size + 2]

                    # Append a pseudo-transaction with amount 0 for pairing
                    transactions.append({
                        "date": next((b["text"] for b in row if self._is_date(b["text"])), ""),  # optional
                        "description": "Balance b/f",
                        "amount": Decimal('0.00'),
                        "original_balance": Decimal('0.00'),
                        "new_balance": Decimal('0.00'),
                        "page_num": page_num
                    })

                    self.balance_replacements.append({
                        "type": "transaction_balance",
                        "original_value": Decimal('0.00'),
                        "bbox": target_bbox,
                        "font_size": font_size,
                        "font": font_name,
                        "page_num": page_num,
                        "y_position": target_bbox[1]
                    })
                    continue  # go to next row

                # Find best candidate for date at far-left
                left_text = row[0]["text"]
                if not self._is_date(left_text):
                    # sometimes date is not the first span due to wrapping; try any span in row
                    date_blk = next((b for b in row if self._is_date(b["text"])), None)
                else:
                    date_blk = row[0]

                if not date_blk:
                    continue

                # Identify amounts by nearest header x
                debit_val, credit_val, balance_blk = None, None, None
                for b in row:
                    num = self._clean_num_text(b["text"])
                    if num is None:
                        continue
                    # choose closest header x among debit/credit/balance
                    x = b["x"]
                    nearest = min(
                        [("DEBIT", abs(x - debit_x)), ("CREDIT", abs(x - credit_x)), ("BALANCE", abs(x - balance_x))],
                        key=lambda t: t[1]
                    )[0]

                    if nearest == "DEBIT":
                        debit_val = num
                    elif nearest == "CREDIT":
                        credit_val = num
                    elif nearest == "BALANCE":
                        balance_blk = b

                # Skip if this is just a header-like or summary line
                if debit_val is None and credit_val is None and balance_blk is None:
                    continue

                # Build amount (debit negative, credit positive)
                amount = Decimal('0.00')
                if debit_val is not None and debit_val != Decimal('0.00'):
                    amount = -debit_val
                elif credit_val is not None and credit_val != Decimal('0.00'):
                    amount = credit_val

                # Description = everything in the row that is not numeric under the 3 numeric columns and not the date
                desc_parts = []
                for b in row:
                    if b is date_blk:
                        continue
                    if self._clean_num_text(b["text"]) is not None:
                        continue
                    desc_parts.append(b["text"])
                description = " ".join(desc_parts).strip()

                transactions.append({
                    "date": date_blk["text"],
                    "description": description,
                    "amount": amount,
                    "original_balance": Decimal('0.00'),
                    "new_balance": Decimal('0.00'),
                    "page_num": page_num
                })

                # Record balance cell for replacement
                # If there's an existing balance number on this row, use its bbox; if not, create a bbox zone aligned to balance_x
                if balance_blk:
                    target_bbox = balance_blk["bbox"]
                    font_size = balance_blk.get("font_size", 12)
                else:
                    rightmost_block = max(row, key=lambda b: b["x"])
                    font_size = rightmost_block.get("font_size", 12)
                    target_bbox = [
                        balance_x - 40, rightmost_block["y"],
                        balance_x + 40, rightmost_block["y"] + font_size + 2
                    ]

                self.balance_replacements.append({
                    "type": "transaction_balance",
                    "original_value": Decimal('0.00'),
                    "bbox": target_bbox,
                    "font_size": font_size,
                    "font": balance_blk.get("font", "") if balance_blk else rightmost_block.get("font", ""),
                    "page_num": page_num,
                    "y_position": target_bbox[1]
                })

        except Exception as e:
            log_func(f"⚠️ Error in OCBC extraction: {e}")

        return transactions

    def _locate_headers(self, blocks: List[Dict]) -> Optional[Dict[str, float]]:
        """Find approximate x positions of DEBIT, CREDIT, BALANCE headers (and DATE if present)."""
        header_x = {}
        for b in blocks:
            t = b["text"].upper().strip()
            if t in ("DEBIT", "DEBIT ", " DEBIT"):
                header_x["DEBIT"] = b["x"]
            elif t in ("CREDIT", "KREDIT", "CREDIT ", " KREDIT"):
                header_x["CREDIT"] = b["x"]
            elif t in ("BALANCE", "BAKI", " BALANCE", " BAKI"):
                header_x["BALANCE"] = b["x"]
            elif t in ("DATE", "TARIKH", " DATE", " TARIKH"):
                header_x["DATE"] = b["x"]

        # Require the three numeric columns
        if all(k in header_x for k in ("DEBIT", "CREDIT", "BALANCE")):
            return header_x
        return None

    def _find_balance_column_position(self, all_blocks: List[Dict], log_func: Callable) -> float:
        """Satisfy abstract API: infer BALANCE column x from headers; fallback to 0."""
        headers = self._locate_headers(all_blocks)
        if headers and "BALANCE" in headers:
            self.balance_column_x = headers["BALANCE"]
            # pick a reasonable right edge in case a synthetic bbox is needed
            self.balance_column_right = self.balance_column_x + 80
            return self.balance_column_x
        return 0.0

    def _parse_transaction_row(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable) -> Optional[Dict]:
        """Not used for OCBC (we parse rows in _extract_transactions_and_balances)."""
        return None