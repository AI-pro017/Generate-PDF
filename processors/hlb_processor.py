#!/usr/bin/env python3
"""
Hong Leong Bank (HLB) PDF Statement Processor
Parses HLB statement table (Deposit, Withdrawal, Balance) and preserves layout by replacing only numbers.
"""

import re
import os
from decimal import Decimal
from typing import List, Dict, Optional, Callable
import fitz

from .base_processor import BaseProcessor


class HLBProcessor(BaseProcessor):
    """Hong Leong Bank-specific PDF statement processor"""

    def __init__(self):
        super().__init__()
        self.bank_name = "Hong Leong Bank (HLB)"
        # carry pending amount across pages (rows without visible balance)
        self._carry = Decimal('0.00')
        self._hlb_balance_right_edge = None
    def _is_amount_text(self, text: str) -> bool:
        t = text.strip()
        # must contain decimals; reject integers and long IDs
        if "." not in t:
            return False
        # allow comma thousands, exactly 2 decimals typical for statements
        return bool(re.match(r'^-?\d{1,3}(,\d{3})*(\.\d{2})$|-?\d+\.\d{2}$', t))
    # HLB dates like "02-06-2025"
    def _is_date(self, text: str) -> bool:
        t = text.strip()
        return bool(re.match(r'^\d{2}-\d{2}-\d{4}$', t))

    def _norm(self, s: str) -> str:
        return re.sub(r'[^A-Z]', '', s.upper())

    # Aliases
    BALANCE_BF_ALIASES = [
        "BALANCEFROMPREVIOUSSTATEMENT",
        "BALANCEBF",
        "BAKIBAWAPREVIOUSSTATEMENT",
        "BAKIBAWA",  # sometimes truncated
    ]

    def extract_transactions_from_pdf(self, pdf_path: str, log_func: Callable = print) -> List[Dict]:
        """Extract transaction data from HLB PDF format"""
        self.original_pdf_path = pdf_path
        self.balance_replacements = []
        self._carry = Decimal('0.00')

        transactions: List[Dict] = []

        try:
            pdf_document = fitz.open(pdf_path)

            log_func(f"📄 Processing {self.bank_name} PDF: {os.path.basename(pdf_path)}")
            log_func(f"📊 Total pages: {len(pdf_document)}")

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text_dict = page.get_text("dict")

                # Parse rows and balance cells
                page_transactions = self._extract_transactions_and_balances(text_dict, page_num + 1, log_func)
                transactions.extend(page_transactions)

                # Summary label (closing balance) usually present at bottom of pages
                self._find_summary_positions_hlb(text_dict, page_num + 1, log_func)

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
        if not self._is_amount_text(text):
            return None
        t = text.replace(',', '').strip()
        try:
            return Decimal(t)
        except Exception:
            return None

    def _locate_headers(self, blocks: List[Dict]) -> Optional[Dict[str, float]]:
        """Find approximate x positions for DEPOSIT, WITHDRAWAL, BALANCE headers, and table top y."""
        header_x, header_y = {}, {}
        for b in blocks:
            t = b["text"].upper().strip()
            if t in ("DEPOSIT", "SIMPANAN"):
                header_x["DEPOSIT"] = b["x"]; header_y["DEPOSIT"] = b["y"]
            elif t in ("WITHDRAWAL", "PENGELUARAN"):
                header_x["WITHDRAWAL"] = b["x"]; header_y["WITHDRAWAL"] = b["y"]
            elif t in ("BALANCE", "BAKI"):
                header_x["BALANCE"] = b["x"]; header_y["BALANCE"] = b["y"]
            elif t in ("DATE", "TARIKH"):
                header_x["DATE"] = b["x"]; header_y["DATE"] = b["y"]

        if all(k in header_x for k in ("DEPOSIT", "WITHDRAWAL", "BALANCE")):
            header_x["_TABLE_TOP_Y"] = min(header_y.values()) if header_y else None
            return header_x
        return None

    def _extract_transactions_and_balances(self, text_dict: dict, page_num: int, log_func: Callable) -> List[Dict]:
        transactions: List[Dict] = []
        try:
            blocks = []
            for b in text_dict.get("blocks", []):
                for ln in b.get("lines", []):
                    for sp in ln.get("spans", []):
                        tx = sp["text"].strip()
                        if not tx:
                            continue
                        x0,y0,x1,y1 = sp["bbox"]
                        blocks.append({"text": tx, "x": x0, "y": y0, "bbox": [x0,y0,x1,y1],
                                       "font_size": sp.get("size",12), "font": sp.get("font","")})
            blocks.sort(key=lambda b: (b["y"], b["x"]))

            headers = self._locate_headers(blocks)
            if not headers:
                return transactions

            dep_x = headers["DEPOSIT"]; wdr_x = headers["WITHDRAWAL"]; bal_x = headers["BALANCE"]
            table_top_y = headers.get("_TABLE_TOP_Y", None)

            # fixed bands: [.. midDW) -> deposit, [midDW .. midWB) -> withdraw, [midWB .. ) -> balance
            midDW = (dep_x + wdr_x) / 2.0
            midWB = (wdr_x + bal_x) / 2.0
            def col_of(x):
                if x < midDW: return "DEPOSIT"
                if x < midWB: return "WITHDRAWAL"
                return "BALANCE"

            # tighter row grouping for HLB
            rows = []
            cur = []; last_y = None; y_tol = 3  # tighter than base to avoid row merges
            for b in blocks:
                if last_y is None or abs(b["y"] - last_y) <= y_tol:
                    cur.append(b)
                else:
                    if cur: rows.append(cur)
                    cur = [b]
                last_y = b["y"]
            if cur: rows.append(cur)

            pending = self._carry

            for row in rows:
                row.sort(key=lambda b: b["x"])
                if not row: 
                    continue

                row_y = min(b["y"] for b in row)
                if table_top_y is not None and row_y < table_top_y + 5:
                    continue

                row_text = " ".join(b["text"] for b in row)
                row_norm = self._norm(row_text)

                # ignore table summaries/labels
                if any(k in row_norm for k in ["TOTALWITHDRAWALS","TOTALDEPOSITS","JUMLAHPENGELUARAN","JUMLAHSIMPANAN","DATETARIKH","PAGE"]):
                    continue

                is_opening = any(alias in row_norm for alias in self.BALANCE_BF_ALIASES)
                date_blk = next((b for b in row if self._is_date(b["text"])), None)

                deposit_val = None; withdraw_val = None; balance_blk = None
                for b in row:
                    num = self._clean_num_text(b["text"])
                    if num is None:
                        continue
                    c = col_of(b["x"])
                    if c == "DEPOSIT":
                        deposit_val = num
                    elif c == "WITHDRAWAL":
                        withdraw_val = num
                    else:  # BALANCE
                        balance_blk = b

                if is_opening and balance_blk is not None:
                    transactions.append({
                        "date": date_blk["text"] if date_blk else "",
                        "description": "Balance from previous statement",
                        "amount": Decimal('0.00'),
                        "original_balance": Decimal('0.00'),
                        "new_balance": Decimal('0.00'),
                        "page_num": page_num
                    })
                    self.balance_replacements.append({
                        "type":"transaction_balance","original_value":Decimal('0.00'),
                        "bbox": balance_blk["bbox"], "font_size": balance_blk.get("font_size",12),
                        "font": balance_blk.get("font",""), "page_num": page_num, "y_position": balance_blk["bbox"][1]
                    })
                    continue

                if deposit_val is None and withdraw_val is None and balance_blk is None:
                    continue

                delta = Decimal('0.00')
                if withdraw_val not in (None, Decimal('0.00')): delta -= withdraw_val
                if deposit_val not in (None, Decimal('0.00')): delta += deposit_val
                pending += delta

                if balance_blk is not None:
                    transactions.append({
                        "date": date_blk["text"] if date_blk else "",
                        "description": "",
                        "amount": pending,
                        "original_balance": Decimal('0.00'),
                        "new_balance": Decimal('0.00'),
                        "page_num": page_num
                    })
                    self.balance_replacements.append({
                        "type":"transaction_balance","original_value":Decimal('0.00'),
                        "bbox": balance_blk["bbox"], "font_size": balance_blk.get("font_size",12),
                        "font": balance_blk.get("font",""), "page_num": page_num, "y_position": balance_blk["bbox"][1]
                    })
                    pending = Decimal('0.00')
                    # Track the right edge of the Balance column for precise right-alignment later
                    x2 = balance_blk["bbox"][2]
                    self._hlb_balance_right_edge = x2 if self._hlb_balance_right_edge is None else max(self._hlb_balance_right_edge, x2)

            self._carry = pending

        except Exception as e:
            log_func(f"⚠️ Error in HLB extraction: {e}")

        return transactions

    def _find_summary_positions_hlb(self, text_dict: dict, page_num: int, log_func: Callable):
        """Find 'Closing Balance / Baki Akhir' and replace using the detected Balance column right edge."""
        try:
            blocks = []
            for b in text_dict.get("blocks", []):
                for ln in b.get("lines", []):
                    for sp in ln.get("spans", []):
                        tx = sp["text"].strip()
                        if not tx:
                            continue
                        x0, y0, x1, y1 = sp["bbox"]
                        blocks.append({
                            "text": tx, "x": x0, "y": y0, "bbox": [x0, y0, x1, y1],
                            "font_size": sp.get("size", 12), "font": sp.get("font", "")
                        })

            # Tolerance for same-baseline grouping
            y_tol = 8

            def is_num_fragment(t: str) -> bool:
                return bool(re.match(r'^-?[\d,\.]+$', t.strip()))

            for lbl in blocks:
                if self._norm(lbl["text"]) not in {"CLOSINGBALANCE", "BAKIAKHIR"}:
                    continue

                # Collect numeric fragments on same row (to the right of label)
                fragments = [b for b in blocks
                             if b["x"] > lbl["x"]
                             and abs(b["y"] - lbl["y"]) <= y_tol
                             and is_num_fragment(b["text"])]

                if not fragments:
                    log_func(f"⚠️ HLB: No numeric found on closing balance row p{page_num}")
                    continue

                fragments.sort(key=lambda b: b["x"])
                x0 = min(b["bbox"][0] for b in fragments)
                y0 = min(b["bbox"][1] for b in fragments)
                x1 = max(b["bbox"][2] for b in fragments)
                y1 = max(b["bbox"][3] for b in fragments)

                merged_text = "".join(b["text"] for b in fragments).replace(" ", "")
                amount_val = self._clean_num_text(merged_text)
                if amount_val is None:
                    # fallback to rightmost valid fragment
                    for b in reversed(fragments):
                        amount_val = self._clean_num_text(b["text"])
                        if amount_val is not None:
                            break
                if amount_val is None:
                    log_func(f"⚠️ HLB: Could not parse closing balance on p{page_num}, text='{merged_text}'")
                    continue

                # Anchor the right edge to the Balance column's detected right edge for perfect alignment
                right_edge = self._hlb_balance_right_edge if self._hlb_balance_right_edge else x1
                right_edge = max(right_edge, x1)  # never smaller than detected fragments

                # AFTER: preserve top y0 so baseline stays correct; expand sides/bottom only
                pad_left_right = 16
                pad_bottom = 8
                expanded = [x0 - pad_left_right, y0, right_edge, y1 + pad_bottom]

                # Tight vertical cover: keep top at y0, bottom just around the glyph height
                # This avoids painting over the horizontal rule below the row.
                pad_lr = 8
                fs = fragments[-1].get("font_size", 12)
                # bottom limited to about one text-height from the top; do NOT expand downward
                bottom = min(y1, y0 + fs * 1.05)
                expanded = [x0 - pad_lr, y0, right_edge, bottom]

                log_func(f"🔎 HLB Closing Balance p{page_num}: value={amount_val}, right_edge={right_edge:.2f}, bbox={expanded}")

                self.balance_replacements.append({
                    "type": "ending_balance",
                    "original_value": amount_val,
                    "bbox": expanded,
                    "font_size": fragments[-1].get("font_size", 12),
                    "font": fragments[-1].get("font", ""),
                    "page_num": page_num,
                    "y_position": y0
                })
                break
        except Exception as e:
            log_func(f"⚠️ Error finding HLB closing balance: {e}")

    # Abstracts (unused by this parser, but required)
    def _find_balance_column_position(self, all_blocks: List[Dict], log_func: Callable) -> float:
        headers = self._locate_headers(all_blocks)
        if headers and "BALANCE" in headers:
            self.balance_column_x = headers["BALANCE"]
            self.balance_column_right = self.balance_column_x + 80
            return self.balance_column_x
        return 0.0

    def _parse_transaction_row(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable) -> Optional[Dict]:
        return None