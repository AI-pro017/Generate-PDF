#!/usr/bin/env python3
"""
Public Bank (PBB) PDF Statement Processor
Handles PBB-specific PDF format and structure
"""

import re
import os
from decimal import Decimal
from typing import List, Dict, Optional, Callable
import fitz

from .base_processor import BaseProcessor


class PBBProcessor(BaseProcessor):
    """Public Bank-specific PDF statement processor"""
    
    def __init__(self):
        super().__init__()
        self.bank_name = "Public Bank (PBB)"
        # keep right edge of balance column for better alignment if needed later
        self._balance_right_edge = None

    def _is_amount_text(self, text: str) -> bool:
        t = text.strip().replace("RM", "")
        if not t:
            return False
        return bool(re.match(r'^-?\d{1,3}(,\d{3})*(\.\d{2})$|-?\d+\.\d{2}$', t))

    def _norm(self, s: str) -> str:
        return re.sub(r'[^A-Z]', '', s.upper())
    
    def extract_transactions_from_pdf(self, pdf_path: str, log_func: Callable = print) -> List[Dict]:
        """Extract transaction data from PBB PDF format"""
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

                # Flatten text blocks
                blocks: List[Dict] = []
                for b in text_dict.get("blocks", []):
                    for ln in b.get("lines", []):
                        for sp in ln.get("spans", []):
                            tx = sp.get("text", "").strip()
                            if not tx:
                                continue
                            x0, y0, x1, y1 = sp.get("bbox")
                            blocks.append({
                                "text": tx,
                                "x": x0,
                                "y": y0,
                                "bbox": [x0, y0, x1, y1],
                                "font_size": sp.get("size", 12),
                                "font": sp.get("font", "")
                            })

                blocks.sort(key=lambda b: (b["y"], b["x"]))

                # 1) Summary: "Baki Penutup / Closing Balance" → replace with user input (type=beginning_balance)
                self._add_summary_closing_replacement(blocks, page_num + 1, log_func)

                # 2) Locate table headers and parse rows
                hdr = self._locate_headers(blocks)
                if not hdr:
                    continue
                deb_x, cre_x, bal_x = hdr["DEBIT"], hdr["CREDIT"], hdr["BALANCE"]
                table_top_y = hdr.get("_TABLE_TOP_Y", None)

                midDC = (deb_x + cre_x) / 2.0
                midCB = (cre_x + bal_x) / 2.0
                def col_of(x: float) -> str:
                    if x < midDC: return "DEBIT"
                    if x < midCB: return "CREDIT"
                    return "BALANCE"

                # tighter row grouping
                rows: List[List[Dict]] = []
                cur: List[Dict] = []
                last_y = None
                y_tol = 3
                for bl in blocks:
                    if last_y is None or abs(bl["y"] - last_y) <= y_tol:
                        cur.append(bl)
                    else:
                        if cur:
                            rows.append(cur)
                        cur = [bl]
                    last_y = bl["y"]
                if cur:
                    rows.append(cur)

                for row in rows:
                    row.sort(key=lambda b: b["x"])
                    if not row:
                        continue
                    row_y = min(b["y"] for b in row)
                    if table_top_y is not None and row_y < table_top_y + 5:
                        # above header band
                        continue

                    row_text_norm = self._norm(" ".join(b["text"] for b in row))
                    is_closing_row = any(k in row_text_norm for k in [
                        "CLOSINGBALANCEINTHISSTATEMENT", "CLOSINGBALANCE", "BAKIPENUTUP"
                    ])

                    debit_val: Optional[Decimal] = None
                    credit_val: Optional[Decimal] = None
                    balance_blk: Optional[Dict] = None

                    for b in row:
                        t = b["text"].replace(" ", "")
                        if self._is_amount_text(t):
                            c = col_of(b["x"])
                            if c == "DEBIT":
                                debit_val = Decimal(t.replace(",", ""))
                            elif c == "CREDIT":
                                credit_val = Decimal(t.replace(",", ""))
                            else:
                                balance_blk = b

                    # Closing summary row inside table: replace directly with user input
                    if is_closing_row and balance_blk is not None:
                        self.balance_replacements.append({
                            "type": "beginning_balance",  # will be set to GUI-entered value
                            "original_value": Decimal('0.00'),
                            "bbox": balance_blk["bbox"],
                            "font_size": balance_blk.get("font_size", 12),
                            "font": balance_blk.get("font", ""),
                            "page_num": page_num + 1,
                            "y_position": balance_blk["bbox"][1]
                        })
                        continue

                    # Regular data row or B/F, C/F balances
                    if balance_blk is None and (debit_val is None and credit_val is None):
                        continue

                    # Track balance column right edge for alignment
                    if balance_blk is not None:
                        x2 = balance_blk["bbox"][2]
                        self._balance_right_edge = x2 if self._balance_right_edge is None else max(self._balance_right_edge, x2)

                    # record replacement for the balance cell on this row
                    if balance_blk is not None:
                        self.balance_replacements.append({
                            "type": "transaction_balance",
                            "original_value": Decimal('0.00'),
                            "bbox": balance_blk["bbox"],
                            "font_size": balance_blk.get("font_size", 12),
                            "font": balance_blk.get("font", ""),
                            "page_num": page_num + 1,
                            "y_position": balance_blk["bbox"][1]
                        })

                    # store a logical transaction used for later balance computation (bottom-up)
                    transactions.append({
                        "date": "",  # PBB balance calc does not require the date
                        "description": "",
                        "amount": (debit_val or Decimal('0.00')) - (credit_val or Decimal('0.00')),
                        "debit": debit_val or Decimal('0.00'),
                        "credit": credit_val or Decimal('0.00'),
                        "new_balance": Decimal('0.00'),
                        "page_num": page_num + 1,
                        "y_position": balance_blk["bbox"][1] if balance_blk else row_y
                    })
            
            pdf_document.close()
            
        except Exception as e:
            log_func(f"❌ Error extracting data from PDF: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        log_func(f"✅ Extracted {len(transactions)} PBB rows to compute balances")
        return transactions
    
    def _analyze_pbb_structure(self, text_dict: dict, page_num: int, log_func: Callable):
        pass
    
    def _extract_transactions_and_balances(self, text_dict: dict, page_num: int, log_func: Callable) -> List[Dict]:
        # Not used; logic handled in extract_transactions_from_pdf
        return []
    
    def _find_balance_column_position(self, all_blocks: List[Dict], log_func: Callable) -> float:
        headers = self._locate_headers(all_blocks)
        if headers and "BALANCE" in headers:
            return headers["BALANCE"]
        return 0.0
    
    def _parse_transaction_row(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable) -> Optional[Dict]:
        return None

    def _locate_headers(self, blocks: List[Dict]) -> Optional[Dict[str, float]]:
        header_x, header_y = {}, {}
        for b in blocks:
            t = b["text"].upper().strip()
            if t in ("DEBIT",):
                header_x["DEBIT"] = b["x"]; header_y["DEBIT"] = b["y"]
            elif t in ("KREDIT", "CREDIT"):
                header_x["CREDIT"] = b["x"]; header_y["CREDIT"] = b["y"]
            elif t in ("BAKI", "BALANCE"):
                header_x["BALANCE"] = b["x"]; header_y["BALANCE"] = b["y"]
        if all(k in header_x for k in ("DEBIT", "CREDIT", "BALANCE")):
            header_x["_TABLE_TOP_Y"] = min(header_y.values()) if header_y else None
            return header_x
        return None

    def _add_summary_closing_replacement(self, blocks: List[Dict], page_num: int, log_func: Callable):
        try:
            y_tol = 8
            for lbl in blocks:
                if self._norm(lbl["text"]) not in {"BAKIPENUTUP", "CLOSINGBALANCE"}:
                    continue
                # collect numeric fragments on same row to the right
                frags = [b for b in blocks if b["x"] > lbl["x"] and abs(b["y"] - lbl["y"]) <= y_tol and self._is_amount_text(b["text"]) ]
                if not frags:
                    continue
                frags.sort(key=lambda b: b["x"])
                x0 = min(f["bbox"][0] for f in frags)
                y0 = min(f["bbox"][1] for f in frags)
                x1 = max(f["bbox"][2] for f in frags)
                y1 = max(f["bbox"][3] for f in frags)
                self.balance_replacements.append({
                    "type": "beginning_balance",  # use GUI input
                    "original_value": Decimal('0.00'),
                    "bbox": [x0, y0, x1, y1],
                    "font_size": frags[-1].get("font_size", 12),
                    "font": frags[-1].get("font", ""),
                    "page_num": page_num,
                    "y_position": y0
                })
                break
        except Exception:
            pass

    # Override balance recomputation for PBB: start from closing balance (GUI input)
    def recalculate_balances(self, transactions: List[Dict], beginning_balance: Decimal, log_func: Callable = print) -> List[Dict]:
        try:
            closing_balance = beginning_balance
            # Compute from bottom (last page, bottom row) upward
            ordered = sorted(transactions, key=lambda t: (t['page_num'], t.get('y_position', 0)))
            rev = list(reversed(ordered))

            running = closing_balance
            total_credit = Decimal('0.00')
            total_debit = Decimal('0.00')
            computed = []
            for t in rev:
                debit = t.get('debit', Decimal('0.00')) or Decimal('0.00')
                credit = t.get('credit', Decimal('0.00')) or Decimal('0.00')
                ct = dict(t)
                # The balance to display for this row (balance *after* this transaction)
                # is the 'running' balance from the previous iteration.
                ct['new_balance'] = running
                
                # Now calculate the balance *before* this transaction, which will be
                # the 'running' balance for the next iteration (the row above).
                # Rule: for credit, apply - (subtract); for debit, apply + (add)
                if credit > 0:
                    running = (running - credit).quantize(Decimal('0.01'))
                if debit > 0:
                    running = (running + debit).quantize(Decimal('0.01'))
                computed.append(ct)
                total_debit += debit
                total_credit += credit

            # Restore top-to-bottom order for mapping to replacements
            computed = list(reversed(computed))

            # Logging similar to base
            log_func(f"💰 Closing balance (input): RM {closing_balance:,.2f}")
            log_func("=" * 90)
            log_func(f"{'#':<3} {'Page':<4} {'Y':<7} {'Debit':>12} {'Credit':>12} {'New Balance':>15}")
            log_func("=" * 90)
            for i, t in enumerate(computed, 1):
                log_func(f"{i:<3} {t['page_num']:<4} {int(t.get('y_position',0)):<7} "
                        f"{t.get('debit',Decimal('0.00')):>12,.2f} {t.get('credit',Decimal('0.00')):>12,.2f} "
                        f"{t['new_balance']:>15,.2f}")
            log_func("=" * 90)
            log_func(f"Totals  Debit: RM {total_debit:,.2f}   Credit: RM {total_credit:,.2f}")

            return computed
        except Exception as e:
            log_func(f"⚠️ Error recalculating PBB balances: {e}")
            return transactions

    # Override populate to ensure only the very last transaction gets the ending balance
    def _populate_new_values(self, transactions: List[Dict], log_func: Callable = print):
        try:
            # Handle summary replacements (closing balance)
            for rep in self.balance_replacements:
                if rep.get('type') == 'beginning_balance':
                    # This is the closing balance - use user input
                    rep['new_value'] = self._format_amount(self.beginning_balance)
                    log_func(f"🔍 PBB closing balance: {rep['new_value']}")

            # Group transactions by page
            from collections import defaultdict
            tx_by_page = defaultdict(list)
            for tx in transactions:
                tx_by_page[tx['page_num']].append(tx)

            reps_by_page = defaultdict(list)
            for rep in self.balance_replacements:
                if rep.get('type') == 'transaction_balance':
                    reps_by_page[rep['page_num']].append(rep)

            # Find the very last transaction across all pages
            all_transactions = sorted(transactions, key=lambda t: (t['page_num'], t.get('y_position', 0)))
            last_transaction = all_transactions[-1] if all_transactions else None

            # Assign transaction balances per page
            for page_num, txs in tx_by_page.items():
                reps = sorted(reps_by_page.get(page_num, []), key=lambda r: r.get('y_position', 0))
                n = min(len(txs), len(reps))
                log_func(f"🔍 PBB page {page_num}: {len(txs)} transactions, {len(reps)} replacements")
                
                for i in range(n):
                    # Check if this is the very last transaction across ALL pages
                    is_last_across_all_pages = (txs[i] == last_transaction)
                    
                    if is_last_across_all_pages:
                        # Only the very last transaction row across all pages gets the ending balance
                        val = self._format_amount(self.beginning_balance)
                        reps[i]['new_value'] = val
                        log_func(f"🔍 PBB last transaction across all pages: {val}")
                    else:
                        # All other rows use calculated values from reverse calculation
                        # This represents the balance AFTER this transaction was applied
                        val = self._format_amount(txs[i]['new_balance'])
                        reps[i]['new_value'] = val
                        log_func(f"🔍 PBB transaction {i}: {val} (balance after transaction)")

            # Debug: show final values for all replacements
            log_func(f"🔍 PBB Final replacement values:")
            for i, rep in enumerate(self.balance_replacements):
                log_func(f"🔍 PBB {i}: type={rep.get('type')}, value={rep.get('new_value')}")

        except Exception as e:
            log_func(f"⚠️ Error populating PBB values: {e}")