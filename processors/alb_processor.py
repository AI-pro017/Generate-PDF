#!/usr/bin/env python3
"""
Alliance Bank (ALB) PDF Statement Processor
Parses a single statement table (Date, Details, Cheque No, Debit, Credit, Balance)
and preserves the "CR" suffix when replacing balance values. Implements reverse
calculation: input is treated as closing balance; balances are computed bottom‑up.
"""

import re
import os
from decimal import Decimal
from typing import List, Dict, Optional, Callable

import fitz

from .base_processor import BaseProcessor


class ALBProcessor(BaseProcessor):
    """Alliance Bank-specific PDF statement processor"""

    def __init__(self):
        super().__init__()
        self.bank_name = "Alliance Bank (ALB)"
        self._right_inset = 2.0  # keep a small gap from the vertical rule

    def _row_numeric_style(self, row: List[Dict], balance_blk: Optional[Dict]) -> Dict[str, object]:
        """Infer typical numeric style (font size/name) from DEBIT/CREDIT cells on this row."""
        try:
            candidates = []
            for b in row:
                if b is balance_blk:
                    continue
                t = b.get("text", "").replace(" ", "")
                if self._is_amount_text(t):
                    candidates.append(b)
            if not candidates:
                return {
                    "font_size": max(9, int((balance_blk or {}).get("font_size", 12)) - 1),
                    "font": (balance_blk or {}).get("font", "helv") or "helv",
                }
            # Prefer the rightmost numeric (credit) which best matches column style
            rightmost = max(candidates, key=lambda b: b["x"]) 
            return {
                "font_size": int(rightmost.get("font_size", 12)),
                "font": rightmost.get("font", "helv") or "helv",
            }
        except Exception:
            return {
                "font_size": max(9, int((balance_blk or {}).get("font_size", 12)) - 1),
                "font": (balance_blk or {}).get("font", "helv") or "helv",
            }

    def _row_date_style(self, row: List[Dict]) -> Dict[str, object]:
        """Return the style (font size/name) of the Date column for this row.
        Prefers a 6‑digit date like DDMMYY; otherwise uses the leftmost span in the row.
        """
        try:
            date_candidates = []
            for b in row:
                t = (b.get("text", "") or "").strip()
                if re.match(r"^\d{6}$", t):
                    date_candidates.append(b)
            target = date_candidates[0] if date_candidates else min(row, key=lambda b: b["x"])  # leftmost
            return {
                "font_size": int(target.get("font_size", 12)),
                "font": target.get("font", "helv") or "helv",
            }
        except Exception:
            return {"font_size": 10, "font": "helv"}

    # Amount fragments can have optional CR suffix
    def _is_amount_text(self, text: str) -> bool:
        if not text:
            return False
        t = text.replace(",", "").replace("RM", "").strip()
        t = t.replace(" CR", "").replace("CR", "")
        if "." not in t:
            return False
        return bool(re.match(r"^-?\d+(?:\.\d{2})$", t))

    def extract_transactions_from_pdf(self, pdf_path: str, log_func: Callable = print) -> List[Dict]:
        self.original_pdf_path = pdf_path
        self.balance_replacements = []

        transactions: List[Dict] = []

        try:
            doc = fitz.open(pdf_path)
            log_func(f"📄 Processing {self.bank_name} PDF: {os.path.basename(pdf_path)}")
            log_func(f"📊 Total pages: {len(doc)}")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_dict = page.get_text("dict")

                blocks: List[Dict] = []
                for b in text_dict.get("blocks", []):
                    for ln in b.get("lines", []):
                        for sp in ln.get("spans", []):
                            tx = sp.get("text", "").strip()
                            if not tx:
                                continue
                            x0, y0, x1, y1 = sp.get("bbox")
                            blocks.append({
                                "text": tx, "x": x0, "y": y0, "bbox": [x0, y0, x1, y1],
                                "font_size": sp.get("size", 12), "font": sp.get("font", "")
                            })

                blocks.sort(key=lambda b: (b["y"], b["x"]))

                # Locate headers for DEBIT/CREDIT/BALANCE
                headers = self._locate_headers(blocks)
                if not headers:
                    continue
                deb_x, cre_x, bal_x = headers["DEBIT"], headers["CREDIT"], headers["BALANCE"]
                table_top_y = headers.get("_TABLE_TOP_Y", None)

                midDC = (deb_x + cre_x) / 2.0
                midCB = (cre_x + bal_x) / 2.0

                def col_of(x: float) -> str:
                    if x < midDC:
                        return "DEBIT"
                    if x < midCB:
                        return "CREDIT"
                    return "BALANCE"

                # Group into rows
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
                    row.sort(key=lambda b: b["x"])  # left to right
                    if not row:
                        continue
                    row_y = min(b["y"] for b in row)
                    if table_top_y is not None and row_y < table_top_y + 5:
                        continue

                    row_text = " ".join(b["text"] for b in row).upper()
                    # Skip totals/footer rows
                    if "TOTAL DEBIT/CREDIT" in row_text or "TOTAL" in row_text:
                        continue
                    norm = re.sub(r"[^A-Z]", "", row_text)
                    is_beginning = "BEGINNINGBALANCE" in norm
                    is_ending = "ENDINGBALANCE" in norm

                    debit_val: Optional[Decimal] = None
                    credit_val: Optional[Decimal] = None
                    balance_blk: Optional[Dict] = None
                    balance_has_cr: bool = False

                    for b in row:
                        t = b["text"].replace(" ", "")
                        if self._is_amount_text(t):
                            c = col_of(b["x"])
                            num_txt = b["text"].replace(",", "").replace("RM", "").strip()
                            has_cr = num_txt.endswith("CR") or num_txt.endswith(" CR")
                            num_txt = num_txt.replace(" CR", "").replace("CR", "")
                            try:
                                num = Decimal(num_txt)
                            except Exception:
                                continue
                            if c == "DEBIT":
                                debit_val = num
                            elif c == "CREDIT":
                                credit_val = num
                            else:
                                balance_blk = b
                                balance_has_cr = has_cr

                    # If there's a balance cell on this row, record a replacement and a logical tx
                    if balance_blk is None and debit_val is None and credit_val is None:
                        continue

                    if balance_blk is not None:
                        # inset right edge to avoid painting over the vertical rule
                        bb = list(balance_blk["bbox"])
                        bb[2] = max(bb[0] + 1.0, bb[2] - self._right_inset)
                        
                        if is_beginning:
                            rep_type = "beginning_balance"
                        elif is_ending:
                            rep_type = "ending_balance"
                        else:
                            rep_type = "transaction_balance"
                        
                        # Use date column style for BEGINNING/ENDING rows; otherwise use numeric style
                        style = self._row_date_style(row) if (is_beginning or is_ending) else self._row_numeric_style(row, balance_blk)
                        final_font_size = style["font_size"]
                        
                        # For BEGINNING BALANCE rows, always preserve CR suffix if the original had it
                        # For other rows, use the detected CR suffix
                        suffix = ""
                        if is_beginning:
                            # For BEGINNING BALANCE rows, we need to check if ANY balance in the PDF has CR suffix
                            # This indicates the account type uses CR suffix
                            # We'll set this to always include CR for BEGINNING BALANCE rows if we detect CR anywhere
                            suffix = " CR"  # Always add CR for BEGINNING BALANCE rows in ALB
                        else:
                            suffix = " CR" if balance_has_cr else ""
                        
                        log_func(f"🔍 ALB creating replacement: type={rep_type}, suffix='{suffix}', is_beginning={is_beginning}, balance_has_cr={balance_has_cr}")
                        self.balance_replacements.append({
                            "type": rep_type,
                            "original_value": Decimal('0.00'),
                            "bbox": bb,
                            "font_size": final_font_size,
                            "font": style["font"],
                            "page_num": page_num + 1,
                            "y_position": bb[1],
                            "suffix": suffix
                        })

                    # Logical transaction for recomputation (bottom-up)
                    # For reverse calculation: we need the signed transaction amount
                    # Credit (+10.00) means money in, so previous balance = current - 10.00
                    # Debit (-20.00) means money out, so previous balance = current + 20.00
                    if debit_val is None and credit_val is None:
                        continue
                    
                    # Skip BEGINNING BALANCE rows - they don't create transactions
                    if is_beginning:
                        continue
                    
                    # The transaction amount is the net effect on the balance
                    # For credit (money in): subtract from balance
                    # For debit (money out): add to balance
                    transaction_amount = (credit_val or Decimal('0.00')) - (debit_val or Decimal('0.00'))
                    
                    log_func(f"🔍 ALB creating transaction: debit={debit_val}, credit={credit_val}, amount={transaction_amount}")
                    
                    transactions.append({
                        "date": "",
                        "description": "",
                        "debit": debit_val or Decimal('0.00'),
                        "credit": credit_val or Decimal('0.00'),
                        "amount": transaction_amount,  # This is the signed transaction amount
                        "new_balance": Decimal('0.00'),
                        "page_num": page_num + 1,
                        "y_position": balance_blk["bbox"][1] if balance_blk else row_y
                    })

                # Do not add separate summary positions to avoid duplicates; handled per row above

            doc.close()

        except Exception as e:
            log_func(f"❌ Error extracting data from PDF: {e}")
            import traceback
            traceback.print_exc()
            return []

        # Ensure at least one synthetic transaction so downstream pipeline continues
        if not transactions and self.balance_replacements:
            first = self.balance_replacements[0]
            transactions.append({
                "date": "",
                "description": "SYNTHETIC",
                "debit": Decimal('0.00'),
                "credit": Decimal('0.00'),
                "amount": Decimal('0.00'),
                "new_balance": Decimal('0.00'),
                "page_num": first.get('page_num', 1),
                "y_position": first.get('y_position', 0)
            })

        log_func(f"✅ Extracted {len(transactions)} ALB rows to compute balances")
        return transactions

    def _locate_headers(self, blocks: List[Dict]) -> Optional[Dict[str, float]]:
        xs, ys = {}, {}
        for b in blocks:
            t = b["text"].upper().strip()
            if t in ("DEBIT",):
                xs["DEBIT"] = b["x"]; ys["DEBIT"] = b["y"]
            elif t in ("CREDIT",):
                xs["CREDIT"] = b["x"]; ys["CREDIT"] = b["y"]
            elif t in ("BALANCE", "BAKI"):
                xs["BALANCE"] = b["x"]; ys["BALANCE"] = b["y"]
        if all(k in xs for k in ("DEBIT", "CREDIT", "BALANCE")):
            xs["_TABLE_TOP_Y"] = min(ys.values()) if ys else None
            return xs
        return None

    def _add_summary_positions(self, blocks: List[Dict], page_num: int):
        # Find ENDING BALANCE and BEGINNING BALANCE rows' balance cell; capture suffix
        labels = {
            "ENDINGBALANCE": "ending_balance",
            "BEGINNINGBALANCE": "transaction_balance",  # map to first row
        }
        y_tol = 6
        blocks_list = list(blocks)
        for b in blocks_list:
            key = re.sub(r"[^A-Z]", "", b.get("text", "").upper())
            if key not in labels:
                continue
            # Numeric on same row to right
            candidates = []
            for c in blocks_list:
                if c["x"] > b["x"] and abs(c["y"] - b["y"]) <= y_tol:
                    t = c["text"].strip()
                    if self._is_amount_text(t):
                        candidates.append(c)
            if not candidates:
                continue
            target = max(candidates, key=lambda x: x["x"])  # rightmost numeric
            txt = target["text"].replace(",", "").strip()
            has_cr = txt.endswith("CR") or txt.endswith(" CR")
            # inset right edge slightly to avoid overlapping right border
            bb = list(target["bbox"]) ; bb[2] = max(bb[0] + 1.0, bb[2] - self._right_inset)
            self.balance_replacements.append({
                "type": labels[key],
                "original_value": Decimal('0.00'),
                "bbox": bb,
                "font_size": target.get("font_size", 12),
                "font": target.get("font", ""),
                "page_num": page_num,
                "y_position": bb[1],
                "suffix": " CR" if has_cr else ""
            })

    # Reverse recomputation: start from user input as closing balance
    def recalculate_balances(self, transactions: List[Dict], beginning_balance: Decimal, log_func: Callable = print) -> List[Dict]:
        try:
            closing = beginning_balance
            ordered = sorted(transactions, key=lambda t: (t['page_num'], t.get('y_position', 0)))
            
            # Start with the closing balance and work backwards through transactions
            running_balance = closing
            computed: List[Dict] = []
            
            # Process from last transaction to first (reverse order)
            for t in reversed(ordered):
                # Set this transaction's balance
                t['new_balance'] = running_balance
                
                # Calculate the previous balance
                debit = t.get('debit', Decimal('0.00')) or Decimal('0.00')
                credit = t.get('credit', Decimal('0.00')) or Decimal('0.00')
                
                # For reverse calculation:
                # Credit (+10.00): money came in, so previous balance = current - 10.00
                # Debit (20.00): money went out, so previous balance = current + 20.00
                # So: previous = current - transaction_amount
                transaction_amount = t.get('amount', Decimal('0.00'))
                running_balance = (running_balance - transaction_amount).quantize(Decimal('0.01'))
                
                computed.append(t)
            
            # Reverse back to original order
            computed.reverse()
            log_func(f"💰 Closing balance (input): RM {closing:,.2f}")
            return computed
        except Exception as e:
            log_func(f"⚠️ Error recalculating ALB balances: {e}")
            return transactions

    # Override populate to keep CR suffix
    def _populate_new_values(self, transactions: List[Dict], log_func: Callable = print):
        try:
            # Ending balance use user input and preserve CR suffix if seen on summary row
            log_func(f"🔍 ALB _populate_new_values: processing {len(self.balance_replacements)} replacements")
            for rep in self.balance_replacements:
                log_func(f"🔍 ALB replacement type: {rep.get('type')}, suffix: '{rep.get('suffix', '')}'")
                if rep.get('type') == 'ending_balance':
                    # if original had CR, keep it
                    suffix = rep.get('suffix', '')
                    rep['new_value'] = self._format_amount(self.beginning_balance) + suffix
                    log_func(f"🔍 ALB ENDING BALANCE: {rep['new_value']}")
                elif rep.get('type') == 'beginning_balance':
                    # BEGINNING BALANCE gets the calculated opening balance
                    suffix = rep.get('suffix', '')
                    log_func(f"🔍 ALB BEGINNING BALANCE: calculating with suffix '{suffix}'")
                    # The opening balance is the final running_balance after reverse calculation
                    # We need to calculate it here since it's not stored in transactions
                    if transactions:
                        # Calculate the opening balance by working backwards from the last transaction
                        closing_balance = self.beginning_balance  # This is the user input (closing balance)
                        opening_balance = closing_balance
                        
                        log_func(f"🔍 ALB transactions: {len(transactions)}")
                        for i, t in enumerate(reversed(transactions)):
                            # For each transaction, calculate the previous balance
                            transaction_amount = t.get('amount', Decimal('0.00'))
                            old_balance = opening_balance
                            opening_balance = (opening_balance - transaction_amount).quantize(Decimal('0.01'))
                            log_func(f"🔍 ALB transaction {i}: amount={transaction_amount}, {old_balance} -> {opening_balance}")
                        
                        log_func(f"🔍 ALB BEGINNING BALANCE calculation: closing={closing_balance}, opening={opening_balance}")
                        rep['new_value'] = self._format_amount(opening_balance) + suffix
                        log_func(f"🔍 ALB BEGINNING BALANCE final value: {rep['new_value']}")
                    else:
                        rep['new_value'] = self._format_amount(self.beginning_balance) + suffix
                        log_func(f"🔍 ALB BEGINNING BALANCE no transactions: {rep['new_value']}")

            # Assign transaction balances per page, preserving CR suffix
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

            for page_num, txs in tx_by_page.items():
                reps = sorted(reps_by_page.get(page_num, []), key=lambda r: r.get('y_position', 0))
                n = min(len(txs), len(reps))
                log_func(f"🔍 ALB page {page_num}: {len(txs)} transactions, {len(reps)} replacements")
                for i in range(n):
                    # Check if this is the very last transaction across ALL pages
                    is_last_across_all_pages = (txs[i] == last_transaction)
                    
                    if is_last_across_all_pages:
                        # Only the very last transaction row across all pages gets the card value
                        val = self._format_amount(self.beginning_balance) + reps[i].get('suffix', '')
                        reps[i]['new_value'] = val
                        log_func(f"🔍 ALB last transaction across all pages: {val}")
                    else:
                        # All other rows use calculated values
                        val = self._format_amount(txs[i]['new_balance']) + reps[i].get('suffix', '')
                        reps[i]['new_value'] = val
                        log_func(f"🔍 ALB transaction {i}: {val}")

            # Totals unchanged (if present)
            ending_balance = transactions[-1]['new_balance'] if transactions else self.beginning_balance
            total_credit = sum(t.get('credit', Decimal('0.00')) for t in transactions)
            total_debit = sum(t.get('debit', Decimal('0.00')) for t in transactions)
            for rep in self.balance_replacements:
                t = rep.get('type')
                if t == 'total_credit':
                    rep['new_value'] = self._format_amount(total_credit)
                elif t == 'total_debit':
                    rep['new_value'] = self._format_amount(total_debit)
                elif t == 'beginning_balance':
                    # BEGINNING BALANCE is already handled above in the main loop
                    # Don't overwrite it here
                    pass

            # Fallback: ensure all replacements have a value to avoid runtime errors
            for rep in self.balance_replacements:
                if 'new_value' not in rep or rep['new_value'] is None:
                    rep['new_value'] = self._format_amount(ending_balance) + rep.get('suffix', '')
            
            # Debug: show final values for all replacements
            log_func(f"🔍 ALB Final replacement values:")
            for i, rep in enumerate(self.balance_replacements):
                log_func(f"🔍 ALB {i}: type={rep.get('type')}, value={rep.get('new_value')}, suffix='{rep.get('suffix', '')}'")
        except Exception as e:
            log_func(f"⚠️ Error populating ALB values: {e}")

    # Tight erase rect: keep off the right rule, and nudge slightly downward to avoid top rule
    def _erase_rect(self, page, bbox):
        import fitz
        x0, y0, x1, y1 = bbox
        left_pad = 8.0
        right_inset = max(self._right_inset, 2.0)
        # nudge down a touch and keep bottom slightly above the next rule
        cell_h = max(0.1, y1 - y0)
        top_inset = 0.8 if cell_h <= 12.0 else 0.4
        bottom_inset = 0.6 if cell_h <= 12.0 else 0.8
        rx0 = x0 - left_pad
        rx1 = x1 - right_inset
        ry0 = y0 + top_inset
        ry1 = max(y0 + 1.2, y1 - bottom_inset)
        r = fitz.Rect(rx0, ry0, rx1, ry1)
        pr = page.rect
        r.x0 = max(pr.x0, r.x0); r.y0 = max(pr.y0, r.y0)
        r.x1 = min(pr.x1, r.x1); r.y1 = min(pr.y1, r.y1)
        return r

    # Abstracts (not used directly; provided to satisfy BaseProcessor)
    def _extract_transactions_and_balances(self, text_dict: dict, page_num: int, log_func: Callable) -> List[Dict]:
        return []

    def _find_balance_column_position(self, all_blocks: List[Dict], log_func: Callable) -> float:
        headers = self._locate_headers(all_blocks)
        if headers and "BALANCE" in headers:
            return headers["BALANCE"]
        return 0.0

    def _parse_transaction_row(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable) -> Optional[Dict]:
        return None


