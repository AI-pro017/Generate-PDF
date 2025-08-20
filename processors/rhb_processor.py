#!/usr/bin/env python3
"""
RHB Bank PDF Statement Processor
Handles RHB-specific PDF format and structure
"""

import re
import os
from decimal import Decimal
from typing import List, Dict, Optional, Callable
import fitz

from .base_processor import BaseProcessor


class RHBProcessor(BaseProcessor):
    """RHB Bank-specific PDF statement processor"""
    
    def __init__(self):
        super().__init__()
        self.bank_name = "RHB Bank"
        self._balance_right_edge = None
        self._summary_open_info = None

    def _row_numeric_font(self, row: List[Dict], balance_blk: Optional[Dict]) -> int:
        """Return a typical numeric font size for this transaction row.

        Prefers the smallest numeric font size among other amount cells on the
        same row (e.g., Debit/Credit), so the Balance value matches the body.
        Falls back to balance cell size minus 1pt, with a floor of 9pt.
        """
        try:
            candidates: List[int] = []
            for b in row:
                if b is balance_blk:
                    continue
                t = b.get("text", "")
                if self._is_amount_text(t):
                    sz = int(b.get("font_size", 12))
                    candidates.append(sz)
            if candidates:
                # Make balance slightly smaller than the smallest numeric on the row
                return max(7, min(candidates) - 3)
            fb = int((balance_blk or {}).get("font_size", 12))
            return max(7, fb - 4)
        except Exception:
            fb = int((balance_blk or {}).get("font_size", 12))
            return max(7, fb - 4)

    def _norm(self, s: str) -> str:
        return re.sub(r'[^A-Z]', '', s.upper())

    def _is_amount_text(self, text: str) -> bool:
        t = text.strip().replace("RM", "")
        if not t or '.' not in t:
            return False
        return bool(re.match(r'^-?\d{1,3}(,\d{3})*(\.\d{2})$|-?\d+\.\d{2}$', t))
    
    def extract_transactions_from_pdf(self, pdf_path: str, log_func: Callable = print) -> List[Dict]:
        """Extract transaction data from RHB PDF format"""
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

                # Summary table replacements on first page
                if page_num == 0:
                    self._add_summary_replacements(blocks, page_num + 1, log_func)

                # Parse transaction table
                headers = self._locate_headers(blocks)
                if not headers:
                    continue
                deb_x = headers["DEBIT"]; cre_x = headers["CREDIT"]; bal_x = headers["BALANCE"]
                table_top_y = headers.get("_TABLE_TOP_Y")

                midDC = (deb_x + cre_x)/2.0
                midCB = (cre_x + bal_x)/2.0
                def col_of(x: float) -> str:
                    if x < midDC: return "DEBIT"
                    if x < midCB: return "CREDIT"
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
                        if cur: rows.append(cur)
                        cur = [bl]
                    last_y = bl["y"]
                if cur: rows.append(cur)

                reached_cf = False
                for row in rows:
                    row.sort(key=lambda b: b["x"]) 
                    if not row:
                        continue
                    if reached_cf:
                        break
                    row_y = min(b["y"] for b in row)
                    if table_top_y is not None and row_y < table_top_y + 5:
                        continue

                    row_text = " ".join(b["text"] for b in row)
                    row_norm = self._norm(row_text)
                    # Skip totals/footer rows
                    if "TOTALCOUNT" in row_norm or row_text.strip().upper() in {"(RM)", "RM"}:
                        continue
                    is_bf = any(k in row_norm for k in ["BFBALANCE","BF"])
                    is_cf = any(k in row_norm for k in ["CFBALANCE","ENDINGBALANCE"]) 

                    debit = None; credit = None; balance_blk = None
                    for b in row:
                        t = b["text"].replace(" ", "")
                        if self._is_amount_text(t):
                            c = col_of(b["x"])
                            if c == "DEBIT":
                                debit = Decimal(t.replace(",", ""))
                            elif c == "CREDIT":
                                credit = Decimal(t.replace(",", ""))
                            else:
                                balance_blk = b

                    # Track balance column right edge
                    if balance_blk is not None:
                        x2 = balance_blk["bbox"][2]
                        self._balance_right_edge = x2 if self._balance_right_edge is None else max(self._balance_right_edge, x2)

                    # Add replacements
                    if balance_blk is not None:
                        row_font_size = self._row_numeric_font(row, balance_blk)
                        if is_cf:
                            # bottom closing balance in table → replace with GUI input
                            self.balance_replacements.append({
                                "type": "beginning_balance",
                                "original_value": Decimal('0.00'),
                                "bbox": balance_blk["bbox"],
                                "font_size": row_font_size,
                                "font": balance_blk.get("font", ""),
                                "page_num": page_num + 1,
                                "y_position": balance_blk["bbox"][1]
                            })
                        else:
                            # normal/bf rows → will be mapped from computed balances
                            self.balance_replacements.append({
                                "type": "transaction_balance",
                                "original_value": Decimal('0.00'),
                                "bbox": balance_blk["bbox"],
                                "font_size": row_font_size,
                                "font": balance_blk.get("font", ""),
                                "page_num": page_num + 1,
                                "y_position": balance_blk["bbox"][1]
                            })

                    # record a logical transaction row for recompute
                    if debit is None and credit is None and balance_blk is None:
                        continue
                    transactions.append({
                        "date": "",
                        "description": "",
                        "debit": debit or Decimal('0.00'),
                        "credit": credit or Decimal('0.00'),
                        "amount": (debit or Decimal('0.00')) - (credit or Decimal('0.00')),
                        "new_balance": Decimal('0.00'),
                        "page_num": page_num + 1,
                        "y_position": balance_blk["bbox"][1] if balance_blk else row_y
                    })

                    if is_cf:
                        reached_cf = True
                        # do not process totals/notes after closing balance
                        continue
            
            pdf_document.close()
            
        except Exception as e:
            log_func(f"❌ Error extracting data from PDF: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        log_func(f"✅ Extracted {len(transactions)} RHB rows to compute balances")
        return transactions
    
    def _analyze_rhb_structure(self, text_dict: dict, page_num: int, log_func: Callable):
        pass
    
    def _extract_transactions_and_balances(self, text_dict: dict, page_num: int, log_func: Callable) -> List[Dict]:
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
            elif t in ("BALANCE", "BAKI"):
                header_x["BALANCE"] = b["x"]; header_y["BALANCE"] = b["y"]
        if all(k in header_x for k in ("DEBIT", "CREDIT", "BALANCE")):
            header_x["_TABLE_TOP_Y"] = min(header_y.values()) if header_y else None
            return header_x
        return None

    def _add_summary_replacements(self, blocks: List[Dict], page_num: int, log_func: Callable):
        try:
            # Build header x/y map for summary table
            header_xy = {}
            for b in blocks:
                n = self._norm(b["text"]) if b.get("text") else ""
                if any(h in n for h in ("OPENINGBALANCE", "BAKIPEMBUKAAN", "ENDINGBALANCE", "BAKIAKHIR", "INTERESTPAIDYTD")):
                    header_xy[n] = (b["x"], b["y"])  # take last; columns increase to right

            # Helper: pick typical numeric font size on the same row as ref block
            def row_numeric_font_size(ref_blk: Dict) -> int:
                try:
                    y_tol = 3.5
                    same_row_nums = [bb for bb in blocks
                                     if self._is_amount_text(bb.get("text", ""))
                                     and abs(bb["y"] - ref_blk["y"]) <= y_tol]
                    if not same_row_nums:
                        return max(10, int(ref_blk.get("font_size", 12)) - 1)
                    # Prefer the rightmost numeric (e.g., Interest Paid YTD column)
                    rightmost = max(same_row_nums, key=lambda bb: bb["x"]) 
                    return int(rightmost.get("font_size", ref_blk.get("font_size", 12)))
                except Exception:
                    return max(10, int(ref_blk.get("font_size", 12)) - 1)

            # helper: value cell under a given header label using neighbor header as right bound
            def value_under(headers_norm_set: set) -> Optional[Dict]:
                # pick the header with largest x among matches
                hs = [(n, xy) for n, xy in header_xy.items() if any(h in n for h in headers_norm_set)]
                if not hs:
                    return None
                hx, hy = max(hs, key=lambda kv: kv[1][0])[1]
                # find next header to the right to define column range
                right_neighbors = [xy[0] for n, xy in header_xy.items() if xy[0] > hx]
                right_bound = min(right_neighbors) if right_neighbors else hx + 400
                left_bound = hx - 40
                # candidates: numeric directly below within the column range
                candidates = [b for b in blocks if self._is_amount_text(b["text"]) and b["x"] >= left_bound and b["x"] <= right_bound and b["y"] > hy + 4 and b["y"] - hy < 120]
                if not candidates:
                    return None
                return min(candidates, key=lambda b: b["y"])  # closest below

            # Ending balance value under the header
            end_cell = value_under({"ENDINGBALANCE", "BAKIAKHIR"})
            if end_cell:
                payload = {
                    "original_value": Decimal('0.00'),
                    "bbox": end_cell["bbox"],
                    # Match typical numeric font size used on this summary row (e.g., Interest Paid YTD)
                    "font_size": row_numeric_font_size(end_cell),
                    "font": end_cell.get("font", ""),
                    "page_num": page_num,
                    "y_position": end_cell["bbox"][1]
                }
                # Only one replacement for the closing balance cell to avoid double drawing (bold look)
                self.balance_replacements.append({"type": "ending_balance", **payload})

            # Opening balance value under the header
            open_cell = value_under({"OPENINGBALANCE", "BAKIPEMBUKAAN"})
            if open_cell:
                self.balance_replacements.append({
                    "type": "transaction_balance",  # map to first-row computed opening balance
                    "original_value": Decimal('0.00'),
                    "bbox": open_cell["bbox"],
                    # Use the same row's typical numeric font size so it matches other columns
                    "font_size": row_numeric_font_size(open_cell),
                    "font": open_cell.get("font", ""),
                    "page_num": page_num,
                    "y_position": open_cell["bbox"][1] - 2.0
                })
                self._summary_open_info = {"page_num": page_num, "y": open_cell["bbox"][1]}

            # Fallback: if either cell missing, use the product row (e.g., MAXSAVE)
            if not end_cell or not open_cell:
                # find a reference row containing the long account number
                acct = next((b for b in blocks if re.match(r"^\d{8,}$", b["text"].replace(" ", ""))), None)
                if acct:
                    ry = acct["y"]
                    row_nums = [b for b in blocks if self._is_amount_text(b["text"]) and abs(b["y"] - ry) <= 3.5]
                    # choose by proximity to header x if headers detected
                    if not open_cell:
                        hx_open = None
                        for n, (hx, hy) in header_xy.items():
                            if "OPENINGBALANCE" in n or "BAKIPEMBUKAAN" in n:
                                hx_open = hx
                        if hx_open is not None and row_nums:
                            oc = min(row_nums, key=lambda b: abs(b["x"] - hx_open))
                            self.balance_replacements.append({
                                "type": "transaction_balance",
                                "original_value": Decimal('0.00'),
                                "bbox": oc["bbox"],
                                "font_size": max(10, oc.get("font_size", 12) - 1),
                                "font": oc.get("font", ""),
                                "page_num": page_num,
                                "y_position": oc["bbox"][1] - 2.0
                            })
                            self._summary_open_info = {"page_num": page_num, "y": oc["bbox"][1]}
                    if not end_cell:
                        hx_end = None
                        for n, (hx, hy) in header_xy.items():
                            if "ENDINGBALANCE" in n or "BAKIAKHIR" in n:
                                hx_end = hx
                        if hx_end is not None and row_nums:
                            ec = min(row_nums, key=lambda b: abs(b["x"] - hx_end))
                            payload = {
                                "original_value": Decimal('0.00'),
                                "bbox": ec["bbox"],
                                # Match the summary-row numeric font size here, too
                                "font_size": row_numeric_font_size(ec),
                                "font": ec.get("font", ""),
                                "page_num": page_num,
                                "y_position": ec["bbox"][1]
                            }
                            # Single replacement for the closing balance cell
                            self.balance_replacements.append({"type": "ending_balance", **payload})
        except Exception:
            pass

    # Tighter erase box: keep height short to avoid wiping blue rules
    def _erase_rect(self, page, bbox):
        import fitz
        x0, y0, x1, y1 = bbox
        left_pad = 6.0
        right_inset = 2.0
        # Expand slightly upward and use no bottom inset for short summary cells to prevent overlap
        cell_h = max(0.1, y1 - y0)
        # Nudge erase box DOWN a bit to avoid touching the blue rule above
        bottom_inset = 0.0 if cell_h <= 10.0 else 0.8
        top_inset = 0.8 if cell_h <= 12.0 else 0.0
        rx0 = x0 - left_pad
        rx1 = x1 - right_inset
        ry0 = y0 + top_inset
        ry1 = max(y0 + 1.2, y1 - bottom_inset)
        r = fitz.Rect(rx0, ry0, rx1, ry1)
        pr = page.rect
        r.x0 = max(pr.x0, r.x0); r.y0 = max(pr.y0, r.y0)
        r.x1 = min(pr.x1, r.x1); r.y1 = min(pr.y1, r.y1)
        return r

    # Bottom-up recomputation: start from user input (ending balance)
    def recalculate_balances(self, transactions: List[Dict], beginning_balance: Decimal, log_func: Callable = print) -> List[Dict]:
        try:
            closing = beginning_balance
            ordered = sorted(transactions, key=lambda t: (t['page_num'], t.get('y_position', 0)))
            rev = list(reversed(ordered))
            running = closing
            computed = []
            total_debit = Decimal('0.00'); total_credit = Decimal('0.00')
            for t in rev:
                debit = t.get('debit', Decimal('0.00')) or Decimal('0.00')
                credit = t.get('credit', Decimal('0.00')) or Decimal('0.00')
                nt = dict(t)
                # The balance to display for this row (balance *after* this transaction)
                # is the 'running' balance from the previous iteration.
                nt['new_balance'] = running
                
                # Now calculate the balance *before* this transaction, which will be
                # the 'running' balance for the next iteration (the row above).
                # Rule: for credit, apply - (subtract); for debit, apply + (add)
                if credit > 0:
                    running = (running - credit).quantize(Decimal('0.01'))
                if debit > 0:
                    running = (running + debit).quantize(Decimal('0.01'))
                computed.append(nt)
                total_debit += debit; total_credit += credit
            computed = list(reversed(computed))

            # Insert synthetic first-page opening balance to drive summary mapping
            if self._summary_open_info:
                p = self._summary_open_info['page_num']
                idx = next((i for i, t in enumerate(computed) if t['page_num'] == p), None)
                if idx is not None:
                    opening_val = computed[idx]['new_balance']
                    synthetic = {
                        'date': '', 'description': 'SUMMARY_OPEN',
                        'debit': Decimal('0.00'), 'credit': Decimal('0.00'), 'amount': Decimal('0.00'),
                        'new_balance': opening_val, 'page_num': p, 'y_position': self._summary_open_info['y'] - 2.0
                    }
                    computed.insert(idx, synthetic)

            log_func(f"💰 Closing balance (input): RM {closing:,.2f}")
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
            log_func(f"⚠️ Error recalculating RHB balances: {e}")
            return transactions

    # Override populate to ensure only the very last transaction gets the ending balance
    def _populate_new_values(self, transactions: List[Dict], log_func: Callable = print):
        try:
            # Handle summary replacements (ending balance and C/F balance)
            for rep in self.balance_replacements:
                if rep.get('type') == 'ending_balance':
                    # This is the ending balance - use user input
                    rep['new_value'] = self._format_amount(self.beginning_balance)
                    log_func(f"🔍 RHB ending balance: {rep['new_value']}")
                elif rep.get('type') == 'beginning_balance':
                    # This is the C/F balance - use user input (ending balance)
                    rep['new_value'] = self._format_amount(self.beginning_balance)
                    log_func(f"🔍 RHB C/F balance: {rep['new_value']}")

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
                log_func(f"🔍 RHB page {page_num}: {len(txs)} transactions, {len(reps)} replacements")
                
                for i in range(n):
                    # Check if this is the very last transaction across ALL pages
                    is_last_across_all_pages = (txs[i] == last_transaction)
                    
                    if is_last_across_all_pages:
                        # Only the very last transaction row across all pages gets the ending balance
                        val = self._format_amount(self.beginning_balance)
                        reps[i]['new_value'] = val
                        log_func(f"🔍 RHB last transaction across all pages: {val}")
                    else:
                        # All other rows use calculated values from reverse calculation
                        # This represents the balance AFTER this transaction was applied
                        val = self._format_amount(txs[i]['new_balance'])
                        reps[i]['new_value'] = val
                        log_func(f"🔍 RHB transaction {i}: {val} (balance after transaction)")

            # Handle opening balance (type: transaction_balance) - it should get the first transaction's calculated balance
            for rep in self.balance_replacements:
                if rep.get('type') == 'transaction_balance' and 'new_value' not in rep:
                    # This is likely the opening balance - find the first transaction
                    if transactions:
                        first_tx = min(transactions, key=lambda t: (t['page_num'], t.get('y_position', 0)))
                        rep['new_value'] = self._format_amount(first_tx['new_balance'])
                        log_func(f"🔍 RHB opening balance: {rep['new_value']}")

            # Debug: show final values for all replacements
            log_func(f"🔍 RHB Final replacement values:")
            for i, rep in enumerate(self.balance_replacements):
                log_func(f"🔍 RHB {i}: type={rep.get('type')}, value={rep.get('new_value')}")

        except Exception as e:
            log_func(f"⚠️ Error populating RHB values: {e}")