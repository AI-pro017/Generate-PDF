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
        # computed opening balance after reverse calculation
        self._computed_opening_balance: Optional[Decimal] = None
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

    def _preferred_row_font_size(self, row, col_of) -> float:
        """Return a font size matching other amount columns in this row."""
        sizes = []
        for b in row:
            tx = b.get("text", "")
            if self._clean_num_text(tx) is None:
                continue
            # Prefer Deposit/Withdrawal columns
            if col_of(b["x"]) != "BALANCE":
                sizes.append(b.get("font_size", 12))
        if sizes:
            return min(sizes)  # keep it slightly smaller if there’s any mismatch

        # Fallback: any numeric in the row
        for b in row:
            if self._clean_num_text(b.get("text", "")) is not None:
                sizes.append(b.get("font_size", 12))
        return min(sizes) if sizes else 12

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

                row_fs = self._preferred_row_font_size(row, col_of)

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
                        "bbox": balance_blk["bbox"], "font_size": row_fs,
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
                        "bbox": balance_blk["bbox"], "font_size": row_fs,
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

    # Override erase box just for HLB so we don't wipe the right table rule
    def _erase_rect(self, page, bbox):
        """Return a tighter erase rectangle for HLB numbers.

        Requirements:
        - Do not increase height (avoid touching horizontal rules)
        - Pull the right edge slightly left so the vertical border line remains visible
        - Keep a small left padding to fully cover any light background behind digits
        """
        import fitz
        x0, y0, x1, y1 = bbox
        left_pad = 6.0
        right_inset = 2.0  # stay this far left of the balance column edge

        # Keep height unchanged; only adjust width/position
        rx0 = x0 - left_pad
        rx1 = x1 - right_inset

        # Ensure at least 1pt width
        if rx1 <= rx0 + 1.0:
            rx1 = rx0 + 1.0

        r = fitz.Rect(rx0, y0, rx1, y1)

        # Clamp to page bounds
        pr = page.rect
        r.x0 = max(pr.x0, r.x0)
        r.y0 = max(pr.y0, r.y0)
        r.x1 = min(pr.x1, r.x1)
        r.y1 = min(pr.y1, r.y1)
        return r

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

    # Reverse recomputation: treat user input as closing balance and compute upwards
    def recalculate_balances(self, transactions: List[Dict], beginning_balance: Decimal, log_func: Callable = print) -> List[Dict]:
        try:
            closing_balance = beginning_balance
            log_func(f"💰 Closing balance (input): RM {closing_balance:,.2f}")
            log_func("=" * 90)
            log_func(f"{'#':<3} {'Date':<10} {'Description':<40} {'Amount':>12} {'New Balance':>15}")
            log_func("=" * 90)

            running = closing_balance
            for t in reversed(transactions):
                amt = t.get('amount', Decimal('0.00')) or Decimal('0.00')
                # The balance printed on this row is AFTER applying its net change
                t['new_balance'] = running
                # Move upward: undo this row's change
                running = (running - amt).quantize(Decimal('0.01'))

            self._computed_opening_balance = running

            for i, t in enumerate(transactions, 1):
                desc = (t.get('description') or '')
                if len(desc) > 37:
                    desc = desc[:37] + '...'
                amt = t.get('amount', Decimal('0.00')) or Decimal('0.00')
                log_func(f"{i:<3} {t.get('date',''):<10} {desc:<40} {amt:>12,.2f} {t['new_balance']:>15,.2f}")

            log_func("=" * 90)
            log_func(f"🟡 Computed opening balance: RM {self._computed_opening_balance:,.2f}")
            return transactions
        except Exception as e:
            log_func(f"⚠️ Error recalculating HLB balances: {e}")
            return transactions

    # Override to place values in summary according to reverse logic
    def _populate_new_values(self, transactions: List[Dict], log_func: Callable = print):
        # Use base logic for mapping per-row balances and totals first
        super()._populate_new_values(transactions, log_func)
        try:
            # Then override summary placements
            for rep in self.balance_replacements:
                t = rep.get('type')
                if t == 'ending_balance':
                    # Set closing balance to the user input
                    rep['new_value'] = self._format_amount(self.beginning_balance)
                elif t == 'beginning_balance':
                    # If any beginning balance label exists, use computed opening
                    opening = (self._computed_opening_balance
                               if self._computed_opening_balance is not None
                               else (transactions[0]['new_balance'] if transactions else self.beginning_balance))
                    rep['new_value'] = self._format_amount(opening)
        except Exception as e:
            log_func(f"⚠️ Error overriding HLB summary values: {e}")

    def generate_updated_pdf(self, transactions: List[Dict], output_path: str, log_func: Callable = print):
        log_func(" Generating updated PDF (HLB style)...")
        # Fill new values first
        self._populate_new_values(transactions, log_func)

        import fitz, os
        orig = fitz.open(self.original_pdf_path)
        out  = fitz.open()

        # copy metadata
        self._copy_metadata(orig, out, log_func)

        # slightly grey text color to match original HLB numbers
        hlb_text_color = (0.20, 0.20, 0.20)
        # pure black for ending balance to make it truly bold like the Date
        ending_balance_color = (0, 0, 0)

        for page_idx in range(len(orig)):
            src = orig.load_page(page_idx)
            dst = out.new_page(width=src.rect.width, height=src.rect.height)
            dst.show_pdf_page(dst.rect, orig, page_idx)

            reps = [r for r in self.balance_replacements if r['page_num'] == page_idx + 1]
            log_func(f" Page {page_idx+1}: {len(reps)} balance positions")

            # erase
            for rep in reps:
                erase_r = self._erase_rect(dst, rep['bbox'])
                bg = self._bg_for(dst, rep['bbox'], rep.get('bg_color'))
                dst.add_redact_annot(erase_r, fill=bg)
            dst.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

            # draw (right-aligned)
            for rep in reps:
                fs   = rep.get('font_size', 12)
                text = str(rep['new_value'])
                
                # Use exact same style as Date row for ending balance
                if rep.get('type') == 'ending_balance':
                    # Make it bold and align exactly like the Date above it
                    bbox = rep['bbox']
                    char_w = fs * 0.5
                    x = bbox[2] - len(text) * char_w
                    y = bbox[1] + fs * 0.8
                    
                    # Fix right alignment to match the Date exactly
                    # The ending balance should have the same right edge as the Date
                    # Move it to the right by reducing the left offset
                    x = x - 2  # Reduced from -6 to -2 to move it further right
                    
                    # Try different font families that might match the original HLB bold style
                    try:
                        # Try Arial-Bold first (often used in statements)
                        dst.insert_text(fitz.Point(x, y), text, fontsize=fs, color=ending_balance_color, fontname="Arial-Bold")
                        log_func(f"🔍 Drawing ending balance '{text}' with Arial-Bold font (size: {fs}, pure black, pos: {x:.1f}, {y:.1f})")
                    except:
                        try:
                            # Fallback to Helvetica-Bold
                            dst.insert_text(fitz.Point(x, y), text, fontsize=fs, color=ending_balance_color, fontname="Helvetica-Bold")
                            log_func(f"🔍 Drawing ending balance '{text}' with Helvetica-Bold font (size: {fs}, pure black, pos: {x:.1f}, {y:.1f})")
                        except:
                            # Final fallback to helv with pure black
                            dst.insert_text(fitz.Point(x, y), text, fontsize=fs, color=ending_balance_color, fontname="helv")
                            log_func(f"🔍 Drawing ending balance '{text}' with helv font (size: {fs}, pure black, pos: {x:.1f}, {y:.1f})")
                else:
                    # Normal style for transaction rows
                    char_w = fs * 0.5
                    x = rep['bbox'][2] - len(text) * char_w
                    y = rep['bbox'][1] + fs * 0.8
                    dst.insert_text(fitz.Point(x, y), text, fontsize=fs, color=hlb_text_color, fontname="helv")
                    log_func(f"🔍 Drawing transaction balance '{text}' with normal style")

        out.save(output_path)
        out.close()
        orig.close()
        log_func(f"✅ Updated PDF saved to: {output_path}")