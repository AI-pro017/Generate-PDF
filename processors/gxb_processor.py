#!/usr/bin/env python3
"""
GXBank (GXB) PDF Statement Processor

Layout highlights (based on sample):
- Top summary with 4 squares: Money in, Money out, Interest earned, Closing balance
  We only replace the rightmost "Closing balance" figure with the user input.
- Transactions table headers: Date, Transaction description, Money in (RM), Money out (RM),
  Interest earned (RM), Closing balance (RM)
  For reverse computation (bottom-up):
    - Treat Money out as debit (+ when walking upward)
    - Treat Money in as credit (- when walking upward)
  So for each row: debit = money_out, credit = money_in.

Critical: Preserve the original font color when writing new numbers.
This processor samples the color from the original number on the same row
and stores it on each replacement. We override PDF writing to use this color.
"""

import re
import os
from decimal import Decimal
from typing import List, Dict, Optional, Callable

import fitz

from .base_processor import BaseProcessor


class GXBProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.bank_name = "GXBank (GXB)"
        self._right_inset = 2.0
        # Fixed brand color requested: #283386
        self._brand_color = (0x28/255.0, 0x33/255.0, 0x86/255.0)

    def _is_amount_text(self, text: str) -> bool:
        if not text:
            return False
        t = text.strip().replace(",", "").replace("RM", "")
        t = t.replace("+", "").replace("-", "")
        if not t:
            return False
        return bool(re.match(r"^\d+(?:\.\d{2})$", t))

    def _norm(self, s: str) -> str:
        return re.sub(r"[^A-Z]", "", (s or "").upper())

    def _span_rgb(self, sp: dict) -> Optional[tuple]:
        try:
            c = sp.get("color")
            if isinstance(c, int):
                # common device gray/black return 0; treat as near-black
                if c == 0:
                    return (0, 0, 0)
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                # already 0..1 floats
                r, g, b = c[0], c[1], c[2]
                # normalize if values > 1
                if r > 1 or g > 1 or b > 1:
                    return (r/255.0, g/255.0, b/255.0)
                return (r, g, b)
        except Exception:
            pass
        return None

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
                            rgb = self._span_rgb(sp)
                            blocks.append({
                                "text": tx,
                                "x": x0,
                                "y": y0,
                                "bbox": [x0, y0, x1, y1],
                                "font_size": sp.get("size", 12),
                                "font": sp.get("font", ""),
                                "color": rgb,
                            })

                blocks.sort(key=lambda b: (b["y"], b["x"]))

                if page_num == 0 and not self._brand_color:
                    self._brand_color = self._detect_brand_color(blocks)

                # Top summary: replace Closing balance only
                self._tag_summary_closing(blocks, page_num + 1)

                # Transaction table
                hdr = self._locate_headers(blocks)
                if not hdr:
                    continue
                x_in, x_out, x_bal = hdr["IN"], hdr["OUT"], hdr["BAL"]
                table_top_y = hdr.get("_TABLE_TOP_Y")

                mid_in_out = (x_in + x_out) / 2.0
                mid_out_bal = (x_out + x_bal) / 2.0

                def col_of(x: float) -> str:
                    if x < mid_in_out:
                        return "IN"
                    if x < mid_out_bal:
                        return "OUT"
                    return "BAL"

                rows: List[List[Dict]] = []
                cur: List[Dict] = []
                last_y = None
                y_tol = 4
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
                    row.sort(key=lambda b: b["x"])  # left -> right
                    if not row:
                        continue
                    row_y = min(b["y"] for b in row)
                    if table_top_y is not None and row_y < table_top_y + 12:
                        continue

                    # Determine date color to enforce for this row (use the left date value)
                    date_color = self._row_date_color(row)

                    in_val = None
                    out_val = None
                    bal_blk = None
                    bal_color = None

                    for b in row:
                        t = b["text"].replace(" ", "")
                        if not self._is_amount_text(t):
                            continue
                        c = col_of(b["x"])
                        raw = b["text"].replace(",", "").replace("RM", "").strip()
                        sgn = 1
                        if raw.startswith("+"):
                            raw = raw[1:]
                        elif raw.startswith("-"):
                            sgn = -1
                            raw = raw[1:]
                        try:
                            num = Decimal(raw)
                        except Exception:
                            continue
                        if c == "IN":
                            in_val = sgn * num
                        elif c == "OUT":
                            out_val = sgn * num
                        else:
                            bal_blk = b
                            bal_color = date_color or b.get("color")

                    # Detect summary band (grey row) mostly numbers and no words
                    has_letters = any(re.search(r"[A-Za-z]", b.get("text", "")) for b in row)
                    amount_count = sum(1 for b in row if self._is_amount_text(b.get("text", "").replace(" ", "")))
                    if not has_letters and amount_count >= 3 and bal_blk is not None:
                        bb = list(bal_blk["bbox"]) ; bb[2] = max(bb[0] + 1.0, bb[2] - self._right_inset)
                        self.balance_replacements.append({
                            "type": "ending_balance",
                            "original_value": Decimal('0.00'),
                            "bbox": bb,
                            "font_size": bal_blk.get("font_size", 12),
                            "font": bal_blk.get("font", ""),
                            "page_num": page_num + 1,
                            "y_position": bb[1],
                            "color": self._brand_color or date_color or bal_color or (0.10, 0.16, 0.70),
                            "is_summary": True,
                        })
                        continue

                    if bal_blk is not None:
                        bb = list(bal_blk["bbox"])
                        bb[2] = max(bb[0] + 1.0, bb[2] - self._right_inset)
                        self.balance_replacements.append({
                            "type": "transaction_balance",
                            "original_value": Decimal('0.00'),
                            "bbox": bb,
                            "font_size": bal_blk.get("font_size", 12),
                            "font": bal_blk.get("font", ""),
                            "page_num": page_num + 1,
                            "y_position": bb[1],
                            "color": date_color or bal_color or self._brand_color or (0.10, 0.16, 0.70),
                        })

                    # Build a transaction row even if no IN/OUT present (e.g., Opening balance)
                    # so that the balance cell still receives a computed value.
                    if in_val is None and out_val is None and bal_blk is None:
                        continue
                    
                    # The transaction amount is already correctly signed from the PDF
                    # in_val and out_val already have the correct signs (+ or -)
                    transaction_amount = (in_val or Decimal('0.00')) + (out_val or Decimal('0.00'))
                    
                    transactions.append({
                        "date": "",
                        "description": "",
                        "debit": Decimal('0.00'),  # Not used in calculation
                        "credit": Decimal('0.00'),  # Not used in calculation
                        "amount": transaction_amount,  # This is the signed transaction amount
                        "new_balance": Decimal('0.00'),
                        "page_num": page_num + 1,
                        "y_position": bal_blk["bbox"][1] if bal_blk else row_y,
                    })

            doc.close()

        except Exception as e:
            log_func(f"❌ Error extracting data from PDF: {e}")
            import traceback
            traceback.print_exc()
            return []

        log_func(f"✅ Extracted {len(transactions)} GXB rows to compute balances")
        return transactions

    def _row_date_color(self, row: List[Dict]) -> Optional[tuple]:
        """Pick a dark blue from the date column. If multiple date/time colors exist,
        choose the bluest and darker one. If the sampled color is too light/grey,
        clamp to a canonical GX dark blue.
        """
        try:
            candidates = []
            # Gather colors of the leftmost few spans (date/time column)
            left_x = min(b["x"] for b in row) if row else None
            for b in row:
                if left_x is None:
                    break
                if b["x"] - left_x > 160:  # stay within date column band
                    continue
                col = b.get("color")
                if not col:
                    continue
                r, g, bch = col
                blue_score = bch - max(r, g)
                luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                candidates.append((blue_score, -luma, col))
            if candidates:
                # Highest blue score; if tie, darker (lower luma)
                candidates.sort(reverse=True)
                r, g, bch = candidates[0][2]
                # If too light/grey, snap to canonical dark blue
                luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                if luma > 0.55 or bch < max(r, g) + 0.02:
                    return (0.10, 0.16, 0.70)
                return (r, g, bch)
            # Fallback: leftmost span color
            left = min(row, key=lambda x: x["x"]) if row else None
            if left and left.get("color"):
                r, g, bch = left.get("color")
                luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                if luma > 0.55 or bch < max(r, g) + 0.02:
                    return (0.10, 0.16, 0.70)
                return left.get("color")
        except Exception:
            pass
        return (0.10, 0.16, 0.70)

    def _detect_brand_color(self, blocks: List[Dict]) -> Optional[tuple]:
        try:
            # focus on the top 25% of page height using y threshold from sample (~300)
            tops = [b for b in blocks if b.get('y', 0) < 300 and b.get('color')]
            scores = []
            for b in tops:
                r,g,bch = b['color']
                blue = bch - max(r,g)
                luma = 0.2126*r + 0.7152*g + 0.0722*bch
                if blue > 0.06 and luma < 0.75:
                    scores.append((blue, -luma, b['color']))
            if scores:
                scores.sort(reverse=True)
                return scores[0][2]
            return None
        except Exception:
            return None

    def _locate_headers(self, blocks: List[Dict]) -> Optional[Dict[str, float]]:
        xs, ys = {}, {}
        # First try direct header text matching
        for b in blocks:
            t = self._norm(b["text"]) if b.get("text") else ""
            if t in ("MONEYINRM", "MONEYIN(RM)", "MONEYIN", "DUITMASUK"):
                xs["IN"], ys["IN"] = b["x"], b["y"]
            elif t in ("MONEYOUTRM", "MONEYOUT(RM)", "MONEYOUT", "DUITKELUAR"):
                xs["OUT"], ys["OUT"] = b["x"], b["y"]
            elif t in ("CLOSINGBALANCERM", "CLOSINGBALANCE(RM)", "CLOSINGBALANCE", "BAKIPENUTUP"):
                xs["BAL"], ys["BAL"] = b["x"], b["y"]
            elif t in ("DATE", "TARIKH"):
                ys["DATEHDR"] = b["y"]
        if all(k in xs for k in ("IN", "OUT", "BAL")):
            # table top around the header band; push a bit lower to avoid skipping first row
            # prefer DATE header if found
            top = ys.get("DATEHDR", None)
            if top is None:
                numeric_y_values = [ys[k] for k in ys if k in ("IN", "OUT", "BAL")]
                top = min(numeric_y_values) if numeric_y_values else None
            if top is not None:
                top = top + 2
            xs["_TABLE_TOP_Y"] = top
            return xs
        # Fallback: infer columns from numeric clusters (rightmost three x positions)
        numeric_blocks = [b for b in blocks if self._is_amount_text(b.get("text", ""))]
        if len(numeric_blocks) < 8:
            return None
        # Optional table top via DATE/TARIKH
        date_hdr_y = next((b["y"] for b in blocks if self._norm(b.get("text", "")) in {"DATE","TARIKH"}), None)
        # cluster xs
        xs_all = sorted(b["x"] for b in numeric_blocks)
        groups: List[float] = []
        tol = 40.0
        for x in xs_all:
            if not groups or abs(x - groups[-1]) > tol:
                groups.append(x)
        if len(groups) < 3:
            # try coarser tolerance
            groups = []
            tol = 60.0
            for x in xs_all:
                if not groups or abs(x - groups[-1]) > tol:
                    groups.append(x)
        if len(groups) < 3:
            return None
        # take the last three groups as IN, OUT, BAL from left to right
        cols = sorted(groups[-3:])
        inferred = {"IN": cols[0], "OUT": cols[1], "BAL": cols[2]}
        # table top: min y of rows containing any of these xs, or DATE header y
        top_candidates = [b["y"] for b in numeric_blocks if any(abs(b["x"] - inferred[k]) < tol for k in ("IN","OUT","BAL"))]
        top = min(top_candidates) if top_candidates else None
        if date_hdr_y is not None:
            top = date_hdr_y
        if top is not None:
            top = top + 2
        inferred["_TABLE_TOP_Y"] = top
        return inferred

    def _tag_summary_closing(self, blocks: List[Dict], page_num: int):
        # Robust search for the summary "Closing balance" value in the top cards
        labels = [b for b in blocks if self._norm(b.get("text")) in {"CLOSINGBALANCE"}]
        target = None
        if labels:
            lbl = max(labels, key=lambda b: b["x"])  # rightmost card label
            candidates = [b for b in blocks if b["x"] >= lbl["x"] - 260 and b["x"] <= lbl["x"] + 360 and 0 < (lbl["y"] - b["y"]) < 120 and self._is_amount_text(b.get("text","")) ]
            if candidates:
                target = max(candidates, key=lambda b: b["x"])  # rightmost numeric in card
        if target is None:
            # fallback: data above the table headers
            date_hdr = next((b for b in blocks if self._norm(b.get("text")) in {"DATE","TARIKH"}), None)
            if date_hdr:
                candidates = [b for b in blocks if b["y"] < date_hdr["y"] - 10 and date_hdr["y"] - b["y"] < 500 and self._is_amount_text(b.get("text",""))]
                if candidates:
                    target = max(candidates, key=lambda b: b["x"])  # far right
        if target is None:
            return
        bb = list(target["bbox"]) ; bb[2] = max(bb[0] + 1.0, bb[2] - self._right_inset)
        self.balance_replacements.append({
            "type": "ending_balance",
            "original_value": Decimal('0.00'),
            "bbox": bb,
            "font_size": target.get("font_size", 14),
            "font": target.get("font", ""),
            "page_num": page_num,
            "y_position": bb[1],
            "color": target.get("color") or self._brand_color or (0.10, 0.16, 0.70),
            "is_card": True,
        })

    # Reverse recomputation (closing balance input)
    def recalculate_balances(self, transactions: List[Dict], beginning_balance: Decimal, log_func: Callable = print) -> List[Dict]:
        try:
            closing = beginning_balance
            ordered = sorted(transactions, key=lambda t: (t['page_num'], t.get('y_position', 0)))
            
            # Start with the closing balance and work backwards through transactions
            # For each transaction: previous_balance = current_balance - transaction_effect
            # Money out (debit) increases balance, Money in (credit) decreases balance
            running_balance = closing
            computed: List[Dict] = []
            
            # Process from last transaction to first (reverse order)
            for t in reversed(ordered):
                # Set this transaction's balance
                t['new_balance'] = running_balance
                
                # Calculate the previous balance
                # The amount field already contains the signed transaction amount
                # For reverse calculation:
                # If this transaction was -14.30 (negative), previous balance = current + 14.30
                # If this transaction was +15.00 (positive), previous balance = current - 15.00
                # So: previous = current - transaction_amount
                transaction_amount = t.get('amount', Decimal('0.00'))
                running_balance = (running_balance - transaction_amount).quantize(Decimal('0.01'))
                
                computed.append(t)
            
            # Reverse back to original order
            computed.reverse()
            log_func(f"💰 Closing balance (input): RM {closing:,.2f}")
            return computed
        except Exception as e:
            log_func(f"⚠️ Error recalculating GXB balances: {e}")
            return transactions

    # Populate with preserved color
    def _populate_new_values(self, transactions: List[Dict], log_func: Callable = print):
        try:
            # Ending balance square = fixed input; also recycle that value to first table balance
            card_value = self.beginning_balance
            for rep in self.balance_replacements:
                if rep.get('type') == 'ending_balance':
                    rep['new_value'] = self._format_amount(card_value)

            # per-row balances - need to find the very last transaction across all pages
            from collections import defaultdict
            by_page = defaultdict(list)
            for t in transactions:
                by_page[t['page_num']].append(t)
            reps_by_page = defaultdict(list)
            for rep in self.balance_replacements:
                if rep.get('type') == 'transaction_balance':
                    reps_by_page[rep['page_num']].append(rep)
            
            # Find the very last transaction across all pages
            all_transactions = sorted(transactions, key=lambda t: (t['page_num'], t.get('y_position', 0)))
            last_transaction = all_transactions[-1] if all_transactions else None
            
            for p, txs in by_page.items():
                reps = sorted(reps_by_page.get(p, []), key=lambda r: r.get('y_position', 0))
                n = min(len(txs), len(reps))
                
                if n > 0 and reps:
                    for i in range(n):
                        if i == n - 1 and txs[i] == last_transaction:
                            # Only the very last transaction row across all pages gets the card value
                            reps[i]['new_value'] = self._format_amount(card_value)
                            reps[i]['color'] = self._brand_color or reps[i].get('color')
                        else:
                            # All other rows use calculated values
                            reps[i]['new_value'] = self._format_amount(txs[i]['new_balance'])
                else:
                    # Fallback: use calculated values
                    for i in range(n):
                        reps[i]['new_value'] = self._format_amount(txs[i]['new_balance'])
                
                # Ensure summary band (if present) shows card value and shares card styling
                for rep in reps:
                    if rep.get('is_summary'):
                        rep['new_value'] = self._format_amount(card_value)
                        rep['color'] = rep.get('color')  # Keep original color for summary
            # Fallback: ensure all have values
            last_val = self._format_amount(transactions[-1]['new_balance']) if transactions else self._format_amount(self.beginning_balance)
            for rep in self.balance_replacements:
                if 'new_value' not in rep or rep['new_value'] is None:
                    rep['new_value'] = last_val
        except Exception as e:
            log_func(f"⚠️ Error populating GXB values: {e}")

    # Override PDF writing to honor color
    def generate_updated_pdf(self, transactions: List[Dict], output_path: str, log_func: Callable = print):
        try:
            self._populate_new_values(transactions, log_func)
            original_pdf = fitz.open(self.original_pdf_path)
            new_pdf = fitz.open()

            # Copy each page and draw
            for page_num in range(len(original_pdf)):
                op = original_pdf.load_page(page_num)
                np = new_pdf.new_page(width=op.rect.width, height=op.rect.height)
                np.show_pdf_page(np.rect, original_pdf, page_num)

                page_reps = [r for r in self.balance_replacements if r['page_num'] == page_num + 1]

                # ERASE
                for rep in page_reps:
                    erase_r = self._erase_rect(np, rep['bbox'])
                    bg = self._bg_for(np, rep['bbox'], rep.get('bg_color'))
                    np.add_redact_annot(erase_r, fill=bg)
                np.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                # DRAW with preserved color (bold); align card value to card label left edge
                for rep in page_reps:
                    font_size = rep.get('font_size', 12)
                    text = str(rep['new_value'])
                    if rep.get('is_card') and not text.startswith('RM'):
                        text = 'RM' + text
                    char_w = font_size * 0.5
                    text_w = len(text) * char_w
                    if rep.get('is_card'):
                        # align to label's left start (more like Money in/out cards)
                        x = rep.get('left_x', rep['bbox'][0])
                    else:
                        x = rep['bbox'][2] - text_w
                    y = rep['bbox'][1] + font_size * 0.8
                    color = self._brand_color or rep.get('color') or (0, 0, 0)
                    # If balance color is near-black, attempt sampling from original page region
                    if color == (0,0,0) or sum(color) < 0.1:
                        try:
                            clip = fitz.Rect(rep['bbox'])
                            pix = op.get_pixmap(clip=clip, alpha=False)
                            data, n = pix.samples, pix.n
                            R=G=B=cnt=0
                            for i in range(0, len(data), n):
                                r8,g8,b8 = data[i], data[i+1], data[i+2]
                                if r8+g8+b8 < 120:  # ignore dark ink
                                    continue
                                R+=r8; G+=g8; B+=b8; cnt+=1
                            if cnt:
                                color = (R/cnt/255.0, G/cnt/255.0, B/cnt/255.0)
                        except Exception:
                            pass
                    # Final clamp: if sampled color still looks light/grey, snap to brand color
                    if color:
                        r,g,bch = color
                        blue = bch - max(r,g)
                        luma = 0.2126*r + 0.7152*g + 0.0722*bch
                        if blue < 0.05 or luma > 0.65:
                            color = self._brand_color or (0.10, 0.16, 0.70)
                    # Emulate bold. Cards need stronger weight than rows.
                    if rep.get('is_card') or rep.get('is_summary'):
                        # Four-pass draw for thicker appearance like native card values
                        np.insert_text(fitz.Point(x, y), text, fontsize=font_size, color=color, fontname="helv")
                        np.insert_text(fitz.Point(x + 0.22, y), text, fontsize=font_size, color=color, fontname="helv")
                        np.insert_text(fitz.Point(x + 0.11, y + 0.12), text, fontsize=font_size, color=color, fontname="helv")
                        np.insert_text(fitz.Point(x + 0.11, y - 0.12), text, fontsize=font_size, color=color, fontname="helv")
                    else:
                        # Two-pass for table rows
                        np.insert_text(fitz.Point(x, y), text, fontsize=font_size, color=color, fontname="helv")
                        np.insert_text(fitz.Point(x + 0.15, y), text, fontsize=font_size, color=color, fontname="helv")

            new_pdf.save(output_path)
            new_pdf.close(); original_pdf.close()
            log_func(f"✅ Updated PDF saved to: {output_path}")
        except Exception as e:
            log_func(f"❌ Error generating updated PDF (GXB): {e}")
            import traceback
            traceback.print_exc()

    # Minimal stubs for abstract API (not used directly)
    def _extract_transactions_and_balances(self, text_dict: dict, page_num: int, log_func: Callable) -> List[Dict]:
        return []

    def _find_balance_column_position(self, all_blocks: List[Dict], log_func: Callable) -> float:
        hdr = self._locate_headers(all_blocks)
        if hdr and "BAL" in hdr:
            return hdr["BAL"]
        return 0.0

    def _parse_transaction_row(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable) -> Optional[Dict]:
        return None


