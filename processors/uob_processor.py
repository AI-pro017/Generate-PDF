#!/usr/bin/env python3
"""
UOB Bank PDF Statement Processor
Parses UOB summary tables and transaction table; preserves background by sampling page colors.
"""

import re
import os
from decimal import Decimal
from typing import List, Dict, Optional, Callable
import fitz

from .base_processor import BaseProcessor

# Canonical UOB row colors
UOB_BLUE  = (203/255.0, 215/255.0, 228/255.0)  # #CBD7E4
UOB_WHITE = (1.0, 1.0, 1.0)

def _clamp_rect_to_page(page, rect):
    pr = page.rect
    x0, y0, x1, y1 = rect
    return [
        max(pr.x0, x0),
        max(pr.y0, y0),
        min(pr.x1, x1),
        min(pr.y1, y1),
    ]

class UOBProcessor(BaseProcessor):
    """UOB Bank-specific PDF statement processor"""

    def __init__(self):
        super().__init__()
        self.bank_name = "UOB Bank"
        self._uob_balance_right_edge = None
        self._carry = Decimal('0.00')

    # UOB formats: "01 Apr", "01 Apr 2025"
    def _is_date(self, text: str) -> bool:
        t = text.strip()
        return bool(re.match(r'^\d{2}\s*[A-Za-z]{3}(?:\s*\d{4})?$', t))

    def _uob_expand_bbox(self, page, bbox, left_pad=12, right_pad=6, bottom_pad=2):
        """Expand to fully cover the light-grey capsule behind the number."""
        x0, y0, x1, y1 = bbox
        rect = [x0 - left_pad, y0, x1 + right_pad, y1 + bottom_pad]
        return _clamp_rect_to_page(page, rect)

    def _uob_bg_color(self, page, bbox):
        """Sample a tiny strip in the cell and snap to UOB_BLUE / UOB_WHITE."""
        import fitz, math
        x0, y0, x1, y1 = bbox
        w  = max(4, (x1 - x0) * 0.12)
        sx0 = min(x0 + 3, x1 - w - 1); sx1 = sx0 + w
        sy0 = y0 + (y1 - y0) * 0.32;   sy1 = y1 - (y1 - y0) * 0.28
        r = fitz.Rect(sx0, sy0, sx1, sy1)
        pix = page.get_pixmap(clip=r, alpha=False)
        data, n = pix.samples, pix.n
        R = G = B = cnt = 0
        for i in range(0, len(data), n):
            r8, g8, b8 = data[i], data[i+1], data[i+2]
            if r8 + g8 + b8 < 120:  # ignore ink
                continue
            R += r8; G += g8; B += b8; cnt += 1
        if not cnt:
            return UOB_WHITE
        rN, gN, bN = R/cnt/255.0, G/cnt/255.0, B/cnt/255.0
        return UOB_BLUE if math.dist((rN,gN,bN), UOB_BLUE) < math.dist((rN,gN,bN), UOB_WHITE) else UOB_WHITE

    def _uob_erase_bbox(self, page, anchor_right, y0, y1, orig_bbox):
        """
        Tight erase rect for UOB number 'chip':
        - width: original number width + 4 (cap 24..34 px)
        - right edge: 1.6 px left of the balance column
        - vertical: shrink from top/bottom so rules remain intact
        """
        # width based on original number width
        num_w = max(6.0, (orig_bbox[2] - orig_bbox[0]))
        chip_w = max(24.0, min(num_w + 4.0, 34.0))  # 24..34 px

        # stay left of the column edge
        x1 = anchor_right - 1.6
        x0 = x1 - chip_w

        # keep off the rules
        top_margin    = 0.6
        bottom_margin = 0.8
        ty0 = y0 + top_margin
        ty1 = y1 - bottom_margin

        rect = [x0, ty0, x1, ty1]
        return _clamp_rect_to_page(page, rect)

    def _append_uob(self, *, label, typ, bbox, page, page_num, original_value, font_size, font, log_func):
        # bbox = original number bbox (from PDF) – use this for anchoring
        anchor_right = min(bbox[2], page.rect.x1 - 2)
        # tight, clamped erase rect (you already have _uob_erase_bbox)
        erase_bbox   = self._uob_erase_bbox(page, anchor_right, bbox[1], bbox[3], bbox)
        # pick bg from the whole row band, not from the chip
        bg           = self._uob_row_bg_color(page, bbox[1], bbox[3])  # ← use vector band

        self.balance_replacements.append({
            "type": typ, "original_value": original_value,
            "bbox": erase_bbox,
            "anchor_right": anchor_right, "text_y0": bbox[1], "text_y1": bbox[3],
            "font_size": font_size, "font": font or "helv",
            "page_num": page_num, "y_position": bbox[1], "bg_color": bg
        })
        log_func(f"[UOB] {label} p{page_num} bg=({bg[0]:.3f},{bg[1]:.3f},{bg[2]:.3f}) erase={erase_bbox} anchor={anchor_right:.1f}")
    def extract_transactions_from_pdf(self, pdf_path: str, log_func: Callable = print) -> List[Dict]:
        """Extract transaction data from UOB PDF format"""
        self.original_pdf_path = pdf_path
        self.balance_replacements = []
        self._uob_balance_right_edge = None
        self._carry = Decimal('0.00')

        transactions: List[Dict] = []

        try:
            pdf_document = fitz.open(pdf_path)

            log_func(f"📄 Processing {self.bank_name} PDF: {os.path.basename(pdf_path)}")
            log_func(f"📊 Total pages: {len(pdf_document)}")

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text_dict = page.get_text("dict")

                # Page 1: summary tables
                if page_num == 0:
                    self._find_uob_summaries(text_dict, page, page_num + 1, log_func)

                # Transactions on any page that has headers
                page_transactions = self._extract_transactions_and_balances(text_dict, page, page_num + 1, log_func)
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

    # ---------- Utilities ----------

    def _is_amount_text(self, text: str) -> bool:
        t = text.strip()
        if "." not in t:
            return False
        return bool(re.match(r'^-?\d{1,3}(,\d{3})*(\.\d{2})$|-?\d+\.\d{2}$', t))

    def _clean_num_text(self, text: str) -> Optional[Decimal]:
        if not self._is_amount_text(text):
            return None
        t = text.replace(',', '').strip()
        try:
            return Decimal(t)
        except Exception:
            return None

    def _flatten_blocks(self, text_dict: dict) -> List[Dict]:
        blocks = []
        for b in text_dict.get("blocks", []):
            if "lines" in b:
                for ln in b["lines"]:
                    for sp in ln["spans"]:
                        tx = sp["text"].strip()
                        if not tx:
                            continue
                        x0, y0, x1, y1 = sp["bbox"]
                        blocks.append({
                            "text": tx, "x": x0, "y": y0, "bbox": [x0, y0, x1, y1],
                            "font_size": sp.get("size", 12), "font": sp.get("font", "")
                        })
        blocks.sort(key=lambda b: (b["y"], b["x"]))
        return blocks

    def _locate_tx_headers(self, blocks: List[Dict]) -> Optional[Dict[str, float]]:
        """Find x-positions for Withdrawals, Deposits, Balance headers on transaction table."""
        xs = {}
        ys = {}
        for b in blocks:
            t = b["text"].upper().strip()
            if t in ("WITHDRAWALS", "PENGELUARAN", "(-)"):
                xs["WITHDRAWAL"] = b["x"]; ys["WITHDRAWAL"] = b["y"]
            elif t in ("DEPOSITS", "DEPOSIT", "(+)"):
                xs["DEPOSIT"] = b["x"]; ys["DEPOSIT"] = b["y"]
            elif t in ("BALANCE", "BAKI"):
                xs["BALANCE"] = b["x"]; ys["BALANCE"] = b["y"]
        if all(k in xs for k in ("WITHDRAWAL", "DEPOSIT", "BALANCE")):
            xs["_TABLE_TOP_Y"] = min(ys.values()) if ys else None
            return xs
        return None

    def _group_rows_tight(self, blocks: List[Dict]) -> List[List[Dict]]:
        rows, cur, last_y = [], [], None
        y_tol = 3
        for b in blocks:
            if last_y is None or abs(b["y"] - last_y) <= y_tol:
                cur.append(b)
            else:
                if cur: rows.append(cur)
                cur = [b]
            last_y = b["y"]
        if cur: rows.append(cur)
        return rows

    def _group_rows(self, blocks, y_tol=3):
        rows, cur, last = [], [], None
        for b in blocks:
            if last is None or abs(b["y"] - last) <= y_tol:
                cur.append(b)
            else:
                if cur: rows.append(cur)
                cur = [b]
            last = b["y"]
        if cur: rows.append(cur)
        return rows

    # ---------- Page 1 summaries ----------

    def _find_uob_summaries(self, text_dict, page, page_num, log_func):
        blocks = self._flatten_blocks(text_dict)
        self._tag_overview_amount(blocks, page, page_num, log_func)

        def first_y(*labels):
            labs = {s.upper() for s in labels}
            ys = [b["y"] for b in blocks if b["text"].strip().upper() in labs]
            return min(ys) if ys else None

        y_overview = first_y("ACCOUNT OVERVIEW AS AT", "GAMBARAN KESELURUHAN AKAUN PADA")
        y_depos_hdr = first_y("DEPOSIT", "DEPOSITS")         # section title (pill)
        y_interest  = first_y("ONE ACCOUNT INTEREST OVERVIEW^",
                          "GAMBARAN KESELURUHAN FAEDAH ONE ACCOUNT^") or 9e9

        # find Amount header x inside first table
        amt_hdr = next((b for b in blocks
                    if b["text"].strip().upper() in {"AMOUNT (RM)", "AMAUN (RM)"}
                    and (y_overview is None or b["y"] >= y_overview-2)
                    and (y_depos_hdr is None or b["y"] < y_depos_hdr-2)), None)
        amount_x = amt_hdr["x"] if amt_hdr else None

        rows = self._group_rows(blocks)

        # 1) FIRST TABLE (Account Overview) – fallback: rightmost amount in the block
        if y_overview is not None and y_depos_hdr is not None:
            first_tbl_nums = [c for c in blocks
                              if (c["y"] > y_overview + 2) and (c["y"] < y_depos_hdr - 2)
                              and self._is_amount_text(c["text"])]
            log_func(f"[UOB] Overview candidates: {len(first_tbl_nums)}")
            if first_tbl_nums:
                tgt = max(first_tbl_nums, key=lambda c: c["x"])  # rightmost numeric in the block
                self._append_uob(
                    label="Overview amount",
                    typ="ending_balance",
                    bbox=tgt["bbox"],
                    page=page, page_num=page_num,
                    original_value=self._clean_num_text(tgt["text"]),
                    font_size=tgt.get("font_size", 12),
                    font=tgt.get("font", "helv"),
                    log_func=log_func
                )
                log_func(f"📍 UOB Overview amount -> {tgt['bbox']}")

        # 2) SECOND TABLE (Deposits section): replace last 'Balance/Baki' column values
        bal_hdr = next((b for b in blocks
                    if b["text"].strip().upper() in {"BALANCE", "BAKI"}
                    and y_depos_hdr is not None and b["y"] >= y_depos_hdr-2
                    and b["y"] < y_interest-2), None)
        if bal_hdr:
            bal_x, y_hdr = bal_hdr["x"], bal_hdr["y"]
            for row in rows:
                y = min(b["y"] for b in row)
                if not (y > y_hdr+2 and y < y_interest-2):
                    continue
                # choose numeric in this row closest to Balance column
                nums = [b for b in row if self._is_amount_text(b["text"])]
                if not nums:
                    continue
                tgt = min(nums, key=lambda b: abs(b["x"] - bal_x))
                self._append_uob(
                    label="Deposits table balance",
                    typ="ending_balance",
                    bbox=tgt["bbox"],
                    page=page, page_num=page_num,
                    original_value=self._clean_num_text(tgt["text"]),
                    font_size=tgt.get("font_size", 12),
                    font=tgt.get("font", ""),
                    log_func=log_func
                )
            log_func(" UOB Deposits table Balance column tagged")

    # ---------- Transactions table ----------

    def _extract_transactions_and_balances(self, text_dict: dict, page, page_num: int, log_func: Callable) -> List[Dict]:
        """Extract transactions and record balance bboxes (row has visible balance)"""
        transactions: List[Dict] = []

        try:
            blocks = self._flatten_blocks(text_dict)
            headers = self._locate_tx_headers(blocks)
            if not headers:
                return transactions

            dep_x = headers["DEPOSIT"]; wdr_x = headers["WITHDRAWAL"]; bal_x = headers["BALANCE"]
            table_top_y = headers.get("_TABLE_TOP_Y", None)

            # fixed bands
            midDW = (dep_x + wdr_x) / 2.0
            midWB = (wdr_x + bal_x) / 2.0

            def col_of(x):
                if x < midDW: return "DEPOSIT"
                if x < midWB: return "WITHDRAWAL"
                return "BALANCE"

            rows = self._group_rows_tight(blocks)
            pending = self._carry

            for row in rows:
                row.sort(key=lambda b: b["x"])
                if not row:
                    continue

                row_y = min(b["y"] for b in row)
                if table_top_y is not None and row_y < table_top_y + 5:
                    continue

                row_text = " ".join(b["text"] for b in row).upper()
                # skip header/summary lines
                if any(k in row_text for k in ["TOTAL", "JUMLAH", "DATE", "TARIKH", "ACCOUNT", "DETAILS"]):
                    # If this summary row has a balance value, tag it to final ending balance
                    bal = next((b for b in row if col_of(b["x"]) == "BALANCE" and self._is_amount_text(b["text"])), None)
                    if bal:
                        self._append_uob(
                            label="Totals balance",
                            typ="ending_balance",
                            bbox=bal["bbox"],
                            page=page, page_num=page_num,
                            original_value=self._clean_num_text(bal["text"]),
                            font_size=bal.get("font_size", 12),
                            font=bal.get("font", ""),
                            log_func=log_func
                        )
                    continue

                is_opening = "BALANCE B/F" in row_text or "BALANCE B F" in row_text

                date_blk = next((b for b in row if self._is_date(b["text"])), None)

                deposit_val = None; withdraw_val = None; balance_blk = None
                for b in row:
                    if not self._is_amount_text(b["text"]):
                        continue
                    c = col_of(b["x"])
                    num = self._clean_num_text(b["text"])
                    if num is None:
                        continue
                    if c == "DEPOSIT":
                        deposit_val = num
                    elif c == "WITHDRAWAL":
                        withdraw_val = num
                    else:
                        balance_blk = b

                if is_opening and balance_blk is not None:
                    # opening balance → amount 0, we replace the visible balance with beginning_balance
                    transactions.append({
                        "date": date_blk["text"] if date_blk else "",
                        "description": "BALANCE B/F",
                        "amount": Decimal('0.00'),
                        "original_balance": Decimal('0.00'),
                        "new_balance": Decimal('0.00'),
                        "page_num": page_num
                    })
                    self._append_uob(
                        label="Txn balance",
                        typ="transaction_balance",
                        bbox=balance_blk["bbox"],
                        page=page, page_num=page_num,
                        original_value=Decimal('0.00'),
                        font_size=balance_blk.get("font_size", 12),
                        font=balance_blk.get("font", ""),
                        log_func=log_func
                    )
                    # track right edge for precise right-alignment (if needed later)
                    self._uob_balance_right_edge = balance_blk["bbox"][2] if self._uob_balance_right_edge is None else max(self._uob_balance_right_edge, balance_blk["bbox"][2])
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
                    self._append_uob(
                        label="Txn balance",
                        typ="transaction_balance",
                        bbox=balance_blk["bbox"],
                        page=page, page_num=page_num,
                        original_value=Decimal('0.00'),
                        font_size=balance_blk.get("font_size", 12),
                        font=balance_blk.get("font", ""),
                        log_func=log_func
                    )
                    # track right edge
                    self._uob_balance_right_edge = balance_blk["bbox"][2] if self._uob_balance_right_edge is None else max(self._uob_balance_right_edge, balance_blk["bbox"][2])
                    pending = Decimal('0.00')

            self._carry = pending

        except Exception as e:
            log_func(f"⚠️ Error in UOB extraction: {e}")

        return transactions

    # Abstracts (unused by this parser, but required)
    def _find_balance_column_position(self, all_blocks: List[Dict], log_func: Callable) -> float:
        return 0.0

    def _parse_transaction_row(self, row_blocks: List[Dict], page_num: int, balance_column_x: float, log_func: Callable) -> Optional[Dict]:
        return None

    def _sample_bg(self, page, bbox):
        try:
            x0, y0, x1, y1 = bbox
            # small inner strip left of digits
            w = max(4, (x1 - x0) * 0.12)
            sx0 = min(x0 + 3, x1 - w - 1)
            sx1 = sx0 + w
            sy0 = y0 + (y1 - y0) * 0.32
            sy1 = y1 - (y1 - y0) * 0.28
            r = fitz.Rect(sx0, sy0, sx1, sy1)

            pix = page.get_pixmap(clip=r, alpha=False)
            data, n = pix.samples, pix.n

            # mean of non-dark pixels
            R = G = B = cnt = 0
            for i in range(0, len(data), n):
                r8, g8, b8 = data[i], data[i+1], data[i+2]
                if r8 + g8 + b8 < 120:   # skip ink/border
                    continue
                R += r8; G += g8; B += b8; cnt += 1
            if cnt == 0:
                return (1, 1, 1)

            rN, gN, bN = R/cnt/255.0, G/cnt/255.0, B/cnt/255.0
            # classify: canonical white vs canonical UOB light blue
            luma = 0.2126*rN + 0.7152*gN + 0.0722*bN
            blue_score = bN - max(rN, gN)

            if blue_score > 0.10 and luma < 0.95:
                # UOB light blue (tuned to avoid grey)
                return (0.86, 0.92, 0.97)
            return (1.0, 1.0, 1.0)
        except Exception:
            return (1.0, 1.0, 1.0)

    def _append_with_bg(self, *, label: str, typ: str, bbox, original_value, page, page_num, font_size, font, y_position, log_func, use_row_bg=True):
        bg = self._sample_row_bg(page, bbox) if use_row_bg else self._sample_bg(page, bbox)
        self.balance_replacements.append({
            "type": typ,
            "original_value": original_value,
            "bbox": bbox,
            "font_size": font_size,
            "font": font,
            "page_num": page_num,
            "y_position": y_position,
            "bg_color": bg,
        })
        log_func(f"[UOB] {label} p{page_num} bgRGB=({bg[0]:.3f},{bg[1]:.3f},{bg[2]:.3f}) bbox={bbox}")

    def _classify_uob_color(self, r, g, b):
        # Return canonical blue or white to avoid grey tints
        # Light UOB blue ~ (0.86, 0.92, 0.97)
        luma = 0.2126*r + 0.7152*g + 0.0722*b
        blue_score = b - max(r, g)
        if blue_score > 0.10 and luma < 0.97:
            return (0.86, 0.92, 0.97)
        return (1.0, 1.0, 1.0)

    def _sample_row_bg(self, page, bbox):
        """
        Sample background across the row (not near the digits) so blue rows stay blue.
        bbox: the cell bbox of the balance value on that row.
        """
        try:
            x0, y0, x1, y1 = bbox
            W = page.rect.width
            # Horizontal band through the row away from borders/columns
            band_left  = max(20, W * 0.12)
            band_right = min(W * 0.72, x0 - 10)  # stop before the balance cell
            band_top   = y0 + (y1 - y0) * 0.25
            band_bot   = y1 - (y1 - y0) * 0.25
            if band_right <= band_left:
                band_left, band_right = max(20, W*0.15), min(W*0.65, x0-6)

            r = fitz.Rect(band_left, band_top, band_right, band_bot)
            pix = page.get_pixmap(clip=r, alpha=False)
            data, n = pix.samples, pix.n

            # histogram on non-dark pixels → average → classify
            R = G = B = cnt = 0
            for i in range(0, len(data), n):
                r8, g8, b8 = data[i], data[i+1], data[i+2]
                if r8 + g8 + b8 < 120:  # drop ink/borders/shadows
                    continue
                R += r8; G += g8; B += b8; cnt += 1
            if cnt == 0:
                return (1, 1, 1)
            rN, gN, bN = R/cnt/255.0, G/cnt/255.0, B/cnt/255.0
            return self._classify_uob_color(rN, gN, bN)
        except Exception:
            return (1, 1, 1) 

    def _tag_overview_amount(self, blocks, page, page_num, log_func):
        def fy(*labels):
            labs = {s.upper() for s in labels}
            ys = [b["y"] for b in blocks if b["text"].strip().upper() in labs]
            return min(ys) if ys else None

        y_over = fy("ACCOUNT OVERVIEW AS AT", "GAMBARAN KESELURUHAN AKAUN PADA")
        y_deps = fy("DEPOSIT", "DEPOSITS")
        amt_hdr = next((b for b in blocks
                    if b["text"].strip().upper() in {"AMOUNT (RM)", "AMAUN (RM)"}
                    and (y_over is None or b["y"] >= y_over - 4)
                    and (y_deps is None or b["y"] <= y_deps + 4)), None)
        amount_x = amt_hdr["x"] if amt_hdr else None

        # Primary: numerics strictly between Overview and Deposits
        cands = [c for c in blocks
                 if (y_over is None or c["y"] > y_over + 2)
                 and (y_deps is None or c["y"] < y_deps - 2)
                 and self._is_amount_text(c["text"])]

        log_func(f"[UOB] Overview candidates: {len(cands)}")

        tgt = None
        if cands:
            tgt = min(cands, key=lambda c: abs(c["x"] - amount_x)) if amount_x is not None else max(cands, key=lambda c: c["x"])
        else:
            # Fallback: any numeric close to the Amount header x on the next row(s)
            if amount_x is not None:
                near = [c for c in blocks
                        if abs(c["x"] - amount_x) <= 80
                        and (y_over is None or c["y"] > y_over)
                        and self._is_amount_text(c["text"])]
                if near:
                    tgt = min(near, key=lambda c: c["y"])

        if not tgt:
            log_func("[UOB] Overview not found")
            return

        self._append_uob(
            label="Overview amount",
            typ="ending_balance",
            bbox=tgt["bbox"],
            page=page, page_num=page_num,
            original_value=self._clean_num_text(tgt["text"]),
            font_size=tgt.get("font_size", 12),
            font=tgt.get("font", "helv"),
            log_func=log_func
        )
        log_func(f"[UOB] Overview tagged at {tgt['bbox']}")

    def _uob_row_bg_color(self, page, row_y0, row_y1):
        """
        Pick bg from the row’s vector band: always UOB_BLUE or UOB_WHITE.
        No pixel sampling → no grey.
        """
        import fitz, math
        cell = fitz.Rect(page.rect.x0 + 2, row_y0 + 0.6, page.rect.x1 - 2, row_y1 - 0.6)

        best_fill = None
        best_overlap = 0.0
        for d in page.get_drawings():
            fill = d.get("fill"); r = d.get("rect")
            if not fill or not r:
                continue
            # overlap with this row
            inter = cell & r
            if inter.is_empty:
                continue
            ov = inter.get_area() / cell.get_area()
            if ov > best_overlap:
                best_overlap = ov
                best_fill = fill

        if not best_fill:
            return UOB_WHITE  # fail-safe

        # snap to canonical white or UOB blue
        return UOB_BLUE if math.dist(best_fill, UOB_BLUE) < math.dist(best_fill, UOB_WHITE) else UOB_WHITE