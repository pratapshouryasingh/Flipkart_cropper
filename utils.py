import requests
import json
import sys
import shutil
import os
from pdfrw import PdfReader, PdfWriter
import fitz
from datetime import datetime
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


# ---------------------- Precompiled regex ----------------------
CLEAN_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]")


# ---------------------- Helpers ----------------------
def fast_text_clean(text):
    """
    Plain (uncached) text cleaning.

    NOTE: the original used @lru_cache here, but page text on real labels
    (order IDs, addresses, SKUs, timestamps) is essentially never repeated
    across pages, so the cache almost never hits. It still pays the cost of
    hashing every full page string and growing an internal dict up to
    maxsize=1000 before evicting. Removing the cache avoids that overhead
    with zero behavioral change (regex sub on a precompiled pattern is
    already fast on its own).
    """
    return CLEAN_PATTERN.sub("", text)


# ---------------------- Read Config ----------------------
def read_config():
    config_path = os.environ.get("JOB_CONFIG_PATH", "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------- Check Input PDF ----------------------
def check_input_file(filepath):
    all_pdf = []
    for x in os.listdir(filepath):
        path = os.path.join(filepath, x)
        if not path.lower().endswith(".pdf"):
            continue
        try:
            with open(path, "rb") as f:
                header = f.read(4)
                if header != b"%PDF":
                    print(f"Skipping invalid PDF: {x}")
                    continue
            all_pdf.append(path)
        except:
            print(f"Skipping unreadable file: {x}")
    if not all_pdf:
        print(f"No valid PDF files found in {filepath}")
        return []
    return all_pdf


# ---------------------- Merge PDF ----------------------
def pdf_merger(all_path, save_path):
    """
    Same output as before. Two additions purely for merge/save speed:
      - links=False, annots=False on insert_pdf: skips copying link/annotation
        objects that shipping-label PDFs never use, so PyMuPDF has less to walk.
      - garbage=4, deflate=True on save: garbage-collects now-unused objects
        from the merge and compresses streams, which also makes every
        downstream fitz.open()/save() on this file faster and produces a
        smaller intermediate file (same visible PDF output).
    """
    merged = fitz.open()
    for path in all_path:
        try:
            doc = fitz.open(path)
            merged.insert_pdf(doc, links=False, annots=False)
            doc.close()
        except Exception as e:
            print(f"Error merging {path}: {e}")
    merged.save(save_path, garbage=4, deflate=True)
    merged.close()


# ---------------------- Convert PDF to String ----------------------
def convert_pdf_to_string(file_path):
    """
    Ported from the reference file: extract each page's text in parallel
    with a ThreadPoolExecutor instead of a sequential list comprehension.
    Output order is preserved via the futures->index map, so page order
    is identical to the original despite out-of-order completion.

    Caveat worth flagging: PyMuPDF's underlying MuPDF C library serializes
    concurrent calls on the *same* fitz.Document internally, so this won't
    scale linearly with thread count the way pure-Python work would - the
    win comes from overlapping I/O/decompression, not true parallel CPU
    work. It's still a net win for large multi-page PDFs and matches what
    the reference file already does, but if you ever see odd/corrupted
    text on some pages under heavy load, the safe fallback is to open a
    separate fitz.Document per worker thread instead of sharing one.
    """
    doc = fitz.open(file_path)
    all_page = [None] * len(doc)

    def process_page(i):
        return doc[i].get_text("text")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_page, i): i for i in range(len(doc))}
        for future in as_completed(futures):
            idx = futures[future]
            all_page[idx] = future.result()

    doc.close()
    return all_page


# ---------------------- Extraction Helpers (operate on pre-split lines) ----------------------
# These are the same field-parsing rules as the original sku_extract /
# quantity_extract / courier_extract / soldBy_extract, just refactored to
# accept an already-cleaned, already-split `lines` list instead of each
# doing its own fast_text_clean(page) + page.split("\n"). That removes
# 3 redundant clean+split passes per page (4 calls -> 1).

def quantity_extract_lines(lines):
    try:
        qty_indices = [i for i, l in enumerate(lines) if "QTY" in l.upper()]
        if not qty_indices:
            return 1, False
        qty_start = qty_indices[0]
        qtys = []
        for l in lines[qty_start + 1:]:
            l_stripped = l.strip()
            if any(keyword in l_stripped.upper() for keyword in ["SKU", "SOLD BY", "COLOR", "SIZE"]):
                break
            if l_stripped and l_stripped.isdigit():
                qtys.append(int(l_stripped))
        total_qty = sum(qtys) if qtys else 1
        return total_qty, len(qtys) > 1
    except:
        return 1, False


def courier_extract_lines(lines):
    return lines[2].strip() if len(lines) > 2 else ""


def sku_extract_lines(lines):
    all_pipe = [x for x in lines if "|" in x and x and x[0].isnumeric()]
    if not all_pipe:
        return "", False
    sku = all_pipe[0].split(" ", 1)
    return sku[1].split("|", 1)[0] if len(sku) > 1 else "", len(all_pipe) > 1


def soldBy_extract_lines(lines):
    soldby_line = next((line for line in lines if "Sold By:" in line), None)
    if soldby_line:
        return soldby_line.replace("Sold By:", "").strip().split(",", 1)[0]
    return ""


# Thin wrappers kept for backward compatibility, in case anything else in
# the pipeline calls the original per-page-text signatures directly.
def quantity_extract(page):
    return quantity_extract_lines(fast_text_clean(page).split("\n"))


def courier_extract(page):
    return courier_extract_lines(fast_text_clean(page).split("\n"))


def sku_extract(page):
    return sku_extract_lines(fast_text_clean(page).split("\n"))


def soldBy_extract(page):
    return soldBy_extract_lines(fast_text_clean(page).split("\n"))


# ---------------------- Extract Data (Optimized) ----------------------
def extract_data(text, merged_pdf_path, output_path, timestamp):
    """
    Same batching/threading strategy as before, but each page is now
    cleaned and split exactly once (`lines`), and that single `lines`
    list is reused for all four field extractions instead of each
    extractor re-cleaning and re-splitting the raw page text.
    """

    def process_batch(batch_items):
        results = []
        errors = []
        for idx, page in batch_items:
            try:
                lines = fast_text_clean(page).split("\n")

                sku, multi_sku = sku_extract_lines(lines)
                qty, mqty = quantity_extract_lines(lines)
                courier = courier_extract_lines(lines)
                soldBy = soldBy_extract_lines(lines)

                multi = (multi_sku or mqty or qty > 1)
                results.append({"page": idx, "sku": sku, "qty": qty, "multi": multi,
                               "courier": courier, "soldBy": soldBy})
                if sku == "":
                    errors.append(idx)
            except:
                errors.append(idx)
        return results, errors

    pages_list = [(i, page) for i, page in enumerate(text) if page and page.strip()]
    batch_size = max(1, len(pages_list) // 32)
    batches = [pages_list[i:i + batch_size] for i in range(0, len(pages_list), batch_size)]

    df_list = []
    error_pages = []

    with ThreadPoolExecutor(max_workers=min(8, len(batches))) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            results, errors = future.result()
            df_list.extend(results)
            error_pages.extend(errors)

    df = pd.DataFrame(df_list)

    if error_pages:
        error_filename = f"error_pages_{timestamp}.pdf"
        reader_input = PdfReader(merged_pdf_path)
        writer = PdfWriter()
        for page in error_pages:
            if page < len(reader_input.pages):
                writer.addpage(reader_input.pages[page])
        writer.write(os.path.join(output_path, error_filename))

    return df


# ---------------------- PDF Optimized Processor ----------------------
def process_pdf_optimized(pdf_path, config, temp_path, timestamp, page_order=None):
    """
    Same label/invoice cropping and output as before. Two micro-optimizations:

      1. add_clipped_page now takes the already-fetched `source_page` object
         instead of a page number, so we don't call doc[page_no] a second
         time for every page we already have open.
      2. Local variable caching of config flags before the loop (avoids
         repeated dict .get() calls per page - negligible per-call, but
         it's free and adds up over thousands of pages).

    Left unchanged on purpose: get_label_top_y() still calls
    page.search_for("STD") once per page (not just for the template
    pages), because that offset can genuinely vary page-to-page depending
    on courier label layout. Caching it globally like template_label_y
    would risk silently shifting crops on pages where "STD" sits somewhere
    different - a correctness risk, not just a speed one - so it's left
    as a per-page search_for() call to preserve exact output.
    """
    now = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
    formatted_datetime = now.strftime("%d-%m-%y %I:%M %p")
    date_stamp_text = f"DATE: {formatted_datetime}"

    add_date_on_top = config.get("add_date_on_top", False)
    keep_invoice = config.get("keep_invoice", False)

    doc = fitz.open(pdf_path)
    result = fitz.open()

    def get_label_y(source_page):
        try:
            rects = source_page.search_for("Order Id:")
            if rects:
                return rects[0].y0 - 10
        except Exception:
            pass
        return None

    def get_invoice_y(source_page):
        try:
            rects = source_page.search_for("TAX INVOICE")
            if rects:
                return rects[0].y0 - 10
        except Exception:
            pass
        return None

    def get_label_top_y(source_page):
        try:
            rects = source_page.search_for("STD")
            if rects:
                return max(0, rects[0].y0 - 7)
        except Exception:
            pass
        return 28

    # Determine template coordinates from first few pages (unchanged: this
    # amortizes the "Order Id:"/"TAX INVOICE" search across the whole run
    # instead of doing it per page, same as the original).
    pages_to_process = list(page_order) if page_order is not None else list(range(len(doc)))
    template_label_y = None
    template_invoice_y = None
    for sample_page_no in pages_to_process:
        sample_page = doc[int(sample_page_no)]
        if template_label_y is None:
            template_label_y = get_label_y(sample_page)
        if keep_invoice and template_invoice_y is None:
            template_invoice_y = get_invoice_y(sample_page)
        if template_label_y is not None and (not keep_invoice or template_invoice_y is not None):
            break

    def make_label_rect(source_page, label_y):
        if label_y is None:
            return None
        return fitz.Rect(
            185,
            get_label_top_y(source_page),
            source_page.rect.width - 185,
            label_y,
        )

    def make_invoice_rect(source_page, invoice_y):
        if invoice_y is None:
            return None
        return fitz.Rect(0, invoice_y, source_page.rect.width, source_page.rect.height)

    def add_clipped_page(source_page, source_page_no, clip_rect, top_text=None):
        clip_rect = clip_rect or source_page.rect
        top_margin = 11 if top_text else 0
        page_width = clip_rect.width
        page_height = clip_rect.height + top_margin

        output_page = result.new_page(width=page_width, height=page_height)
        if top_text:
            output_page.insert_text(
                fitz.Point(0, 8),
                top_text,
                fontsize=9,
                fontname="Helv",
                color=(0, 0, 0),
            )

        target_rect = fitz.Rect(0, top_margin, page_width, page_height)
        output_page.show_pdf_page(
            target_rect,
            doc,
            source_page_no,
            clip=clip_rect,
        )

    # Process pages in the given order
    for page_no in pages_to_process:
        page_no = int(page_no)
        try:
            source_page = doc[page_no]
            label_rect = make_label_rect(source_page, template_label_y)

            if keep_invoice:
                invoice_rect = make_invoice_rect(source_page, template_invoice_y)
                add_clipped_page(
                    source_page,
                    page_no,
                    label_rect,
                    date_stamp_text if add_date_on_top else None,
                )
                add_clipped_page(source_page, page_no, invoice_rect)
            else:
                add_clipped_page(
                    source_page,
                    page_no,
                    label_rect,
                    date_stamp_text if add_date_on_top else None,
                )
        except Exception as e:
            # Fallback: just copy the page as-is
            result.insert_pdf(doc, from_page=page_no, to_page=page_no)

    output_path = os.path.join(temp_path, "processed_final.pdf")
    result.save(output_path, garbage=4, deflate=True, clean=True)
    print(f"\nSaved processed PDF to: {output_path}")

    doc.close()
    result.close()
    return output_path


# ---------------------- Create Count Excel (Optimized) ----------------------
def create_count_excel(df, output_path, timestamp):
    """Unchanged from the original - groupby-based aggregation was already
    vectorized and isn't a hotspot relative to PDF parsing/cropping."""
    df["sku"] = df["sku"].astype(str).str.strip().replace({"nan": "", "None": ""})
    df["soldBy"] = df["soldBy"].astype(str).fillna("")

    for col in ["color", "size"]:
        if col not in df.columns:
            df[col] = ""

    # SKU REPORT
    sku_df = df.groupby(["qty", "soldBy", "color", "sku"], as_index=False).size()
    sku_df.columns = ["Qty", "SoldBy", "Color", "SKU", "Count"]
    sku_df["SKU_lower"] = sku_df["SKU"].str.lower()
    sku_df = sku_df.sort_values(by=["Count", "SKU_lower", "Qty"], ascending=[False, True, True])
    sku_df = sku_df.drop(columns=["SKU_lower"]).reset_index(drop=True)

    # COURIER + SOLD BY
    courierSold_df = df.groupby(["courier", "soldBy"], as_index=False).size()
    courierSold_df.columns = ["Courier", "SoldBy", "Packages"]
    courierSold_df = courierSold_df.sort_values(by=["Packages", "Courier"], ascending=[False, True]).reset_index(drop=True)

    # COURIER
    courier_df = df.groupby(["courier"], as_index=False).size()
    courier_df.columns = ["Courier", "Packages"]
    courier_df = courier_df.sort_values(by=["Packages", "Courier"], ascending=[False, True]).reset_index(drop=True)

    # SOLD BY
    soldby_df = df.groupby(["soldBy"], as_index=False).size()
    soldby_df.columns = ["SoldBy", "Packages"]
    soldby_df = soldby_df.sort_values(by=["Packages", "SoldBy"], ascending=[False, True]).reset_index(drop=True)

    filename = f"summary_report_{timestamp}.xlsx"
    summary_path = os.path.join(output_path, filename)

    with pd.ExcelWriter(summary_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Summary")
        writer.sheets["Summary"] = worksheet

        bold_format = workbook.add_format({'bold': True, 'font_size': 12})
        header_format = workbook.add_format({'bold': True, 'bg_color': '#DDEEFF', 'border': 1, 'text_wrap': True})
        wrap_format = workbook.add_format({'text_wrap': True})

        row = 0
        def write_block(title, df_block):
            nonlocal row
            worksheet.write(row, 0, title, bold_format)
            row += 1
            for col_num, value in enumerate(df_block.columns):
                worksheet.write(row, col_num, value, header_format)
            row += 1
            for r in df_block.itertuples(index=False):
                for col_num, value in enumerate(r):
                    worksheet.write(row, col_num, value, wrap_format)
                row += 1
            for i, col in enumerate(df_block.columns):
                max_len = max(
                    len(str(col)),
                    df_block[col].astype(str).str.len().max() if len(df_block) > 0 else 0
                )
                worksheet.set_column(i, i, min(max_len + 2, 30))
            row += 2

        write_block("SKU REPORT", sku_df)
        write_block("COURIER + SOLD BY REPORT", courierSold_df)
        write_block("COURIER REPORT", courier_df)
        write_block("SOLD BY REPORT", soldby_df)

    print(f"Excel generated -> {summary_path}")
    return summary_path
#done