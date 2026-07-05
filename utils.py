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
from functools import lru_cache
import numpy as np


# ---------------------- Precompiled regex ----------------------
CLEAN_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]")


# ---------------------- Helpers ----------------------
@lru_cache(maxsize=1000)
def fast_text_clean(text):
    """Cached text cleaning for repeated patterns"""
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
    merged = fitz.open()
    for path in all_path:
        try:
            doc = fitz.open(path)
            merged.insert_pdf(doc)
            doc.close()
        except Exception as e:
            print(f"Error merging {path}: {e}")
    merged.save(save_path)
    merged.close()


# ---------------------- Convert PDF to String ----------------------
def convert_pdf_to_string(file_path):
    """Extract PDF text in page order."""
    doc = fitz.open(file_path)
    all_page = [doc[i].get_text("text") for i in range(len(doc))]
    doc.close()
    return all_page


# ---------------------- Extraction Helpers ----------------------
def quantity_extract(page):
    page_clean = fast_text_clean(page)
    lines = page_clean.split("\n")
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


def courier_extract(page):
    page = fast_text_clean(page)
    lines = page.split("\n")
    return lines[2].strip() if len(lines) > 2 else ""


def sku_extract(page):
    page = fast_text_clean(page)
    lines = page.split("\n")
    all_pipe = [x for x in lines if "|" in x and x and x[0].isnumeric()]
    if not all_pipe:
        return "", False
    sku = all_pipe[0].split(" ", 1)
    return sku[1].split("|", 1)[0] if len(sku) > 1 else "", len(all_pipe) > 1


def soldBy_extract(page):
    page = fast_text_clean(page)
    lines = page.split("\n")
    soldby_line = next((line for line in lines if "Sold By:" in line), None)
    if soldby_line:
        return soldby_line.replace("Sold By:", "").strip().split(",", 1)[0]
    return ""


# ---------------------- Extract Data (Optimized) ----------------------
def extract_data(text, merged_pdf_path, output_path, timestamp):
    """Optimized data extraction with better parallel processing"""

    def process_batch(batch_items):
        results = []
        errors = []
        for idx, page in batch_items:
            try:
                sku, multi_sku = sku_extract(page)
                qty, mqty = quantity_extract(page)
                courier = courier_extract(page)
                soldBy = soldBy_extract(page)
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
    Super-optimized PDF processor with improved label cropping and higher date stamp.
    Processes only pages in page_order (if provided) to avoid extra passes.
    """
    now = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")  # we need the datetime object for formatting
    formatted_datetime = now.strftime("%d-%m-%y %I:%M %p")
    date_stamp_text = f"DATE: {formatted_datetime}"

    add_date_on_top = config.get("add_date_on_top", False)

    doc = fitz.open(pdf_path)
    result = fitz.open()

    keep_invoice = config.get("keep_invoice", False)

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

    # Determine template coordinates from first few pages
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

    def add_clipped_page(source_page_no, clip_rect, top_text=None):
        source_page = doc[source_page_no]
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
                    page_no,
                    label_rect,
                    date_stamp_text if add_date_on_top else None,
                )
                add_clipped_page(page_no, invoice_rect)
            else:
                add_clipped_page(
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
    """Optimized Excel creation with vectorized operations"""
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