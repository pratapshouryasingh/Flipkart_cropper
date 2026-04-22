import requests
import json
import sys
import shutil
import os
from pdfrw import PdfReader, PdfWriter
from tqdm import tqdm
import fitz
from datetime import datetime
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed



# ---------------------- Create Directories ----------------------
def create_filedir(temp_path="temp", output_path="output"):
    shutil.rmtree(temp_path, ignore_errors=True)
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

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

# ---------------------- Read Config ----------------------
def read_config():
    with open("config.json", "r") as f:
        return json.load(f)

# ---------------------- Merge PDF ----------------------
def pdf_merger(all_path, save_path):
    writer = PdfWriter()
    for path in all_path:
        reader = PdfReader(path)
        for page in reader.pages:
            writer.addpage(page)
    writer.write(save_path)

# ---------------------- Convert PDF to String (Parallel) ----------------------
def convert_pdf_to_string(file_path):
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

# ---------------------- Precompiled regex ----------------------
CLEAN_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]")

# ---------------------- Extraction Helpers ----------------------
def quantity_extract(page):
    page_clean = CLEAN_PATTERN.sub("", page)
    lines = page_clean.split("\n")
    try:
        qty_start = next(i for i, l in enumerate(lines) if "QTY" in l.upper())
        qtys = []
        for l in lines[qty_start + 1:]:
            l_stripped = l.strip()
            if any(keyword in l_stripped.upper() for keyword in ["SKU", "SOLD BY", "COLOR", "SIZE"]):
                break
            if l_stripped.isdigit():
                qtys.append(int(l_stripped))
        total_qty = sum(qtys) if qtys else 1
        return total_qty, len(qtys) > 1
    except StopIteration:
        return 1, False

def courier_extract(page):
    page = CLEAN_PATTERN.sub("", page)
    page = page.split("\n")
    try:
        return page[2].strip()
    except:
        return ""

def sku_extract(page):
    page = CLEAN_PATTERN.sub("", page)
    page = page.split("\n")
    all_pipe = [x for x in page if "|" in x]
    try:
        skus = [x for x in all_pipe if x[0].isnumeric()]
    except:
        return "", False
    if not skus:
        return "", False
    sku = skus[0].split(" ", 1)
    return sku[1].split("|", 1)[0], len(skus) > 1

def soldBy_extract(page):
    page = CLEAN_PATTERN.sub("", page)
    page = page.split("\n")
    try:
        soldby_idx = [x for x in range(len(page)) if "Sold By:" in page[x]][0]
        return page[soldby_idx].replace("Sold By:", "").strip().split(",", 1)[0]
    except:
        return ""

# ---------------------- Extract Data (Parallel) ----------------------
def extract_data(text, merged_pdf_path, output_path):
    df_list = []
    error_pages = []

    def process_page(idx, page):
        try:
            sku, multi_sku = sku_extract(page)
            qty, mqty = quantity_extract(page)
            courier = courier_extract(page)
            soldBy = soldBy_extract(page)
            multi = (multi_sku or mqty or qty > 1)
            return {"page": idx, "sku": sku, "qty": qty, "multi": multi, "courier": courier, "soldBy": soldBy}, idx if sku == "" else None
        except:
            return None, idx

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_page, i, page) for i, page in enumerate(text)]
        for future in as_completed(futures):
            result, error_idx = future.result()
            if result:
                df_list.append(result)
            if error_idx is not None:
                error_pages.append(error_idx)

    df = pd.DataFrame(df_list)

    # Handle error pages
    if error_pages:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        error_filename = f"error_pages_{timestamp}.pdf"
        reader_input = PdfReader(merged_pdf_path)
        writer = PdfWriter()
        for page in error_pages:
            writer.addpage(reader_input.pages[page])
        writer.write(os.path.join(output_path, error_filename))

    return df

# ---------------------- PDF Whitespace ----------------------
def pdf_whitespace(pdf_path, temp_path):
    """Remove whitespace and save to temp directory only"""
    doc = fitz.open(pdf_path)
    for page_no in tqdm(range(len(doc)), desc="Removing whitespace"):
        try:
            text_instances = doc[page_no].search_for("TAX INVOICE")[0]
            crop_rect = fitz.Rect(0, 0, doc[page_no].rect.width - 8, text_instances.y0 + 20)
            doc[page_no].set_cropbox(crop_rect)
        except:
            pass
    save_path = os.path.join(temp_path, "whitespace_removed.pdf")
    doc.save(save_path, garbage=4, deflate=True, clean=True)
    doc.close()
    return save_path

def pdf_cropper(pdf_path, config, temp_path):
    now = datetime.now()
    formatted_datetime = now.strftime("%d-%m-%y %I:%M %p")
    doc = fitz.open(pdf_path)
    result = fitz.open()

    for page_no in tqdm(range(len(doc)), desc="Cropping pages"):
        try:
            if config.get("keep_invoice", False):
                # Insert full page twice: first = Label, second = Invoice
                result.insert_pdf(doc, from_page=page_no, to_page=page_no)
                result.insert_pdf(doc, from_page=page_no, to_page=page_no)

                label_page = result[-2]
                invoice_page = result[-1]

                # ---- CROP LABEL ----
                text_instances = label_page.search_for("Order Id:")
                if text_instances:
                    label_rect = fitz.Rect(
                        182, 15,
                        label_page.rect.width - 175,
                        text_instances[0].y0 - 10
                    )
                    label_page.set_cropbox(label_rect)

                if config.get("add_date_on_top", False):
                    label_page.insert_text(fitz.Point(12, 10), formatted_datetime, fontsize=11)

                # ---- CROP INVOICE (from TAX INVOICE downwards) ----
                text_instances = invoice_page.search_for("TAX INVOICE")
                if text_instances:
                    invoice_rect = fitz.Rect(
                        0, text_instances[0].y0 - 10,
                        invoice_page.rect.width,
                        invoice_page.rect.height
                    )
                    invoice_page.set_cropbox(invoice_rect)
                else:
                    # fallback
                    invoice_page.set_cropbox(invoice_page.rect)

            else:
                # Only label
                result.insert_pdf(doc, from_page=page_no, to_page=page_no)
                label_page = result[-1]

                text_instances = label_page.search_for("Order Id:")
                if text_instances:
                    label_rect = fitz.Rect(
                        185, 15,
                        label_page.rect.width - 175,
                        text_instances[0].y0 - 10
                    )
                    label_page.set_cropbox(label_rect)

                if config.get("add_date_on_top", False):
                    label_page.insert_text(fitz.Point(12, 10), formatted_datetime, fontsize=11)

        except Exception as e:
            print(f"⚠️ Error cropping page {page_no}: {e}")
            result.insert_pdf(doc, from_page=page_no, to_page=page_no)

    doc.close()
    output_filename = os.path.join(temp_path, "cropped_final.pdf")
    result.save(output_filename, garbage=4, deflate=True, clean=True)
    result.close()
    return output_filename

# ---------------------- Create Count Excel ----------------------
# ---------------------- Create Count Excel (Formatted like second script) ----------------------
def create_count_excel(df, output_path):
    df["sku"] = df["sku"].astype(str).str.strip().replace({"nan": "", "None": ""})
    df["soldBy"] = df["soldBy"].astype(str).fillna("")
    df["color"] = df.get("color", "")  # ensure column exists
    df["size"] = df.get("size", "")    # ensure column exists

    # SKU REPORT
    sku_df = df[["qty", "soldBy", "color", "sku"]].value_counts().reset_index()
    sku_df.columns = ["Qty", "SoldBy", "Color", "SKU", "Count"]
    sku_df["SKU_lower"] = sku_df["SKU"].str.lower()
    sku_df = sku_df.sort_values(by=["Count", "SKU_lower", "Qty"], ascending=[False, True, True])
    sku_df = sku_df.drop(columns=["SKU_lower"]).reset_index(drop=True)

    # COURIER + SOLD BY REPORT
    courierSold_df = df[["courier", "soldBy"]].value_counts().reset_index()
    courierSold_df.columns = ["Courier", "SoldBy", "Packages"]
    courierSold_df = courierSold_df.sort_values(by=["Packages", "Courier"], ascending=[False, True]).reset_index(drop=True)

    # COURIER REPORT
    courier_df = df[["courier"]].value_counts().reset_index()
    courier_df.columns = ["Courier", "Packages"]
    courier_df = courier_df.sort_values(by=["Packages", "Courier"], ascending=[False, True]).reset_index(drop=True)

    # SOLD BY REPORT
    soldby_df = df[["soldBy"]].value_counts().reset_index()
    soldby_df.columns = ["SoldBy", "Packages"]
    soldby_df = soldby_df.sort_values(by=["Packages", "SoldBy"], ascending=[False, True]).reset_index(drop=True)

    # Save Excel with formatting
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
            for r in df_block.values:
                for col_num, value in enumerate(r):
                    worksheet.write(row, col_num, value, wrap_format)
                row += 1
            for i, col in enumerate(df_block.columns):
                max_len = max([len(str(col))] + [len(str(val)) for val in df_block[col]])
                worksheet.set_column(i, i, min(max_len + 2, 30))
            row += 2

        # Write all blocks
        write_block("SKU REPORT", sku_df)
        write_block("COURIER + SOLD BY REPORT", courierSold_df)
        write_block("COURIER REPORT", courier_df)
        write_block("SOLD BY REPORT", soldby_df)

    print(f"✅ Excel generated -> {summary_path}")
    return summary_path
