import os
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tempfile import TemporaryDirectory
from pdfrw import PdfReader, PdfWriter
from utils import (
    
    create_filedir,
    check_input_file,
    pdf_merger,
    convert_pdf_to_string,
    read_config,
    extract_data,
    pdf_whitespace,
    pdf_cropper,
    create_count_excel,
)

def process_folder(input_path, output_path):
    folder_name = os.path.basename(input_path)
    print(f"\n=== Processing folder: {folder_name} ===")
    
    try:
        # Use a temporary directory for all intermediate files
        with TemporaryDirectory() as temp_path:
            create_filedir(temp_path, output_path)

            # Input validation
            all_pdf = check_input_file(input_path)
            if not all_pdf:
                print(f"⚠ No PDFs found in {input_path}")
                return

            # Read config
            config = read_config()

            # Merge PDFs
            print("Merging all the PDF Files")
            merged_pdf = os.path.join(temp_path, "output.pdf")
            pdf_merger(all_pdf, save_path=merged_pdf)
            print(f"Merge Completed -> {merged_pdf}")

            # Convert to text
            print("Converting PDF to Text")
            all_page = convert_pdf_to_string(merged_pdf)
            print("Conversion Completed")

            # Extract data
            print("Extracting Data...")
            df = extract_data(all_page, merged_pdf, output_path)
            print("Extraction Completed")

            # Clean & prepare sorting
            df["sku"] = df["sku"].str.strip().fillna("")
            df["courier"] = df["courier"].str.strip().fillna("")
            df["soldBy"] = df["soldBy"].str.strip().fillna("")
            df["sku_lower"] = df["sku"].str.lower()

            sort_list = ["multi"]
            ascending_list = [True]
            if config.get("sku_sort", False):
                sort_list.append("sku_lower")
                ascending_list.append(False)
            if config.get("courier_sort", False):
                sort_list.append("courier")
                ascending_list.append(True)
            if config.get("soldBy_sort", False):
                sort_list.append("soldBy")
                ascending_list.append(True)

            print("\nSorting by:", sort_list)
            print("Ascending order:", ascending_list)

            df = df.sort_values(by=sort_list, ascending=ascending_list, na_position="last")
            df = df.drop(columns=["sku_lower"])
            whole_data = df.copy(deep=True)

            # Create sorted PDF
            reader_input = PdfReader(merged_pdf)
            writer_output = PdfWriter()
            for page in df.page.values:
                writer_output.addpage(reader_input.pages[page])

            sorted_pdf_path = os.path.join(temp_path, "output_sorted.pdf")
            writer_output.write(sorted_pdf_path)
            print(f"Sorted PDF created -> {sorted_pdf_path}")

            # Process PDF (whitespace + crop)
            whitespace_pdf_path = pdf_whitespace(sorted_pdf_path, temp_path)
            cropped_pdf_path = pdf_cropper(whitespace_pdf_path, config, temp_path)

            # Save final PDF to output folder
            final_name = f"result_pdf_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
            final_path = os.path.join(output_path, final_name)
            shutil.copy(cropped_pdf_path, final_path)
            print(f"Final PDF saved as: {final_path}")

            # Export Excel summary report
            print("Generating Excel summary report...")
            summary_path = create_count_excel(whole_data, output_path)
            print(f"Summary report saved to {summary_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def main():


    input_root = "input"
    output_root = "output"

    subfolders = [
        f for f in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, f))
    ]
    if not subfolders:
        print("No subfolders found in 'input'.")
        return

    futures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for folder in subfolders:
            future = executor.submit(
                process_folder,
                os.path.join(input_root, folder),
                os.path.join(output_root, folder),
            )
            futures.append(future)

        # Wait for all processes to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Process error: {e}")

    print("\nAll folders processed successfully.")


if __name__ == "__main__":
    main()
