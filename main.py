import os
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tempfile import TemporaryDirectory
from utils import (
    check_input_file,
    pdf_merger,
    convert_pdf_to_string,
    read_config,
    extract_data,
    process_pdf_optimized,
    create_count_excel,
)

def process_folder(input_path, output_path, config, timestamp):
    folder_name = os.path.basename(input_path)
    logs = []
    logs.append(f"\n=== Processing folder: {folder_name} ===")

    try:
        with TemporaryDirectory() as temp_path:
            os.makedirs(output_path, exist_ok=True)

            all_pdf = check_input_file(input_path)
            if not all_pdf:
                logs.append(f"⚠ No PDFs found in {input_path}")
                print("\n".join(logs))
                return

            # Merge PDFs
            logs.append("Merging all PDF Files")
            merged_pdf = os.path.join(temp_path, "output.pdf")
            pdf_merger(all_pdf, save_path=merged_pdf)

            # Convert to text
            logs.append("Converting PDF to Text")
            all_page = convert_pdf_to_string(merged_pdf)

            # Extract data
            logs.append("Extracting Data...")
            df = extract_data(all_page, merged_pdf, output_path, timestamp)

            # Clean & prepare sorting
            for col in ("sku", "courier", "soldBy"):
                df[col] = df[col].fillna("").str.strip()

            df["sku_lower"] = df["sku"].str.lower()

            # Build sort lists efficiently
            sort_list = ["multi"]
            ascending_list = [True]
            if config.get("sku_sort", False):
                sort_list = ["qty", "sku_lower"] + sort_list
                ascending_list = [True, True] + ascending_list
            if config.get("courier_sort", False):
                sort_list.append("courier")
                ascending_list.append(True)
            if config.get("soldBy_sort", False):
                sort_list.append("soldBy")
                ascending_list.append(True)

            logs.append(f"Sorting by: {sort_list}")
            logs.append(f"Ascending order: {ascending_list}")

            # Sort with ignore_index to avoid extra work
            df = df.sort_values(
                by=sort_list,
                ascending=ascending_list,
                na_position="last",
                ignore_index=True,
            )
            df.drop(columns=["sku_lower"], inplace=True)

            # Use the same df for Excel (no copy)
            whole_data = df

            # Process PDF in sorted page order
            page_order = df['page'].tolist()
            logs.append("Processing PDF (cropping and date addition)...")
            processed_pdf_path = process_pdf_optimized(
                merged_pdf, config, temp_path, timestamp, page_order=page_order
            )

            # Save final PDF (move instead of copy)
            final_name = f"result_pdf_{timestamp}.pdf"
            final_path = os.path.join(output_path, final_name)
            shutil.move(processed_pdf_path, final_path)
            logs.append(f"Final PDF saved as: {final_path}")

            # Generate Excel summary
            logs.append("Generating Excel summary report...")
            summary_path = create_count_excel(whole_data, output_path, timestamp)
            logs.append(f"Summary report saved to {summary_path}")

    except Exception as e:
        logs.append(f"Error processing {input_path}: {e}")

    print("\n".join(logs))


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

    # Read config once
    config = read_config()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Use fewer workers to reduce disk contention
    cpu_count = os.cpu_count() or 1
    max_workers = min(4, len(subfolders), max(1, cpu_count // 2))

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for folder in subfolders:
            future = executor.submit(
                process_folder,
                os.path.join(input_root, folder),
                os.path.join(output_root, folder),
                config,
                timestamp,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Process error: {e}")

    print("\nAll folders processed successfully.")


if __name__ == "__main__":
    main()