import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# --- Configuration ---
BATCHES_FOLDER = "H:/syntorion/pharmabot/pharma_new/final/qembed/FINAL_QA_embedding_batches"
FINAL_OUTPUT_FILE = "finalfinal_QA_embeddings.parquet"

def merge_parquet_files_robustly():
    """
    Merges Parquet files efficiently (low RAM) while also handling
    minor schema inconsistencies between batches.
    """
    print(f"--- Starting robust, low-memory merge from '{BATCHES_FOLDER}' ---")

    # 1. Check if the folder exists
    if not os.path.isdir(BATCHES_FOLDER):
        print(f"\n❌ ERROR: The folder '{BATCHES_FOLDER}' was not found.")
        return

    # 2. Find and sort all the .parquet batch files
    try:
        batch_files = [os.path.join(BATCHES_FOLDER, f) for f in os.listdir(BATCHES_FOLDER) if f.endswith('.parquet')]
        if not batch_files:
            print(f"❌ ERROR: No .parquet files were found in '{BATCHES_FOLDER}'.")
            return

        batch_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        num_batches_found = len(batch_files)
        print(f"Found {num_batches_found} batch files to merge.")

    except (ValueError, IndexError):
        print("\n❌ ERROR: Could not sort the batch files. Ensure names are like 'batch_1.parquet'.")
        return

    # 3. Establish the "master" schema from the very first file.
    try:
        master_schema = pq.read_schema(batch_files[0])
        print("Established master schema from the first batch file.")
    except Exception as e:
        print(f"❌ ERROR: Could not read the schema from the first batch file '{batch_files[0]}'. Error: {e}")
        return

    # 4. Use a ParquetWriter to write files one by one, conforming them to the master schema.
    print("Writing batches to the final file...")
    with pq.ParquetWriter(FINAL_OUTPUT_FILE, schema=master_schema) as writer:
        for file_path in tqdm(batch_files, desc="Merging batches"):
            try:
                # Read the current batch into a pyarrow Table
                current_table = pq.read_table(file_path)

                # **This is the key step**: Cast the current table to match the master schema.
                # This fixes inconsistencies like a 'null' column being cast to a 'list' column.
                casted_table = current_table.cast(master_schema, safe=False)

                # Write the conformed table to the output file.
                writer.write_table(casted_table)
            except Exception as e:
                print(f"\n❌ ERROR: Could not process file '{file_path}'. Error: {e}")
                print("Stopping the merge process.")
                return

    # 5. Final summary
    final_row_count = pq.read_metadata(FINAL_OUTPUT_FILE).num_rows
    print("\n--- ✅ Merge Complete! ---")
    print(f"Total number of batches merged: {num_batches_found}")
    print(f"Total records in the final file: {final_row_count}")


if __name__ == "__main__":
    merge_parquet_files_robustly()