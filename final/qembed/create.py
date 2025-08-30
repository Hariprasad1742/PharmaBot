import json
import os
import time
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# --- Configuration ---
INPUT_FILE = "H:/syntorion/pharmabot/pharma_new/final/MID_Q&A_Chunks_kalyan.jsonl"
EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
BATCH_SIZE = 5000
CHECKPOINT_FILE = "embedding_checkpoint.txt"
OUTPUT_DIR = "FINAL_embedding_batches"
FINAL_OUTPUT_FILE = "final_embeddings.parquet"


# --- Smart Helper Functions to find data ---
def find_text_in_record(record):
    """
    Searches for the text content in a few possible locations within the JSON object.
    """
    # Pattern 1: Top-level 'text' key
    if record.get("text"):
        return record.get("text")
    # Pattern 2: Top-level 'text_chunk' key
    if record.get("text_chunk"):
        return record.get("text_chunk")
    # Pattern 3: Nested inside a 'root' object
    if isinstance(record.get("root"), dict):
        root = record.get("root")
        if root.get("text"):
            return root.get("text")
        if root.get("text_chunk"):
            return root.get("text_chunk")
    # If none of the patterns match, return None
    return None

def find_metadata_in_record(record):
    """
    Searches for the metadata object in a few possible locations.
    """
    # Pattern 1: Top-level 'metadata' key
    if isinstance(record.get("metadata"), dict):
        return record.get("metadata")
    # Pattern 2: Nested inside a 'root' object
    if isinstance(record.get("root"), dict):
        root = record.get("root")
        if isinstance(root.get("metadata"), dict):
            return root.get("metadata")
    # If no metadata is found, return an empty dictionary
    return {}


# --- Core Processing Functions (no changes here) ---
def read_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE): return 0
    try:
        with open(CHECKPOINT_FILE, 'r') as f: return int(f.read().strip())
    except: return 0

def write_checkpoint(line_number):
    with open(CHECKPOINT_FILE, 'w') as f: f.write(str(line_number))

def process_and_save_batch(model, batch_data, batch_index):
    last_line_num = batch_data[-1]["line_num"]
    print(f"Processing Batch #{batch_index} (ending at line {last_line_num})...")
    texts_to_encode = [item['text'] for item in batch_data]
    embeddings = model.encode(texts_to_encode, convert_to_numpy=True, show_progress_bar=False)
    df = pd.DataFrame(batch_data)
    df['embedding'] = list(embeddings)
    batch_filename = os.path.join(OUTPUT_DIR, f"batch_{batch_index}.parquet")
    df.to_parquet(batch_filename)
    print(f"  - Saved batch to {batch_filename}")
    write_checkpoint(last_line_num)
    print(f"  - Checkpoint updated to line {last_line_num}")

def concatenate_batches():
    print(f"\nConcatenating batch results into {FINAL_OUTPUT_FILE}...")
    if not os.path.exists(OUTPUT_DIR) or not os.listdir(OUTPUT_DIR):
        print("⚠ No batch files were created, so there is nothing to concatenate.")
        return
    try:
        batch_files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith('.parquet')]
        batch_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        all_dfs = [pd.read_parquet(file) for file in batch_files]
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_parquet(FINAL_OUTPUT_FILE)
        print(f"✅ Successfully concatenated {len(batch_files)} batches into {FINAL_OUTPUT_FILE} ({len(final_df)} records).")
    except Exception as e:
        print(f"Error during concatenation: {e}")


def generate_embeddings():
    """Main function that now uses the smart helper functions."""
    start_time = time.time()
    print("--- Starting Embedding Generation ---")

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print("Model loaded successfully.")

    start_index = read_checkpoint()
    batch_counter = (start_index // BATCH_SIZE) + 1
    print(f"Resuming from line: {start_index + 1}")

    current_line_num = 0
    batch_data = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        if start_index > 0:
            for _ in range(start_index): next(f)
            current_line_num = start_index

        for line in f:
            current_line_num += 1
            try:
                record = json.loads(line)
                
                # --- THIS IS THE NEW LOGIC ---
                # Use the smart functions to find the data, no matter the structure.
                text = find_text_in_record(record)
                metadata = find_metadata_in_record(record)
                
                if not text:
                    # If the smart search can't find text, the line is truly unusable.
                    continue
                
                unique_id = current_line_num
                
                batch_data.append({
                    "id": unique_id, 
                    "text": text, 
                    "metadata": metadata,
                    "line_num": current_line_num
                })
                # --- END OF NEW LOGIC ---

                if len(batch_data) >= BATCH_SIZE:
                    process_and_save_batch(model, batch_data, batch_counter)
                    batch_data = []
                    batch_counter += 1

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"❌ An unexpected error occurred at line {current_line_num}: {e}")

    if batch_data:
        process_and_save_batch(model, batch_data, batch_counter)

    print("\n--- All batches processed. ---")
    concatenate_batches()

    end_time = time.time()
    print(f"\n--- Script finished in {end_time - start_time:.2f} seconds. ---")


if __name__ == "__main__":
    generate_embeddings()