import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import math
from tqdm import tqdm
import re # <-- ADDED for keyword extraction

# --- CONFIGURATION ---
INPUT_FILE = '../Cleaned_MID_data.csv'
OUTPUT_DIR = 'medication_completed_batches'
CHECKPOINT_FILE = 'medication_checkpoint.txt'
MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'
BATCH_SIZE = 1000

# --- HELPER FUNCTIONS ---

def get_last_processed_row():
    """Reads the checkpoint file to find out where to resume."""
    if not os.path.exists(CHECKPOINT_FILE): return 0
    with open(CHECKPOINT_FILE, 'r') as f:
        try: return int(f.read().strip())
        except ValueError: return 0

def update_checkpoint(row_number):
    """Updates the checkpoint file with the latest processed row number."""
    with open(CHECKPOINT_FILE, 'w') as f: f.write(str(row_number))

# <-- ADDED FROM YOUR SCRIPT ---
def extract_keywords_simple(text: str) -> str:
    """A simple function to extract capitalized words as keywords and return as a string."""
    if not isinstance(text, str):
        return ""
    keywords = re.findall(r'\b[A-Z][a-z]+\b', text)
    unique_keywords = sorted(list(set([kw.lower() for kw in keywords])))
    return ", ".join(unique_keywords)
# --- END OF ADDED FUNCTION ---


# --- MAIN SCRIPT ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
start_row = get_last_processed_row()
print(f"Resuming from CSV row: {start_row}")

print("\nInitializing the embedding model...")
model = SentenceTransformer(MODEL_NAME, device='cuda')
print(f"Model '{MODEL_NAME}' loaded successfully.")

fields_to_embed = [
    'ProductIntroduction', 'ProductUses', 'ProductBenefits', 'SideEffect', 'HowToUse',
    'HowWorks', 'QuickTips', 'SafetyAdvice'
]

try:
    chunk_iterator = pd.read_csv(
        INPUT_FILE, 
        chunksize=BATCH_SIZE, 
        skiprows=range(1, start_row + 1)
    )
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILE}' was not found.")
    exit()

print("\nStarting batch processing...")
for i, df_batch in enumerate(chunk_iterator):
    batch_num = (start_row // BATCH_SIZE) + i
    print(f"\n--- Processing CSV Batch {i+1} (Overall Batch {batch_num}) ---")
    
    records_to_embed = []
    for index, row in df_batch.iterrows():
        drug_name = str(row.get("Name", "Unknown")).strip()
        contains = str(row.get("Contains", "")).strip()
        
        for field in fields_to_embed:
            content = row.get(field)
            if isinstance(content, str) and len(content.strip()) > 0:
                
                # Using the robust composite ID for uniqueness
                unique_product_id = f"{drug_name}_{contains}"
                chunk_id = f"{unique_product_id}_{field}"
                
                # <-- KEYWORD EXTRACTION ADDED HERE ---
                keywords = extract_keywords_simple(content)
                
                records_to_embed.append({
                    'id': chunk_id,
                    'text': f"{drug_name}. {content}",
                    'embedding': None, # Placeholder for the vector
                    'metadata': { 
                        'drug_name': drug_name,
                        'section_title': field,
                        'contains': contains,
                        'text': content,
                        'keywords': keywords, # <-- KEYWORDS FIELD ADDED
                        'therapeutic_class': row.get('Therapeutic_Class', ''),
                        'chemical_class': row.get('Chemical_Class', ''),
                        'habit_forming': row.get('Habit_Forming', ''),
                        'action_class': row.get('Action_Class', ''),
                        'record_type': 'sheet',
                        'source': 'medication.csv'
                    }
                })

    if not records_to_embed:
        print("No valid chunks created in this batch. Skipping.")
        continue

    batch_texts = [rec['text'] for rec in records_to_embed]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=True, batch_size=32)

    for record, embedding in zip(records_to_embed, batch_embeddings):
        record['embedding'] = embedding.tolist()
    
    # Flatten the data for the Parquet file
    final_data = []
    for record in records_to_embed:
        flat_data = {
            'id': record['id'],
            'text': record['text'],
            'vector': record['embedding']
        }
        flat_data.update(record['metadata'])
        final_data.append(flat_data)
        
    final_df = pd.DataFrame(final_data)
    
    output_filename = os.path.join(OUTPUT_DIR, f'medication_batch_{batch_num}.parquet')
    final_df.to_parquet(output_filename, index=False)
    print(f"✅ Batch {batch_num} saved to {output_filename}")
    
    last_row_in_batch = start_row + (i + 1) * BATCH_SIZE
    update_checkpoint(last_row_in_batch)
    print(f"Checkpoint updated to CSV row: {last_row_in_batch}")

print("\n--- ALL BATCHES PROCESSED ---")