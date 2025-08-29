# Install required packages (run in terminal if not already installed)
# pip install pyarrow chromadb pandas

import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
import chromadb

# --- CONFIGURATION ---
parquet_path = 'finalmedication_embeddings_final.parquet'
batch_size = 100   # How many rows to read into memory at a time
max_entries = 10000  # The total number of entries to process
collection_name = '1medication_embeddings_test'
db_path = './chroma_db'   # Directory for DB storage

# --- HELPER FUNCTION ---
# Handles None or unsupported data types for ChromaDB metadata
def fix_metadata_value(value):
    if value is None:
        return "None"
    elif isinstance(value, (int, float, bool)):
        return value
    else:
        return str(value)

# --- SCRIPT START ---

# 1. Set up ChromaDB client and collection
print(f"Setting up ChromaDB client at path: {db_path}")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={'hnsw:space': 'cosine'}
)

# 2. Read parquet file and process the first 15,000 entries
dataset = ds.dataset(parquet_path, format='parquet')
scanner = dataset.scanner(batch_size=batch_size)

total_added = 0
print(f"Starting to process the first {max_entries} entries...")

for batch in scanner.to_batches():
    df_batch = batch.to_pandas()
    
    # Determine how many rows to take from this batch
    remaining = max_entries - total_added
    if len(df_batch) > remaining:
        df_to_add = df_batch.head(remaining)
    else:
        df_to_add = df_batch
    
    # Prepare data for ChromaDB
    ids = df_to_add['id'].astype(str).tolist()
    documents = df_to_add['text'].tolist()
    embeddings = df_to_add['vector'].tolist()
    
    # Create metadata, ensuring 'doc_id' is included for your main app
    metadatas = []
    for idx, (_, row) in enumerate(df_to_add.iterrows()):
        metadata = {
            'doc_id': ids[idx],
            'text': fix_metadata_value(row.get('text')),
            'drug_name': fix_metadata_value(row.get('drug_name')),
            'section_title': fix_metadata_value(row.get('section_title')),
            'contains': fix_metadata_value(row.get('contains')),
            'keywords': fix_metadata_value(row.get('keywords')),
            'therapeutic_class': fix_metadata_value(row.get('therapeutic_class')),
            'chemical_class': fix_metadata_value(row.get('chemical_class')),
            'habit_forming': fix_metadata_value(row.get('habit_forming')),
            'action_class': fix_metadata_value(row.get('action_class')),
            'record_type': fix_metadata_value(row.get('record_type')),
            'source': fix_metadata_value(row.get('source'))
        }
        metadatas.append(metadata)
    
    # Add the prepared data to the collection
    if metadatas:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    total_added += len(df_to_add)
    print(f"Added {len(df_to_add)} entries from batch (total added: {total_added}).")

    # Stop if we've reached the limit
    if total_added >= max_entries:
        break

# --- FIX: The client.persist() method is removed as it's no longer needed ---
# All changes are now automatically persisted to disk by the PersistentClient.

print(f"\n✅ Success! The first {total_added} entries were added to the ChromaDB collection '{collection_name}'.")