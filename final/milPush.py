# Install required packages (run in terminal if not already installed)
# pip install pymilvus pyarrow pandas

import pyarrow.dataset as ds
import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# --- CONFIGURATION ---
parquet_path = 'finalmedication_embeddings_final.parquet'
batch_size = 100
max_entries = 10000
collection_name = 'medication_embeddings_milvus_test'

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# --- HELPER FUNCTION (adapted for clarity) ---
def sanitize_value(value):
    """Ensures value is a string if it's not already a basic type."""
    if value is None:
        return "None"
    # Milvus schema handles types, so we just need to ensure it's not a complex object
    return str(value)

# --- SCRIPT START ---

# 1. Connect to Milvus and define the collection
print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# Drop collection if it already exists (for a clean run)
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Dropped existing collection: {collection_name}")

# Define the schema based on your ChromaDB metadata
# This is the most critical step for Milvus
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=255),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768), # IMPORTANT: Set the correct dimension
    FieldSchema(name="drug_name", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="contains", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="therapeutic_class", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="chemical_class", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="habit_forming", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="action_class", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="record_type", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
]
schema = CollectionSchema(fields, "Medication embeddings collection")
collection = Collection(name=collection_name, schema=schema)
print(f"Collection '{collection_name}' created successfully.")

# 2. Read parquet file and process in batches
dataset = ds.dataset(parquet_path, format='parquet')
scanner = dataset.scanner(batch_size=batch_size)

total_added = 0
print(f"Starting to process the first {max_entries} entries...")

for batch in scanner.to_batches():
    df_batch = batch.to_pandas()
    
    remaining = max_entries - total_added
    df_to_add = df_batch.head(remaining) if len(df_batch) > remaining else df_batch
    
    # Prepare data for Milvus in a columnar format (list of lists)
    data_for_milvus = [
        df_to_add['id'].astype(str).tolist(),
        df_to_add['text'].apply(sanitize_value).tolist(),
        df_to_add['vector'].tolist(),
        df_to_add['drug_name'].apply(sanitize_value).tolist(),
        df_to_add['section_title'].apply(sanitize_value).tolist(),
        df_to_add['contains'].apply(sanitize_value).tolist(),
        df_to_add['keywords'].apply(sanitize_value).tolist(),
        df_to_add['therapeutic_class'].apply(sanitize_value).tolist(),
        df_to_add['chemical_class'].apply(sanitize_value).tolist(),
        df_to_add['habit_forming'].apply(sanitize_value).tolist(),
        df_to_add['action_class'].apply(sanitize_value).tolist(),
        df_to_add['record_type'].apply(sanitize_value).tolist(),
        df_to_add['source'].apply(sanitize_value).tolist(),
    ]
    
    # Insert the prepared data into the collection
    if not df_to_add.empty:
        collection.insert(data_for_milvus)
    
    total_added += len(df_to_add)
    print(f"Inserted {len(df_to_add)} entries from batch (total inserted: {total_added}).")

    if total_added >= max_entries:
        break

# 3. Finalize the collection for searching
print("\nFlushing data to disk...")
collection.flush()

print(f"Creating index for the vector field...")
index_params = {
    "metric_type": "COSINE", # Matching your ChromaDB setup
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}
collection.create_index(field_name="vector", index_params=index_params)
print("Index created successfully.")

print("Loading collection into memory...")
collection.load()
print("Collection loaded.")

print(f"\n✅ Success! The first {total_added} entries were added to the Milvus collection '{collection_name}'.")