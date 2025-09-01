# Install required packages (run in terminal if not already installed)
# pip install pymilvus pyarrow pandas

import pyarrow.dataset as ds
import pandas as pd
import gc # IMPORTED: Garbage Collector
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# --- CONFIGURATION ---
parquet_path = r'H:\syntorion\pharmabot\pharma_new\final\finalmedication_embeddings_final.parquet'
batch_size = 100
collection_name = 'medication'

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# --- HELPER FUNCTION ---
def sanitize_value(value):
    if value is None:
        return "None"
    return str(value)

# --- SCRIPT START ---

# 1. Connect to Milvus and define the collection
print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Dropped existing collection: {collection_name}")

fields = [
    
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
    
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
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
print("Starting to process the entire Parquet file...")

for batch_num, batch in enumerate(scanner.to_batches()):
    df_batch = batch.to_pandas()
    
    data_for_milvus = [
        df_batch['id'].astype(str).tolist(),
        df_batch['text'].apply(sanitize_value).tolist(),
        df_batch['vector'].tolist(),
        df_batch['drug_name'].apply(sanitize_value).tolist(),
        df_batch['section_title'].apply(sanitize_value).tolist(),
        df_batch['contains'].apply(sanitize_value).tolist(),
        df_batch['keywords'].apply(sanitize_value).tolist(),
        df_batch['therapeutic_class'].apply(sanitize_value).tolist(),
        df_batch['chemical_class'].apply(sanitize_value).tolist(),
        df_batch['habit_forming'].apply(sanitize_value).tolist(),
        df_batch['action_class'].apply(sanitize_value).tolist(),
        df_batch['record_type'].apply(sanitize_value).tolist(),
        df_batch['source'].apply(sanitize_value).tolist(),
    ]
    
    if not df_batch.empty:
        collection.upsert(data_for_milvus)
    
    total_added += len(df_batch)
    print(f"Upserted batch {batch_num + 1} with {len(df_batch)} entries (total upserted: {total_added}).")
    
    # --- ADDED: Explicitly free up memory after each batch ---
    del df_batch
    del data_for_milvus
    gc.collect()
    # ---------------------------------------------------------

# 3. Finalize the collection for searching
print("\nFlushing data to disk...")
collection.flush()

print("Creating index for the vector field...")
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}
collection.create_index(field_name="vector", index_params=index_params)
print("Index created successfully.")

print("Loading collection into memory...")
collection.load()
print("Collection loaded.")

print(f"\n✅ Success! All {total_added} entries were added to the Milvus collection '{collection_name}'.")