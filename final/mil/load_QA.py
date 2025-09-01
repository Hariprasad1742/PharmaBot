# load_data.py (Modified to process ALL data)
import pandas as pd
import pyarrow.parquet as pq
from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    connections,
)
import numpy as np
import gc # Garbage Collector interface

# --- 1. CONFIGURATION ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "QA"
DATA_SOURCE_FILE = r"H:\syntorion\pharmabot\pharma_new\final\qembed\QA_embeddings.parquet"
BATCH_SIZE = 100

# --- Column names from your Parquet file ---
ID_COLUMN = 'id'
TEXT_COLUMN = 'text'
METADATA_COLUMN = 'metadata'
EMBEDDING_COLUMN = 'embedding'
# -------------------------------------------

# --- 2. CONNECT TO MILVUS ---
print("Connecting to Milvus...")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
print("Connection successful.")

# --- 3. SETUP COLLECTION ---
if utility.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists. Dropping it for a clean start.")
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="drug_name", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, "Medication Q&A collection")
collection = Collection(COLLECTION_NAME, schema)
print(f"Collection '{COLLECTION_NAME}' created.")

# --- 4. RAM-EFFICIENT BATCH PROCESSING ---
print(f"Opening data source: '{DATA_SOURCE_FILE}'")
try:
    parquet_file = pq.ParquetFile(DATA_SOURCE_FILE)
except FileNotFoundError:
    print(f"ERROR: The file '{DATA_SOURCE_FILE}' was not found.")
    exit()

# REMOVED: Data subset logic to process the entire file.
total_rows = parquet_file.metadata.num_rows
total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
print(f"Total records in file: {total_rows}")
print(f"Starting upsert of all data in {total_batches} batches of size {BATCH_SIZE}...")

# Iterate over the file in chunks (batches)
for i, batch in enumerate(parquet_file.iter_batches(batch_size=BATCH_SIZE)):
    batch_df = batch.to_pandas()
    
    # Prepare data for the current batch
    ids = batch_df[ID_COLUMN].astype(str).tolist()
    texts = batch_df[TEXT_COLUMN].astype(str).tolist()
    drug_names = batch_df[METADATA_COLUMN].apply(lambda x: x.get('drug_name', 'General Info')).tolist()
    section_titles = batch_df[METADATA_COLUMN].apply(lambda x: x.get('section_title', 'General Topic')).tolist()
    embeddings = [e.tolist() if isinstance(e, np.ndarray) else e for e in batch_df[EMBEDDING_COLUMN]]
    
    data_to_upsert = [ids, texts, drug_names, section_titles, embeddings]
    
    # Upsert the batch
    collection.upsert(data_to_upsert)
    
    print(f"Upserted batch {i + 1}/{total_batches}.")
    
    # --- Clean up memory ---
    del batch_df
    del data_to_upsert
    gc.collect()
    # -----------------------

collection.flush()
print("\nData upsert complete.")

# --- 5. CREATE INDEX AND LOAD ---
print("Creating index for vector search...")
index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
collection.create_index(field_name="vector", index_params=index_params)
print("Index created.")

collection.load()
print(f"Collection '{COLLECTION_NAME}' loaded into memory.")

connections.disconnect("default")
print("Disconnected from Milvus.")