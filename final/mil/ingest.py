# ingest_to_milvus.py
import pyarrow.dataset as ds
import pandas as pd
import gc
import pickle
from pymilvus import MilvusClient, DataType
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# --- CONFIGURATION ---
parquet_path = r'H:\syntorion\pharmabot\pharma_new\final\finalmedication_embeddings_final.parquet'
batch_size = 100
collection_name = 'medicatdion'
MILVUS_URI = "http://localhost:19530"

# --- RUN CONFIGURATION ---
# For a new, clean run, set this to 1.
# To resume a failed run, set this to the batch number that failed.
START_FROM_BATCH = 1

# --- HELPER FUNCTIONS ---
# Ensure you have downloaded the necessary NLTK data first
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def tokenize(text):
    return [
        word for word in word_tokenize(str(text).lower())
        if word.isalpha() and word not in stop_words
    ]

def sanitize_value(value):
    if pd.isna(value) or value is None:
        return ""
    return str(value)

# --- SCRIPT START ---
print("🚀 Starting Part 2: Data Ingestion into Milvus (Bulk-Load Strategy)")

# === Step 1: Load Pre-trained Models ===
print("⚙️  Loading pre-trained TF-IDF model and dictionary...")
try:
    with open('sparse_model.pkl', 'rb') as f:
        models = pickle.load(f)
    dictionary = models['dictionary']
    tfidf_model = models['tfidf_model']
    print("✅ Models loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: sparse_model.pkl not found. Please run the 'prepare_models.py' script first.")
    exit()

# === Step 2: Connect to Milvus and Create Collection (WITHOUT INDEX) ===
print(f"⚙️  Connecting to Milvus at {MILVUS_URI}")
client = MilvusClient(uri=MILVUS_URI)

if START_FROM_BATCH == 1 and client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)
    print(f"Dropped existing collection: {collection_name}")

if not client.has_collection(collection_name=collection_name):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    
    # --- FULL SCHEMA DEFINITION ---
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
    schema.add_field(field_name="drug_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="section_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="contains", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="keywords", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="therapeutic_class", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="chemical_class", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="habit_forming", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="action_class", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="record_type", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    
    client.create_collection(collection_name, schema=schema)
    print(f"✅ Collection '{collection_name}' created without index.")
else:
    print(f"Collection '{collection_name}' already exists. Resuming insertion.")

# === Step 3: Process and Upsert ALL Data ===
print(f"\n🚀 Processing and upserting data, starting from batch {START_FROM_BATCH}...")
total_added_this_run = 0
skipped_count = 0
scanner = ds.dataset(parquet_path, format='parquet').scanner(batch_size=batch_size)
for batch_num, batch in enumerate(scanner.to_batches(), 1):
    if batch_num < START_FROM_BATCH:
        if batch_num % 500 == 0:
            print(f"Fast-forwarding... currently at batch {batch_num}")
        continue

    df_batch = batch.to_pandas()
    
    batch_tokens = df_batch['text'].apply(tokenize).tolist()
    batch_bow = [dictionary.doc2bow(tokens) for tokens in batch_tokens]
    tfidf_vectors = tfidf_model[batch_bow]
    
    sparse_vectors_milvus = []
    for doc_vector in tfidf_vectors:
        sparse_dict = {term_id: float(score) for term_id, score in doc_vector}
        sparse_vectors_milvus.append(sparse_dict)

    data_for_milvus = []
    for i in range(len(df_batch)):
        if not sparse_vectors_milvus[i]:
            skipped_count += 1
            doc_id = str(df_batch['id'].iloc[i])
            print(f"⚠️  WARNING: Skipping document with empty sparse vector. ID: {doc_id}")
            continue

        # --- FULL DATA ROW MAPPING ---
        row_data = {
            "id": str(df_batch['id'].iloc[i]),
            "text": sanitize_value(df_batch['text'].iloc[i]),
            "vector": df_batch['vector'].iloc[i],
            "drug_name": sanitize_value(df_batch['drug_name'].iloc[i]),
            "section_title": sanitize_value(df_batch['section_title'].iloc[i]),
            "contains": sanitize_value(df_batch['contains'].iloc[i]),
            "keywords": sanitize_value(df_batch['keywords'].iloc[i]),
            "therapeutic_class": sanitize_value(df_batch['therapeutic_class'].iloc[i]),
            "chemical_class": sanitize_value(df_batch['chemical_class'].iloc[i]),
            "habit_forming": sanitize_value(df_batch['habit_forming'].iloc[i]),
            "action_class": sanitize_value(df_batch['action_class'].iloc[i]),
            "record_type": sanitize_value(df_batch['record_type'].iloc[i]),
            "source": sanitize_value(df_batch['source'].iloc[i]),
            "sparse_vector": sparse_vectors_milvus[i]
        }
        data_for_milvus.append(row_data)

    if data_for_milvus:
        client.upsert(collection_name, data=data_for_milvus, timeout=120)
        total_added_this_run += len(data_for_milvus)

    print(f"-> Upserted batch {batch_num}, new entries in this run: {total_added_this_run} (skipped: {skipped_count})")
    
    del df_batch, data_for_milvus
    gc.collect()

# === Step 4: Create Index AFTER All Data is Inserted ===
print("\n⚙️  All data inserted. Now creating indexes... (This may take a while)")
client.flush(collection_name)
print("Flushed data successfully.")

print("Creating dense vector index...")
index_params_dense = MilvusClient.prepare_index_params()
index_params_dense.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
client.create_index(collection_name, index_params_dense)
print("✅ Dense vector index created.")

print("Creating sparse vector index...")
index_params_sparse = MilvusClient.prepare_index_params()
index_params_sparse.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
client.create_index(collection_name, index_params_sparse)
print("✅ Sparse vector index created.")

# === Step 5: Finalize ===
print("\nLoading collection into memory for searching...")
client.load_collection(collection_name)
current_entities = client.query(collection_name, output_fields=["count(*)"])[0]["count(*)"]
print(f"\n🎉 Success! The collection '{collection_name}' now contains {current_entities} total entries.")
print(f"Total entries skipped in this run due to empty text: {skipped_count}")