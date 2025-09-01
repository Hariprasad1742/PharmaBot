# check_schema.py
from pymilvus import utility, connections, Collection

# --- CONFIGURATION ---
# Make sure these match your setup
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "medication_embeddings_milvus_test"
# ---------------------

try:
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Connection successful.")

    if not utility.has_collection(COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
    else:
        collection = Collection(name=COLLECTION_NAME)
        schema = collection.schema

        print("\n--- Schema for collection: " + COLLECTION_NAME + " ---")
        for field in schema.fields:
            print(f"- Field Name: {field.name:<20} Data Type: {field.dtype}")
        print("--------------------------------------------------")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    connections.disconnect("default")
    print("Disconnected.")