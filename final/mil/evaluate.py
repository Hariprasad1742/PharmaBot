# evaluate.py (Final version with Groq LLM and local embeddings)
import os
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables (like your GROQ_API_KEY) from a .env file
load_dotenv()

# --- CONFIGURATION ---
EVALUATION_DATA_FILE = "evaluation_data.xlsx"
EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
# ---------------------


# --- 1. LOAD YOUR COMPLETED DATASET ---
print(f"Loading evaluation data from '{EVALUATION_DATA_FILE}'...")
try:
    df = pd.read_excel(EVALUATION_DATA_FILE)
    df['contexts'] = df['contexts'].apply(lambda x: eval(x))
    print(f"Found {len(df)} records to evaluate.")
except FileNotFoundError:
    print(f"ERROR: The file '{EVALUATION_DATA_FILE}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the Excel file: {e}")
    exit()

dataset = Dataset.from_pandas(df)


# --- 2. CONFIGURE THE EVALUATION MODELS ---
# Configure the LLM for judging (Groq)
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Make sure it's in your .env file.")

groq_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

# Configure the embedding model (your local S-BioBert)
hf_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cuda'} # Use GPU
)


# --- 3. RUN THE EVALUATION ---
print("Running RAGAs evaluation... (This may take a few minutes)")
# The first time you run this, it may take extra time to download the embedding model.

result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    # --- THIS IS THE KEY CHANGE ---
    # We pass both our Groq LLM and our local embedding model.
    llm=groq_llm,
    embeddings=hf_embeddings
)

print("Evaluation complete.")


# --- 4. VIEW THE RESULTS ---
df_results = result.to_pandas()
print("\n--- RAG Evaluation Results ---")
print(df_results)
print("----------------------------")