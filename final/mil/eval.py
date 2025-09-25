import os
import time
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm # For a nice progress bar

# --- 1. Load Environment Variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("🔴 GROQ_API_KEY not found. Make sure it's in your .env file.")
print("✅ API key loaded successfully.")

# --- 2. Configuration ---
EVALUATION_DATASET_PATH = "medgemma_ragas_dataset.xlsx"
EMBEDDING_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'
# This is an example model name; Groq supports models like 'llama3-70b-8192' or 'mixtral-8x7b-32768'
# Please check Groq's documentation for currently available models. 'moonshotai/kimi-k2-instruct-0905' is not a standard Groq model.
# Using a valid Groq model for this example:
LLM_MODEL_NAME = 'gemma2-9b-it' 

# --- 3. Load and Prepare the Dataset ---
try:
    df = pd.read_excel(EVALUATION_DATASET_PATH)
    print(f"📄 Dataset '{EVALUATION_DATASET_PATH}' loaded successfully with {len(df)} rows.")
except FileNotFoundError:
    raise FileNotFoundError(f"🔴 Error: The file '{EVALUATION_DATASET_PATH}' was not found.")

# Rename columns to meet Ragas requirements
df.rename(columns={
    'user_input': 'question',
    'response': 'answer',
    'reference': 'ground_truth',
    'retrieved_contexts': 'context' # Renaming to 'context' to be processed next
}, inplace=True)
print("✅ Renamed columns to meet Ragas requirements.")

# Convert the context string into a list of strings
if 'context' in df.columns and 'contexts' not in df.columns:
    # Ensure context is treated as a string before splitting
    df['contexts'] = df['context'].astype(str).apply(lambda x: [doc.strip() for doc in x.split('---')])
    print("✅ Converted 'context' column to 'contexts' list.")

required_columns = ['question', 'ground_truth', 'contexts', 'answer']
if not all(col in df.columns for col in required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    raise ValueError(f"🔴 Dataset is missing required columns: {missing}")

evaluation_dataset = Dataset.from_pandas(df)

# --- 4. Initialize Models ---
llm = ChatGroq(model_name=LLM_MODEL_NAME, groq_api_key=groq_api_key, temperature=0)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cuda'}  # Use 'cuda' if you have a GPU, otherwise 'cpu'
)
print(f"✅ LLM ({LLM_MODEL_NAME}) and Embedding Model ({EMBEDDING_MODEL_NAME}) initialized.")
# --- 5. Define Metrics and Run Evaluation with Delays ---
metrics_to_evaluate = [
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
    answer_similarity,
]

# Copy to store results
result_df = df.copy()

print("\n🚀 Starting Ragas evaluation with per-record and per-metric delay...")

# Iterate through each record (row) in the dataset
for i in tqdm(range(len(df)), desc="Evaluating Records"):
    row_dataset = Dataset.from_pandas(df.iloc[[i]])  # one record at a time
    
    for metric in metrics_to_evaluate:
        # Run evaluation for this record and metric
        result = evaluate(
            dataset=row_dataset,
            metrics=[metric],
            llm=llm,
            embeddings=embeddings
        )

        # Extract the score
        score = result.to_pandas()[metric.name].iloc[0]
        result_df.loc[i, metric.name] = score

        print(f"✅ Record {i+1}, Metric '{metric.name}' = {score}")
        
        # Sleep per-record, per-metric
        time.sleep(2)  # adjust to 5/10 sec as you need

print("\n🎉 Evaluation complete!")

# --- 6. Save Results ---
output_filename = "kimi_results_gran.xlsx"
result_df.to_excel(output_filename, index=False)
print(f"\n✅ DataFrame with all results saved to '{output_filename}'")
