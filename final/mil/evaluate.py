import os
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

# --- 1. Load Environment Variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("🔴 GROQ_API_KEY not found. Make sure it's in your .env file.")
print("✅ API key loaded successfully.")

# --- 2. Configuration ---
EVALUATION_DATASET_PATH = "ragas_evaluation_dataset.xlsx"
EMBEDDING_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'
LLM_MODEL_NAME = 'moonshotai/kimi-k2-instruct-0905'  # Correct Groq model ID

# --- 3. Load and Prepare the Dataset ---
try:
    df = pd.read_excel(EVALUATION_DATASET_PATH)
    print(f"📄 Dataset '{EVALUATION_DATASET_PATH}' loaded successfully with {len(df)} rows.")
except FileNotFoundError:
    raise FileNotFoundError(f"🔴 Error: The file '{EVALUATION_DATASET_PATH}' was not found.")

#  Ragas expects
df.rename(columns={
    'user_input': 'question',
    'response': 'answer',
    'reference': 'ground_truth',
    'retrieved_contexts': 'context' # Renaming to 'context' to be processed next
}, inplace=True)
print("✅ Renamed columns to meet Ragas requirements.")

if 'context' in df.columns and 'contexts' not in df.columns:
    df['contexts'] = df['context'].apply(lambda x: [doc.strip() for doc in str(x).split('---')])
    print("✅ Converted 'context' column to 'contexts' list.")

required_columns = ['question', 'ground_truth', 'contexts', 'answer']
if not all(col in df.columns for col in required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    raise ValueError(f"🔴 Dataset is missing required columns: {missing}")

evaluation_dataset = Dataset.from_pandas(df)

# --- 4. Initialize Models ---
llm = ChatGroq(model_name=LLM_MODEL_NAME, groq_api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cuda'}  # Use 'cuda' if you have a GPU
)
print(f"✅ LLM ({LLM_MODEL_NAME}) and Embedding Model ({EMBEDDING_MODEL_NAME}) initialized.")

# --- 5. Define Metrics and Run Evaluation ---
metrics_to_evaluate = [
    faithfulness,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
]

print("\n🚀 Starting Ragas evaluation... This may take a while depending on your dataset size.")
result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics_to_evaluate,
    llm=llm,
    embeddings=embeddings
)
print("🎉 Evaluation complete!")

# --- 6. Display Results ---
print("\n📊 Evaluation Results (Raw):")
print(result)

result_df = result.to_pandas()
print("\n📊 Evaluation Results (DataFrame):")
print(result_df.head())

result_df.to_excel("kimi_RAGAS_eval_report.xlsx", index=False)
print("\n✅ DataFrame saved '")