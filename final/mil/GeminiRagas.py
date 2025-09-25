import os
import pandas as pd
import ast
import re
import ftfy
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_similarity, answer_correctness
# 🔄 Switched to Google's Gemini model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. Load Environment Variables ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("🔴 GOOGLE_API_KEY not found in .env file.")
print("✅ Google API key loaded successfully.")

# --- 2. Initialize Models ---
# Using the free Gemini Pro model as the evaluator LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)
embeddings = HuggingFaceEmbeddings(model_name='pritamdeka/S-BioBert-snli-multinli-stsb')
print("✅ Models initialized (Gemini Pro, S-BioBert).")

# --- 3. Define the Cleaning Function ---
def clean_text(text):
    text = str(text)
    text = ftfy.fix_text(text)
    text = re.sub(r'_x[0-9a-fA-F]{4}_', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- 4. Load, Clean, and Prepare the Dataset ---
df = pd.read_excel("ragas_evaluation_dataset.xlsx")
print(f"📄 Dataset loaded with {len(df)} rows.")

# Rename columns to what Ragas expects
df.rename(columns={
    'user_input': 'question',
    'response': 'answer',
    'reference': 'ground_truth',
    'retrieved_contexts': 'context'
}, inplace=True)

# Apply the cleaning function to all text columns
print("🧼 Cleaning and sanitizing all text data...")
for col in ['question', 'answer', 'ground_truth', 'context']:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)
print("✅ Data sanitization complete.")

# Process contexts and ground_truth after cleaning
df['contexts'] = df['context'].apply(lambda x: [doc.strip() for doc in x.split('---') if doc.strip()])
df['ground_truth'] = df['ground_truth'].apply(lambda gt: str(gt).strip("[]'\""))

evaluation_dataset = Dataset.from_pandas(df)

# --- 5. Run Evaluation ---
metrics_to_evaluate = [faithfulness, context_precision, context_recall, answer_similarity, answer_correctness]

print("\n🚀 Starting Ragas evaluation with Gemini Pro...")
result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics_to_evaluate,
    llm=llm,
    embeddings=embeddings,
    raise_exceptions=False
)
print("🎉 Evaluation complete!")

# --- 6. Display and Save Results ---
result_df = result.to_pandas()
print("\n📊 Evaluation Results (DataFrame):")
print(result_df)
result_df.to_excel("ragas_gemini111.xlsx", index=False)
print("\n✅ DataFrame saved to 'ragas_evaluation_report_gemini.xlsx'")