# evaluate.py
import os
import pandas as pd
import ast  # NEW: For safely evaluating string literals
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,  # NEW: Added for factual accuracy checking
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# --- CONFIGURATION ---
EVALUATION_DATA_FILE = "eval_data2.xlsx"
EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
LLM_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
RESULTS_OUTPUT_FILE = "ragas_evaluation_results.xlsx"

def load_and_prepare_data(filepath: str) -> Dataset:
    """Loads data from an Excel file and prepares it for RAGAS evaluation."""
    print(f"Loading evaluation data from '{filepath}'...")
    try:
        df = pd.read_excel(filepath)
        
        # --- IMPROVED: Robust column handling ---
        # Rename columns to what RAGAS expects, handling potential variations
        rename_map = {
            "context": "contexts",
            "ground_truth": "ground_truths"
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Ensure 'contexts' and 'ground_truths' are lists, not strings
        if 'contexts' not in df.columns or 'ground_truths' not in df.columns:
            raise ValueError("Dataset must contain 'contexts' and 'ground_truths' columns.")

        # --- IMPROVED: Safer list conversion ---
        # Use ast.literal_eval instead of the unsafe eval()
        df['contexts'] = df['contexts'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        print(f"Found {len(df)} records to evaluate.")
        return Dataset.from_pandas(df)

    except FileNotFoundError:
        print(f"ERROR: The file '{filepath}' was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading or preparing the data: {e}")
        exit()

def run_evaluation(dataset: Dataset) -> pd.DataFrame:
    """Configures models and runs the RAGAS evaluation."""
    load_dotenv()
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Make sure it's in your .env file.")

    # Configure the LLM and Embedding models
    groq_llm = ChatGroq(model_name=LLM_MODEL_NAME, groq_api_key=groq_api_key)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'} # Use 'cpu' if you don't have a GPU
    )

    print("Running RAGAs evaluation... (This may take a few minutes)")
    
    # Define the metrics for evaluation
    metrics = [
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        answer_correctness,  # NEW: Evaluate against the ground truth
    ]
    
    # Run the evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=groq_llm,
        embeddings=hf_embeddings
    )
    
    print("Evaluation complete.")
    return result.to_pandas()

def report_and_save_results(df_results: pd.DataFrame, output_filename: str):
    """Prints results, summary statistics, and saves to an Excel file."""
    # Force pandas to display all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n--- RAG Evaluation Results (Full) ---")
    print(df_results)
    print("--------------------------------------")

    # --- NEW: Display summary statistics for a quick overview ---
    print("\n--- Summary Statistics ---")
    metric_cols = ['context_precision', 'faithfulness', 'answer_relevancy', 'context_recall', 'answer_correctness']
    print(df_results[metric_cols].describe())
    print("--------------------------")
    # -----------------------------------------------------------

    try:
        df_results.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"\n✅ Successfully saved evaluation results to '{output_filename}'")
    except Exception as e:
        print(f"\nError saving to Excel: {e}")

def main():
    """Main function to run the entire evaluation pipeline."""
    evaluation_dataset = load_and_prepare_data(EVALUATION_DATA_FILE)
    results_df = run_evaluation(evaluation_dataset)
    report_and_save_results(results_df, RESULTS_OUTPUT_FILE)

if __name__ == "__main__":
    main()