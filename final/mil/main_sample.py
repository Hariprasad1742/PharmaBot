# main.py
import os
import pickle
from functools import lru_cache
from typing import List, Dict

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from scipy.sparse import coo_matrix, csr_matrix

# --- 1. CONFIGURATION ---
load_dotenv()

# Service Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = 'medication'
# Schema Field Names
PK_FIELD = "id"
VECTOR_FIELD = "vector"
SPARSE_VECTOR_FIELD = "sparse_vector"
TEXT_CHUNK_FIELD = "text"
METADATA_FIELDS = ["drug_name", "section_title", TEXT_CHUNK_FIELD]

# Model and Search Configuration
EMBEDDING_MODEL = 'pritamdeka/S-BioBert-snli-multinli-stsb'
LLM_MODEL = 'meta-llama/llama-4-maverick-17b-128e-instruct' # Adjusted to a common Groq model
NUM_CANDIDATES = 10
DEFAULT_TOP_K = 6

# Path to saved BM25 model
BM25_MODEL_PATH = 'bm25_model.pkl'
# In main.py, replace the old FINAL_SYSTEM_PROMPT

# --- 2. PROMPT TEMPLATE ---
FINAL_SYSTEM_PROMPT = """
### Persona
You are Pharmabot, an AI assistant with the persona of a skilled and precise medical writer.
### Core Instructions
1.  **Strictly Adhere to Context:** You MUST answer the user's question using ONLY the information provided in the "CONTEXT" section below. Do not use any prior knowledge.
2.  **Synthesize, Don't Apologize:** Your primary goal is to synthesize a direct answer from the provided text. Combine information from different sources if needed. Do NOT apologize or state that the information is incomplete. If the context provides any relevant information, use it to answer the question directly.
3.  **Handle Genuinely Missing Information:** Only if the CONTEXT contains absolutely NO relevant information to answer the question, you MUST state: "The provided information does not contain an answer to your question." Do not use this phrase if you can extract even a partial answer.
### Formatting Rules
- Use Markdown for clarity (e.g., bullet points for lists).
- Always **bold** the drug's name whenever it is mentioned.
### CONTEXT
{context}
### USER QUESTION
{query}
### Final Instruction
- Always end your entire response with this exact disclaimer on a new line, with no extra formatting:
This is for informational purposes only. Please consult a healthcare professional for medical advice.
"""

# --- 3. PYDANTIC MODELS & FASTAPI APP ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict]

app = FastAPI(
    title="Pharmabot API",
    description="A RAG system for medication questions."
)

# --- 4. CACHED MODELS & CLIENTS ---
@lru_cache(maxsize=None)
def get_embedding_model():
    """Initializes and returns the embedding model, cached for performance."""
    print("Initializing embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL)

@lru_cache(maxsize=None)
def get_groq_client():
    """Initializes and returns the Groq client, cached for performance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    print("Initializing Groq client...")
    return Groq(api_key=api_key)

@lru_cache(maxsize=None)
def get_bm25_encoder():
    """Loads the pre-fitted BM25 encoder."""
    if not os.path.exists(BM25_MODEL_PATH):
        raise ValueError(f"BM25 model file not found at {BM25_MODEL_PATH}. Save it during ingestion.")
    with open(BM25_MODEL_PATH, 'rb') as f:
        return pickle.load(f)

# --- 5. APPLICATION LIFECYCLE (STARTUP) ---
@app.on_event("startup")
def startup_event():
    """Connects to Milvus and loads the collection into memory."""
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token=TOKEN)
        if not client.has_collection(collection_name=COLLECTION_NAME):
            raise ConnectionError(f"FATAL: Collection '{COLLECTION_NAME}' does not exist.")
        
        print("Loading Milvus collection into memory...")
        client.load_collection(collection_name=COLLECTION_NAME)
        app.state.milvus_client = client
        print("✅ Collection loaded successfully. Application is ready.")
    except Exception as e:
        print(f"FATAL: Could not connect or load Milvus collection. Error: {e}")
        raise SystemExit(1) from e

# --- 6. CORE RAG FUNCTIONS ---
STOP_WORDS = set(["what", "are", "the", "is", "of", "for", "a", "an", "mg", "capsule", "tablet", "side", "effects", "uses", "how", "to", "take"])

# In main.py

def hybrid_retrieve_and_fuse(client: MilvusClient, query: str, top_k: int) -> List[Dict]:
    """Performs hybrid retrieval with sparse (BM25) and dense vectors."""
    # --- START: CORRECTED FILTER LOGIC ---
    
    # Find potential drug name words from the query
    drug_name_candidates = [
        word for word in query.split() 
        if word.istitle() or (len(word) > 4 and word.lower() not in STOP_WORDS)
    ]
    print(f"Potential drug name words for filtering: {drug_name_candidates}")

    # Build a more robust filter expression
    filter_parts = [f"drug_name like '%{word}%'" for word in drug_name_candidates]
    filter_expr = " and ".join(filter_parts) if filter_parts else ""
    
    print(f"Constructed filter expression: '{filter_expr}'")
    
    # --- END: CORRECTED FILTER LOGIC ---


    # Load BM25 encoder and encode query to sparse vector
    bm25_encoder = get_bm25_encoder()
    sparse_query_coo = bm25_encoder.encode_queries([query])[0]

    # Convert COO to dict format for Milvus (index: value)
    sparse_query_vector = {int(i): float(v) for i, v in zip(sparse_query_coo.col, sparse_query_coo.data)}

    # Sparse (BM25) search request
    sparse_params = {
        "metric_type": "IP",
        "params": {},
        "expr": filter_expr
    }
    sparse_request = AnnSearchRequest(
        data=[sparse_query_vector],
        anns_field=SPARSE_VECTOR_FIELD,
        param=sparse_params,
        limit=NUM_CANDIDATES
    )

    # Dense (semantic) search request
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()
    dense_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
        "expr": filter_expr
    }
    dense_request = AnnSearchRequest(
        data=[query_vector],
        anns_field=VECTOR_FIELD,
        param=dense_params,
        limit=NUM_CANDIDATES
    )

    # Hybrid search with reranking
    results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[sparse_request, dense_request],
        ranker=WeightedRanker(0.6, 0.4),
        limit=top_k,
        output_fields=METADATA_FIELDS + [PK_FIELD]
    )

    # Extract and return fused results
    fused_docs = [hit['entity'] for hit in results[0]]
    if not fused_docs:
        return []
    return fused_docs

def generate_response(query: str, context: List[Dict]) -> str:
    """Generates a response by directly feeding the context to a robust prompt."""
    client = get_groq_client()
    
    formatted_context = "\n---\n".join(
        f"Drug: {doc.get('drug_name', 'N/A')}\nSection: {doc.get('section_title', 'N/A')}\nInformation: {doc.get(TEXT_CHUNK_FIELD, '')}"
        for doc in context
    )
    
    # SIMPLIFIED LOGIC: The robust main prompt handles all cases, removing the need
    # for a separate, error-prone answerability check.
    final_prompt = FINAL_SYSTEM_PROMPT.format(context=formatted_context, query=query)
        
    completion = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": final_prompt}], temperature=0.1
    )
    return completion.choices[0].message.content

# --- 7. API ENDPOINT ---
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: Request, query_request: QueryRequest):
    """Handles a user's question by retrieving context, generating an answer, and returning it with source documents."""
    try:
        milvus_client = request.app.state.milvus_client
        fused_docs = hybrid_retrieve_and_fuse(milvus_client, query_request.question, NUM_CANDIDATES)
        if not fused_docs:
            raise HTTPException(status_code=404, detail="Could not find any relevant documents in the database.")

        final_context = fused_docs[:query_request.top_k]
        answer = generate_response(query_request.question, final_context)
        
        return QueryResponse(answer=answer, source_documents=final_context)
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")