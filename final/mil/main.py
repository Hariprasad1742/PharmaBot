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

# --- NEW IMPORTS ---
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
# --------------------

# --- 1. CONFIGURATION ---
load_dotenv()

# Service Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = 'medication' # Updated to your new collection name
# Schema Field Names
PK_FIELD = "id"
VECTOR_FIELD = "vector"
SPARSE_VECTOR_FIELD = "sparse_vector"
TEXT_CHUNK_FIELD = "text"
METADATA_FIELDS = ["drug_name", "section_title", TEXT_CHUNK_FIELD]

# Model and Search Configuration
EMBEDDING_MODEL = 'pritamdeka/S-BioBert-snli-multinli-stsb'
LLM_MODEL = 'llama-3.3-70b-versatile'
NUM_CANDIDATES = 10
DEFAULT_TOP_K = 6

# --- UPDATED: Path to the new sparse model ---
SPARSE_MODEL_PATH = 'sparse_model.pkl'

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
- Don't use starter phrases like "According to the provided information".
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
    print("Initializing embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL)

@lru_cache(maxsize=None)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    print("Initializing Groq client...")
    return Groq(api_key=api_key)

# --- UPDATED: Function to load the new Gensim sparse model components ---
@lru_cache(maxsize=None)
def get_sparse_model_components():
    """Loads the pre-fitted Gensim dictionary and TF-IDF model."""
    if not os.path.exists(SPARSE_MODEL_PATH):
        raise ValueError(f"Sparse model file not found at {SPARSE_MODEL_PATH}. Please run the 'prepare_models.py' script first.")
    with open(SPARSE_MODEL_PATH, 'rb') as f:
        models = pickle.load(f)
    return models['dictionary'], models['tfidf_model']

# --- 5. APPLICATION LIFECYCLE (STARTUP) ---
@app.on_event("startup")
def startup_event():
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
# --- NEW: Text processing function (must be identical to ingestion script) ---
stop_words_nltk = set(stopwords.words('english'))
def tokenize(text):
    return [
        word for word in word_tokenize(str(text).lower())
        if word.isalpha() and word not in stop_words_nltk
    ]

def hybrid_retrieve_and_fuse(client: MilvusClient, query: str, top_k: int) -> List[Dict]:
    """Performs hybrid retrieval with sparse (TF-IDF) and dense vectors."""
    drug_name_candidates = [
        word for word in query.split() 
        if word.istitle() or (len(word) > 4 and word.lower() not in STOP_WORDS)
    ]
    filter_parts = [f"drug_name like '%{word}%'" for word in drug_name_candidates]
    filter_expr = " and ".join(filter_parts) if filter_parts else ""
    print(f"Constructed filter expression: '{filter_expr}'")

    # --- UPDATED: Sparse vector generation using Gensim TF-IDF ---
    # 1. Load the dictionary and TF-IDF model
    dictionary, tfidf_model = get_sparse_model_components()
    
    # 2. Tokenize the query using the same function as ingestion
    query_tokens = tokenize(query)
    
    # 3. Convert tokenized query to a Bag-of-Words (BoW) vector
    query_bow = dictionary.doc2bow(query_tokens)
    
    # 4. Convert BoW vector to a TF-IDF vector
    query_tfidf = tfidf_model[query_bow]
    
    # 5. Convert Gensim's list-of-tuples format to Milvus' dictionary format
    sparse_query_vector = {term_id: float(score) for term_id, score in query_tfidf}
    # -------------------------------------------------------------

    # Sparse (TF-IDF) search request
    sparse_params = { "metric_type": "IP", "params": {}, "expr": filter_expr }
    sparse_request = AnnSearchRequest(
        data=[sparse_query_vector],
        anns_field=SPARSE_VECTOR_FIELD,
        param=sparse_params,
        limit=NUM_CANDIDATES
    )

    # Dense (semantic) search request
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()
    dense_params = { "metric_type": "COSINE", "params": {"nprobe": 10}, "expr": filter_expr }
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
        ranker=WeightedRanker(0.7, 0.3  ), # Adjusted weights for TF-IDF vs semantic
        limit=top_k,
        output_fields=METADATA_FIELDS + [PK_FIELD]
    )

    fused_docs = [hit['entity'] for hit in results[0]] if results else []
    return fused_docs

def generate_response(query: str, context: List[Dict]) -> str:
    client = get_groq_client()
    formatted_context = "\n---\n".join(
        f"Drug: {doc.get('drug_name', 'N/A')}\nSection: {doc.get('section_title', 'N/A')}\nInformation: {doc.get(TEXT_CHUNK_FIELD, '')}"
        for doc in context
    )
    final_prompt = FINAL_SYSTEM_PROMPT.format(context=formatted_context, query=query)
    completion = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": final_prompt}], temperature=0.1
    )
    return completion.choices[0].message.content

# --- 7. API ENDPOINT ---
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: Request, query_request: QueryRequest):
    try:
        milvus_client = request.app.state.milvus_client
        fused_docs = hybrid_retrieve_and_fuse(milvus_client, query_request.question, query_request.top_k)
        if not fused_docs:
            raise HTTPException(status_code=404, detail="Could not find any relevant documents in the database.")
        
        final_context = fused_docs # Use all returned docs up to top_k
        answer = generate_response(query_request.question, final_context)
        
        return QueryResponse(answer=answer, source_documents=final_context)
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")