# main.py
import os
from functools import lru_cache
from typing import List, Dict

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# --- 1. CONFIGURATION ---
load_dotenv()

# Service Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
TOKEN = os.getenv("MILVUS_TOKEN") # Set your token in .env if needed
COLLECTION_NAME = 'medication'

# Schema Field Names
PK_FIELD = "id"
VECTOR_FIELD = "vector"
TEXT_CHUNK_FIELD = "text"
METADATA_FIELDS = ["drug_name", "section_title", TEXT_CHUNK_FIELD]

# Model and Search Configuration
EMBEDDING_MODEL = 'pritamdeka/S-BioBert-snli-multinli-stsb'
LLM_MODEL = 'llama3-8b-8192'
NUM_CANDIDATES = 10 # Number of results to fetch from each search before fusion
DEFAULT_TOP_K = 3   # Final number of documents to use for the answer

# --- 2. PROMPT TEMPLATE ---
FINAL_SYSTEM_PROMPT = """
### Persona
You are Pharmabot, an AI assistant with the persona of a skilled and precise medical writer.
### Core Instructions
1.  **Strictly Adhere to Context:** You MUST answer the user's question using ONLY the information provided in the "CONTEXT" section below. Do not use any prior knowledge.
2.  **Be Direct and Focused:** Answer ONLY the user's specific question.
3.  **Handle Missing Information:** If the "CONTEXT" does not contain the answer, you MUST state: "The provided information does not contain an answer to your question."
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
    title="Pharmabot API (Hybrid Search Edition)",
    description="A RAG system using Milvus with Hybrid Search (Keyword + Vector) and Rank Fusion."
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
        app.state.milvus_client = client # Store client in app state
        print("✅ Collection loaded successfully. Application is ready.")
    except Exception as e:
        print(f"FATAL: Could not connect or load Milvus collection. Error: {e}")
        # Exit if Milvus connection fails on startup
        raise SystemExit(1) from e

# --- 6. CORE RAG FUNCTIONS ---
def hybrid_retrieve_and_fuse(client: MilvusClient, query: str, top_k: int) -> List[Dict]:
    """
    Performs hybrid retrieval using keyword and vector search, then fuses the results.
    """
    # 1. Keyword Search
    keyword_expr = " or ".join([f"{TEXT_CHUNK_FIELD} like '%{word}%'" for word in query.split()])
    keyword_results = client.query(
        collection_name=COLLECTION_NAME,
        filter=keyword_expr,
        output_fields=METADATA_FIELDS + [PK_FIELD],
        limit=top_k
    )

    # 2. Vector Search
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    vector_search_hits = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field=VECTOR_FIELD,
        search_params=search_params,
        output_fields=METADATA_FIELDS + [PK_FIELD],
        limit=top_k
    )
    vector_results = [hit['entity'] for hit in vector_search_hits[0]]

    # 3. Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    all_results = [keyword_results, vector_results]
    for results in all_results:
        for rank, doc in enumerate(results):
            doc_id = doc[PK_FIELD]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (60 + rank + 1) # RRF with k=60

    if not fused_scores:
        return []

    reranked_ids = [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    
    # 4. Fetch final documents in the new order
    id_filter = f"{PK_FIELD} in {reranked_ids}"
    final_docs = client.query(COLLECTION_NAME, id_filter, output_fields=METADATA_FIELDS + [PK_FIELD])
    doc_map = {doc[PK_FIELD]: doc for doc in final_docs}
    
    return [doc_map[doc_id] for doc_id in reranked_ids if doc_id in doc_map]

def generate_response(query: str, context: List[Dict]) -> str:
    """Generates a response using the LLM based on the provided context."""
    formatted_context = "\n---\n".join(
        f"Drug: {doc.get('drug_name', 'N/A')}\nSection: {doc.get('section_title', 'N/A')}\nInformation: {doc.get(TEXT_CHUNK_FIELD, '')}"
        for doc in context
    )
    
    prompt = FINAL_SYSTEM_PROMPT.format(context=formatted_context, query=query)
    client = get_groq_client()
    
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return completion.choices[0].message.content

# --- 7. API ENDPOINT ---
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: Request, query_request: QueryRequest):
    """
    Handles a user's question by retrieving context, generating an answer,
    and returning it along with the source documents.
    """
    try:
        milvus_client = request.app.state.milvus_client
        
        # Retrieve a pool of candidate documents using hybrid search
        fused_docs = hybrid_retrieve_and_fuse(milvus_client, query_request.question, NUM_CANDIDATES)
        if not fused_docs:
            raise HTTPException(status_code=404, detail="Could not find any relevant documents in the database.")

        # Select the final top_k documents to use as context
        final_context = fused_docs[:query_request.top_k]
        
        # Generate the final answer
        answer = generate_response(query_request.question, final_context)
        
        return QueryResponse(answer=answer, source_documents=final_context)
        
    except HTTPException as http_exc:
        # Re-raise HTTPException to let FastAPI handle it
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")