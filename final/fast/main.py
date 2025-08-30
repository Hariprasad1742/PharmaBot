import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from functools import lru_cache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

# --- Import for BM25 ---
from rank_bm25 import BM25Okapi

# --- 1. CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Make sure it is set in your .env file.")
groq_client = Groq(api_key=api_key)

DB_PATH = 'H:/syntorion/pharmabot/pharma_new/final/chroma_db' 
COLLECTION_NAME = '1medication_embeddings_test' 
EMBEDDING_MODEL = 'pritamdeka/S-BioBert-snli-multinli-stsb'
LLM_MODEL = 'llama3-8b-8192'
NUM_CANDIDATE_RESULTS = 10 
FINAL_TOP_K = 3 

# --- NEW: Mapping query keywords to metadata section titles ---
TOPIC_KEYWORD_TO_SECTION_MAP = {
    "side effect": "SideEffect",
    "benefit": "ProductUses",
    "use": "ProductUses",
    "used for": "ProductUses",
    "how it works": "HowWorks",
    "mechanism": "HowWorks",
    "warning": "SafetyAdvice",
    "advice": "SafetyAdvice",
    "safety": "SafetyAdvice",
    "tip": "QuickTips",
    "dosage": "HowToUse",
    "how to take": "HowToUse"
}

# --- 2. Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = FINAL_TOP_K
class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict]

# --- 3. FastAPI App Initialization ---
app = FastAPI(title="Pharmabot API", description="An API for querying the medication RAG system.")

# --- 4. Global Variables & Cached Models --
bm25_index: Optional[BM25Okapi] = None
document_store: Optional[List[Dict]] = None
doc_id_to_doc: Optional[Dict[str, Dict]] = None

@lru_cache(maxsize=None)
def get_embedding_model():
    print("Initializing embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL, device='cuda')

@lru_cache(maxsize=None)
def get_db_collection():
    print(f"Connecting to ChromaDB collection: '{COLLECTION_NAME}'...")
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)

# --- 5. Application Startup Logic ---
@app.on_event("startup")
def build_bm25_index():
    # (This function remains unchanged)
    global bm25_index, document_store, doc_id_to_doc
    # ... (code is the same)
    print("Building BM25 index from ChromaDB documents...")
    try:
        collection = get_db_collection()
        all_docs = collection.get(include=["metadatas", "documents"])
    except Exception as e:
        print(f"FATAL: Could not connect to ChromaDB. Please check DB_PATH and COLLECTION_NAME. Error: {e}")
        return
    if not all_docs or not all_docs['documents']:
        print(f"Warning: No documents found in collection '{COLLECTION_NAME}'. BM25 index will be empty.")
        return
    document_store = [{"id": id, "document": doc, "metadata": meta} for id, doc, meta in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas'])]
    doc_id_to_doc = {doc['id']: doc for doc in document_store}
    tokenized_corpus = [doc['document'].split(" ") for doc in document_store]
    bm25_index = BM25Okapi(tokenized_corpus)
    print("BM25 index built successfully.")

# --- 6. Core RAG Functions ---
def extract_drug_name(query: str, all_drug_names: set) -> Optional[str]:
    # (This function remains unchanged)
    # ... (code is the same)
    query_lower = query.lower()
    for drug in all_drug_names:
        if drug and drug.lower() in query_lower:
            return drug
    return None

def rerank_context(query: str, documents: List[Dict], top_k: int) -> List[Dict]:
    # (This function remains unchanged)
    # ... (code is the same)
    if not documents: return []
    reranked_docs = []
    for doc in documents:
        rerank_prompt = f"""User Question: "{query}"\n\nDocument: "{doc['metadata'].get('text', '')}"\n\nBased on the document, is it highly relevant to answering the user's question? Respond with only 'Yes' or 'No'."""
        try:
            completion = groq_client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": rerank_prompt}], max_tokens=5, temperature=0.0)
            choice = completion.choices[0].message.content.strip().lower()
            if 'yes' in choice: reranked_docs.append(doc)
        except Exception as e:
            print(f"Error during reranking document ID {doc.get('id', 'N/A')}: {e}")
            continue 
    return reranked_docs[:top_k]

# --- MODIFIED RETRIEVAL FUNCTION ---
def hybrid_retrieve_context(query: str, top_k: int) -> List[Dict]:
    """Retrieves documents using a two-tiered metadata filter."""
    if bm25_index is None or doc_id_to_doc is None or document_store is None:
        raise HTTPException(status_code=503, detail="Server is starting up. Please try again in a moment.")
    
    query_lower = query.lower()
    all_drug_names = set(doc['metadata'].get('drug_name') for doc in document_store if doc['metadata'].get('drug_name'))
    
    # Build the filter list
    filters = []
    
    # Tier 1: Filter by drug name
    extracted_drug = extract_drug_name(query, all_drug_names)
    if extracted_drug:
        filters.append({"drug_name": {"$eq": extracted_drug}})

    # Tier 2: Filter by topic keyword
    for keyword, section in TOPIC_KEYWORD_TO_SECTION_MAP.items():
        if keyword in query_lower:
            filters.append({"section_title": {"$eq": section}})
            break # Use the first topic keyword found

    # Construct the final where clause for ChromaDB
    if len(filters) > 1:
        metadata_filter = {"$and": filters}
    elif len(filters) == 1:
        metadata_filter = filters[0]
    else:
        metadata_filter = None # No filter if no drug or topic found

    print(f"Using metadata filter: {metadata_filter}")

    # Perform ChromaDB query
    model = get_embedding_model()
    collection = get_db_collection()
    query_vector = model.encode(query).tolist()
    
    query_args = {
        "query_embeddings": [query_vector],
        "n_results": top_k,
        "include": ["metadatas"]
    }
    if metadata_filter:
        query_args["where"] = metadata_filter
    
    semantic_results = collection.query(**query_args)
    
    # The rest of the function (RRF) remains the same to fuse with keyword search
    semantic_doc_ids = [meta['doc_id'] for meta in semantic_results.get('metadatas', [[]])[0] if 'doc_id' in meta]
    tokenized_query = query.split(" ")
    keyword_docs = bm25_index.get_top_n(tokenized_query, document_store, n=top_k)
    keyword_doc_ids = [doc['id'] for doc in keyword_docs]

    fused_scores = {}
    k = 60
    for i, doc_id in enumerate(semantic_doc_ids):
        if doc_id not in fused_scores: fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + i + 1)
    for i, doc_id in enumerate(keyword_doc_ids):
        if doc_id not in fused_scores: fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + i + 1)

    reranked_ids = sorted(fused_scores.keys(), key=fused_scores.get, reverse=True)
    
    final_docs = [doc_id_to_doc[doc_id] for doc_id in reranked_ids if doc_id in doc_id_to_doc]
    return final_docs[:top_k]
# In main.py

def generate_response(query: str, context: list) -> str:
    """Generates the final answer using the refined context."""
    formatted_context = "\n\n---\n\n".join([item['metadata'].get('text', '') for item in context])
    
    # --- THIS IS THE FULL, CORRECT SYSTEM PROMPT ---
    system_prompt = """
    **Persona:** You are Pharmabot, an AI assistant. Your persona is that of a skilled medical writer who is precise, clear, and prioritizes structured information.

    **Core Principles:**
    1.  **Context is King:** You MUST answer exclusively using the provided `CONTEXT`.
    2.  **Be Direct and Focused:** This is your most important rule. Answer **only** the user's specific question. Do not add extra sections or volunteer information that was not asked for. For example, if the user asks for "side effects," provide only the side effects.
    3.  **Focus on the Subject:** The `CONTEXT` may contain info on multiple drugs. Your answer must ONLY be about the specific drug in the `USER QUESTION`. Ignore irrelevant information.
    4.  **Handle Missing Information:** If the context does not contain the answer to the specific question asked, state that clearly (e.g., "The provided context does not list the side effects for this medication.").

    **Formatting Rules:**
    - Use Markdown for clarity: Bold drug names (`**Drug Name**`) and use bullet points (`*`) for lists.

    **Final Instruction:**
    - Always end your entire response with this exact disclaimer on a new line: "This is for informational purposes only. Please consult a healthcare professional for medical advice."
    """
    
    user_prompt = f"CONTEXT FROM MEDICATION DATABASE:\n{formatted_context}\n\nUSER QUESTION:\n{query}"
    try:
        completion = groq_client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.1)
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with the LLM.")

# --- 7. API Endpoint ---
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    # (This function remains unchanged)
    # ... (code is the same)
    try:
        candidate_docs = hybrid_retrieve_context(request.question, NUM_CANDIDATE_RESULTS)
        if not candidate_docs:
            raise HTTPException(status_code=404, detail="Could not find any potentially relevant documents.")
        final_context_data = rerank_context(request.question, candidate_docs, request.top_k)
        if not final_context_data:
            raise HTTPException(status_code=404, detail="Could not find highly relevant information after reranking.")
        answer = generate_response(request.question, final_context_data)
        source_documents = [doc['metadata'] for doc in final_context_data]
        return QueryResponse(answer=answer, source_documents=source_documents)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred")