# main.py
import os
from functools import lru_cache
from typing import List, Dict, Optional

from pymilvus import Collection, utility, connections
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Make sure it is set in your .env file.")
groq_client = Groq(api_key=api_key)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = 'medication'

# --- Schema Field Names ---
# NOTE: Your loading script used 'vector', but your RAG app code used 'v'. 
# I've changed VECTOR_FIELD to 'vector' to match the loading script. Please verify.
PK_FIELD = "id"
VECTOR_FIELD = "vector" 
TEXT_CHUNK_FIELD = "text"
METADATA_FIELDS = ["drug_name", "section_title", "text"]

# --- Model and Search Configuration ---
EMBEDDING_MODEL = 'pritamdeka/S-BioBert-snli-multinli-stsb'
LLM_MODEL = 'llama-3.1-8b-instant'
NUM_CANDIDATE_RESULTS = 20
FINAL_TOP_K = 3
    
# --- NEW: Limit for BM25 Index ---
MAX_DOCS_FOR_BM25 = 100000 # 1 Lakh

TOPIC_KEYWORD_TO_SECTION_MAP = {
    "side effect": "SideEffect", "benefit": "ProductUses", "use": "ProductUses",
    "used for": "ProductUses", "how it works": "HowWorks", "mechanism": "HowWorks",
    "warning": "SafetyAdvice", "advice": "SafetyAdvice", "safety": "SafetyAdvice",
    "tip": "QuickTips", "dosage": "HowToUse", "how to take": "HowToUse",
    "alternatives": "alternatives", "options": "alternatives"
}

class QueryRequest(BaseModel):
    question: str
    top_k: int = FINAL_TOP_K
class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict]

app = FastAPI(title="Pharmabot API (Milvus Edition)", description="An API for querying the medication RAG system using Milvus.")

bm25_index: Optional[BM25Okapi] = None
document_store: Optional[List[Dict]] = None
doc_id_to_doc: Optional[Dict[str, Dict]] = None

@lru_cache(maxsize=None)
def get_embedding_model():
    print("Initializing embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL, device='cuda')

def get_db_collection():
    return Collection(name=COLLECTION_NAME)

@app.on_event("startup")
def startup_event():
    """Connect to Milvus, load a subset of a collection, and build the BM25 index."""
    global bm25_index, document_store, doc_id_to_doc
    
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        if not utility.has_collection(COLLECTION_NAME):
            raise ConnectionError(f"FATAL: Collection '{COLLECTION_NAME}' does not exist.")
        
        collection = get_db_collection()
        print("Loading Milvus collection into memory...")
        collection.load()
        print("Collection loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not connect to or load Milvus collection. Error: {e}")
        return

    print("Building BM25 index from a subset of Milvus documents...")
    try:
        num_entities = collection.num_entities
        print(f"Found {num_entities} total entities in the collection.")
        
        all_docs_milvus = []
        
        iterator = collection.query_iterator(
            expr=f'{PK_FIELD} != ""',
            output_fields=[PK_FIELD, TEXT_CHUNK_FIELD] + METADATA_FIELDS,
            batch_size=1000
        )
        
        # --- MODIFIED LOOP with a limit ---
        while True:
            batch = iterator.next()
            if not batch:
                break
            all_docs_milvus.extend(batch)
            print(f"Fetched {len(all_docs_milvus)} of {num_entities} entities...")
            
            # Stop once we've reached our defined limit
            if len(all_docs_milvus) >= MAX_DOCS_FOR_BM25:
                print(f"Reached the limit of {MAX_DOCS_FOR_BM25} documents for the BM25 index.")
                # We also need to trim the list to be exactly the max size
                all_docs_milvus = all_docs_milvus[:MAX_DOCS_FOR_BM25]
                break
        
        iterator.close()
        # --- END OF MODIFICATION ---

        if not all_docs_milvus:
            print(f"Warning: No documents found. BM25 index will be empty.")
            return

        print(f"Finished fetching. Building BM25 index from {len(all_docs_milvus)} documents (this may take a moment)...")
        document_store = []
        for doc_data in all_docs_milvus:
            metadata = {field: doc_data.get(field) for field in METADATA_FIELDS}
            document_store.append({
                "id": doc_data[PK_FIELD],
                "document": doc_data[TEXT_CHUNK_FIELD],
                "metadata": metadata
            })

        doc_id_to_doc = {doc['id']: doc for doc in document_store}
        tokenized_corpus = [doc['document'].split(" ") for doc in document_store]
        bm25_index = BM25Okapi(tokenized_corpus)
        print("✅ BM25 index built successfully.")

    except Exception as e:
        print(f"Error building BM25 index: {e}")

@app.on_event("shutdown")
def shutdown_event():
    print("Disconnecting from Milvus...")
    connections.disconnect("default")

def extract_drug_name(query: str, all_drug_names: set) -> Optional[str]:
    query_lower = query.lower()
    for drug in all_drug_names:
        if drug and drug.lower() in query_lower:
            return drug
    return None

def rerank_context(query: str, documents: List[Dict], top_k: int) -> List[Dict]:
    if not documents: return []
    reranked_docs = []
    for doc in documents:
        rerank_prompt = f"""User Question: "{query}"\n\nDocument: "{doc['metadata'].get('text', '')}"\n\nBased on the document, is it highly relevant to answering the user's question? Respond with only 'Yes' or 'No'."""
        try:
            completion = groq_client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": rerank_prompt}], max_tokens=5, temperature=0.0)
            choice = completion.choices[0].message.content.strip().lower()
            if 'yes' in choice: reranked_docs.append(doc)
        except Exception as e:
            print(f"Error during reranking: {e}")
            continue 
    return reranked_docs[:top_k]

def hybrid_retrieve_context(query: str, top_k: int) -> List[Dict]:
    if bm25_index is None or doc_id_to_doc is None:
        raise HTTPException(status_code=503, detail="Server is starting up, BM25 index not ready. Please try again.")

    query_lower = query.lower()
    all_drug_names = set(doc['metadata'].get('drug_name') for doc in document_store if doc['metadata'].get('drug_name'))

    filter_parts = []
    extracted_drug = extract_drug_name(query, all_drug_names)
    if extracted_drug:
        filter_parts.append(f'drug_name == "{extracted_drug}"')

    for keyword, section in TOPIC_KEYWORD_TO_SECTION_MAP.items():
        if keyword in query_lower:
            filter_parts.append(f'section_title == "{section}"')
            break

    milvus_filter_expr = " and ".join(filter_parts) if filter_parts else ""
    print(f"Using Milvus filter: '{milvus_filter_expr}'")

    model = get_embedding_model()
    collection = get_db_collection()
    query_vector = model.encode(query).tolist()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    semantic_results = collection.search(
        data=[query_vector], anns_field=VECTOR_FIELD, param=search_params,
        limit=top_k, expr=milvus_filter_expr, output_fields=[PK_FIELD]
    )

    semantic_doc_ids = [hit.entity.get(PK_FIELD) for hit in semantic_results[0]]
    tokenized_query = query.split(" ")
    keyword_docs = bm25_index.get_top_n(tokenized_query, document_store, n=top_k)
    keyword_doc_ids = [doc['id'] for doc in keyword_docs]

    fused_scores, k = {}, 60
    for i, doc_id in enumerate(semantic_doc_ids):
        if doc_id not in fused_scores: fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + i + 1)
    for i, doc_id in enumerate(keyword_doc_ids):
        if doc_id not in fused_scores: fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + i + 1)

    reranked_ids = sorted(fused_scores.keys(), key=fused_scores.get, reverse=True)
    final_docs = [doc_id_to_doc[doc_id] for doc_id in reranked_ids if doc_id in doc_id_to_doc]
    return final_docs[:top_k]

def generate_response(query: str, context: list) -> str:
    formatted_context = "\n\n---\n\n".join([item['metadata'].get('text', '') for item in context])
    system_prompt = """
    **Persona:** You are Pharmabot, an AI assistant. Your persona is that of a skilled medical writer who is precise, clear, and prioritizes structured information.
    **Core Principles:**
    1.  **Context is King:** You MUST answer exclusively using the provided `CONTEXT`.
    2.  **Be Direct and Focused:** Answer **only** the user's specific question.
    3.  **Focus on the Subject:** Your answer must ONLY be about the specific drug in the `USER QUESTION`.
    4.  **Handle Missing Information:** If the context does not contain the answer, state that clearly.
    **Formatting Rules:**
    - Use Markdown for clarity: Bold drug names and use bullet points for lists.
    **Final Instruction:**
    - Always end your entire response with this exact disclaimer on a new line: "This is for informational purposes only. Please consult a healthcare professional for medical advice."
    """
    user_prompt = f"CONTEXT:\n{formatted_context}\n\nUSER QUESTION:\n{query}"
    completion = groq_client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.1)
    return completion.choices[0].message.content

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
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
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred")