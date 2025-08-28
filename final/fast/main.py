import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from functools import lru_cache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# --- CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Make sure it is set in your .env file.")
groq_client = Groq(api_key=api_key)

DB_PATH = '../chroma_db_test'
COLLECTION_NAME = 'medication_embeddings_test'
EMBEDDING_MODEL = 'pritamdeka/S-BioBert-snli-multinli-stsb'
LLM_MODEL = 'llama3-8b-8192'
NUM_RESULTS_TO_FETCH = 3

# --- Pydantic Models for API Data Validation ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = NUM_RESULTS_TO_FETCH

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Pharmabot API",
    description="An API for querying the medication RAG system.",
)

# --- Caching Models for Performance ---
@lru_cache(maxsize=None)
def get_embedding_model():
    print("Initializing embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL, device='cuda')

@lru_cache(maxsize=None)
def get_db_collection():
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)

# --- CORE RAG FUNCTIONS ---
def retrieve_context(query: str, top_k: int) -> list:
    model = get_embedding_model()
    collection = get_db_collection()
    
    query_vector = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["metadatas"]
    )
    return results['metadatas'][0]

def generate_response(query: str, context: list) -> str:
    # Use .get() for safe access to the 'text' key
    formatted_context = "\n\n---\n\n".join([item.get('text', '') for item in context])
    
    # --- NEW, TUNED SYSTEM PROMPT BASED ON YOUR TEMPLATE ---
    # This is the new system_prompt
    system_prompt = """
    **Persona:** You are Pharmabot, a knowledgeable and helpful AI assistant specializing in pharmaceuticals.

    **Core Instruction:** Your primary goal is to answer the user's question accurately and exclusively using the provided `CONTEXT`. Do not use any external knowledge.

    **Tone and Style:** Your response must be conversational, clear, and easy to understand. Avoid robotic, structured lists or headers like "Relevant Details:".
     
    **Response Flow:**      
    1.  Start with a direct answer to the user's question.
    2.  Naturally integrate important details from the context, such as what the medication is used for, how it works, and any relevant warnings.
    3.  If the context does not contain the answer, state that clearly.
    4.  **Crucially, always end your response with this exact disclaimer:** "This is for informational purposes only. Please consult a healthcare professional for medical advice."
    """
    
    user_prompt = f"""CONTEXT FROM MEDICATION DATABASE:
    {formatted_context}

    USER QUESTION:
    {query}
    """

    try:
        completion = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Low temperature for factual, less creative answers
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with the LLM.")

# --- API ENDPOINT ---
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        context_data = retrieve_context(request.question, request.top_k)
        if not context_data:
            raise HTTPException(status_code=404, detail="Could not find relevant information in the database.")

        answer = generate_response(request.question, context_data)
        
        return QueryResponse(
            answer=answer,
            source_documents=context_data
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")