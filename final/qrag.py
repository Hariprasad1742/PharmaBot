# File: query_rag.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- Configuration ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "medications_testing" # Must match the name in the indexing script

# --- Initialize API Clients and Models ---
load_dotenv()
client = QdrantClient(url=QDRANT_URL)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Loading embedding model for queries...")
embedding_model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb') # <-- CHANGE THIS
print("Embedding model loaded.")


def answer_question(query: str):
    """
    Performs a HYBRID RAG process: Embed, Retrieve using two methods, and Generate.
    """
    # 1. Embed the user's query
    query_embedding = embedding_model.encode(query).tolist()
    
    # 2. Define two parallel search queries (semantic and keyword)
    semantic_prefetch = models.Prefetch(
        query=query_embedding,
        limit=5
    )
    
    keyword_prefetch = models.Prefetch(
        query=query_embedding,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="text",
                    match=models.MatchText(text=query)
                )
            ]
        ),
        limit=5
    )

    # 3. Execute the hybrid search using Reciprocal Rank Fusion (RRF)
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[semantic_prefetch, keyword_prefetch],
        query=query_embedding, # Use RRF to rank the combined results
        limit=3 # Return the final top 3 results
    )
    
    # 4. Format the context from the results
    context = ""
    for result in search_results.points:
        context += f"Medication: {result.payload.get('medication_name', 'N/A')}\n"
        context += f"Information: {result.payload.get('text', 'N/A')}\n\n"
        
    # 5. Generate an answer using the improved prompt
    prompt = f"""
    You are a helpful pharmacy assistant. Your task is to answer the user's question by summarizing the provided context.
    Synthesize the information into a clear, easy-to-read paragraph.

    **Instructions:**
    - **Do not** use a question-and-answer (Q&A) format.
    - **Do not** repeat questions found in the context.
    - Base your answer **only** on the information given in the context below.

    **Context:**
    ---
    {context}
    ---

    **User's Question:** {query}

    **Your Answer:**
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    
    return chat_completion.choices[0].message.content

# --- Main execution block for interactive chat ---
if __name__ == '__main__':
    print("\n--- RAG System with Hybrid Search is Ready ---")
    # Keep asking questions until the user types 'exit'
    while True:
        user_query = input("\nAsk a question about medications (or type 'exit'): ")
        if user_query.lower() == 'exit':
            break
        
        final_answer = answer_question(user_query)
        print("\nAnswer:")
        print(final_answer)