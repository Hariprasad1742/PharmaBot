import chromadb
import os
from groq import Groq
from typing import List, Dict, Optional
import json
import time


def load_env_file():
    """Load environment variables from .env file manually"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        return True
    except FileNotFoundError:
        return False


class MedicalRAGGroq:
    """Medical RAG system using Groq API for fast inference"""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "full_medication_database"):
        # Load environment variables manually
        load_env_file()
        
        # Connect to your existing ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)
        
        # Initialize Groq client with API key from .env file
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")
        
        self.groq_client = Groq(api_key=api_key)
        print(f"Groq API initialized (key: {api_key[:10]}...)")
    
    def create_medical_prompt_template(self, query: str, retrieved_docs: List[str], 
                                     custom_settings: Optional[Dict] = None):
        """Create structured prompt using your template"""
        
        template = {
            "persona": "You are a knowledgeable medical information assistant with expertise in pharmaceuticals and medication guidance.",
            "instruction": f"Answer the user's question about medications based ONLY on the provided context. If the information isn't in the context, say so clearly. Provide accurate, helpful medical information while emphasizing that users should consult healthcare professionals.",
            "context": f"Retrieved medication information:\n\n{self._format_retrieved_docs(retrieved_docs)}",
            "data_format": "Provide a clear, structured response with:\n1. Direct answer to the question\n2. Relevant medication details\n3. Important warnings or considerations\n4. Recommendation to consult healthcare provider",
            "audience": "General users seeking medication information who need clear, accurate, and safe guidance",
            "text": f"User Question: {query}",
            "tone": "Professional, helpful, and cautious. Use clear medical terminology with explanations when needed.",
            "data": "Base your response strictly on the provided medication database context. Do not add information from general knowledge."
        }
        
        if custom_settings:
            template.update(custom_settings)
            
        return self._build_final_prompt(template)
    
    def _format_retrieved_docs(self, docs: List[str]) -> str:
        """Format retrieved documents for context"""
        formatted = ""
        for i, doc in enumerate(docs, 1):
            formatted += f"--- Medication Record {i} ---\n{doc}\n\n"
        return formatted
    
    def _build_final_prompt(self, template: Dict) -> str:
        """Build the final prompt from template components"""
        prompt = f"""
# PERSONA
{template['persona']}
# INSTRUCTION
{template['instruction']}
# CONTEXT
{template['context']}
# DATA FORMAT
{template['data_format']}
# AUDIENCE
{template['audience']}
# TEXT
{template['text']}
# TONE
{template['tone']}
# DATA
{template['data']}
Please provide your response:
"""
        return prompt
    
    def retrieve_relevant_docs(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant documents from ChromaDB"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas']
            )
            
            documents = results['documents'][0] if results['documents'] else []
            return documents
            
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_response(self, prompt: str, model: str = "gemma2-9b-it") -> str:
        """Generate response using Groq API - ULTRA FAST!"""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a medical information assistant. Provide accurate, helpful information based only on the provided context."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=model,
                temperature=0.1,
                max_tokens=1000,
                top_p=0.9
            )
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self, user_query: str, n_docs: int = 3, model: str = "gemma2-9b-it", 
             custom_template: Optional[Dict] = None) -> Dict:
        """Main RAG chat function using Groq API"""
        
        print(f"Searching for: {user_query}")
        
        retrieved_docs = self.retrieve_relevant_docs(user_query, n_docs)
        
        if not retrieved_docs:
            return {
                "query": user_query,
                "response": "I couldn't find relevant information in the medication database.",
                "sources": 0,
                "model": model
            }
        
        print(f"Found {len(retrieved_docs)} relevant documents")
        
        prompt = self.create_medical_prompt_template(
            user_query, 
            retrieved_docs, 
            custom_template
        )
        
        print(f"Generating response with Groq ({model})...")
        response = self.generate_response(prompt, model)
        
        return {
            "query": user_query,
            "response": response,
            "sources": len(retrieved_docs),
            "retrieved_docs": retrieved_docs[:2],
            "model": model
        }


GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Most capable, best for complex medical queries",
    "llama-3.1-70b-versatile": "Very capable, slightly faster",
    "llama-3.1-8b-instant": "Fastest model, good for simple queries",
    "mixtral-8x7b-32768": "Good balance of speed and capability",
    "gemma2-9b-it": "Lightweight but capable"
}


def interactive_groq_rag():
    """Interactive interface for your medical RAG system using Groq"""
    
    print("Medical RAG System - Powered by Groq API")
    print("=" * 60)
    print("Ultra-fast medical information retrieval!")
    
    try:
        rag = MedicalRAGGroq()
        print("Connected to medication database and Groq API")
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        print("Make sure your .env file contains: GROQ_API_KEY=your_api_key_here")
        return
    
    print("\nExample queries:")
    print("• 'What medication helps with diabetes?'")
    print("• 'Side effects of pain medication'")
    print("• 'Blood pressure medications'")
    print("• 'Antibiotics for infection'")
    
    print("\nCommands:")
    print("• Type your question to get medical information")
    print("• 'models' - Change AI model")
    print("• 'quit' - Exit")
    print()
    
    current_model = "gemma2-9b-it"
    
    while True:
        try:
            user_query = input("Ask about medications: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_query.lower() == 'models':
                print("\nSelect model:")
                models_list = list(GROQ_MODELS.keys())
                for i, model in enumerate(models_list, 1):
                    print(f"{i}. {model}")
                
                try:
                    choice = int(input("Enter number: ")) - 1
                    if 0 <= choice < len(models_list):
                        current_model = models_list[choice]
                        print(f"Switched to: {current_model}")
                    else:
                        print("Invalid choice")
                except ValueError:
                    print("Please enter a valid number")
                continue
            
            if not user_query:
                continue
            
            start_time = time.time()
            result = rag.chat(user_query, model=current_model)
            end_time = time.time()
            response_time = end_time - start_time
            
            print("\n" + "="*60)
            print(f"GROQ MEDICAL RAG RESPONSE")
            print("="*60)
            print(f"Query: {result['query']}")
            print(f"Model: {result['model']}")
            print(f"Sources: {result['sources']} medications found")
            print(f"Response Time: {response_time:.2f} seconds")
            print("\nResponse:")
            print("-" * 40)
            print(result['response'])
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def check_setup():
    """Check if .env file and dependencies are properly set up"""
    print("CHECKING SETUP...")
    print("=" * 30)
    
    # Check .env file
    if os.path.exists('.env'):
        print(".env file found")
        
        # Load and check API key
        load_env_file()
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            print(f"GROQ_API_KEY found (starts with: {api_key[:10]}...)")
        else:
            print("GROQ_API_KEY not found in .env file")
            print("Add this line to your .env file:")
            print("   GROQ_API_KEY=your_actual_api_key_here")
            return False
    else:
        print(".env file not found")
        print("Create a .env file with:")
        print("   GROQ_API_KEY=your_actual_api_key_here")
        return False
    
    # Check dependencies
    try:
        import chromadb
        print("chromadb installed")
    except ImportError:
        print("chromadb not installed. Run: pip install chromadb")
        return False
    
    try:
        from groq import Groq
        print("groq installed")
    except ImportError:
        print("groq not installed. Run: pip install groq")
        return False
    
    # Check ChromaDB database
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("full_medication_database")
        count = collection.count()
        print(f"ChromaDB database found with {count} medications")
    except Exception as e:
        print(f"ChromaDB database issue: {str(e)}")
        return False
    
    print("\nAll checks passed! Ready to run RAG system.")
    return True


if __name__ == "__main__":
    print("Medical RAG System with Groq API")
    print("=" * 40)
    
    if check_setup():
        print("\nStarting interactive RAG system...")
        interactive_groq_rag()
    else:
        print("\nSetup incomplete. Please fix the issues above.")
