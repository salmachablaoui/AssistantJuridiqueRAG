# app/services/chat_service.py
import requests
from app.config import OLLAMA_URL, CHAT_MODEL

def ask_llm(question: str, context: str) -> str:
    """Ask the LLM a question with context"""
    
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context provided.
If you cannot answer using the context, say "I don't have enough information to answer that."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"