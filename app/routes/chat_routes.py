# app/routes/chat_routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.services.embedding_service import generate_embedding
from app.services.vector_service import search_similar_chunks
from app.services.chat_service import ask_llm

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/chat")
async def chat(request: QuestionRequest, db: Session = Depends(get_db)):
    """Ask a question about the documents"""
    
    if not request.question or not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    
    # Generate embedding for the question
    try:
        query_embedding = generate_embedding(request.question)
    except Exception as e:
        raise HTTPException(500, f"Failed to generate query embedding: {str(e)}")
    
    # Search for similar chunks
    similar_chunks = search_similar_chunks(db, query_embedding)
    
    if not similar_chunks:
        return {
            "response": "I couldn't find any relevant information in the documents.",
            "sources": []
        }
    
    # Build context from top chunks
    context_parts = []
    for chunk in similar_chunks[:3]:
        context_parts.append(f"[From document {chunk['document_id']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)
    
    # Generate answer
    try:
        answer = ask_llm(request.question, context)
    except Exception as e:
        raise HTTPException(500, f"Failed to generate answer: {str(e)}")
    
    return {
        "response": answer,
        "sources": [
            {
                "text": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                "similarity": round(c["similarity"], 3),
                "document_id": c["document_id"]
            }
            for c in similar_chunks[:3]
        ]
    }