# app/routes/upload_routes.py
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
import os
import shutil

from app.database import get_db
from app.models import Document
from app.config import UPLOAD_DIR
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunk_service import chunk_text
from app.services.embedding_service import generate_embedding
from app.services.vector_service import store_chunk

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload and index a PDF file"""
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are accepted")
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {str(e)}")
    
    # Extract text
    try:
        text = extract_text_from_pdf(file_path)
        if not text or not text.strip():
            raise HTTPException(400, "PDF contains no extractable text")
    except Exception as e:
        raise HTTPException(500, f"Failed to extract text: {str(e)}")
    
    # Create document record
    doc = Document(filename=file.filename, content=text[:10000])
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    # Process chunks
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "No text chunks generated")
    
    # Generate embeddings and store
    for idx, chunk in enumerate(chunks):
        try:
            embedding = generate_embedding(chunk)
            store_chunk(db, doc.id, chunk, idx, embedding)
        except Exception as e:
            print(f"Warning: Failed to process chunk {idx}: {e}")
            continue
    
    db.commit()
    
    return {
        "message": "PDF processed successfully",
        "filename": file.filename,
        "total_chunks": len(chunks),
        "successful_chunks": len(chunks)
    }