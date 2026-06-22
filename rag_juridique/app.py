# =========================================================
# MULTILINGUAL LEGAL RAG SYSTEM
# FULL WORKING VERSION WITH OCR
# FASTAPI + FAISS + OLLAMA + OCR
# =========================================================

import os
import fitz
import faiss
import requests
import numpy as np
import pytesseract

from pdf2image import convert_from_path
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# =========================================================
# TESSERACT PATH (WINDOWS)
# =========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =========================================================
# CONFIG
# =========================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2"

DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

# =========================================================
# EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...")

embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding model loaded.")

# =========================================================
# GLOBAL STORAGE
# =========================================================

chunks_store = []
metadata_store = []

index = None
bm25 = None

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Multilingual Legal RAG",
    version="1.0"
)

# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "RAG API running successfully"
    }

# =========================================================
# PDF EXTRACTION + OCR
# =========================================================

def extract_text(pdf_path):

    text = ""

    try:

        # =====================================
        # NORMAL PDF EXTRACTION
        # =====================================

        doc = fitz.open(pdf_path)

        for page in doc:

            page_text = page.get_text("text")

            if page_text.strip():
                text += page_text + "\n"

        # =====================================
        # OCR IF TEXT EMPTY
        # =====================================

        if len(text.strip()) < 20:

            print("\nOCR MODE ACTIVATED")

            images = convert_from_path(pdf_path)

            for img in images:

                ocr_text = pytesseract.image_to_string(
                    img,
                    lang="ara+fra+eng"
                )

                text += ocr_text + "\n"

        print("\n========== EXTRACTED TEXT ==========")
        print(text[:1000])

        return text.strip()

    except Exception as e:

        print("\nEXTRACTION ERROR:")
        print(e)

        return ""

# =========================================================
# CHUNKING
# =========================================================

def chunk_text(text, chunk_size=250):

    words = text.split()

    chunks = []
    current = []

    for word in words:

        current.append(word)

        if len(current) >= chunk_size:

            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks

# =========================================================
# BUILD FAISS
# =========================================================

def build_faiss():

    global index

    if len(chunks_store) == 0:
        return

    print("\nCreating embeddings...")

    embeddings = embedder.encode(chunks_store)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("FAISS READY")

# =========================================================
# BUILD BM25
# =========================================================

def build_bm25():

    global bm25

    tokenized = [
        c.lower().split()
        for c in chunks_store
    ]

    bm25 = BM25Okapi(tokenized)

# =========================================================
# INGEST PDF
# =========================================================

def ingest_pdf(pdf_path, filename):

    global chunks_store
    global metadata_store

    print(f"\nPROCESSING FILE: {filename}")

    text = extract_text(pdf_path)

    if len(text.strip()) == 0:

        return "No text extracted"

    chunks = chunk_text(text)

    print(f"\nTOTAL CHUNKS CREATED: {len(chunks)}")

    for chunk in chunks:

        chunks_store.append(chunk)

        metadata_store.append({
            "document": filename,
            "text": chunk
        })

    build_faiss()
    build_bm25()

    return f"{len(chunks)} chunks added"

# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(question, top_k=5):

    global index
    global bm25

    if index is None:
        return []

    # =====================================
    # VECTOR SEARCH
    # =====================================

    q_embedding = embedder.encode([question])

    q_embedding = np.array(q_embedding).astype("float32")

    distances, indices = index.search(
        q_embedding,
        top_k
    )

    faiss_results = []

    for idx in indices[0]:

        if idx < len(chunks_store):

            faiss_results.append(
                chunks_store[idx]
            )

    # =====================================
    # BM25 SEARCH
    # =====================================

    bm25_scores = bm25.get_scores(
        question.lower().split()
    )

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:top_k]

    bm25_results = []

    for idx in bm25_indices:

        if idx < len(chunks_store):

            bm25_results.append(
                chunks_store[idx]
            )

    # =====================================
    # FUSION
    # =====================================

    results = []

    for r in faiss_results + bm25_results:

        if r not in results:
            results.append(r)

    return results[:top_k]

# =========================================================
# OLLAMA
# =========================================================

def ask_ollama(prompt):

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            },
            timeout=120
        )

        data = response.json()

        print("\n========== OLLAMA RESPONSE ==========")
        print(data)

        if "message" in data:

            return data["message"]["content"]

        if "response" in data:

            return data["response"]

        if "error" in data:

            return f"Ollama error: {data['error']}"

        return str(data)

    except Exception as e:

        return str(e)

# =========================================================
# RAG ANSWER
# =========================================================

def rag_answer(question):

    docs = hybrid_search(question)

    if len(docs) == 0:

        return "Aucun document trouvé."

    print("\n========== RETRIEVED CHUNKS ==========")

    for i, d in enumerate(docs):

        print(f"\n--- CHUNK {i+1} ---")
        print(d[:500])

    context = "\n\n".join(docs)

    prompt = f"""
Tu es un assistant juridique multilingue.

RÈGLES STRICTES:
- répondre uniquement en français
- comprendre arabe, français et anglais
- si le contexte est en arabe :
  traduire mentalement avant réponse
- utiliser uniquement le contexte
- ne jamais inventer
- réponse concise et précise
- si information absente :
"Information non trouvée"

CONTEXTE:
{context}

QUESTION:
{question}

RÉPONSE:
"""

    return ask_ollama(prompt)

# =========================================================
# ROUTES
# =========================================================

# =====================================
# UPLOAD
# =====================================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    try:

        save_path = os.path.join(
            DATA_DIR,
            file.filename
        )

        with open(save_path, "wb") as f:

            f.write(await file.read())

        result = ingest_pdf(
            save_path,
            file.filename
        )

        return {
            "status": "success",
            "filename": file.filename,
            "result": result,
            "total_chunks": len(chunks_store)
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

# =====================================
# QUESTION
# =====================================

class Question(BaseModel):

    question: str

@app.post("/ask")
def ask_question(q: Question):

    answer = rag_answer(q.question)

    return {
        "question": q.question,
        "answer": answer
    }

# =========================================================
# START
# =========================================================

print("\n===================================")
print("MULTILINGUAL LEGAL RAG READY")
print("===================================")

print("\nSwagger UI:")
print("http://127.0.0.1:8000/docs")