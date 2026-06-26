# app/api/routes.py — version définitive
import os
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.config import settings

logger = structlog.get_logger()
router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)  # ← min=1 pas 3
    user_id: Optional[int] = None
    dossier_id: Optional[int] = None
    document_type: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float
    mode: str
    reasoning: Optional[str] = None


class IndexRequest(BaseModel):
    document_id: int
    dossier_id: int
    chemin_fichier: str
    nom_fichier: str
    type_document: Optional[str] = None
    numero_dossier: Optional[str] = None
    affaire: Optional[str] = None
    force_reindex: bool = False


class IndexResponse(BaseModel):
    success: bool
    document_id: int
    n_chunks: int
    message: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # ── Imports ───────────────────────────────────────────────
    from app.services.chitchat_service import detect_chitchat
    from app.services.query_router import route_with_llm_fallback, SearchMode
    from app.services.sql_service import (
        execute_sql_query,
        format_sql_results_as_context,
    )
    from app.services.vector_service import search_documents
    from app.services.hybrid_service import execute_hybrid_search
    from app.services.llm_service import (
        generate_sql_answer,
        generate_vector_answer,
        format_sql_results_direct,
    )

    question = request.question.strip()
    logger.info("chat_request", question=question[:100])

    # ── ÉTAPE 1 : Chitchat (instantané, sans LLM) ─────────────
    chitchat = detect_chitchat(question)
    if chitchat.is_chitchat:
        logger.info("chitchat_response", question=question[:60])
        return ChatResponse(
            answer=chitchat.response,
            sources=[],
            confidence=1.0,
            mode="CHITCHAT",
            reasoning="Chitchat détecté",
        )

    # ── ÉTAPE 2 : Routing + exécution ─────────────────────────
    try:
        decision = await route_with_llm_fallback(question)
        result = {}

        # ── SQL ───────────────────────────────────────────────
        if decision.mode == SearchMode.SQL:
            if decision.sql_query_key:
                sql_results = await execute_sql_query(
                    decision.sql_query_key,
                    decision.sql_params or {},
                )
                sql_context = format_sql_results_as_context(sql_results, decision.sql_query_key)

                # Format direct pour les cas simples (sans LLM)
                direct = format_sql_results_direct(sql_results, decision.sql_query_key)
                if direct:
                    result = {
                        "answer": direct,
                        "confidence": 0.95,
                        "sources": [{"type": "database", "query": decision.sql_query_key}],
                        "mode": "SQL",
                    }
                else:
                    result = await generate_sql_answer(question, sql_context, decision.sql_query_key)
                    result["mode"] = "SQL"
            else:
                # Pas de requête prédéfinie → SQL dynamique
                try:
                    from app.services.sql_service import generate_dynamic_sql, execute_dynamic_sql
                    dynamic_sql = await generate_dynamic_sql(question)
                    if dynamic_sql:
                        sql_results = await execute_dynamic_sql(dynamic_sql)
                        sql_context = format_sql_results_as_context(sql_results, "dynamic")
                        result = await generate_sql_answer(question, sql_context, "dynamic")
                        result["mode"] = "SQL"
                    else:
                        decision.mode = SearchMode.VECTOR
                except Exception:
                    decision.mode = SearchMode.VECTOR

        # ── HYBRID ────────────────────────────────────────────
        if decision.mode == SearchMode.HYBRID:
            result = await execute_hybrid_search(question, decision)

        # ── VECTOR ────────────────────────────────────────────
        if decision.mode == SearchMode.VECTOR:
            chunks = await search_documents(
                question=question,
                top_k=settings.TOP_K_RESULTS,
                dossier_id=request.dossier_id,
                document_type=request.document_type,
            )
            result = await generate_vector_answer(question, chunks)
            result["mode"] = "VECTOR"

        return ChatResponse(
            answer=result.get("answer", "Aucune réponse générée."),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            mode=result.get("mode", "UNKNOWN"),
            reasoning=decision.reasoning if settings.DEBUG else None,
        )

    except Exception as e:
        logger.error("chat_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index-document", response_model=IndexResponse)
async def index_document(request: IndexRequest):
    from app.services.pdf_service import get_text_with_ocr_fallback
    from app.services.vector_service import index_document_chunks
    from app.services.qdrant_service import delete_document_chunks

    full_path = os.path.join(
        settings.PDF_STORAGE_PATH,
        request.chemin_fichier.lstrip("/")
    )

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Fichier non trouvé: {full_path}")

    try:
        if request.force_reindex:
            delete_document_chunks(request.document_id)

        text = get_text_with_ocr_fallback(full_path)

        if not text or len(text.strip()) < 20:
            return IndexResponse(
                success=False,
                document_id=request.document_id,
                n_chunks=0,
                message="Texte insuffisant extrait du PDF"
            )

        n_chunks = await index_document_chunks(
            text=text,
            document_id=request.document_id,
            dossier_id=request.dossier_id,
            document_type=request.type_document or "unknown",
            nom_fichier=request.nom_fichier,
            numero_dossier=request.numero_dossier or "",
            affaire=request.affaire or "",
        )

        return IndexResponse(
            success=True,
            document_id=request.document_id,
            n_chunks=n_chunks,
            message=f"Indexé avec succès ({n_chunks} chunks)"
        )
    except Exception as e:
        logger.error("index_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    from app.db.postgres import check_connection
    from app.services.qdrant_service import get_collection_stats, get_qdrant_client

    pg_ok = await check_connection()

    qdrant_ok = False
    qdrant_stats = {}
    try:
        client = get_qdrant_client()
        client.get_collections()
        qdrant_ok = True
        qdrant_stats = get_collection_stats()
    except Exception as e:
        qdrant_stats = {"error": str(e)}

    ollama_ok = False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.OLLAMA_URL}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    status = "healthy" if (pg_ok and qdrant_ok and ollama_ok) else "degraded"
    return {
        "status": status,
        "services": {
            "postgresql": "up" if pg_ok else "down",
            "qdrant":     "up" if qdrant_ok else "down",
            "ollama":     "up" if ollama_ok else "down",
        },
        "qdrant_stats":    qdrant_stats,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model":       settings.CHAT_MODEL,
    }