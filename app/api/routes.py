import os
import re
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.config import settings

logger = structlog.get_logger()
router = APIRouter()

_DOSSIER_RAW = re.compile(r'\b[A-Z]{2,4}-\d{3,}-\d{3,}\b', re.IGNORECASE)
_DOSSIER_OK  = re.compile(r'\b[A-Z]{2,4}-\d{4}-\d{4}\b',   re.IGNORECASE)


class ChatRequest(BaseModel):
    question:      str = Field(..., min_length=1, max_length=2000)
    user_id:       Optional[int] = None
    dossier_id:    Optional[int] = None
    document_type: Optional[str] = None


class ChatResponse(BaseModel):
    answer:    str
    sources:   list[dict]
    confidence: float
    mode:      str
    reasoning: Optional[str] = None


class IndexRequest(BaseModel):
    document_id:    int
    dossier_id:     int
    chemin_fichier: str
    nom_fichier:    str
    type_document:  Optional[str] = None
    numero_dossier: Optional[str] = None
    affaire:        Optional[str] = None
    force_reindex:  bool = False


class IndexResponse(BaseModel):
    success:     bool
    document_id: int
    n_chunks:    int
    message:     str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    from app.services.query_router import route_with_llm_fallback, SearchMode
    from app.services.sql_service   import execute_sql_query, format_sql_results_as_context
    from app.services.vector_service import search_documents
    from app.services.hybrid_service import execute_hybrid_search
    from app.services.llm_service    import generate_sql_answer, generate_vector_answer

    question = request.question.strip()
    logger.info("chat_request", question=question[:100])

    try:
        # ── Guard : numéro de dossier mal formaté ─────────────
        raw_match = _DOSSIER_RAW.search(question)
        ok_match  = _DOSSIER_OK.search(question)
        if raw_match and not ok_match:
            return ChatResponse(
                answer=(
                    f"Le format du numéro **{raw_match.group(0)}** semble incorrect. "
                    f"Format attendu : `PREFIX-AAAA-NNNN` (ex: DSS-2026-0004)."
                ),
                sources=[], confidence=0.99, mode="VALIDATION",
            )

        decision = await route_with_llm_fallback(question)

        # ── CHITCHAT ──────────────────────────────────────────
        if decision.mode == SearchMode.CHITCHAT:
            return ChatResponse(
                answer=decision.chitchat_reply,
                sources=[], confidence=1.0, mode="CHITCHAT",
                reasoning=decision.reasoning if settings.DEBUG else None,
            )

        # ── SQL ───────────────────────────────────────────────
        if decision.mode == SearchMode.SQL and decision.sql_query_key:
            sql_results = await execute_sql_query(
                decision.sql_query_key,
                decision.sql_params or {}
            )

            # Guard dossier inexistant
            if not sql_results and decision.sql_params and "numero" in decision.sql_params:
                num = decision.sql_params["numero"].strip("%")
                return ChatResponse(
                    answer=f"Le dossier **{num}** n'existe pas dans le système ANP Legal.",
                    sources=[], confidence=0.99, mode="SQL",
                    reasoning=decision.reasoning if settings.DEBUG else None,
                )

            sql_context = format_sql_results_as_context(sql_results, decision.sql_query_key)
            result = await generate_sql_answer(question, sql_context, decision.sql_query_key)
            result["mode"] = "SQL"

        # ── HYBRID ────────────────────────────────────────────
        elif decision.mode == SearchMode.HYBRID:
            # Guard existence dossier avant Qdrant
            if decision.sql_params and "numero" in decision.sql_params:
                check = await execute_sql_query(
                    "dossier_by_numero",
                    {"numero": decision.sql_params["numero"]}
                )
                if not check:
                    num = decision.sql_params["numero"].strip("%")
                    return ChatResponse(
                        answer=f"Le dossier **{num}** n'existe pas dans le système ANP Legal.",
                        sources=[], confidence=0.99, mode="HYBRID",
                        reasoning=decision.reasoning if settings.DEBUG else None,
                    )
            result = await execute_hybrid_search(question, decision)

        # ── VECTOR ────────────────────────────────────────────
        else:
            chunks = await search_documents(
                question=question,
                top_k=settings.TOP_K_RESULTS,
                dossier_id=request.dossier_id,
                document_type=request.document_type,
            )
            result = await generate_vector_answer(question, chunks)
            result["mode"] = "VECTOR"

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            mode=result["mode"],
            reasoning=decision.reasoning if settings.DEBUG else None,
        )

    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index-document", response_model=IndexResponse)
async def index_document(request: IndexRequest):
    from app.services.pdf_service    import get_text_with_ocr_fallback
    from app.services.vector_service import index_document_chunks
    from app.services.qdrant_service import delete_document_chunks

    full_path = os.path.join(
        settings.PDF_STORAGE_PATH,
        request.chemin_fichier.lstrip("/")
    )
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404,
                            detail=f"Fichier non trouvé: {request.chemin_fichier}")

    try:
        if request.force_reindex:
            delete_document_chunks(request.document_id)

        text = get_text_with_ocr_fallback(full_path)
        if not text or len(text.strip()) < 20:
            return IndexResponse(success=False, document_id=request.document_id,
                                 n_chunks=0, message="Texte insuffisant")

        n_chunks = await index_document_chunks(
            text=text,
            document_id=request.document_id,
            dossier_id=request.dossier_id,
            document_type=request.type_document or "unknown",
            nom_fichier=request.nom_fichier,
            numero_dossier=request.numero_dossier or "",
            affaire=request.affaire or "",
        )
        return IndexResponse(success=True, document_id=request.document_id,
                             n_chunks=n_chunks,
                             message=f"Indexé avec succès ({n_chunks} chunks)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    from app.db.postgres import check_connection
    from app.services.qdrant_service import get_collection_stats, get_qdrant_client

    pg_ok = await check_connection()
    qdrant_ok, qdrant_stats = False, {}
    try:
        get_qdrant_client().get_collections()
        qdrant_ok    = True
        qdrant_stats = get_collection_stats()
    except Exception as e:
        qdrant_stats = {"error": str(e)}

    ollama_ok = False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as c:
            ollama_ok = (await c.get(f"{settings.OLLAMA_URL}/api/tags")).status_code == 200
    except Exception:
        pass

    return {
        "status": "healthy" if (pg_ok and qdrant_ok and ollama_ok) else "degraded",
        "services": {
            "postgresql": "up" if pg_ok     else "down",
            "qdrant":     "up" if qdrant_ok else "down",
            "ollama":     "up" if ollama_ok else "down",
        },
        "qdrant_stats":    qdrant_stats,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model":       settings.CHAT_MODEL,
    }


@router.get("/metrics")
async def get_metrics():
    from app.services.query_router import get_router_metrics
    from app.services.sql_service  import get_sql_metrics
    return {"router": get_router_metrics(), "sql": get_sql_metrics()}


@router.post("/metrics/reset")
async def reset_metrics():
    from app.services.query_router import _router_metrics
    from app.services.sql_service  import reset_sql_metrics
    for store in _router_metrics.values():
        if hasattr(store, "clear"):
            store.clear()
    reset_sql_metrics()
    return {"reset": True}