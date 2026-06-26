# app/services/vector_service.py
import structlog
from app.config import settings
from app.services.embedding_service import get_embeddings, get_single_embedding
from app.services.qdrant_service import search_vectors, upsert_chunks

logger = structlog.get_logger()

# ← Abaissé de 0.45 à 0.30 pour retrouver plus de chunks
MIN_RELEVANCE_SCORE = 0.30


async def validate_dossier_exists(numero: str) -> bool:
    try:
        from app.db.postgres import execute_query
        results = await execute_query(
            "SELECT id FROM public.dossiers WHERE numero_dossier ILIKE :numero LIMIT 1",
            {"numero": f"%{numero}%"}
        )
        return len(results) > 0
    except Exception as e:
        logger.warning("dossier_validation_failed", error=str(e))
        return True


async def search_documents(
    question: str,
    top_k: int = None,
    dossier_id: int = None,
    document_type: str = None,
    dossier_numero: str = None,
) -> list[dict]:
    top_k = top_k or settings.TOP_K_RESULTS

    if dossier_numero:
        exists = await validate_dossier_exists(dossier_numero)
        if not exists:
            logger.warning("dossier_not_found_skip_vector", numero=dossier_numero)
            return []

    try:
        query_embedding = get_single_embedding(question)

        filters = {}
        if dossier_id:
            filters["dossier_id"] = dossier_id
        if document_type:
            filters["document_type"] = document_type

        # Chercher plus large puis filtrer
        results = search_vectors(
            query_embedding=query_embedding,
            top_k=top_k * 3,
            filters=filters if filters else None,
        )

        # Filtrer par score minimum
        results = [r for r in results if r.get("score", 0) >= MIN_RELEVANCE_SCORE]

        # Dedup par document_id + chunk_index
        seen = set()
        deduped = []
        for r in results:
            key = (r.get("document_id"), r.get("chunk_index", 0))
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        deduped.sort(key=lambda x: x.get("score", 0), reverse=True)
        deduped = deduped[:top_k]

        logger.info(
            "vector_search_complete",
            question=question[:80],
            n_raw=len(results),
            n_deduped=len(deduped),
            scores=[round(r.get("score", 0), 2) for r in deduped[:3]],
        )
        return deduped

    except Exception as e:
        logger.error("vector_search_error", error=str(e))
        return []


async def index_document_chunks(
    text: str,
    document_id: int,
    dossier_id: int,
    document_type: str,
    nom_fichier: str,
    numero_dossier: str = "",
    affaire: str = "",
) -> int:
    from app.services.chunk_service import chunk_text

    chunks = chunk_text(text=text, document_id=document_id, document_type=document_type)
    if not chunks:
        logger.warning("no_chunks_generated", document_id=document_id)
        return 0

    for chunk in chunks:
        chunk["dossier_id"]     = dossier_id
        chunk["nom_fichier"]    = nom_fichier
        chunk["numero_dossier"] = numero_dossier
        chunk["affaire"]        = affaire

    texts      = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)
    n_indexed  = upsert_chunks(chunks, embeddings)

    logger.info("document_indexed", document_id=document_id, n_chunks=n_indexed)
    return n_indexed


class VectorService:
    def add(self, embeddings, chunks):
        if chunks and embeddings:
            upsert_chunks(chunks, embeddings)

    def search(self, query_embedding, k=None, filters=None):
        k = k or settings.TOP_K_RESULTS
        return search_vectors(query_embedding, top_k=k, filters=filters)