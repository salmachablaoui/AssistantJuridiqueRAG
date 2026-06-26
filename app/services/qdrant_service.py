# ============================================================
# app/services/qdrant_service.py
# Compatible qdrant-client >= 1.7 ET anciennes versions
# ============================================================

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import uuid
import structlog
from typing import Optional
from app.config import settings

logger = structlog.get_logger()

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        try:
            _client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=5,
                prefer_grpc=False,
            )
            _client.get_collections()
            logger.info("qdrant_connected_server")
        except Exception:
            import os
            qdrant_path = os.path.join(os.getcwd(), "qdrant_storage")
            os.makedirs(qdrant_path, exist_ok=True)
            _client = QdrantClient(path=qdrant_path)
            logger.info("qdrant_connected_local", path=qdrant_path)
    return _client


def ensure_collection_exists() -> None:
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.QDRANT_VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        logger.info("qdrant_collection_created", collection=settings.QDRANT_COLLECTION)
    else:
        logger.info("qdrant_collection_exists", collection=settings.QDRANT_COLLECTION)


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    if not chunks or not embeddings:
        return 0
    client = get_qdrant_client()
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text":           chunk.get("text", ""),
                    "document_id":    chunk.get("document_id"),
                    "dossier_id":     chunk.get("dossier_id"),
                    "document_type":  chunk.get("document_type", "unknown"),
                    "chunk_index":    chunk.get("chunk_index", 0),
                    "nom_fichier":    chunk.get("nom_fichier", ""),
                    "affaire":        chunk.get("affaire", ""),
                    "numero_dossier": chunk.get("numero_dossier", ""),
                },
            )
        )
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points, wait=True)
    logger.info("chunks_indexed", n_points=len(points))
    return len(points)


def search_vectors(
    query_embedding: list[float],
    top_k: int = None,
    filters: Optional[dict] = None,
) -> list[dict]:
    client = get_qdrant_client()
    top_k = top_k or settings.TOP_K_RESULTS

    qdrant_filter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items() if v is not None
        ]
        if conditions:
            qdrant_filter = Filter(must=conditions)

    raw_results = []
    try:
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        raw_results = response.points
        logger.info("qdrant_search_api", method="query_points", n=len(raw_results))
    except AttributeError:
        raw_results = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            score_threshold=0.3,
        )
        logger.info("qdrant_search_api", method="search_legacy", n=len(raw_results))
    except Exception as e:
        logger.error("qdrant_search_error", error=str(e))
        return []

    results = []
    for r in raw_results:
        payload = r.payload or {}
        score = getattr(r, "score", 0.0)
        if score >= 0.3:
            results.append({
                "text":           payload.get("text", ""),
                "score":          round(score, 4),
                "document_id":    payload.get("document_id"),
                "dossier_id":     payload.get("dossier_id"),
                "document_type":  payload.get("document_type", ""),
                "nom_fichier":    payload.get("nom_fichier", ""),
                "affaire":        payload.get("affaire", ""),
                "numero_dossier": payload.get("numero_dossier", ""),
                "chunk_index":    payload.get("chunk_index", 0),
            })

    logger.info("qdrant_search_done", n_results=len(results), filters=filters)
    return results


def delete_document_chunks(document_id: int) -> None:
    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]),
            wait=True,
        )
        logger.info("document_chunks_deleted", document_id=document_id)
    except Exception as e:
        logger.warning("qdrant_delete_failed", error=str(e))


def get_collection_stats() -> dict:
    try:
        client = get_qdrant_client()
        info = client.get_collection(settings.QDRANT_COLLECTION)
        return {
            "vectors_count": info.points_count or 0,
            "status":        str(info.status),
        }
    except Exception as e:
        return {"error": str(e)}