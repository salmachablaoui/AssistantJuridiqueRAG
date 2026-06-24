# ============================================================
# app/services/qdrant_service.py
# REMPLACE : faiss.IndexFlatL2 (perdu à chaque redémarrage)
# NOUVEAU   : Qdrant — persistant, filtrable, scalable
#
# Avantages vs FAISS pour ce cas d'usage :
#   - Persistance sur disque (survit aux redémarrages)
#   - Filtrage par métadonnées (dossier_id, type_document)
#   - Scores de similarité normalisés (0-1) pour confidence score
#   - API REST native + client Python
# ============================================================

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)
import uuid
import structlog
from typing import Optional
from app.config import settings

logger = structlog.get_logger()

# ── Client Singleton ──────────────────────────────────────────
_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        try:
            # Essayer le serveur Docker d'abord
            _client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=5,
                check_compatibility=False,
                prefer_grpc=False,
            )
            # Tester la connexion
            _client.get_collections()
            logger.info("qdrant_connected_server")
        except Exception:
            # Fallback : mode fichier local (pas besoin de Docker)
            import os
            qdrant_path = os.path.join(os.getcwd(), "qdrant_storage")
            os.makedirs(qdrant_path, exist_ok=True)
            _client = QdrantClient(path=qdrant_path)
            logger.info("qdrant_connected_local", path=qdrant_path)
    return _client


def ensure_collection_exists() -> None:
    """
    Crée la collection Qdrant si elle n'existe pas.
    Appelé au démarrage de l'application.
    """
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.QDRANT_VECTOR_SIZE,
                distance=Distance.COSINE,       # Meilleur pour texte
            ),
        )
        logger.info(
            "qdrant_collection_created",
            collection=settings.QDRANT_COLLECTION,
            vector_size=settings.QDRANT_VECTOR_SIZE,
        )
    else:
        logger.info(
            "qdrant_collection_exists",
            collection=settings.QDRANT_COLLECTION,
        )


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Indexe des chunks dans Qdrant avec leurs embeddings et métadonnées.

    REMPLACE l'ancienne approche :
        self.index.add(vectors)
        self.chunks.extend(chunks)
        # ← tout perdu au redémarrage

    Args:
        chunks: Liste de dicts {text, document_id, dossier_id, ...}
        embeddings: Vecteurs correspondants (même ordre)

    Returns:
        Nombre de points insérés
    """
    if not chunks or not embeddings:
        return 0

    client = get_qdrant_client()

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk["text"],
                "document_id": chunk.get("document_id"),
                "dossier_id": chunk.get("dossier_id"),
                "document_type": chunk.get("document_type", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "nom_fichier": chunk.get("nom_fichier", ""),
                "affaire": chunk.get("affaire", ""),
                "numero_dossier": chunk.get("numero_dossier", ""),
            },
        )
        points.append(point)

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=points,
        wait=True,
    )

    logger.info(
        "chunks_indexed",
        n_points=len(points),
        document_id=chunks[0].get("document_id") if chunks else None,
    )
    return len(points)


def search_vectors(
    query_embedding: list[float],
    top_k: int = None,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Recherche sémantique dans Qdrant.

    REMPLACE l'ancienne approche :
        distances, indices = self.index.search(query_vector, k)
        # ← aucun score normalisé, aucun filtre possible

    Args:
        query_embedding: Vecteur de la requête
        top_k: Nombre de résultats
        filters: Filtres optionnels {dossier_id: 5, document_type: "jugement"}

    Returns:
        Liste de dicts {text, score, document_id, dossier_id, ...}
    """
    client = get_qdrant_client()
    top_k = top_k or settings.TOP_K_RESULTS

    # Construction des filtres Qdrant
    qdrant_filter = None
    if filters:
        conditions = []
        for key, value in filters.items():
            if value is not None:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        if conditions:
            qdrant_filter = Filter(must=conditions)

    results = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
        score_threshold=0.3,        # Ignorer les résultats non pertinents
    )

    return [
        {
            "text": r.payload.get("text", ""),
            "score": round(r.score, 4),
            "document_id": r.payload.get("document_id"),
            "dossier_id": r.payload.get("dossier_id"),
            "document_type": r.payload.get("document_type"),
            "nom_fichier": r.payload.get("nom_fichier", ""),
            "affaire": r.payload.get("affaire", ""),
            "numero_dossier": r.payload.get("numero_dossier", ""),
            "chunk_index": r.payload.get("chunk_index", 0),
        }
        for r in results
    ]


def delete_document_chunks(document_id: int) -> int:
    """
    Supprime tous les chunks d'un document (pour réindexation).
    Utilisé par POST /reindex.
    """
    client = get_qdrant_client()

    result = client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        ),
        wait=True,
    )

    logger.info("document_chunks_deleted", document_id=document_id)
    return result.status

def get_collection_stats() -> dict:
    try:
        client = get_qdrant_client()
        info = client.get_collection(settings.QDRANT_COLLECTION)
        return {
            "vectors_count": info.points_count or 0,
            "status": str(info.status),
        }
    except Exception as e:
        return {"error": str(e)}