# ============================================================
# app/services/vector_service.py
# REFACTORÉ — Interface unifiée pour la recherche vectorielle
#
# Délègue à qdrant_service (stockage) et embedding_service (vecteurs)
# Conserve l'interface publique pour ne pas casser les imports existants
# ============================================================

import structlog
from app.config import settings
from app.services.embedding_service import get_embeddings, get_single_embedding
from app.services.qdrant_service import search_vectors, upsert_chunks

logger = structlog.get_logger()


class VectorService:
    """
    Refactored VectorService.

    Ancienne implémentation (inutilisable) :
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []
        # ← perdu à chaque redémarrage, embeddings aléatoires

    Nouvelle implémentation :
        - Qdrant (persistant)
        - BGE-M3 embeddings (production)
        - Filtrage par métadonnées
    """

    def add(self, embeddings: list, chunks: list) -> None:
        """
        Conservé pour compatibilité avec l'ancien code.
        Délègue vers Qdrant.
        """
        if not chunks or not embeddings:
            return
        upsert_chunks(chunks, embeddings)

    def search(
        self,
        query_embedding: list[float],
        k: int = None,
        filters: dict = None,
    ) -> list[dict]:
        """
        Conservé pour compatibilité.
        Retourne maintenant des dicts enrichis (anciennement juste des strings).
        """
        k = k or settings.TOP_K_RESULTS
        return search_vectors(query_embedding, top_k=k, filters=filters)


# ── Fonctions utilitaires standalone ─────────────────────────

async def search_documents(
    question: str,
    top_k: int = None,
    dossier_id: int = None,
    document_type: str = None,
) -> list[dict]:
    """
    Recherche sémantique complète :
    1. Encode la question en vecteur
    2. Recherche dans Qdrant avec filtres optionnels
    3. Retourne les chunks pertinents avec scores

    Args:
        question: Question de l'utilisateur
        top_k: Nombre de résultats
        dossier_id: Filtrer par dossier (utilisé en mode HYBRID)
        document_type: Filtrer par type_document

    Returns:
        Liste de chunks avec scores de similarité
    """
    top_k = top_k or settings.TOP_K_RESULTS

    try:
        # Encoder la requête
        query_embedding = get_single_embedding(question)

        # Construire les filtres
        filters = {}
        if dossier_id:
            filters["dossier_id"] = dossier_id
        if document_type:
            filters["document_type"] = document_type

        # Recherche Qdrant
        results = search_vectors(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters if filters else None,
        )

        logger.info(
            "vector_search_complete",
            question=question[:80],
            n_results=len(results),
            filters=filters,
        )
        return results

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
    """
    Indexe les chunks d'un document dans Qdrant.

    Pipeline :
    1. Chunking sémantique du texte
    2. Génération des embeddings BGE-M3
    3. Upsert dans Qdrant avec métadonnées

    Returns:
        Nombre de chunks indexés
    """
    from app.services.chunk_service import chunk_text

    # 1. Chunking sémantique
    chunks = chunk_text(
        text=text,
        document_id=document_id,
        document_type=document_type,
    )

    if not chunks:
        logger.warning("no_chunks_generated", document_id=document_id)
        return 0

    # 2. Enrichir les chunks avec métadonnées du dossier
    for chunk in chunks:
        chunk["dossier_id"] = dossier_id
        chunk["nom_fichier"] = nom_fichier
        chunk["numero_dossier"] = numero_dossier
        chunk["affaire"] = affaire

    # 3. Générer les embeddings
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)

    # 4. Indexer dans Qdrant
    n_indexed = upsert_chunks(chunks, embeddings)

    logger.info(
        "document_indexed",
        document_id=document_id,
        n_chunks=n_indexed,
        nom_fichier=nom_fichier,
    )
    return n_indexed