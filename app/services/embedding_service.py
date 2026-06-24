# ============================================================
# app/services/embedding_service.py
# REMPLACE : np.random.rand(384) — inutilisable en production
# NOUVEAU   : BGE-M3 multilingue (FR/AR/EN) — dim=1024
#
# BGE-M3 est le meilleur choix pour ce contexte car :
#   - Supporte le français, l'arabe, l'anglais en un seul modèle
#   - Produit des embeddings denses (1024d) + sparse (BM25-like)
#   - Pas besoin d'API externe — tourne en local
# ============================================================

import numpy as np
import structlog
from typing import Union
from functools import lru_cache
from app.config import settings

logger = structlog.get_logger()


@lru_cache(maxsize=1)
def _load_model():
    """
    Charge le modèle une seule fois (singleton via lru_cache).
    Le premier appel prend ~30s pour télécharger/charger le modèle.
    """
    try:
        from FlagEmbedding import BGEM3FlagModel
        logger.info("loading_embedding_model", model=settings.EMBEDDING_MODEL)
        model = BGEM3FlagModel(
            settings.EMBEDDING_MODEL,
            use_fp16=True,          # Réduit la mémoire de moitié
            device="cpu",           # Changer en "cuda" si GPU disponible
        )
        logger.info("embedding_model_loaded", model=settings.EMBEDDING_MODEL)
        return model
    except ImportError:
        # Fallback vers sentence-transformers si FlagEmbedding non installé
        logger.warning("flagembedding_not_found_using_sentence_transformers")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("intfloat/multilingual-e5-large")


def get_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    valid_texts = [t.strip() for t in texts if t and t.strip()]
    if not valid_texts:
        return []

    try:
        model = _load_model()

        # BGE-M3 native
        output = model.encode(
            valid_texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        import numpy as np
        dense = output["dense_vecs"]
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        normalized = (dense / (norms + 1e-8)).tolist()
        return normalized

    except Exception as e:
        logger.error("embedding_error", error=str(e), n_texts=len(texts))
        raise RuntimeError(f"Embedding generation failed: {e}") from e

def get_single_embedding(text: str) -> list[float]:
    """Raccourci pour un seul texte (utilisé pour les requêtes)"""
    results = get_embeddings([text])
    return results[0] if results else []