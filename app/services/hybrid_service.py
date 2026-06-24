# ============================================================
# app/services/hybrid_service.py
# NOUVEAU — Orchestrateur du mode HYBRID
#
# Workflow HYBRID :
# 1. SQL → Récupère les IDs/métadonnées des dossiers concernés
# 2. VECTOR → Recherche sémantique filtrée sur ces dossier_ids
# 3. LLM → Combine les deux contextes pour la réponse finale
#
# Exemple concret :
#   Question: "Que disent les jugements du dossier 2024/ANP/001 ?"
#   → SQL: trouve dossier_id=42 pour ce numéro
#   → VECTOR: cherche dans les chunks de dossier_id=42 avec filtre
#   → LLM: génère une réponse complète
# ============================================================

import structlog
from app.services.sql_service import execute_sql_query, format_sql_results_as_context
from app.services.vector_service import search_documents
from app.services.llm_service import generate_hybrid_answer
from app.services.query_router import RouterDecision
from app.config import settings

logger = structlog.get_logger()


async def execute_hybrid_search(
    question: str,
    decision: RouterDecision,
) -> dict:
    """
    Exécute une recherche hybride SQL + VECTOR.

    Args:
        question: Question originale de l'utilisateur
        decision: Décision du router (contient sql_query_key, params, etc.)

    Returns:
        {answer, sources, confidence, mode, sql_results, vector_results}
    """
    sql_context = ""
    sql_results = []
    vector_results = []
    dossier_ids = []

    # ── ÉTAPE 1 : Requête SQL pour les métadonnées ────────────
    if decision.sql_query_key:
        try:
            sql_results = await execute_sql_query(
                decision.sql_query_key,
                decision.sql_params or {},
            )
            sql_context = format_sql_results_as_context(sql_results, decision.sql_query_key)

            # Extraire les dossier_ids pour filtrer la recherche vectorielle
            dossier_ids = [
                row["id"]
                for row in sql_results
                if "id" in row and row["id"] is not None
            ]

            logger.info(
                "hybrid_sql_step",
                n_results=len(sql_results),
                n_dossier_ids=len(dossier_ids),
            )

        except Exception as e:
            logger.warning("hybrid_sql_failed", error=str(e))
            sql_context = ""

    # ── ÉTAPE 2 : Recherche vectorielle filtrée ───────────────
    # Si on a trouvé des dossier_ids spécifiques, filtrer sur eux
    # Sinon, recherche globale (plus générale)

    if dossier_ids:
        # Recherche dans chaque dossier pertinent
        # Prendre le plus pertinent si plusieurs dossiers
        primary_dossier_id = dossier_ids[0] if dossier_ids else None

        vector_results = await search_documents(
            question=question,
            top_k=settings.TOP_K_RESULTS,
            dossier_id=primary_dossier_id,
        )

        # Si peu de résultats dans le dossier spécifique → recherche globale
        if len(vector_results) < 2:
            global_results = await search_documents(
                question=question,
                top_k=settings.TOP_K_RESULTS,
            )
            # Fusionner en évitant les doublons
            existing_ids = {r.get("document_id") for r in vector_results}
            for r in global_results:
                if r.get("document_id") not in existing_ids:
                    vector_results.append(r)

    else:
        # Pas de dossier spécifique → recherche vectorielle générale
        vector_results = await search_documents(
            question=question,
            top_k=settings.TOP_K_RESULTS,
        )

    logger.info(
        "hybrid_vector_step",
        n_results=len(vector_results),
        dossier_ids=dossier_ids[:3],
    )

    # ── ÉTAPE 3 : Génération LLM ──────────────────────────────
    llm_result = await generate_hybrid_answer(
        question=question,
        sql_context=sql_context,
        chunks=vector_results,
    )

    return {
        **llm_result,
        "mode": "HYBRID",
        "sql_results_count": len(sql_results),
        "vector_results_count": len(vector_results),
    }