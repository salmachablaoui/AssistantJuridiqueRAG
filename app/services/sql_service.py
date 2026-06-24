# ============================================================
# app/services/sql_service.py
# NOUVEAU — Répond aux questions structurées depuis PostgreSQL
#
# Ce service traduit les intentions utilisateur en requêtes SQL
# sur le schéma Laravel existant SANS le modifier.
#
# Exemples de questions traitées ici :
#   "Combien de dossiers sont en cours ?"
#   "Quels sont les dossiers de l'avocat Dupont ?"
#   "Montrer les honoraires impayés"
#   "Liste des séances de la semaine prochaine"
# ============================================================

import re
import structlog
from typing import Optional
from app.db.postgres import execute_query

logger = structlog.get_logger()

# ── Mapping des intentions vers les requêtes SQL ──────────────
# Ces requêtes correspondent exactement au schéma PostgreSQL analysé.
# Aucune modification du schéma n'est requise.

SQL_QUERIES = {

    # ── Dossiers ──────────────────────────────────────────────
    "count_dossiers": """
        SELECT
            statut,
            COUNT(*) as total
        FROM public.dossiers
        GROUP BY statut
        ORDER BY total DESC
    """,

    "dossiers_en_cours": """
        SELECT
            d.numero_dossier,
            d.affaire,
            d.objet_litige,
            d.statut,
            d.date_creation,
            a.nom || ' ' || a.prenom AS avocat,
            c.nom_client AS partie_adverse
        FROM public.dossiers d
        LEFT JOIN public.avocats a ON d.avocat_id = a.id
        LEFT JOIN public.clients c ON d.partie_adverse_id = c.id
        WHERE d.statut = 'en_cours'
        ORDER BY d.date_creation DESC
        LIMIT :limit
    """,

    "dossier_by_numero": """
        SELECT
            d.*,
            a.nom || ' ' || a.prenom AS avocat_nom,
            a.email AS avocat_email,
            a.telephone AS avocat_telephone,
            c.nom_client AS partie_adverse_nom
        FROM public.dossiers d
        LEFT JOIN public.avocats a ON d.avocat_id = a.id
        LEFT JOIN public.clients c ON d.partie_adverse_id = c.id
        WHERE d.numero_dossier ILIKE :numero
    """,

    "dossiers_by_avocat": """
        SELECT
            d.numero_dossier,
            d.affaire,
            d.statut,
            d.date_creation,
            a.nom || ' ' || a.prenom AS avocat
        FROM public.dossiers d
        JOIN public.avocats a ON d.avocat_id = a.id
        WHERE LOWER(a.nom || ' ' || a.prenom) LIKE LOWER(:name)
        ORDER BY d.date_creation DESC
    """,

    # ── Séances ───────────────────────────────────────────────
    "seances_a_venir": """
        SELECT
            s.date_seance,
            s.lieu,
            s.statut,
            d.numero_dossier,
            d.affaire,
            st.type AS stade
        FROM public.seances s
        JOIN public.dossiers d ON s.dossier_id = d.id
        JOIN public.stades st ON s.stade_id = st.id
        WHERE s.date_seance >= NOW()
            AND s.statut = 'programmee'
        ORDER BY s.date_seance ASC
        LIMIT :limit
    """,

    "seances_by_dossier": """
        SELECT
            s.date_seance,
            s.lieu,
            s.statut,
            s.decision,
            s.jugement,
            st.type AS stade
        FROM public.seances s
        JOIN public.stades st ON s.stade_id = st.id
        WHERE s.dossier_id = :dossier_id
        ORDER BY s.date_seance DESC
    """,

    # ── Honoraires ────────────────────────────────────────────
    "honoraires_impayes": """
        SELECT
            h.montant_total,
            h.montant_paye,
            h.reste_a_payer,
            h.statut,
            h.date_limite,
            d.numero_dossier,
            d.affaire,
            a.nom || ' ' || a.prenom AS avocat
        FROM public.honoraires h
        JOIN public.dossiers d ON h.dossier_id = d.id
        LEFT JOIN public.avocats a ON h.avocat_id = a.id
        WHERE h.statut IN ('impaye', 'partiel')
        ORDER BY h.date_limite ASC NULLS LAST
        LIMIT :limit
    """,

    "honoraires_by_dossier": """
        SELECT
            h.*,
            a.nom || ' ' || a.prenom AS avocat_nom
        FROM public.honoraires h
        LEFT JOIN public.avocats a ON h.avocat_id = a.id
        WHERE h.dossier_id = :dossier_id
    """,

    "total_honoraires": """
        SELECT
            SUM(montant_total) AS total_montant,
            SUM(montant_paye) AS total_paye,
            SUM(reste_a_payer) AS total_reste,
            COUNT(*) AS nombre_dossiers
        FROM public.honoraires
    """,

    # ── Avocats ───────────────────────────────────────────────
    "list_avocats": """
        SELECT
            a.nom,
            a.prenom,
            a.specialite,
            a.ville,
            a.statut,
            COUNT(d.id) AS nb_dossiers
        FROM public.avocats a
        LEFT JOIN public.dossiers d ON d.avocat_id = a.id
        WHERE a.statut = 'actif'
        GROUP BY a.id, a.nom, a.prenom, a.specialite, a.ville, a.statut
        ORDER BY nb_dossiers DESC
    """,

    # ── Documents ─────────────────────────────────────────────
    "documents_by_dossier": """
        SELECT
            doc.nom_fichier,
            doc.type_document,
            doc.statut,
            doc.statut_document,
            doc.date_scan,
            doc.created_at
        FROM public.documents doc
        WHERE doc.dossier_id = :dossier_id
        ORDER BY doc.created_at DESC
    """,

    "documents_en_attente": """
        SELECT
            doc.nom_fichier,
            doc.type_document,
            doc.statut_validation,
            doc.created_at,
            d.numero_dossier,
            d.affaire
        FROM public.documents doc
        JOIN public.dossiers d ON doc.dossier_id = d.id
        WHERE doc.statut_validation = 'en_attente'
        ORDER BY doc.created_at ASC
        LIMIT :limit
    """,

    # ── Statistiques globales ──────────────────────────────────
    "dashboard_stats": """
        SELECT
            (SELECT COUNT(*) FROM public.dossiers WHERE statut = 'en_cours') AS dossiers_en_cours,
            (SELECT COUNT(*) FROM public.dossiers WHERE statut = 'cloture') AS dossiers_clotures,
            (SELECT COUNT(*) FROM public.dossiers WHERE statut = 'suspendu') AS dossiers_suspendus,
            (SELECT COUNT(*) FROM public.seances WHERE date_seance >= NOW() AND statut = 'programmee') AS seances_a_venir,
            (SELECT COUNT(*) FROM public.honoraires WHERE statut = 'impaye') AS honoraires_impayes,
            (SELECT SUM(reste_a_payer) FROM public.honoraires) AS total_a_recouvrer,
            (SELECT COUNT(*) FROM public.documents WHERE statut_validation = 'en_attente') AS documents_en_attente,
            (SELECT COUNT(*) FROM public.avocats WHERE statut = 'actif') AS avocats_actifs
    """,
}


async def execute_sql_query(
    query_key: str,
    params: Optional[dict] = None
) -> list[dict]:
    """
    Exécute une requête SQL prédéfinie.

    Args:
        query_key: Clé dans SQL_QUERIES
        params: Paramètres de la requête (:limit, :dossier_id, etc.)

    Returns:
        Liste de résultats sous forme de dicts
    """
    if query_key not in SQL_QUERIES:
        raise ValueError(f"Unknown SQL query key: {query_key}")

    sql = SQL_QUERIES[query_key]
    params = params or {}

    # Valeurs par défaut
    if ":limit" in sql and "limit" not in params:
        params["limit"] = 20

    try:
        results = await execute_query(sql, params)
        logger.info(
            "sql_query_executed",
            query_key=query_key,
            n_results=len(results),
        )
        return results
    except Exception as e:
        logger.error("sql_query_error", query_key=query_key, error=str(e))
        raise


def format_sql_results_as_context(
    results: list[dict],
    query_key: str
) -> str:
    if not results:
        return "Aucun résultat trouvé dans la base de données."

    lines = []

    headers = {
        "count_dossiers": "Statistiques des dossiers par statut",
        "dossiers_en_cours": "Dossiers juridiques en cours",
        "honoraires_impayes": "Honoraires impayés ou partiellement payés",
        "seances_a_venir": "Séances programmées à venir",
        "dashboard_stats": "Tableau de bord — Vue d'ensemble",
        "list_avocats": "Liste des avocats actifs",
    }

    if query_key in headers:
        lines.append(f"=== {headers[query_key]} ===\n")

    # Colonnes qui sont vraiment des montants financiers (pas des comptages)
    MONTANT_COLUMNS = {"montant_total", "montant_paye", "reste_a_payer", 
                       "total_montant", "total_paye", "total_reste", "total_a_recouvrer"}

    for i, row in enumerate(results, 1):
        parts = []
        for key, val in row.items():
            if val is not None:
                # Formater UNIQUEMENT les vraies colonnes de montants
                if key in MONTANT_COLUMNS:
                    try:
                        val = f"{float(val):,.2f} MAD"
                    except (ValueError, TypeError):
                        pass
                parts.append(f"{key}: {val}")
        lines.append(f"{i}. " + " | ".join(parts))

    return "\n".join(lines)