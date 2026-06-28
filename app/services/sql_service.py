# ============================================================
# app/services/sql_service.py — v3.1
# Fix : AmbiguousColumnError partie_adverse_nom dans CTE
#       → remplacer SELECT d.* par colonnes explicites
# ============================================================

import time
import re
import structlog
from typing import Optional
from collections import defaultdict
from app.db.postgres import execute_query

logger = structlog.get_logger()

# ── Métriques ─────────────────────────────────────────────────
_metrics = {
    "calls":         defaultdict(int),
    "errors":        defaultdict(int),
    "total_time_ms": defaultdict(float),
    "last_error":    {},
}

def get_sql_metrics() -> dict:
    out = {}
    for key in _metrics["calls"]:
        calls  = _metrics["calls"][key]
        errors = _metrics["errors"][key]
        avg_ms = _metrics["total_time_ms"][key] / calls if calls else 0.0
        out[key] = {
            "calls":        calls,
            "errors":       errors,
            "success_rate": round((calls - errors) / calls * 100, 1) if calls else 0,
            "avg_ms":       round(avg_ms, 1),
            "last_error":   _metrics["last_error"].get(key),
        }
    return out

def reset_sql_metrics():
    for store in _metrics.values():
        store.clear()

# ── Cache TTL 60s ──────────────────────────────────────────────
_CACHE: dict[str, tuple[list, float]] = {}
_CACHE_TTL  = 60
_CACHEABLE  = {"dashboard_stats", "list_avocats", "count_dossiers"}

def _cache_get(key: str) -> Optional[list]:
    if key in _CACHE:
        data, ts = _CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            return data
        del _CACHE[key]
    return None

def _cache_set(key: str, data: list):
    _CACHE[key] = (data, time.time())

def invalidate_cache(key: Optional[str] = None):
    if key:
        _CACHE.pop(key, None)
    else:
        _CACHE.clear()


# ══════════════════════════════════════════════════════════════
# REQUÊTES SQL
# ══════════════════════════════════════════════════════════════
SQL_QUERIES = {

    # ── Comptages ─────────────────────────────────────────────
    "count_dossiers": """
        SELECT statut, COUNT(*) AS total
        FROM public.dossiers
        GROUP BY statut
        ORDER BY total DESC
    """,

    # ── Listes dossiers ───────────────────────────────────────
    "dossiers_en_cours": """
        SELECT
            d.numero_dossier,
            d.affaire,
            d.objet_litige,
            d.statut,
            d.date_creation,
            d.enjeu_financier,
            a.nom || ' ' || a.prenom                     AS avocat,
            COALESCE(c.nom_client, d.partie_adverse_nom) AS partie_adverse
        FROM public.dossiers d
        LEFT JOIN public.avocats a ON d.avocat_id = a.id
        LEFT JOIN public.clients c ON d.partie_adverse_id = c.id
        WHERE d.statut = 'en_cours'
        ORDER BY d.date_creation DESC
        LIMIT :limit
    """,

    "dossiers_clotures": """
        SELECT
            d.numero_dossier,
            d.affaire,
            d.objet_litige,
            d.statut,
            d.date_creation,
            d.date_cloture,
            d.motif_cloture,
            d.decision_finale,
            a.nom || ' ' || a.prenom                     AS avocat,
            COALESCE(c.nom_client, d.partie_adverse_nom) AS partie_adverse
        FROM public.dossiers d
        LEFT JOIN public.avocats a ON d.avocat_id = a.id
        LEFT JOIN public.clients c ON d.partie_adverse_id = c.id
        WHERE d.statut = 'cloture'
        ORDER BY d.date_cloture DESC NULLS LAST
        LIMIT :limit
    """,

    "dossiers_suspendus": """
        SELECT
            d.numero_dossier,
            d.affaire,
            d.objet_litige,
            d.statut,
            d.date_creation,
            d.motif_suspension,
            a.nom || ' ' || a.prenom                     AS avocat,
            COALESCE(c.nom_client, d.partie_adverse_nom) AS partie_adverse
        FROM public.dossiers d
        LEFT JOIN public.avocats a ON d.avocat_id = a.id
        LEFT JOIN public.clients c ON d.partie_adverse_id = c.id
        WHERE d.statut = 'suspendu'
        ORDER BY d.date_creation DESC
        LIMIT :limit
    """,

    "dossiers_all": """
        SELECT
            d.numero_dossier,
            d.affaire,
            d.objet_litige,
            d.statut,
            d.date_creation,
            a.nom || ' ' || a.prenom                     AS avocat,
            COALESCE(c.nom_client, d.partie_adverse_nom) AS partie_adverse
        FROM public.dossiers d
        LEFT JOIN public.avocats a ON d.avocat_id = a.id
        LEFT JOIN public.clients c ON d.partie_adverse_id = c.id
        ORDER BY
            CASE d.statut
                WHEN 'en_cours'  THEN 1
                WHEN 'suspendu'  THEN 2
                WHEN 'cloture'   THEN 3
                ELSE 4
            END,
            d.date_creation DESC
        LIMIT :limit
    """,

    # ── Dossier par numéro (check existence) ──────────────────
    "dossier_by_numero": """
        SELECT
            d.id,
            d.numero_dossier,
            d.affaire,
            d.objet_litige,
            d.statut,
            d.date_creation,
            d.enjeu_financier,
            d.decision_finale,
            d.motif_cloture,
            d.motif_suspension,
            d.notes,
            a.nom || ' ' || a.prenom                     AS avocat_nom,
            a.email                                      AS avocat_email,
            a.telephone                                  AS avocat_telephone,
            a.specialite                                 AS avocat_specialite,
            a.ville                                      AS avocat_ville,
            COALESCE(c.nom_client, d.partie_adverse_nom) AS partie_adverse_nom,
            c.email                                      AS partie_adverse_email,
            c.telephone                                  AS partie_adverse_telephone
        FROM public.dossiers d
        LEFT JOIN public.avocats a ON d.avocat_id = a.id
        LEFT JOIN public.clients c ON d.partie_adverse_id = c.id
        WHERE d.numero_dossier ILIKE :numero
    """,

    # ── Détail complet — FIX : colonnes explicites, pas SELECT d.* ──
    # SELECT d.* causait AmbiguousColumnError car d.partie_adverse_nom
    # entrait en conflit avec l'alias COALESCE(...) AS partie_adverse_nom
    "dossier_detail_complet": """
        SELECT
            d.id                                         AS dossier_id,
            d.numero_dossier,
            d.affaire,
            d.objet_litige,
            d.enjeu_financier,
            d.statut,
            d.date_creation,
            d.date_cloture,
            d.decision_finale,
            d.motif_cloture,
            d.motif_suspension,
            d.notes,
            a.nom || ' ' || a.prenom                     AS avocat_nom,
            a.email                                      AS avocat_email,
            a.telephone                                  AS avocat_telephone,
            a.specialite                                 AS avocat_specialite,
            a.ville                                      AS avocat_ville,
            COALESCE(c.nom_client, d.partie_adverse_nom) AS partie_adverse_nom,
            c.email                                      AS partie_adverse_email,
            c.telephone                                  AS partie_adverse_telephone,
            (SELECT COUNT(*)
             FROM public.seances s
             WHERE s.dossier_id = d.id)                  AS nb_seances_total,
            (SELECT COUNT(*)
             FROM public.seances s
             WHERE s.dossier_id = d.id
               AND s.date_seance >= NOW()
               AND s.statut = 'programmee')              AS nb_seances_a_venir,
            (SELECT COUNT(*)
             FROM public.documents doc
             WHERE doc.dossier_id = d.id)                AS nb_documents,
            (SELECT h.montant_total
             FROM public.honoraires h
             WHERE h.dossier_id = d.id LIMIT 1)          AS honoraires_total,
            (SELECT h.montant_paye
             FROM public.honoraires h
             WHERE h.dossier_id = d.id LIMIT 1)          AS honoraires_paye,
            (SELECT h.reste_a_payer
             FROM public.honoraires h
             WHERE h.dossier_id = d.id LIMIT 1)          AS honoraires_reste,
            (SELECT h.statut
             FROM public.honoraires h
             WHERE h.dossier_id = d.id LIMIT 1)          AS honoraires_statut,
            (SELECT STRING_AGG(st.type || ' (' || st.statut || ')', ', ')
             FROM public.stades st
             WHERE st.dossier_id = d.id)                 AS stades
        FROM public.dossiers d
        LEFT JOIN public.avocats a ON d.avocat_id = a.id
        LEFT JOIN public.clients c ON d.partie_adverse_id = c.id
        WHERE d.numero_dossier ILIKE :numero
    """,

    # ── Séances ───────────────────────────────────────────────
    "seances_by_dossier": """
        SELECT
            s.date_seance,
            s.heure,
            s.lieu,
            s.statut,
            s.decision,
            s.jugement,
            s.motif_report,
            s.nouvelle_date,
            st.type AS stade
        FROM public.seances s
        JOIN public.stades st ON s.stade_id = st.id
        WHERE s.dossier_id = (
            SELECT id FROM public.dossiers
            WHERE numero_dossier ILIKE :numero LIMIT 1
        )
        ORDER BY s.date_seance DESC
    """,

    "seances_a_venir": """
        SELECT
            s.date_seance,
            s.heure,
            s.lieu,
            s.statut,
            d.numero_dossier,
            d.affaire,
            st.type AS stade
        FROM public.seances s
        JOIN public.dossiers d ON s.dossier_id = d.id
        JOIN public.stades st  ON s.stade_id   = st.id
        WHERE s.date_seance >= NOW()
          AND s.statut = 'programmee'
        ORDER BY s.date_seance ASC
        LIMIT :limit
    """,

    "seances_cette_semaine": """
        SELECT
            s.date_seance,
            s.heure,
            s.lieu,
            s.statut,
            d.numero_dossier,
            d.affaire,
            st.type AS stade
        FROM public.seances s
        JOIN public.dossiers d ON s.dossier_id = d.id
        JOIN public.stades st  ON s.stade_id   = st.id
        WHERE s.date_seance >= DATE_TRUNC('week', NOW())
          AND s.date_seance <  DATE_TRUNC('week', NOW()) + INTERVAL '7 days'
        ORDER BY s.date_seance ASC
    """,

    "seances_passees": """
        SELECT
            s.date_seance,
            s.lieu,
            s.statut,
            s.decision,
            d.numero_dossier,
            d.affaire,
            st.type AS stade
        FROM public.seances s
        JOIN public.dossiers d ON s.dossier_id = d.id
        JOIN public.stades st  ON s.stade_id   = st.id
        WHERE s.date_seance < NOW()
        ORDER BY s.date_seance DESC
        LIMIT :limit
    """,

    # ── Honoraires ────────────────────────────────────────────
    "honoraires_by_dossier": """
        SELECT
            h.montant_total,
            h.montant_paye,
            h.reste_a_payer,
            h.statut,
            h.date_limite,
            h.montant_premiere_instance,
            h.montant_appel,
            h.montant_cassation,
            h.notes,
            a.nom || ' ' || a.prenom AS avocat_nom
        FROM public.honoraires h
        LEFT JOIN public.avocats a ON h.avocat_id = a.id
        WHERE h.dossier_id = (
            SELECT id FROM public.dossiers
            WHERE numero_dossier ILIKE :numero LIMIT 1
        )
    """,

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
        JOIN public.dossiers d     ON h.dossier_id = d.id
        LEFT JOIN public.avocats a ON h.avocat_id  = a.id
        WHERE h.statut IN ('impaye', 'partiel')
        ORDER BY h.reste_a_payer DESC, h.date_limite ASC NULLS LAST
        LIMIT :limit
    """,

    "honoraires_by_avocat": """
        SELECT
            a.nom || ' ' || a.prenom AS avocat,
            COUNT(h.id)              AS nb_dossiers,
            SUM(h.montant_total)     AS total_montant,
            SUM(h.montant_paye)      AS total_paye,
            SUM(h.reste_a_payer)     AS total_reste,
            SUM(CASE WHEN h.statut = 'impaye'  THEN 1 ELSE 0 END) AS nb_impayes,
            SUM(CASE WHEN h.statut = 'partiel' THEN 1 ELSE 0 END) AS nb_partiels,
            SUM(CASE WHEN h.statut = 'paye'    THEN 1 ELSE 0 END) AS nb_payes
        FROM public.honoraires h
        JOIN public.avocats a ON h.avocat_id = a.id
        GROUP BY a.id, a.nom, a.prenom
        ORDER BY total_reste DESC
    """,

    "total_honoraires": """
        SELECT
            SUM(montant_total)  AS total_montant,
            SUM(montant_paye)   AS total_paye,
            SUM(reste_a_payer)  AS total_reste,
            COUNT(*)            AS nombre_dossiers
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
            a.email,
            a.telephone,
            COUNT(d.id) AS nb_dossiers
        FROM public.avocats a
        LEFT JOIN public.dossiers d ON d.avocat_id = a.id
        WHERE a.statut = 'actif'
        GROUP BY a.id, a.nom, a.prenom, a.specialite,
                 a.ville, a.statut, a.email, a.telephone
        ORDER BY nb_dossiers DESC
    """,

    "avocat_by_name": """
        SELECT
            a.nom, a.prenom, a.email, a.telephone,
            a.specialite, a.ville, a.statut, a.numero_barreau,
            COUNT(DISTINCT d.id)                                        AS nb_dossiers_total,
            COUNT(DISTINCT CASE WHEN d.statut='en_cours' THEN d.id END) AS nb_en_cours,
            COALESCE(SUM(h.reste_a_payer), 0)                           AS honoraires_restants
        FROM public.avocats a
        LEFT JOIN public.dossiers d   ON d.avocat_id = a.id
        LEFT JOIN public.honoraires h ON h.avocat_id = a.id
        WHERE LOWER(a.nom || ' ' || a.prenom) LIKE LOWER(:name)
        GROUP BY a.id, a.nom, a.prenom, a.email, a.telephone,
                 a.specialite, a.ville, a.statut, a.numero_barreau
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

    # ── Documents ─────────────────────────────────────────────
    "documents_by_dossier": """
        SELECT
            doc.id,
            doc.nom_fichier,
            doc.type_document,
            doc.statut,
            doc.statut_document,
            doc.statut_validation,
            doc.description,
            doc.nombre_pages,
            doc.date_scan,
            doc.created_at
        FROM public.documents doc
        WHERE doc.dossier_id = (
            SELECT id FROM public.dossiers
            WHERE numero_dossier ILIKE :numero LIMIT 1
        )
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

    # ── Stades ────────────────────────────────────────────────
    "stades_by_dossier": """
        SELECT
            st.type,
            st.statut,
            st.date_debut,
            st.date_fin,
            st.observations
        FROM public.stades st
        WHERE st.dossier_id = (
            SELECT id FROM public.dossiers
            WHERE numero_dossier ILIKE :numero LIMIT 1
        )
        ORDER BY st.date_debut ASC NULLS LAST
    """,

    # ── Partie adverse ────────────────────────────────────────
    "dossiers_by_client": """
        SELECT
            d.numero_dossier,
            d.affaire,
            d.statut,
            d.date_creation,
            COALESCE(c.nom_client, d.partie_adverse_nom) AS partie_adverse,
            a.nom || ' ' || a.prenom                     AS avocat
        FROM public.dossiers d
        LEFT JOIN public.clients c ON d.partie_adverse_id = c.id
        LEFT JOIN public.avocats a ON d.avocat_id = a.id
        WHERE LOWER(COALESCE(c.nom_client, d.partie_adverse_nom, '')) LIKE LOWER(:name)
        ORDER BY d.date_creation DESC
    """,

    # ── Dashboard ─────────────────────────────────────────────
    "dashboard_stats": """
        SELECT
            (SELECT COUNT(*) FROM public.dossiers WHERE statut = 'en_cours')  AS dossiers_en_cours,
            (SELECT COUNT(*) FROM public.dossiers WHERE statut = 'cloture')   AS dossiers_clotures,
            (SELECT COUNT(*) FROM public.dossiers WHERE statut = 'suspendu')  AS dossiers_suspendus,
            (SELECT COUNT(*) FROM public.dossiers)                            AS dossiers_total,
            (SELECT COUNT(*) FROM public.seances
             WHERE date_seance >= NOW() AND statut = 'programmee')            AS seances_a_venir,
            (SELECT COUNT(*) FROM public.honoraires WHERE statut = 'impaye')  AS honoraires_impayes,
            (SELECT COUNT(*) FROM public.honoraires WHERE statut = 'partiel') AS honoraires_partiels,
            (SELECT COALESCE(SUM(reste_a_payer),0) FROM public.honoraires)   AS total_a_recouvrer,
            (SELECT COUNT(*) FROM public.documents
             WHERE statut_validation = 'en_attente')                          AS documents_en_attente,
            (SELECT COUNT(*) FROM public.avocats WHERE statut = 'actif')      AS avocats_actifs
    """,
}


# ── Exécution ─────────────────────────────────────────────────
async def execute_sql_query(query_key: str, params: Optional[dict] = None) -> list[dict]:
    if query_key not in SQL_QUERIES:
        raise ValueError(f"Unknown SQL query key: {query_key}")

    params = params or {}
    sql    = SQL_QUERIES[query_key]

    if ":limit" in sql and "limit" not in params:
        params["limit"] = 20

    use_cache = query_key in _CACHEABLE and not params
    if use_cache:
        cached = _cache_get(query_key)
        if cached is not None:
            _metrics["calls"][query_key] += 1
            return cached

    _metrics["calls"][query_key] += 1
    t0 = time.perf_counter()

    try:
        results = await execute_query(sql, params)
        elapsed = (time.perf_counter() - t0) * 1000
        _metrics["total_time_ms"][query_key] += elapsed
        logger.info("sql_executed", key=query_key, n=len(results), ms=round(elapsed, 1))
        if use_cache:
            _cache_set(query_key, results)
        return results
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        _metrics["total_time_ms"][query_key] += elapsed
        _metrics["errors"][query_key] += 1
        _metrics["last_error"][query_key] = str(e)
        logger.error("sql_error", key=query_key, error=str(e))
        raise


# ── Formatage ─────────────────────────────────────────────────
_MONTANT_COLS = {
    "montant_total", "montant_paye", "reste_a_payer",
    "total_montant", "total_paye", "total_reste", "total_a_recouvrer",
    "honoraires_total", "honoraires_paye", "honoraires_reste",
    "honoraires_restants", "montant_premiere_instance",
    "montant_appel", "montant_cassation", "enjeu_financier",
}
_SKIP_KEYS = {"dossier_id", "avocat_id", "partie_adverse_id", "stade_id"}

_HEADERS = {
    "count_dossiers":        "Répartition des dossiers par statut",
    "dossiers_en_cours":     "Dossiers en cours",
    "dossiers_clotures":     "Dossiers clôturés",
    "dossiers_suspendus":    "Dossiers suspendus",
    "dossiers_all":          "Tous les dossiers",
    "honoraires_impayes":    "Honoraires impayés / partiels",
    "honoraires_by_avocat":  "Honoraires par avocat",
    "seances_a_venir":       "Séances à venir",
    "seances_cette_semaine": "Séances cette semaine",
    "seances_passees":       "Séances passées",
    "dashboard_stats":       "Tableau de bord",
    "list_avocats":          "Avocats actifs",
    "dossier_detail_complet":"Détail complet du dossier",
    "documents_by_dossier":  "Documents du dossier",
    "seances_by_dossier":    "Séances du dossier",
    "honoraires_by_dossier": "Honoraires du dossier",
    "stades_by_dossier":     "Stades du dossier",
}


def _fmt(key: str, val) -> str:
    if val is None:
        return "—"
    if key in _MONTANT_COLS:
        try:
            return f"{float(val):,.2f} MAD"
        except (ValueError, TypeError):
            pass
    return str(val)


def format_sql_results_as_context(results: list[dict], query_key: str) -> str:
    if not results:
        return "Aucun résultat trouvé dans la base de données."

    lines = []
    if query_key in _HEADERS:
        lines.append(f"=== {_HEADERS[query_key]} ===\n")

    # Dashboard
    if query_key == "dashboard_stats":
        r = results[0]
        lines += [
            f"• Dossiers en cours    : {r.get('dossiers_en_cours', 0)}",
            f"• Dossiers clôturés    : {r.get('dossiers_clotures', 0)}",
            f"• Dossiers suspendus   : {r.get('dossiers_suspendus', 0)}",
            f"• Total dossiers       : {r.get('dossiers_total', 0)}",
            f"• Séances à venir      : {r.get('seances_a_venir', 0)}",
            f"• Honoraires impayés   : {r.get('honoraires_impayes', 0)}",
            f"• Honoraires partiels  : {r.get('honoraires_partiels', 0)}",
            f"• Total à recouvrer    : {_fmt('total_a_recouvrer', r.get('total_a_recouvrer'))}",
            f"• Documents en attente : {r.get('documents_en_attente', 0)}",
            f"• Avocats actifs       : {r.get('avocats_actifs', 0)}",
        ]
        return "\n".join(lines)

    # Count dossiers
    if query_key == "count_dossiers":
        for row in results:
            lines.append(
                f"• {row.get('statut','?').replace('_',' ').title()} "
                f": {row.get('total', 0)} dossier(s)"
            )
        return "\n".join(lines)

    # Détail complet
    if query_key == "dossier_detail_complet" and results:
        r = results[0]
        lines += [
            f"Numéro           : {r.get('numero_dossier','—')}",
            f"Affaire          : {r.get('affaire','—')}",
            f"Objet du litige  : {r.get('objet_litige','—')}",
            f"Enjeu financier  : {_fmt('enjeu_financier', r.get('enjeu_financier'))}",
            f"Statut           : {r.get('statut','—')}",
            f"Date création    : {r.get('date_creation','—')}",
        ]
        if r.get("date_cloture"):
            lines.append(f"Date clôture     : {r['date_cloture']}")
        if r.get("motif_cloture"):
            lines.append(f"Motif clôture    : {r['motif_cloture']}")
        if r.get("decision_finale"):
            lines.append(f"Décision finale  : {r['decision_finale']}")
        if r.get("motif_suspension"):
            lines.append(f"Motif suspension : {r['motif_suspension']}")
        if r.get("stades"):
            lines.append(f"Stades           : {r['stades']}")
        lines += [
            "",
            f"Avocat           : {r.get('avocat_nom','—')}",
            f"  Spécialité     : {r.get('avocat_specialite','—')}",
            f"  Email          : {r.get('avocat_email','—')}",
            f"  Tél            : {r.get('avocat_telephone','—')}",
            f"  Ville          : {r.get('avocat_ville','—')}",
            "",
            f"Partie adverse   : {r.get('partie_adverse_nom','—')}",
            f"  Email          : {r.get('partie_adverse_email','—')}",
            f"  Tél            : {r.get('partie_adverse_telephone','—')}",
            "",
            f"Séances total    : {r.get('nb_seances_total', 0)}",
            f"Séances à venir  : {r.get('nb_seances_a_venir', 0)}",
            f"Documents        : {r.get('nb_documents', 0)}",
            "",
            f"Honoraires total : {_fmt('honoraires_total', r.get('honoraires_total'))}",
            f"Honoraires payé  : {_fmt('honoraires_paye',  r.get('honoraires_paye'))}",
            f"Reste à payer    : {_fmt('honoraires_reste',  r.get('honoraires_reste'))}",
            f"Statut honoraires: {r.get('honoraires_statut','—')}",
        ]
        if r.get("notes"):
            lines.append(f"Notes            : {r['notes']}")
        return "\n".join(lines)

    # Générique
    for i, row in enumerate(results, 1):
        parts = []
        for key, val in row.items():
            if key in _SKIP_KEYS:
                continue
            parts.append(f"{key.replace('_',' ')}: {_fmt(key, val)}")
        lines.append(f"{i}. " + " | ".join(parts))

    return "\n".join(lines)


# ── SQL dynamique ─────────────────────────────────────────────
_DANGEROUS = frozenset([
    "insert","update","delete","drop","truncate",
    "alter","create","replace","merge","upsert","grant","revoke",
])

_SCHEMA_CONTEXT = """
Tables PostgreSQL disponibles (lecture seule) :

dossiers(id, numero_dossier, affaire, objet_litige, enjeu_financier,
         statut, avocat_id, partie_adverse_id, partie_adverse_nom,
         date_creation, date_cloture, motif_cloture, decision_finale,
         motif_suspension, notes)
  statuts : 'en_cours' | 'cloture' | 'suspendu'
  NOTE : partie_adverse_nom est un champ texte libre (si partie_adverse_id est NULL)
         → utiliser COALESCE(c.nom_client, d.partie_adverse_nom) pour la partie adverse

avocats(id, nom, prenom, email, telephone, specialite, ville, numero_barreau, statut)
clients(id, nom_client, email, telephone, type_client, ville, statut)
seances(id, dossier_id, stade_id, date_seance, heure, lieu, statut, decision, jugement)
honoraires(id, dossier_id, avocat_id, montant_total, montant_paye, reste_a_payer,
           statut, date_limite, montant_premiere_instance, montant_appel, montant_cassation)
documents(id, dossier_id, nom_fichier, type_document, statut, statut_validation,
          statut_document, nombre_pages, date_scan, created_at)
stades(id, dossier_id, type, statut, date_debut, date_fin, observations)
  types : 'premiere_instance' | 'appel' | 'cassation'

IMPORTANT : NE JAMAIS faire SELECT d.* avec des JOIN qui créent des alias du même nom.
Toujours lister les colonnes explicitement.
"""


async def generate_dynamic_sql(question: str, schema_context: Optional[str] = None) -> Optional[str]:
    from app.services.llm_service import call_ollama
    schema = schema_context or _SCHEMA_CONTEXT
    prompt = f"""Tu es expert SQL PostgreSQL. Génère UNE requête SELECT.

SCHÉMA :
{schema}

RÈGLES :
- SELECT uniquement
- Lister les colonnes explicitement (jamais SELECT d.*)
- COALESCE(c.nom_client, d.partie_adverse_nom) pour la partie adverse
- Résoudre numero_dossier via sous-requête si besoin
- LIMIT 20 par défaut
- SQL brut sans markdown

QUESTION : {question}
SQL :"""
    try:
        raw = await call_ollama(prompt, temperature=0.1)
        sql = raw.strip().replace("```sql","").replace("```","").strip()
        if not sql.lower().startswith("select"):
            return None
        for word in _DANGEROUS:
            if re.search(r"\b" + re.escape(word) + r"\b", sql.lower()):
                logger.warning("dynamic_sql_rejected", word=word)
                return None
        return sql
    except Exception as e:
        logger.error("dynamic_sql_error", error=str(e))
        return None


async def execute_dynamic_sql(sql: str) -> list[dict]:
    if not sql.strip().lower().startswith("select"):
        raise ValueError("SELECT uniquement autorisé")
    for word in _DANGEROUS:
        if re.search(r"\b" + re.escape(word) + r"\b", sql.lower()):
            raise ValueError(f"Mot interdit: {word}")
    _metrics["calls"]["_dynamic"] += 1
    t0 = time.perf_counter()
    try:
        results = await execute_query(sql, {})
        _metrics["total_time_ms"]["_dynamic"] += (time.perf_counter() - t0) * 1000
        return results
    except Exception as e:
        _metrics["errors"]["_dynamic"] += 1
        _metrics["last_error"]["_dynamic"] = str(e)
        raise