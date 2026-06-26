# app/services/dossier_detail_service.py
# Récupère toutes les données d'un dossier (SQL + VECTOR)

import structlog
from app.db.postgres import execute_query

logger = structlog.get_logger()

DOSSIER_FULL_DETAIL_SQL = """
WITH dossier AS (
    SELECT d.*, 
           a.nom || ' ' || a.prenom AS avocat_nom,
           a.email AS avocat_email, a.telephone AS avocat_tel,
           c.nom_client AS partie_adverse
    FROM public.dossiers d
    LEFT JOIN public.avocats a ON d.avocat_id = a.id
    LEFT JOIN public.clients c ON d.partie_adverse_id = c.id
    WHERE d.numero_dossier ILIKE :numero
),
seances AS (
    SELECT s.date_seance, s.lieu, s.statut, s.decision, st.type AS stade
    FROM public.seances s
    JOIN public.stades st ON s.stade_id = st.id
    WHERE s.dossier_id = (SELECT id FROM dossier LIMIT 1)
    ORDER BY s.date_seance DESC LIMIT 5
),
honoraires AS (
    SELECT h.montant_total, h.montant_paye, h.reste_a_payer, h.statut
    FROM public.honoraires h
    WHERE h.dossier_id = (SELECT id FROM dossier LIMIT 1)
),
docs AS (
    SELECT doc.nom_fichier, doc.type_document, doc.statut, doc.created_at
    FROM public.documents doc
    WHERE doc.dossier_id = (SELECT id FROM dossier LIMIT 1)
    ORDER BY doc.created_at DESC
)
SELECT 
    (SELECT row_to_json(dossier) FROM dossier LIMIT 1) AS dossier_info,
    (SELECT json_agg(seances) FROM seances) AS seances,
    (SELECT json_agg(honoraires) FROM honoraires) AS honoraires,
    (SELECT json_agg(docs) FROM docs) AS documents
"""

async def get_dossier_full_context(numero_dossier: str) -> str:
    """
    Récupère toutes les données SQL d'un dossier pour le contexte LLM.
    """
    try:
        results = await execute_query(
            DOSSIER_FULL_DETAIL_SQL, 
            {"numero": f"%{numero_dossier}%"}
        )
        if not results:
            return f"Dossier {numero_dossier} non trouvé."
        
        row = results[0]
        lines = ["=== DONNÉES DU DOSSIER ==="]
        
        if row.get("dossier_info"):
            d = row["dossier_info"]
            lines += [
                f"Numéro : {d.get('numero_dossier', 'N/A')}",
                f"Affaire : {d.get('affaire', 'N/A')}",
                f"Objet : {d.get('objet_litige', 'N/A')}",
                f"Statut : {d.get('statut', 'N/A')}",
                f"Avocat : {d.get('avocat_nom', 'N/A')} | {d.get('avocat_email', '')}",
                f"Partie adverse : {d.get('partie_adverse', 'N/A')}",
                f"Date création : {d.get('date_creation', 'N/A')}",
            ]
        
        if row.get("seances"):
            lines.append("\n=== SÉANCES ===")
            for s in row["seances"]:
                lines.append(
                    f"  • {s.get('date_seance','?')} — {s.get('stade','?')} "
                    f"— {s.get('statut','?')} — {s.get('lieu','?')}"
                )
        
        if row.get("honoraires"):
            lines.append("\n=== HONORAIRES ===")
            for h in row["honoraires"]:
                lines.append(
                    f"  • Total: {h.get('montant_total',0)} MAD | "
                    f"Payé: {h.get('montant_paye',0)} MAD | "
                    f"Reste: {h.get('reste_a_payer',0)} MAD | "
                    f"Statut: {h.get('statut','?')}"
                )
        
        if row.get("documents"):
            lines.append("\n=== DOCUMENTS ===")
            for doc in row["documents"]:
                lines.append(f"  • {doc.get('nom_fichier','?')} ({doc.get('type_document','?')})")
        
        return "\n".join(lines)
    
    except Exception as e:
        logger.error("dossier_full_detail_error", error=str(e))
        return f"Erreur lors de la récupération du dossier : {str(e)}"