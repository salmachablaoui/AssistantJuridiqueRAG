# ============================================================
# app/services/query_router.py
# NOUVEAU — Cerveau du système : décide SQL, VECTOR ou HYBRID
#
# Logique de décision :
#
#  SQL     → Questions sur des données structurées
#             "combien", "liste", "statut", "date", "montant"
#             Entités : dossiers, avocat, séance, honoraire
#
#  VECTOR  → Questions sur le contenu des documents
#             "que dit", "contenu", "texte", "clause", "article"
#             Recherche sémantique dans les PDFs indexés
#
#  HYBRID  → Questions qui croisent structure + contenu
#             "documents du dossier X concernant Y"
#             D'abord SQL pour trouver les IDs, puis VECTOR filtré
#
# Le router utilise d'abord des règles déterministes (rapide),
# puis un LLM comme fallback pour les cas ambigus.
# ============================================================

import re
import structlog
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = structlog.get_logger()


class SearchMode(str, Enum):
    SQL = "SQL"
    VECTOR = "VECTOR"
    HYBRID = "HYBRID"


@dataclass
class RouterDecision:
    mode: SearchMode
    confidence: float
    sql_query_key: Optional[str] = None
    sql_params: Optional[dict] = None
    vector_filters: Optional[dict] = None
    reasoning: str = ""


# ── Patterns déterministes ────────────────────────────────────
# Mots-clés qui indiquent clairement le mode de recherche

SQL_KEYWORDS = {
    # Comptage et statistiques
    r"\bcombien\b", r"\bnombre\b", r"\bcount\b", r"\btotal\b",
    r"\bstatistique", r"\btableau de bord\b", r"\bdashboard\b",
    # Listage
    r"\bliste\b", r"\bénumérer\b", r"\bmontrer\b", r"\bafficher\b",
    r"\bquels sont\b", r"\bquelles sont\b",
    # Entités structurées
    r"\bdossier\b", r"\bhonorai", r"\bséance\b", r"\baudience\b",
    r"\bavocat\b", r"\bclient\b", r"\bpartie adverse\b",
    # Statuts et dates
    r"\ben cours\b", r"\bclotur", r"\bimpay", r"\bpartiel\b",
    r"\bprogram", r"\bà venir\b", r"\bprochain", r"\bdate\b",
    # Montants
    r"\bmontant\b", r"\brecouvr", r"\bpayé\b", r"\breste à payer\b",
    r"\bMAD\b", r"\bDH\b",
    # Questions de statut
    r"\bquel est le statut\b", r"\bquel est l'état\b",
}

VECTOR_KEYWORDS = {
    # Contenu documentaire
    r"\bque dit\b", r"\bcontenu\b", r"\btexte\b", r"\bdocument\b",
    r"\bclause\b", r"\barticle\b", r"\bdisposition\b",
    r"\bjugement\b", r"\bdécision\b", r"\bordonnance\b",
    r"\brequête\b", r"\blettre\b", r"\bmise en demeure\b",
    r"\bextrait\b", r"\bpassage\b", r"\bmentionn", r"\bprescri",
    # Recherche sémantique
    r"\bchercher dans\b", r"\btrouver dans\b", r"\brecherc",
    r"\bsimilaire\b", r"\bcomparable\b", r"\banalogue\b",
}

HYBRID_PATTERNS = [
    # Document d'un dossier spécifique
    r"document[s]?\s+(?:du|de|des)\s+dossier",
    r"dossier\s+\w+.*(?:document|jugement|décision)",
    r"(?:pièce|acte|lettre).*dossier",
    # Contenu lié à une affaire
    r"dans\s+l[ae]\s+(?:dossier|affaire)\s+.*(?:dit|mentionne|contient)",
]

# Mapping patterns → clés SQL
INTENT_TO_SQL = {
    r"combien.*dossier": "count_dossiers",
    r"statut.*dossier|dossier.*statut": "count_dossiers",
    r"dossier.*en cours|en cours.*dossier": "dossiers_en_cours",
    r"séance[s]?\s+à venir|prochaine[s]?\s+audience": "seances_a_venir",
    r"séance[s]?\s+programme": "seances_a_venir",
    r"honoraire[s]?\s+impay|impay.*honoraire": "honoraires_impayes",
    r"total.*honoraire|honoraire.*total": "total_honoraires",
    r"liste.*avocat|avocat.*liste": "list_avocats",
    r"tableau de bord|vue d'ensemble|statistique": "dashboard_stats",
    r"document.*en attente|en attente.*document": "documents_en_attente",
}


def route_query(question: str) -> RouterDecision:
    """
    Analyse la question et décide du mode de recherche optimal.

    Stratégie :
    1. Vérifier les patterns HYBRID (prioritaire)
    2. Compter les indices SQL vs VECTOR
    3. Fallback LLM si ambiguïté
    4. Défaut VECTOR si rien ne match

    Args:
        question: Question de l'utilisateur en français/arabe/anglais

    Returns:
        RouterDecision avec mode, confidence, et paramètres SQL/Vector
    """
    q_lower = question.lower().strip()

    # ── 1. Test HYBRID en priorité ────────────────────────────
    for pattern in HYBRID_PATTERNS:
        if re.search(pattern, q_lower):
            # Extraire le numéro de dossier si présent
            dossier_match = re.search(
                r'dossier\s+([A-Z0-9\-/]+)',
                question,
                re.IGNORECASE
            )
            params = {}
            if dossier_match:
                params["numero"] = f"%{dossier_match.group(1)}%"

            logger.info("router_decision", mode="HYBRID", question=q_lower[:80])
            return RouterDecision(
                mode=SearchMode.HYBRID,
                confidence=0.85,
                sql_query_key="dossier_by_numero" if params else None,
                sql_params=params if params else None,
                reasoning="Pattern HYBRID détecté : question croise dossier + contenu document",
            )

    # ── 2. Compter les indices SQL vs VECTOR ──────────────────
    sql_score = sum(
        1 for pattern in SQL_KEYWORDS
        if re.search(pattern, q_lower)
    )
    vector_score = sum(
        1 for pattern in VECTOR_KEYWORDS
        if re.search(pattern, q_lower)
    )

    # ── 3. Identifier la requête SQL spécifique ───────────────
    sql_query_key = None
    sql_params = {}

    for pattern, query_key in INTENT_TO_SQL.items():
        if re.search(pattern, q_lower):
            sql_query_key = query_key
            break

    # Extraire des paramètres contextuels
    if sql_query_key:
        sql_params = _extract_sql_params(question, sql_query_key)

    # ── 4. Décision finale ────────────────────────────────────
    total = sql_score + vector_score

    if total == 0:
        # Aucun indice → défaut VECTOR (recherche sémantique générale)
        logger.info("router_decision", mode="VECTOR", reason="no_keywords")
        return RouterDecision(
            mode=SearchMode.VECTOR,
            confidence=0.5,
            reasoning="Aucun indice clair — recherche sémantique par défaut",
        )

    if sql_score > vector_score and sql_query_key:
        confidence = min(0.95, 0.6 + (sql_score / max(total, 1)) * 0.4)
        logger.info("router_decision", mode="SQL", sql_key=sql_query_key)
        return RouterDecision(
            mode=SearchMode.SQL,
            confidence=confidence,
            sql_query_key=sql_query_key,
            sql_params=sql_params,
            reasoning=f"Indices SQL ({sql_score}) > VECTOR ({vector_score}), requête: {sql_query_key}",
        )

    if sql_score > vector_score and not sql_query_key:
        # SQL probable mais pas de requête mappée → HYBRID
        logger.info("router_decision", mode="HYBRID", reason="sql_no_mapping")
        return RouterDecision(
            mode=SearchMode.HYBRID,
            confidence=0.65,
            reasoning="Question structurée mais sans requête SQL directe",
        )

    if vector_score >= sql_score:
        confidence = min(0.95, 0.6 + (vector_score / max(total, 1)) * 0.4)
        logger.info("router_decision", mode="VECTOR", vector_score=vector_score)
        return RouterDecision(
            mode=SearchMode.VECTOR,
            confidence=confidence,
            reasoning=f"Indices VECTOR ({vector_score}) >= SQL ({sql_score})",
        )

    # Fallback
    return RouterDecision(
        mode=SearchMode.VECTOR,
        confidence=0.5,
        reasoning="Fallback par défaut",
    )


def _extract_sql_params(question: str, query_key: str) -> dict:
    """
    Extrait les paramètres SQL depuis la question naturelle.
    Exemple : "dossier 2024/001" → {"numero": "%2024/001%"}
    """
    params = {}

    # Numéro de dossier
    dossier_match = re.search(
        r'dossier\s+(?:n[°o]?\s*)?([A-Z0-9\-/]+)',
        question,
        re.IGNORECASE
    )
    if dossier_match and query_key in ("dossier_by_numero",):
        params["numero"] = f"%{dossier_match.group(1)}%"

    # Nom d'avocat
    avocat_match = re.search(
        r'avocat\s+([A-ZÀ-Ö][a-zà-ö]+(?:\s+[A-ZÀ-Ö][a-zà-ö]+)?)',
        question,
        re.IGNORECASE
    )
    if avocat_match and query_key == "dossiers_by_avocat":
        params["name"] = f"%{avocat_match.group(1)}%"

    return params


async def route_with_llm_fallback(question: str) -> RouterDecision:
    """
    Version améliorée : utilise le LLM pour les cas ambigus.
    Seulement appelée si la décision déterministe a une confiance < 0.7.
    """
    decision = route_query(question)

    if decision.confidence >= 0.7:
        return decision

    # Appel LLM pour clarification
    try:
        from app.services.llm_service import call_ollama_raw

        prompt = f"""Tu es un assistant juridique. Classifie la question suivante.
Réponds UNIQUEMENT avec un de ces mots : SQL, VECTOR, HYBRID

- SQL : question sur des données structurées (comptages, listes, statuts, dates, montants)
- VECTOR : question sur le contenu des documents juridiques
- HYBRID : question qui croise données structurées ET contenu de documents

Question : {question}

Réponse (un seul mot) :"""

        llm_response = await call_ollama_raw(prompt, max_tokens=10)
        llm_mode = llm_response.strip().upper()

        if llm_mode in ("SQL", "VECTOR", "HYBRID"):
            decision.mode = SearchMode(llm_mode)
            decision.confidence = 0.8
            decision.reasoning += f" | LLM override: {llm_mode}"
            logger.info("router_llm_override", mode=llm_mode)

    except Exception as e:
        logger.warning("router_llm_fallback_failed", error=str(e))

    return decision