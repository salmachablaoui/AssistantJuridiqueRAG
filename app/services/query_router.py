# app/services/query_router.py — FIXED
import re
import structlog
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = structlog.get_logger()


class SearchMode(str, Enum):
    SQL    = "SQL"
    VECTOR = "VECTOR"
    HYBRID = "HYBRID"


@dataclass
class RouterDecision:
    mode: SearchMode
    confidence: float
    sql_query_key: Optional[str] = None
    sql_params:    Optional[dict] = None
    vector_filters: Optional[dict] = None
    reasoning: str = ""


# ── Patterns SQL ──────────────────────────────────────────────
SQL_KEYWORDS = {
    r"\bcombien\b", r"\bnombre\b", r"\bcount\b", r"\btotal\b",
    r"\bstatistique", r"\btableau de bord\b", r"\bdashboard\b",
    r"\bliste\b", r"\bénumérer\b", r"\bmontrer\b", r"\bafficher\b",
    r"\bquels sont\b", r"\bquelles sont\b",
    r"\bdossier\b", r"\bhonorai", r"\bséance\b", r"\baudience\b",
    r"\bavocat\b", r"\bclient\b", r"\bpartie adverse\b",
    r"\ben cours\b", r"\bclotur", r"\bimpay", r"\bpartiel\b",
    r"\bprogram", r"\bà venir\b", r"\bprochain", r"\bdate\b",
    r"\bmontant\b", r"\brecouvr", r"\bpayé\b", r"\breste à payer\b",
    r"\bMAD\b", r"\bDH\b",
    r"\bquel est le statut\b", r"\bquel est l'état\b",
    # ← AJOUT : patterns manquants
    r"\bnombre total\b", r"\btous les dossiers\b", r"\bdossiers actifs\b",
    r"\bséances liées\b", r"\bséances du dossier\b",
}

VECTOR_KEYWORDS = {
    r"\bque dit\b", r"\bcontenu\b", r"\btexte\b", r"\bdocument\b",
    r"\bclause\b", r"\barticle\b", r"\bdisposition\b",
    r"\bjugement\b", r"\bdécision\b", r"\bordonnance\b",
    r"\brequête\b", r"\blettre\b", r"\bmise en demeure\b",
    r"\bextrait\b", r"\bpassage\b", r"\bmentionn", r"\bprescri",
    r"\bchercher dans\b", r"\btrouver dans\b", r"\brecherc",
    r"\bsimilaire\b", r"\bcomparable\b", r"\banalogue\b",
    r"\bbrief\b", r"\bbrèf\b", r"\bbrièvement\b", r"\bre[s]?um",
}

HYBRID_PATTERNS = [
    r"document[s]?\s+(?:du|de|des)\s+dossier",
    r"dossier\s+\w+.*(?:document|jugement|d[eé]cision)",
    r"(?:pi[eè]ce|acte|lettre).*dossier",
    r"dans\s+l[ae]\s+(?:dossier|affaire)\s+.*(?:dit|mentionne|contient)",
    r"tous les d[eé]tails.*dossier",
    r"d[eé]tails? complets?.*dossier",
    r"r[eé]sum[eé] complet.*dossier",
    r"tout sur.*dossier",
    r"(?:pr[eé]sente|montre|donne).*dossier\s+[A-Z0-9\-]+",
    # ← AJOUT : patterns manquants
    r"documents\s+associ[eé]s?\s+au\s+dossier",
    r"dossier\s+[A-Z0-9\-]+\s*\??\s*$",   # "dossier DSS-2026-0004 ?"
]

# ── Mapping intentions → SQL ──────────────────────────────────
INTENT_TO_SQL = {
    r"combien.*dossier|nombre.*dossier|nombre total.*dossier": "count_dossiers",
    r"statut.*dossier|dossier.*statut":           "count_dossiers",
    r"dossier.*en cours|en cours.*dossier":       "dossiers_en_cours",
    r"s[eé]ance[s]?\s+[àa]\s+venir|prochaine[s]?\s+audience": "seances_a_venir",
    r"s[eé]ance[s]?\s+programm":                 "seances_a_venir",
    # ← AJOUT : séances d'un dossier → SQL pas VECTOR
    r"s[eé]ance[s]?\s+li[eé]es?\s+[àa]|s[eé]ance[s]?\s+du\s+dossier": "seances_a_venir",
    r"honoraire[s]?\s+impay|impay.*honoraire":    "honoraires_impayes",
    r"total.*honoraire|honoraire.*total":         "total_honoraires",
    r"liste.*avocat|avocat.*liste":               "list_avocats",
    r"tableau de bord|vue d'ensemble|statistique|dashboard": "dashboard_stats",
    r"document.*en attente|en attente.*document": "documents_en_attente",
}

# ── Numéros de dossier valides (regex) ───────────────────────
# ← AJOUT : validation pour éviter d'inventer des dossiers
DOSSIER_NUMBER_PATTERN = re.compile(
    r'\b([A-Z]{2,4}-\d{4}-\d{4})\b',
    re.IGNORECASE
)


def extract_dossier_number(question: str) -> Optional[str]:
    """Extrait le numéro de dossier de la question."""
    match = DOSSIER_NUMBER_PATTERN.search(question)
    return match.group(1).upper() if match else None


def route_query(question: str) -> RouterDecision:
    q_lower = question.lower().strip()

    # ── 1. Test HYBRID en priorité ────────────────────────────
    for pattern in HYBRID_PATTERNS:
        if re.search(pattern, q_lower):
            dossier_num = extract_dossier_number(question)
            params = {"numero": f"%{dossier_num}%"} if dossier_num else {}

            logger.info("router_decision", mode="HYBRID", question=q_lower[:80])
            return RouterDecision(
                mode=SearchMode.HYBRID,
                confidence=0.85,
                sql_query_key="dossier_by_numero" if params else None,
                sql_params=params if params else None,
                reasoning="Pattern HYBRID : question croise dossier + contenu document",
            )

    # ── 2. Scores SQL vs VECTOR ───────────────────────────────
    sql_score = sum(1 for p in SQL_KEYWORDS if re.search(p, q_lower))
    vector_score = sum(1 for p in VECTOR_KEYWORDS if re.search(p, q_lower))

    # ── 3. Mapping SQL spécifique ─────────────────────────────
    sql_query_key = None
    sql_params = {}
    for pattern, query_key in INTENT_TO_SQL.items():
        if re.search(pattern, q_lower):
            sql_query_key = query_key
            break

    if sql_query_key:
        sql_params = _extract_sql_params(question, sql_query_key)

    # ── 4. Décision ───────────────────────────────────────────
    total = sql_score + vector_score

    if total == 0:
        return RouterDecision(
            mode=SearchMode.VECTOR,
            confidence=0.5,
            reasoning="Aucun indice — recherche sémantique par défaut",
        )

    if sql_score > vector_score and sql_query_key:
        confidence = min(0.95, 0.6 + (sql_score / max(total, 1)) * 0.4)
        return RouterDecision(
            mode=SearchMode.SQL,
            confidence=confidence,
            sql_query_key=sql_query_key,
            sql_params=sql_params,
            reasoning=f"SQL ({sql_score}) > VECTOR ({vector_score}), requête: {sql_query_key}",
        )

    if sql_score > vector_score and not sql_query_key:
        return RouterDecision(
            mode=SearchMode.HYBRID,
            confidence=0.65,
            reasoning="Question structurée sans requête SQL directe",
        )

    confidence = min(0.95, 0.6 + (vector_score / max(total, 1)) * 0.4)
    return RouterDecision(
        mode=SearchMode.VECTOR,
        confidence=confidence,
        reasoning=f"VECTOR ({vector_score}) >= SQL ({sql_score})",
    )


def _extract_sql_params(question: str, query_key: str) -> dict:
    params = {}
    dossier_num = extract_dossier_number(question)
    if dossier_num and query_key == "dossier_by_numero":
        params["numero"] = f"%{dossier_num}%"

    avocat_match = re.search(
        r'avocat\s+([A-ZÀ-Ö][a-zà-ö]+(?:\s+[A-ZÀ-Ö][a-zà-ö]+)?)',
        question, re.IGNORECASE
    )
    if avocat_match and query_key == "dossiers_by_avocat":
        params["name"] = f"%{avocat_match.group(1)}%"
    return params


async def route_with_llm_fallback(question: str) -> RouterDecision:
    decision = route_query(question)
    if decision.confidence >= 0.7:
        return decision

    try:
        from app.services.llm_service import call_ollama_raw
        prompt = f"""Classifie la question suivante.
Réponds UNIQUEMENT avec : SQL, VECTOR, ou HYBRID

- SQL : comptages, listes, statuts, dates, montants, séances
- VECTOR : contenu de documents juridiques (jugements, requêtes)
- HYBRID : croise données structurées ET contenu de documents

Question : {question}

Réponse :"""
        llm_response = await call_ollama_raw(prompt, max_tokens=10)
        llm_mode = llm_response.strip().upper().split()[0]
        if llm_mode in ("SQL", "VECTOR", "HYBRID"):
            decision.mode = SearchMode(llm_mode)
            decision.confidence = 0.8
            decision.reasoning += f" | LLM: {llm_mode}"
    except Exception as e:
        logger.warning("router_llm_fallback_failed", error=str(e))

    return decision