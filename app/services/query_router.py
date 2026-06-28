# ============================================================
# app/services/query_router.py — v3.0
# ANP Legal — Hybrid RAG
#
# Fixes v3.0 :
#   - "avocats disponibles" → SQL (plus VECTOR)
#   - "affiche détails dossier DSS-2026-0004" → HYBRID
#   - honoraires/séances/documents + numéro → HYBRID
#     (le service résout numero→id en sous-requête SQL)
#   - Chitchat conservé
#   - _extract_sql_params : toujours "numero" (jamais dossier_id)
# ============================================================

import re
import structlog
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

logger = structlog.get_logger()

# ── Métriques ─────────────────────────────────────────────────
_router_metrics = {
    "calls":               defaultdict(int),
    "chitchat_hits":       0,
    "llm_fallback_calls":  0,
    "llm_fallback_errors": 0,
}

def get_router_metrics() -> dict:
    return {
        "by_mode":             dict(_router_metrics["calls"]),
        "chitchat_hits":       _router_metrics["chitchat_hits"],
        "llm_fallback_calls":  _router_metrics["llm_fallback_calls"],
        "llm_fallback_errors": _router_metrics["llm_fallback_errors"],
    }


class SearchMode(str, Enum):
    CHITCHAT = "CHITCHAT"
    SQL      = "SQL"
    VECTOR   = "VECTOR"
    HYBRID   = "HYBRID"


@dataclass
class RouterDecision:
    mode:           SearchMode
    confidence:     float
    sql_query_key:  Optional[str]  = None
    sql_params:     Optional[dict] = None
    vector_filters: Optional[dict] = None
    reasoning:      str            = ""
    chitchat_reply: Optional[str]  = None


# ── Chitchat ──────────────────────────────────────────────────
_CHITCHAT_RULES: list[tuple[str, str]] = [
    (r"^(bonjour|salut|hello|hi|salam|bonsoir)\b",
     "Bonjour ! Je suis votre assistant juridique ANP Legal. "
     "Comment puis-je vous aider ?"),
    (r"^(bonne journée|bonne soirée|bonne nuit|bonne continuation)\b",
     "Merci, bonne journée à vous également !"),
    (r"\b(merci|thank you|thanks)\b",
     "Avec plaisir ! N'hésitez pas si vous avez d'autres questions."),
    (r"^(ok|okay|d'accord|parfait|très bien|nickel|super|bien reçu"
     r"|compris|vu|noté|c'est bon|c'est clair)\b",
     "Très bien. Que puis-je faire d'autre pour vous ?"),
    (r"\b(au revoir|bye|à bientôt|à plus|ciao|bonne fin de journée)\b",
     "Au revoir ! À bientôt."),
    (r"qui es.?tu|c'est quoi ton rôle|à quoi tu sers|tu es quoi|what are you",
     "Je suis l'assistant juridique ANP Legal. Je consulte la base de données "
     "et les documents juridiques pour répondre à vos questions."),
    (r"^(aide|help|aidez.moi|comment.*(utiliser|fonctionn))\b",
     "Questions possibles :\n"
     "• « Quels sont les dossiers clôturés ? »\n"
     "• « Honoraires impayés »\n"
     "• « Détails du dossier DSS-2026-0004 »\n"
     "• « Séances de la semaine »\n"
     "• « Que dit le jugement DSS-2026-0004 ? »"),
    (r"\b(blague|joke|météo|weather|foot|ballon)\b",
     "Je suis spécialisé dans les affaires juridiques ANP."),
]

_COMPILED_CHITCHAT = [
    (re.compile(p, re.IGNORECASE | re.UNICODE), r)
    for p, r in _CHITCHAT_RULES
]


def detect_chitchat(question: str) -> Optional[RouterDecision]:
    q = question.strip()
    for pat, reply in _COMPILED_CHITCHAT:
        if pat.search(q):
            _router_metrics["chitchat_hits"] += 1
            _router_metrics["calls"]["CHITCHAT"] += 1
            return RouterDecision(
                mode=SearchMode.CHITCHAT, confidence=1.0,
                reasoning=f"Chitchat: {pat.pattern[:40]}",
                chitchat_reply=reply,
            )
    return None


# ── Numéro de dossier ─────────────────────────────────────────
_DOSSIER_RE = re.compile(r'\b([A-Z]{2,4}-\d{4}-\d{4})\b', re.IGNORECASE)

def extract_dossier_number(question: str) -> Optional[str]:
    m = _DOSSIER_RE.search(question)
    return m.group(1).upper() if m else None


# ── Patterns HYBRID — dossier + contenu document ─────────────
_HYBRID_PATTERNS = [
    # jugement/requête/ordonnance + dossier ou numéro
    r"(?:jugement|requ[eê]te|ordonnance|d[eé]cision|lettre|mise en demeure)"
    r".*(?:dossier|[A-Z]{2,4}-\d{4}-\d{4})",
    r"(?:dossier|[A-Z]{2,4}-\d{4}-\d{4})"
    r".*(?:jugement|requ[eê]te|ordonnance|d[eé]cision|lettre|mise en demeure)",
    # "que dit / que contient" + dossier/numéro
    r"(?:que\s+dit|que\s+contient|que\s+mentionne|que\s+pr[eé]voit|que\s+stipule)"
    r".*(?:dossier|[A-Z]{2,4}-\d{4}-\d{4})",
    # documents d'un dossier numéroté
    r"document[s]?\s+(?:du|de|des|associ[eé]s?|li[eé]s?)\s+(?:dossier\s+)?[A-Z]{2,4}-\d{4}-\d{4}",
    # tous les détails / fiche + numéro explicite
    r"(?:tous\s+les?\s+d[eé]tails?|d[eé]tails?\s+complets?|fiche\s+compl[eè]te|tout\s+sur)"
    r"\s+(?:le\s+|du\s+)?(?:dossier\s+)?[A-Z]{2,4}-\d{4}-\d{4}",
    # résumé + numéro
    r"r[eé]sum[eé]\s+complet.*[A-Z]{2,4}-\d{4}-\d{4}",
]

# ── Patterns SQL — mots-clés de score ────────────────────────
_SQL_KEYWORDS = {
    r"\bcombien\b", r"\bnombre\b", r"\btotal\b", r"\bstatistique",
    r"\btableau de bord\b", r"\bdashboard\b",
    r"\bliste\b", r"\bénumérer\b", r"\bafficher\b",
    r"\bquels sont\b", r"\bquelles sont\b",
    r"\bdossier\b", r"\bhonorai", r"\bséance\b", r"\baudience\b",
    r"\bavocat\b", r"\bclient\b", r"\bpartie adverse\b",
    r"\ben cours\b", r"\bcl[oô]tur", r"\bsuspendu\b", r"\bimpay",
    r"\bpartiel\b", r"\bprogramm", r"\bà venir\b", r"\bprochain",
    r"\bmontant\b", r"\brecouvr", r"\bpayé\b", r"\breste à payer\b",
    r"\bMAD\b", r"\bDH\b",
    r"\bdonne.?moi\b", r"\bmontre.?moi\b", r"\baffiche.?moi\b",
    r"\bdonne\b", r"\bmontre\b", r"\baffiche\b",
    r"\bdisponible\b", r"\bactif\b", r"\bactifs\b",
    r"\bdétails\b", r"\bfiche\b", r"\binfo", r"\brenseign",
}

_VECTOR_KEYWORDS = {
    r"\bque dit\b", r"\bcontenu\b", r"\btexte\b",
    r"\bclause\b", r"\barticle\b", r"\bdisposition\b",
    r"\bjugement\b", r"\bd[eé]cision\b", r"\bordonnance\b",
    r"\brequ[eê]te\b", r"\blettre\b", r"\bmise en demeure\b",
    r"\bextrait\b", r"\bpassage\b", r"\bmentionn", r"\bprescri",
    r"\bchercher dans\b", r"\btrouver dans\b",
    r"\bresum", r"\bque contient\b",
    r"\bque pr[eé]voit\b", r"\bque stipule\b",
}

# ── Mapping intention → clé SQL ───────────────────────────────
# IMPORTANT : les patterns avec numéro explicite sont traités
# en HYBRID avant d'arriver ici. Ces patterns couvrent
# les questions SANS numéro de dossier.
_INTENT_TO_SQL: list[tuple[str, str]] = [

    # Dossier detail (sans numéro → SQL retourne [] proprement)
    (r"(?:tous\s+les?\s+d[eé]tails?|d[eé]tails?\s+complets?|fiche\s+compl[eè]te|tout\s+sur)"
     r"\s+(?:le\s+|du\s+|ce\s+)?dossier\b",
     "dossier_detail_complet"),

    # Infos/détails d'un dossier par numéro → SQL direct
    (r"(?:info|d[eé]tail|fiche|donn[eé]e|renseign).*dossier\s+[A-Z]{2,4}-\d{4}-\d{4}"
     r"|dossier\s+[A-Z]{2,4}-\d{4}-\d{4}.*(?:info|d[eé]tail|fiche)",
     "dossier_detail_complet"),

    # Afficher/montrer un dossier par numéro → SQL direct
    (r"(?:affiche|montre|donne|show)\s+(?:moi\s+)?(?:le\s+|les?\s+)?(?:d[eé]tails?\s+)?(?:du\s+|de\s+)?dossier\s+[A-Z]{2,4}-\d{4}-\d{4}",
     "dossier_detail_complet"),

    # Comptages
    (r"combien.*dossier|nombre.*dossier|r[eé]partition.*dossier|statut.*dossier",
     "count_dossiers"),

    # Dossiers par statut
    (r"(?:(?:donne.?moi|liste|affiche|montre|quels\s+sont)\s+(?:les?\s+|la\s+liste\s+des?\s+)?)?"
     r"dossier[s]?\s+en\s+cours",
     "dossiers_en_cours"),
    (r"(?:(?:donne.?moi|liste|affiche|montre|quels\s+sont)\s+(?:les?\s+|la\s+liste\s+des?\s+)?)?"
     r"dossier[s]?\s+cl[oô][tû]ur|dossier[s]?\s+termin",
     "dossiers_clotures"),
    (r"(?:(?:donne.?moi|liste|affiche|montre|quels\s+sont)\s+(?:les?\s+|la\s+liste\s+des?\s+)?)?"
     r"dossier[s]?\s+suspendu",
     "dossiers_suspendus"),
    (r"tous\s+les\s+dossiers?|liste\s+(?:de\s+tous\s+les\s+)?dossiers?|dossiers?\s+cr[eé][eé]s?",
     "dossiers_all"),

    # Dossiers par avocat / client
    (r"dossiers?\s+(?:de\s+l'avocat|trait[eé]s?\s+par|g[eé]r[eé]s?\s+par)",
     "dossiers_by_avocat"),
    (r"dossiers?\s+(?:de|du|pour)\s+(?:le\s+client|la\s+partie|la\s+soci[eé]t[eé])",
     "dossiers_by_client"),

    # Séances avec numéro → SQL (sous-requête resolves numero→id)
    (r"s[eé]ance[s]?\s+(?:du|de|pour|li[eé]es?\s+[àa])\s+(?:le\s+)?dossier\s+[A-Z]{2,4}-\d{4}-\d{4}"
     r"|s[eé]ance[s]?\s+[A-Z]{2,4}-\d{4}-\d{4}",
     "seances_by_dossier"),
    (r"s[eé]ance[s]?\s+[àa]\s+venir|prochaine[s]?\s+(?:s[eé]ance|audience)",
     "seances_a_venir"),
    (r"s[eé]ance[s]?\s+(?:cette\s+semaine|de\s+la\s+semaine)",
     "seances_cette_semaine"),
    (r"s[eé]ance[s]?\s+pass[eé]es?|historique.*s[eé]ance|s[eé]ance[s]?\s+tenues?",
     "seances_passees"),

    # Honoraires avec numéro → SQL direct (sous-requête)
    (r"honoraire[s]?\s+(?:du|de|pour)\s+(?:le\s+)?dossier\s+[A-Z]{2,4}-\d{4}-\d{4}"
     r"|honoraire[s]?\s+[A-Z]{2,4}-\d{4}-\d{4}",
     "honoraires_by_dossier"),
    (r"honoraire[s]?\s+impay|impay.*honoraire",
     "honoraires_impayes"),
    (r"honoraire[s]?\s+par\s+avocat|r[eé]partition.*honoraire",
     "honoraires_by_avocat"),
    (r"total.*honoraire|honoraire.*total",
     "total_honoraires"),

    # Documents avec numéro → SQL direct (sous-requête)
    (r"document[s]?\s+(?:du|de|pour|associ[eé]s?)\s+(?:le\s+)?dossier\s+[A-Z]{2,4}-\d{4}-\d{4}"
     r"|document[s]?\s+[A-Z]{2,4}-\d{4}-\d{4}",
     "documents_by_dossier"),
    (r"document[s]?\s+en\s+attente|en\s+attente.*document",
     "documents_en_attente"),

    # Stades
    (r"stade[s]?\s+(?:du|de|pour)\s+(?:le\s+)?dossier\s+[A-Z]{2,4}-\d{4}-\d{4}",
     "stades_by_dossier"),

    # Avocats — toutes formulations incluant "disponibles"
    (r"liste.*avocat|avocat.*liste|tous\s+les\s+avocats?"
     r"|avocats?\s+disponibles?|avocats?\s+actifs?"
     r"|affiche.*avocats?|montre.*avocats?|quels?\s+(?:sont\s+les?\s+)?avocats?",
     "list_avocats"),
    (r"(?:profil|fiche|d[eé]tails?).*avocat\s+\w+",
     "avocat_by_name"),

    # Documents génériques (sans numéro)
    (r"document[s]?\s+(?:du|de|des|li[eé]s?|associ[eé]s?)\s+(?:ce\s+)?dossier\b",
     "documents_by_dossier"),

    # Dashboard
    (r"tableau\s+de\s+bord|vue\s+d'ensemble|statistiques?|dashboard|[eé]tat\s+g[eé]n[eé]ral",
     "dashboard_stats"),
]

# Clés qui attendent un :numero
_NUMERO_KEYED = {
    "dossier_by_numero", "dossier_detail_complet",
    "seances_by_dossier", "honoraires_by_dossier",
    "documents_by_dossier", "stades_by_dossier",
}


def _extract_sql_params(question: str, query_key: str) -> dict:
    params = {}
    num = extract_dossier_number(question)
    if num and query_key in _NUMERO_KEYED:
        params["numero"] = f"%{num}%"

    if query_key in ("dossiers_by_avocat", "avocat_by_name"):
        m = re.search(
            r'avocat\s+([A-ZÀ-Ö][a-zà-ö]+(?:\s+[A-ZÀ-Ö][a-zà-ö]+)?)',
            question, re.IGNORECASE
        )
        if m:
            params["name"] = f"%{m.group(1)}%"

    if query_key == "dossiers_by_client":
        m = re.search(
            r'(?:client|partie|société)\s+([A-ZÀ-Ö][A-Za-zÀ-öÙ-ü\s]+?)(?:\s*\?|$)',
            question, re.IGNORECASE
        )
        if m:
            params["name"] = f"%{m.group(1).strip()}%"

    return params


# ── Routeur principal ─────────────────────────────────────────
def route_query(question: str) -> RouterDecision:
    q_lower = question.lower().strip()

    # 1. Chitchat
    cc = detect_chitchat(question)
    if cc:
        return cc

    # 2. HYBRID (dossier + contenu document)
    for pattern in _HYBRID_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            num = extract_dossier_number(question)
            sql_params = {"numero": f"%{num}%"} if num else {}
            _router_metrics["calls"]["HYBRID"] += 1
            logger.info("router_decision", mode="HYBRID", q=q_lower[:80])
            return RouterDecision(
                mode=SearchMode.HYBRID,
                confidence=0.88,
                sql_query_key="dossier_detail_complet" if sql_params else None,
                sql_params=sql_params or None,
                reasoning="HYBRID: dossier + contenu document",
            )

    # 3. Scores
    sql_score    = sum(1 for p in _SQL_KEYWORDS    if re.search(p, q_lower))
    vector_score = sum(1 for p in _VECTOR_KEYWORDS if re.search(p, q_lower))

    # 4. Mapping → clé SQL
    sql_query_key = None
    for pattern, key in _INTENT_TO_SQL:
        if re.search(pattern, q_lower, re.IGNORECASE):
            sql_query_key = key
            break

    sql_params = {}
    if sql_query_key:
        sql_params = _extract_sql_params(question, sql_query_key)

    # 5. Décision
    total = sql_score + vector_score

    if total == 0:
        _router_metrics["calls"]["VECTOR"] += 1
        return RouterDecision(mode=SearchMode.VECTOR, confidence=0.50,
                              reasoning="Aucun indice → VECTOR")

    if sql_score > vector_score and sql_query_key:
        confidence = min(0.95, 0.60 + (sql_score / max(total, 1)) * 0.40)
        _router_metrics["calls"]["SQL"] += 1
        logger.info("router_decision", mode="SQL", key=sql_query_key,
                    conf=round(confidence, 2), q=q_lower[:80])
        return RouterDecision(
            mode=SearchMode.SQL, confidence=confidence,
            sql_query_key=sql_query_key, sql_params=sql_params,
            reasoning=f"SQL({sql_score})>VECTOR({vector_score}), clé:{sql_query_key}",
        )

    if sql_score > vector_score:
        _router_metrics["calls"]["SQL"] += 1
        return RouterDecision(mode=SearchMode.SQL, confidence=0.60,
                              reasoning="SQL net, pas de clé → SQL dynamique")

    confidence = min(0.95, 0.60 + (vector_score / max(total, 1)) * 0.40)
    _router_metrics["calls"]["VECTOR"] += 1
    return RouterDecision(mode=SearchMode.VECTOR, confidence=confidence,
                          reasoning=f"VECTOR({vector_score})>=SQL({sql_score})")


async def route_with_llm_fallback(question: str) -> RouterDecision:
    decision = route_query(question)
    if decision.mode == SearchMode.CHITCHAT or decision.confidence >= 0.70:
        return decision

    _router_metrics["llm_fallback_calls"] += 1
    try:
        from app.services.llm_service import call_ollama_raw
        prompt = (
            "Classifie. Réponds UNIQUEMENT: SQL, VECTOR, ou HYBRID\n"
            "SQL: listes, comptages, statuts, montants, séances, avocats\n"
            "VECTOR: contenu documents juridiques (jugements, requêtes)\n"
            "HYBRID: dossier spécifique + contenu de ses documents\n\n"
            f"Question: {question}\nRéponse:"
        )
        raw = await call_ollama_raw(prompt, max_tokens=10)
        llm_mode = raw.strip().upper().split()[0]
        if llm_mode in ("SQL", "VECTOR", "HYBRID"):
            decision.mode = SearchMode(llm_mode)
            decision.confidence = 0.80
            decision.reasoning += f" | LLM:{llm_mode}"
    except Exception as e:
        _router_metrics["llm_fallback_errors"] += 1
        logger.warning("llm_fallback_failed", error=str(e))

    return decision