# app/services/llm_service.py — v3
import httpx
import structlog
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings

logger = structlog.get_logger()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def call_ollama(prompt: str, system_prompt: str = None, temperature: float = None) -> str:
    temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
    payload = {
        "model": settings.CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 400,
            "num_ctx": 2048,
            "num_thread": 4,
            "repeat_penalty": 1.1,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(f"{settings.OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"].strip()
    except Exception as e:
        logger.error("ollama_error", error=str(e))
        return f"[LLM indisponible] {str(e)}"


async def call_ollama_raw(prompt: str, max_tokens: int = 10) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.ROUTER_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": max_tokens},
            },
        )
        response.raise_for_status()
        return response.json()["response"].strip()


# ── Prompts système ───────────────────────────────────────────

SYSTEM_PROMPT_SQL = """Tu es un assistant juridique ANP Legal. 
Réponds en français, de manière courte et claire.
Utilise UNIQUEMENT les données fournies. Ne fabrique rien.
Si les données sont vides, dis "Aucun résultat trouvé."."""

SYSTEM_PROMPT_VECTOR = """Tu es un assistant juridique ANP Legal spécialisé dans l'analyse de documents.
RÈGLES ABSOLUES :
1. Utilise UNIQUEMENT les extraits fournis. Ne déduis rien d'autre.
2. Si l'information demandée n'est pas dans les extraits : réponds "Cette information ne figure pas dans les documents disponibles."
3. Cite le nom du fichier source entre parenthèses après chaque information.
4. Réponds en français, de manière concise et structurée.
5. Ne mentionne jamais de dossiers ou documents absents des extraits."""

SYSTEM_PROMPT_HYBRID = """Tu es un assistant juridique ANP Legal.
Tu combines données structurées (base de données) et contenu de documents.
RÈGLES :
1. Utilise UNIQUEMENT les informations explicitement fournies.
2. Si un dossier n'est pas trouvé : dis-le clairement en une phrase.
3. Distingue données base vs documents : marque [Base] ou [Document].
4. Réponds en français, de manière structurée et concise."""


# ── Format direct sans LLM ────────────────────────────────────

def format_sql_results_direct(results: list[dict], query_key: str) -> Optional[str]:
    DIRECT_KEYS = {"dashboard_stats", "count_dossiers", "total_honoraires"}
    if query_key not in DIRECT_KEYS or not results:
        return None

    if query_key == "dashboard_stats":
        r = results[0]
        def fmt(v):
            try: return f"{float(v):,.2f} MAD"
            except: return f"{v} MAD"
        return (
            f"📊 Tableau de bord ANP Legal\n\n"
            f"📁 Dossiers\n"
            f"• En cours    : {r.get('dossiers_en_cours', 0)}\n"
            f"• Clôturés    : {r.get('dossiers_clotures', 0)}\n"
            f"• Suspendus   : {r.get('dossiers_suspendus', 0)}\n\n"
            f"📅 Séances à venir : {r.get('seances_a_venir', 0)}\n\n"
            f"💰 Honoraires\n"
            f"• Impayés     : {r.get('honoraires_impayes', 0)}\n"
            f"• À recouvrer : {fmt(r.get('total_a_recouvrer', 0))}\n\n"
            f"📄 Documents en attente : {r.get('documents_en_attente', 0)}\n"
            f"👨‍⚖️ Avocats actifs      : {r.get('avocats_actifs', 0)}"
        )

    if query_key == "count_dossiers":
        lines = ["📁 Dossiers par statut :"]
        for row in results:
            statut = row.get('statut', '?')
            emoji = {"en_cours": "🟢", "cloture": "✅", "suspendu": "⏸️"}.get(statut, "•")
            lines.append(f"  {emoji} {statut} : {row.get('total', 0)}")
        return "\n".join(lines)

    if query_key == "total_honoraires":
        r = results[0]
        def fmt(v):
            try: return f"{float(v):,.2f} MAD"
            except: return f"{v} MAD"
        return (
            f"💰 Honoraires globaux\n"
            f"• Total facturé  : {fmt(r.get('total_montant', 0))}\n"
            f"• Total payé     : {fmt(r.get('total_paye', 0))}\n"
            f"• Reste à payer  : {fmt(r.get('total_reste', 0))}\n"
            f"• Dossiers       : {r.get('nombre_dossiers', 0)}"
        )
    return None


# ── Génération réponses ───────────────────────────────────────

async def generate_sql_answer(question: str, sql_context: str, query_key: str) -> dict:
    try:
        prompt = f"""Données :
{sql_context}

Question : {question}

Réponse courte et claire en français :"""
        answer = await call_ollama(prompt, system_prompt=SYSTEM_PROMPT_SQL)
        return {
            "answer": answer,
            "confidence": 0.90,
            "sources": [{"type": "database", "query": query_key}],
        }
    except Exception as e:
        logger.error("llm_sql_error", error=str(e))
        return {"answer": sql_context, "confidence": 0.70, "sources": [{"type": "database"}]}


async def generate_vector_answer(question: str, chunks: list[dict]) -> dict:
    # ← Réponse claire si aucun chunk
    if not chunks:
        return {
            "answer": "Je n'ai pas trouvé de documents pertinents pour cette question.\n\nSi vous cherchez le contenu d'un dossier spécifique, vérifiez que :\n• Le numéro de dossier est correct\n• Les documents ont été indexés",
            "confidence": 0.0,
            "sources": [],
        }

    context_parts = []
    sources = []
    seen_docs = set()

    for i, chunk in enumerate(chunks, 1):
        doc_id = chunk.get("document_id")
        nom = chunk.get("nom_fichier", "Document")
        # Nettoyer le nom pour affichage
        nom_affiche = nom.replace('.pdf', '').split('_')[-1] if '_' in nom else nom
        context_parts.append(
            f"[Extrait {i} — {nom_affiche} | Score: {chunk.get('score', 0):.2f}]\n"
            f"{chunk['text']}"
        )
        if doc_id and doc_id not in seen_docs:
            seen_docs.add(doc_id)
            sources.append({
                "type":          "document",
                "document_id":   doc_id,
                "nom_fichier":   nom,
                "dossier":       chunk.get("numero_dossier", ""),
                "score":         chunk.get("score", 0),
                "type_document": chunk.get("document_type", ""),
            })

    context = "\n\n---\n\n".join(context_parts)
    best_score = max((c.get("score", 0) for c in chunks), default=0)
    confidence = round(min(0.95, best_score * 1.1), 2)

    prompt = f"""Extraits de documents juridiques :

{context}

Question : {question}

Réponds de manière concise en te basant UNIQUEMENT sur ces extraits.
Si l'information n'y est pas, dis-le clairement en une phrase.
Réponse :"""

    try:
        answer = await call_ollama(prompt, system_prompt=SYSTEM_PROMPT_VECTOR)
        return {"answer": answer, "confidence": confidence, "sources": sources}
    except Exception as e:
        logger.error("llm_vector_error", error=str(e))
        return {"answer": "Erreur lors de la génération de la réponse.", "confidence": 0.0, "sources": sources}


async def generate_hybrid_answer(question: str, sql_context: str, chunks: list[dict]) -> dict:
    sources = []
    if sql_context:
        sources.append({"type": "database"})

    seen_docs = set()
    for chunk in chunks:
        doc_id = chunk.get("document_id")
        if doc_id and doc_id not in seen_docs:
            seen_docs.add(doc_id)
            sources.append({
                "type":        "document",
                "document_id": doc_id,
                "nom_fichier": chunk.get("nom_fichier", ""),
                "score":       chunk.get("score", 0),
            })

    # Cas dossier introuvable — réponse directe sans LLM
    if not sql_context and not chunks:
        return {
            "answer": "Ce dossier n'existe pas dans le système. Vérifiez le numéro et réessayez.",
            "confidence": 0.0,
            "sources": [],
        }

    doc_context = "\n\n".join(
        f"[{chunk.get('nom_fichier', 'Doc').split('_')[-1]}]: {chunk['text']}"
        for chunk in chunks
    ) if chunks else "Aucun document indexé pour ce dossier."

    best_score = max((c.get("score", 0) for c in chunks), default=0)
    confidence = round(min(0.92, 0.7 + best_score * 0.25), 2)

    prompt = f"""Réponds à la question en combinant ces sources :

[Base de données]
{sql_context or "Aucune donnée trouvée."}

[Documents]
{doc_context}

Question : {question}

Réponse structurée et concise. Si une source ne contient pas l'info, ignore-la.
Réponse :"""

    try:
        answer = await call_ollama(prompt, system_prompt=SYSTEM_PROMPT_HYBRID)
        return {"answer": answer, "confidence": confidence, "sources": sources}
    except Exception as e:
        logger.error("llm_hybrid_error", error=str(e))
        return {"answer": "Erreur LLM.", "confidence": 0.0, "sources": sources}