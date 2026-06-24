# ============================================================
# app/services/llm_service.py
# REFACTORÉ depuis chat_service.py
#
# Améliorations :
#   - Prompts différenciés par mode (SQL / VECTOR / HYBRID)
#   - Réponse structurée avec sources et score de confiance
#   - Retry automatique sur timeout
#   - Support async natif
# ============================================================

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings

logger = structlog.get_logger()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def call_ollama(
    prompt: str,
    system_prompt: str = None,
    temperature: float = None,
) -> str:
    temperature = temperature or settings.LLM_TEMPERATURE

    payload = {
    "model": settings.CHAT_MODEL,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.1,
        "num_predict": 100,   # ← très court
        "num_ctx": 512,       # ← contexte réduit
    },
}

    if system_prompt:
        payload["system"] = system_prompt

    try:
        async with httpx.AsyncClient(timeout=300) as client:  # ← 300 secondes
            response = await client.post(
                f"{settings.OLLAMA_URL}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
    except Exception as e:
        logger.error("ollama_error", error=str(e))
        return f"[LLM indisponible] {str(e)}"


async def call_ollama_raw(prompt: str, max_tokens: int = 100) -> str:
    """Appel minimal pour le router (classification courte)"""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.ROUTER_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
    "temperature": temperature,
    "top_p": 0.9,
    "num_predict": 256,    # ← Réduit à 256 pour réponses plus rapides
},
            },
        )
        response.raise_for_status()
        return response.json()["response"].strip()


# ── Prompts par mode de recherche ─────────────────────────────

SYSTEM_PROMPT_SQL = """Tu es un assistant juridique expert du système de gestion de dossiers ANP Legal.
Tu réponds aux questions basées sur des données structurées de la base de données.
Réponds en français de manière claire, concise et professionnelle.
Si des montants sont présents, précise toujours la devise (MAD).
Ne fabrique JAMAIS d'informations. Utilise UNIQUEMENT les données fournies."""

SYSTEM_PROMPT_VECTOR = """Tu es un assistant juridique expert en droit marocain.
Tu réponds aux questions basées sur le contenu des documents juridiques fournis.
Cite toujours la source (nom du document) quand tu extrais une information.
Réponds en français de manière précise et professionnelle.
Si tu ne trouves pas l'information dans les documents fournis, dis-le clairement."""

SYSTEM_PROMPT_HYBRID = """Tu es un assistant juridique expert du système ANP Legal.
Tu as accès à la fois aux données structurées (dossiers, dates, montants) et au contenu des documents.
Combine les deux sources pour donner une réponse complète et précise.
Distingue clairement ce qui vient de la base de données vs des documents.
Réponds en français de manière professionnelle."""


async def generate_sql_answer(
    question: str,
    sql_context: str,
    query_key: str,
) -> dict:
    try:
        prompt = f"""Voici les données extraites de la base de données.

DONNÉES :
{sql_context}

QUESTION : {question}

Réponds de manière claire et professionnelle.
RÉPONSE :"""

        answer = await call_ollama(prompt, system_prompt=SYSTEM_PROMPT_SQL)
        return {
            "answer": answer,
            "confidence": 0.90,
            "sources": [{"type": "database", "query": query_key}],
        }
    except Exception as e:
        logger.error("llm_sql_error", error=str(e))
        # Retourner les données brutes si LLM échoue
        return {
            "answer": sql_context,
            "confidence": 0.70,
            "sources": [{"type": "database", "query": query_key}],
        }

async def generate_vector_answer(
    question: str,
    chunks: list[dict],
) -> dict:
    """
    Génère une réponse basée sur des chunks de documents.

    Returns:
        {answer, confidence, sources}
    """
    if not chunks:
        return {
            "answer": "Je n'ai pas trouvé de documents pertinents pour répondre à cette question.",
            "confidence": 0.10,
            "sources": [],
        }

    # Construire le contexte depuis les chunks
    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Document {i}: {chunk.get('nom_fichier', 'N/A')} | "
            f"Dossier: {chunk.get('numero_dossier', 'N/A')}]\n"
            f"{chunk['text']}"
        )
        if chunk.get("document_id"):
            sources.append({
                "type": "document",
                "document_id": chunk["document_id"],
                "nom_fichier": chunk.get("nom_fichier", ""),
                "dossier": chunk.get("numero_dossier", ""),
                "score": chunk.get("score", 0),
                "type_document": chunk.get("document_type", ""),
            })

    context = "\n\n---\n\n".join(context_parts)

    # Score de confiance basé sur le meilleur score Qdrant
    best_score = max((c.get("score", 0) for c in chunks), default=0)
    confidence = round(min(0.95, best_score * 1.1), 2)

    prompt = f"""Voici des extraits de documents juridiques pertinents.

DOCUMENTS :
{context}

QUESTION : {question}

Réponds en te basant UNIQUEMENT sur ces documents. Cite le document source quand pertinent.
RÉPONSE :"""

    try:
        answer = await call_ollama(prompt, system_prompt=SYSTEM_PROMPT_VECTOR)
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
        }
    except Exception as e:
        logger.error("llm_vector_error", error=str(e))
        return {
            "answer": f"Erreur lors de la génération : {str(e)}",
            "confidence": 0.0,
            "sources": sources,
        }


async def generate_hybrid_answer(
    question: str,
    sql_context: str,
    chunks: list[dict],
) -> dict:
    """
    Génère une réponse combinant SQL et recherche vectorielle.

    Returns:
        {answer, confidence, sources}
    """
    sources = []

    # Sources SQL
    if sql_context:
        sources.append({"type": "database"})

    # Sources documents
    for chunk in chunks:
        if chunk.get("document_id"):
            sources.append({
                "type": "document",
                "document_id": chunk["document_id"],
                "nom_fichier": chunk.get("nom_fichier", ""),
                "score": chunk.get("score", 0),
            })

    # Contexte combiné
    doc_context = "\n\n".join(
        f"[{chunk.get('nom_fichier', 'Doc')}]: {chunk['text']}"
        for chunk in chunks
    )

    best_score = max((c.get("score", 0) for c in chunks), default=0)
    confidence = round(min(0.92, 0.7 + best_score * 0.25), 2)

    prompt = f"""Tu dois répondre à la question en utilisant ces deux sources :

=== DONNÉES STRUCTURÉES (base de données) ===
{sql_context or "Aucune donnée structurée disponible."}

=== CONTENU DES DOCUMENTS ===
{doc_context or "Aucun document trouvé."}

QUESTION : {question}

Combine intelligemment les deux sources pour une réponse complète.
RÉPONSE :"""

    try:
        answer = await call_ollama(prompt, system_prompt=SYSTEM_PROMPT_HYBRID)
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
        }
    except Exception as e:
        logger.error("llm_hybrid_error", error=str(e))
        return {
            "answer": f"Erreur LLM : {str(e)}",
            "confidence": 0.0,
            "sources": sources,
        }