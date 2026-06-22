import requests

from app.services.pdf_service import get_text_with_ocr_fallback
from app.services.chunk_service import chunk_text
from app.services.embedding_service import get_embeddings
from app.services.vector_service import VectorService

# =========================================
# PDF PATH
# =========================================

pdf_path = "modèle de jugement.pdf"

# =========================================
# EXTRACTION TEXTE
# =========================================

print("📄 Extraction du texte...")

text = get_text_with_ocr_fallback(pdf_path)

print("\n✅ TEXTE EXTRAIT (preview):\n")
print(text[:1000])

# =========================================
# CHUNKING
# =========================================

chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=100
)

print(f"\n✂️ Nombre de chunks : {len(chunks)}")

# =========================================
# EMBEDDINGS
# =========================================

embeddings = get_embeddings(chunks)

print("\n🧠 Embeddings générés")

# =========================================
# VECTOR STORE (FAISS)
# =========================================

vs = VectorService(dim=384)
vs.add(embeddings, chunks)

print("\n📦 Base vectorielle prête")

# =========================================
# CHAT LOOP
# =========================================

while True:

    query = input("\n❓ Pose une question ('exit' pour quitter) : ")

    if query.lower() == "exit":
        break

    # =====================================
    # CAS RÉSUMÉ
    # =====================================

    if "résumé" in query.lower() or "resume" in query.lower():

        prompt = f"""
Tu es un assistant juridique professionnel.

Fais un résumé clair, structuré et précis du document suivant :

{text[:4000]}

Réponds directement sans poser de question.
"""

    # =====================================
    # CAS RAG
    # =====================================

    else:

        query_emb = get_embeddings([query])[0]

        results = vs.search(query_emb, k=3)

        context = "\n".join(results)

        prompt = f"""
Tu es un assistant juridique intelligent.

Réponds directement à la question en utilisant le contexte fourni.
Ne demande jamais à l'utilisateur de poser une autre question.

QUESTION:
{query}

CONTEXTE:
{context}

RÉPONSE:
"""

    # =====================================
    # APPEL OLLAMA
    # =====================================

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            },
            timeout=120
        )

        data = response.json()

        # DEBUG (très important)
        print("\nDEBUG OLLAMA:", data)

        # gestion erreurs
        if "error" in data:
            print("❌ Erreur Ollama:", data["error"])
            continue

        if "message" not in data:
            print("❌ Format inattendu:", data)
            continue

        answer = data["message"].get("content", "").strip()

        if not answer:
            print("❌ Réponse vide du modèle")
        else:
            print("\n🤖 RÉPONSE :\n")
            print(answer)

    except Exception as e:
        print("❌ Erreur requête Ollama:", str(e))