import requests
import numpy as np
import re
import json
import os
from datetime import datetime

# Configuration
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

# ============================================================
# 1. DECOUPAGE INTELLIGENT AVEC METADONNEES
# ============================================================
def chunk_with_metadata(text, document_name="document", chunk_size=600, overlap=100):
    """
    Découpage qui garde la trace des sections et métadonnées
    """
    chunks = []
    
    # Identifier les sections (mots en MAJUSCULES)
    lines = text.split('\n')
    current_section = "General"
    current_chunk = ""
    chunk_index = 0
    
    for line in lines:
        # Detection de nouvelle section (ex: "ARTICLE 1 :", "1. IDENTIFICATION")
        if re.match(r'^(\d+\.|\w+\s+\d+:|[A-Z]{3,}\s)', line.strip()):
            # Sauvegarder l'ancien chunk s'il existe
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "section": current_section,
                    "chunk_id": chunk_index,
                    "document": document_name,
                    "keywords": extract_keywords(current_chunk)
                })
                chunk_index += 1
                current_chunk = ""
            current_section = line.strip()[:50]
            continue
        
        # Ajouter la ligne au chunk courant
        if len(current_chunk) + len(line) < chunk_size:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "section": current_section,
                    "chunk_id": chunk_index,
                    "document": document_name,
                    "keywords": extract_keywords(current_chunk)
                })
                chunk_index += 1
                # Garder un chevauchement pour le contexte
                last_sentences = " ".join(current_chunk.split('.')[-2:]) if current_chunk else ""
                current_chunk = last_sentences + " " + line + "\n"
            else:
                current_chunk = line + "\n"
    
    # Dernier chunk
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "section": current_section,
            "chunk_id": chunk_index,
            "document": document_name,
            "keywords": extract_keywords(current_chunk)
        })
    
    return chunks

def extract_keywords(text):
    """Extraire les mots-clés importants d'un chunk"""
    # Mots à rechercher
    important_words = [
        'dossier', 'numero', 'client', 'avocat', 'honoraire', 'statut',
        'facture', 'echeance', 'tribunal', 'audience', 'contrat',
        'salaire', 'periode', 'confidentialite', 'non-concurrence'
    ]
    
    text_lower = text.lower()
    found = [word for word in important_words if word in text_lower]
    return found

# ============================================================
# 2. GENERATION D'EMBEDDINGS OPTIMISEE
# ============================================================
def generate_embedding(text, max_length=1500):
    """Générer un embedding avec troncature intelligente"""
    if len(text) > max_length:
        # Prendre le début et la fin (contexte important)
        text = text[:max_length//2] + " " + text[-max_length//2:]
    
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=30
    )
    return response.json()["embedding"]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ============================================================
# 3. RECHERCHE AVANCEE AVEC SCORING MULTI-CRITERES
# ============================================================
def search_advanced(query, chunks_data, chunk_embeddings, top_k=5):
    """
    Recherche combinant:
    - Similarité cosinus
    - Correspondance de mots-clés
    - Score de section
    """
    query_embedding = generate_embedding(query)
    query_lower = query.lower()
    
    scores = []
    for i, chunk_info in enumerate(chunks_data):
        # Score 1: Similarité sémantique
        semantic_sim = cosine_similarity(query_embedding, chunk_embeddings[i])
        
        # Score 2: Mots-clés (bonus)
        keyword_bonus = 0
        query_words = query_lower.split()
        for kw in chunk_info["keywords"]:
            if kw in query_lower:
                keyword_bonus += 0.15
            for qw in query_words:
                if qw in kw or kw in qw:
                    keyword_bonus += 0.1
        
        # Score 3: Section pertinente (bonus pour certaines sections)
        section_bonus = 0
        important_sections = ['honoraires', 'avocat', 'client', 'dossier', 'statut']
        for sec in important_sections:
            if sec in chunk_info["section"].lower():
                section_bonus += 0.1
        
        # Score final
        final_score = semantic_sim * 0.7 + keyword_bonus * 0.2 + section_bonus * 0.1
        scores.append((final_score, i))
    
    scores.sort(reverse=True)
    
    results = []
    for score, idx in scores[:top_k]:
        if score >= 0.2:  # Seuil plus bas
            results.append({
                "chunk": chunks_data[idx],
                "score": score,
                "semantic": semantic_sim
            })
    
    return results

# ============================================================
# 4. GENERATION DE REPONSE AVEC CONTEXTE RICHE
# ============================================================
def ask_with_context(question, chunks_data, chunk_embeddings):
    """Répondre à une question avec le contexte enrichi"""
    
    print(f"\n🔍 Recherche pour: '{question}'")
    
    # Recherche avancée
    results = search_advanced(question, chunks_data, chunk_embeddings)
    
    if not results:
        return "❌ Je n'ai pas trouvé d'information pertinente."
    
    print(f"📊 Top résultats:")
    for r in results:
        print(f"   - Score: {r['score']:.3f} | Section: {r['chunk']['section'][:40]}")
    
    # Construction du contexte
    context_parts = []
    for r in results[:3]:
        context_parts.append(f"[Section: {r['chunk']['section']}]\n{r['chunk']['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Prompt amélioré
    prompt = f"""Tu es un assistant juridique expert. Réponds PRÉCISÉMENT à la question.

📌 RÈGLES IMPORTANTES:
1. Utilise UNIQUEMENT le contexte fourni
2. Si l'information n'est pas dans le contexte, dis "Non trouvé dans le document"
3. Cite la section d'où vient l'information
4. Sois concis mais complet

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE STRUCTURÉE:"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Très bas pour être précis
                "top_k": 40
            }
        },
        timeout=60
    )
    
    return response.json()["response"], results

# ============================================================
# 5. GESTION MULTI-DOCUMENTS
# ============================================================
class MultiDocumentRAG:
    def __init__(self):
        self.documents = {}  # {nom: {"chunks": [], "embeddings": []}}
    
    def add_document(self, file_path, doc_name=None):
        """Ajouter un document à l'index"""
        if doc_name is None:
            doc_name = os.path.basename(file_path)
        
        print(f"📄 Ajout de '{doc_name}'...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Chunking avec métadonnées
        chunks = chunk_with_metadata(text, doc_name)
        print(f"   → {len(chunks)} chunks créés")
        
        # Génération des embeddings
        embeddings = []
        for chunk in chunks:
            emb = generate_embedding(chunk["text"])
            embeddings.append(emb)
        
        self.documents[doc_name] = {
            "chunks": chunks,
            "embeddings": embeddings
        }
        
        print(f"   ✅ Document '{doc_name}' indexé")
    
    def search_all(self, query, top_k=5):
        """Rechercher dans tous les documents"""
        all_results = []
        
        for doc_name, doc_data in self.documents.items():
            results = search_advanced(query, doc_data["chunks"], doc_data["embeddings"], top_k)
            for r in results:
                r["chunk"]["document"] = doc_name
                all_results.append(r)
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
    
    def ask_question(self, question):
        """Poser une question sur tous les documents"""
        results = self.search_all(question)
        
        if not results:
            return "❌ Aucune information pertinente trouvée dans les documents."
        
        print(f"\n📊 Résultats de recherche:")
        for i, r in enumerate(results):
            print(f"   {i+1}. [{r['chunk']['document']}] Score: {r['score']:.3f} - Section: {r['chunk']['section'][:40]}")
        
        # Contexte enrichi avec source du document
        context_parts = []
        for r in results[:3]:
            context_parts.append(f"[Document: {r['chunk']['document']} | Section: {r['chunk']['section']}]\n{r['chunk']['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""Tu es un assistant juridique. Réponds à la question en citant les sources.

CONTEXTE (avec sources):
{context}

QUESTION: {question}

RÉPONSE (en mentionnant le document source):"""

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=60
        )
        
        answer = response.json()["response"]
        
        # Ajouter les sources
        sources_text = "\n\n📚 SOURCES:\n"
        for r in results[:3]:
            sources_text += f"   - Document: {r['chunk']['document']}\n"
        
        return answer + sources_text

# ============================================================
# 6. INTERFACE PRINCIPALE
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("   RAG PRO - Multi-Documents")
    print("="*60)
    
    # Vérifier Ollama
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        print("✅ Ollama prêt\n")
    except:
        print("❌ Ollama non démarré!")
        print("   Lance: ollama serve dans un autre terminal")
        exit(1)
    
    # Initialiser le gestionnaire multi-documents
    rag = MultiDocumentRAG()
    
    # Ajouter le document
    rag.add_document("document.txt", "Dossier_Juridique_2024")
    
    print("\n" + "="*60)
    print("📌 DOCUMENT INDEXÉ!")
    print("="*60)
    
    while True:
        print("\n" + "-"*60)
        question = input("❓ Votre question (ou 'quit'): ")
        
        if question.lower() == 'quit':
            print("👋 Au revoir!")
            break
        
        if not question.strip():
            continue
        
        try:
            answer = rag.ask_question(question)
            print("\n" + "="*60)
            print("📝 RÉPONSE:")
            print("="*60)
            print(answer)
            print()
            
        except Exception as e:
            print(f"\n❌ Erreur: {e}")