# ============================================================
# app/services/chunk_service.py
# REFACTORÉ : Chunking sémantique via LangChain
#
# Problème de l'ancien code :
#   chunk = text[start:end]  # ← coupe en plein milieu d'une phrase légale
#
# Nouveau comportement :
#   - Découpe sur les fins de phrases (., ?, !)
#   - Respecte les structures de paragraphes légaux
#   - Maintient overlap cohérent entre chunks
#   - Filtre les chunks trop courts (numéros de page, en-têtes)
# ============================================================

import re
import structlog
from typing import Optional
from app.config import settings

logger = structlog.get_logger()


def _clean_legal_text(text: str) -> str:
    """
    Nettoie le texte extrait d'un PDF juridique marocain.
    Conservé compatible avec l'OCR existant (EasyOCR).
    """
    # Supprimer les répétitions de numéros de page
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Normaliser les espaces multiples
    text = re.sub(r' {3,}', '  ', text)
    # Normaliser les sauts de ligne
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Supprimer les lignes vides en début/fin
    text = text.strip()
    return text


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
    document_id: Optional[int] = None,
    document_type: Optional[str] = None,
) -> list[dict]:
    """
    Découpe sémantique d'un texte en chunks exploitables pour le RAG.

    REMPLACE l'ancienne implémentation :
        chunk = text[start:end]  # découpe arbitraire

    Retourne une liste de dicts avec métadonnées pour Qdrant.

    Args:
        text: Texte brut extrait du PDF
        chunk_size: Taille cible en caractères (défaut: config)
        overlap: Chevauchement en caractères (défaut: config)
        document_id: ID du document dans la table `documents`
        document_type: type_document de la table `documents`

    Returns:
        Liste de dicts {text, chunk_index, document_id, ...}
    """
    if not text or not text.strip():
        return []

    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    # Nettoyage préalable
    text = _clean_legal_text(text)

    try:
        # Chunking sémantique via LangChain
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Séparateurs adaptés aux documents juridiques français/arabe
        separators = [
            "\n\n",          # Paragraphes
            "\n",            # Lignes
            ". ",            # Fin de phrase française
            ".\n",
            "؟",             # Point d'interrogation arabe
            "،",             # Virgule arabe
            " ",             # Mots
            "",              # Caractères
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

        raw_chunks = splitter.split_text(text)

    except ImportError:
        # Fallback si langchain non installé — version améliorée de l'ancien code
        logger.warning("langchain_not_found_using_sentence_splitter")
        raw_chunks = _sentence_aware_chunking(text, chunk_size, overlap)

    # Filtrer et enrichir avec métadonnées
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()

        # Ignorer les chunks trop courts (numéros de page, titres seuls)
        if len(chunk_text) < settings.MIN_CHUNK_CHARS:
            continue

        chunk = {
            "text": chunk_text,
            "chunk_index": i,
            "document_id": document_id,
            "document_type": document_type or "unknown",
            "char_count": len(chunk_text),
        }
        chunks.append(chunk)

    logger.info(
        "chunking_complete",
        n_chunks=len(chunks),
        document_id=document_id,
        total_chars=len(text),
    )
    return chunks


def _sentence_aware_chunking(
    text: str, chunk_size: int, overlap: int
) -> list[str]:
    """
    Fallback : chunking qui respecte les fins de phrases.
    Amélioration de l'ancien code qui découpait arbitrairement.
    """
    # Découper en phrases
    sentences = re.split(r'(?<=[.!?؟])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current)
                # Overlap : réintégrer la dernière partie
                words = current.split()
                overlap_words = words[-max(1, overlap // 10):]
                current = " ".join(overlap_words) + " " + sentence
            else:
                # Phrase unique plus longue que chunk_size
                chunks.append(sentence[:chunk_size])
                current = sentence[chunk_size - overlap:]

    if current:
        chunks.append(current)

    return chunks