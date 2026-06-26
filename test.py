# debug_index.py — à lancer depuis C:\Users\salma\rag-project\
import asyncio, os, sys
sys.path.insert(0, ".")

async def main():
    from app.config import settings
    
    # 1. Vérifier les chemins
    for doc_id, chemin in [(10, "documents/4/1781513733_10_Requete.pdf"),
                            (9,  "documents/4/1781606107_9_Jugement.pdf")]:
        full = os.path.join(settings.PDF_STORAGE_PATH, chemin)
        exists = os.path.exists(full)
        size = os.path.getsize(full) if exists else 0
        print(f"  doc_id={doc_id}: {'✓' if exists else '✗'} {full} ({size} bytes)")
    
    # 2. Test Qdrant
    from app.services.qdrant_service import get_qdrant_client, get_collection_stats
    client = get_qdrant_client()
    print(f"\n  Qdrant collections: {[c.name for c in client.get_collections().collections]}")
    print(f"  Stats: {get_collection_stats()}")
    
    # 3. Test extraction PDF
    full = os.path.join(settings.PDF_STORAGE_PATH, "documents/4/1781513733_10_Requete.pdf")
    if os.path.exists(full):
        from app.services.pdf_service import get_text_with_ocr_fallback
        text = get_text_with_ocr_fallback(full)
        print(f"\n  Texte extrait ({len(text)} chars) : {text[:200]!r}")

asyncio.run(main())