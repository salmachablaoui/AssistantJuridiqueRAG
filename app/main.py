# ============================================================
# main.py — Point d'entrée FastAPI
# Lance : uvicorn main:app --host 0.0.0.0 --port 8000
# ============================================================

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialisation au démarrage, nettoyage à l'arrêt.
    Remplace les anciens @app.on_event("startup").
    """
    logger.info("app_starting", debug=settings.DEBUG)

    # ── 1. Connexion PostgreSQL ───────────────────────────────
    from app.db.postgres import check_connection
    pg_ok = await check_connection()
    if pg_ok:
        logger.info("postgresql_connected", db=settings.POSTGRES_DB)
    else:
        logger.error("postgresql_connection_failed")

    # ── 2. Initialiser Qdrant ─────────────────────────────────
    from app.services.qdrant_service import ensure_collection_exists
    try:
        ensure_collection_exists()
        logger.info("qdrant_ready", collection=settings.QDRANT_COLLECTION)
    except Exception as e:
        logger.error("qdrant_init_failed", error=str(e))

    # ── 3. Préchargement du modèle d'embedding ────────────────
    # Charge le modèle en mémoire une seule fois au démarrage
    # Évite la latence sur la première requête utilisateur
    try:
        from app.services.embedding_service import _load_model
        _load_model()
        logger.info("embedding_model_preloaded", model=settings.EMBEDDING_MODEL)
    except Exception as e:
        logger.warning("embedding_model_preload_failed", error=str(e))

    logger.info("app_ready")
    yield

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("app_shutting_down")
    from app.db.postgres import engine
    await engine.dispose()
    logger.info("database_connections_closed")


# ── Application FastAPI ───────────────────────────────────────

app = FastAPI(
    title="ANP Legal RAG API",
    description=(
        "Hybrid Legal RAG System — Combines PostgreSQL structured queries "
        "with semantic vector search for intelligent legal document assistance."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,   # Désactiver en prod
    redoc_url="/redoc" if settings.DEBUG else None,
)

# ── CORS — Autoriser Laravel ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "http://127.0.0.1",
        "http://127.0.0.1:8080",
        # Ajouter le domaine Laravel en production
        # "https://your-laravel-app.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────
from app.api.routes import router
app.include_router(router, prefix="")


# ── Route racine ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "ANP Legal RAG API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.DEBUG,
        log_level="info",
    )