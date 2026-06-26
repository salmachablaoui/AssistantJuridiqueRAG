import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_starting")

    from app.db.postgres import check_connection
    pg_ok = await check_connection()
    if pg_ok:
        logger.info("postgresql_connected")
    else:
        logger.error("postgresql_connection_failed")

    from app.services.qdrant_service import ensure_collection_exists
    try:
        ensure_collection_exists()
        logger.info("qdrant_ready")
    except Exception as e:
        logger.error("qdrant_init_failed", error=str(e))

    try:
        from app.services.embedding_service import _load_model
        _load_model()
        logger.info("embedding_model_preloaded")
    except Exception as e:
        logger.warning("embedding_model_preload_failed", error=str(e))

    logger.info("app_ready")
    yield

    logger.info("app_shutting_down")
    from app.db.postgres import engine
    await engine.dispose()


app = FastAPI(
    title="ANP Legal RAG API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

from app.api.routes import router
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"name": "ANP Legal RAG API", "version": "2.0.0", "status": "running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)