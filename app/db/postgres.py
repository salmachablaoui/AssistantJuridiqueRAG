# ============================================================
# app/db/postgres.py
# Connexion async au PostgreSQL existant (schéma Laravel intact)
# Utilise asyncpg via SQLAlchemy pour les performances
# ============================================================

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import structlog
from app.config import settings

logger = structlog.get_logger()

# ── Moteur async ──────────────────────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,           # Vérifie la connexion avant utilisation
    pool_recycle=3600,            # Recycle les connexions après 1h
    echo=settings.DEBUG,
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db():
    """Dependency injection pour FastAPI"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def execute_query(sql: str, params: dict = None) -> list[dict]:
    """
    Exécute une requête SQL brute et retourne une liste de dicts.
    Utilisé par sql_service pour les requêtes dynamiques.
    """
    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(text(sql), params or {})
            rows = result.fetchall()
            columns = result.keys()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error("db_query_error", sql=sql[:100], error=str(e))
            raise


async def check_connection() -> bool:
    """Vérifie que la base est accessible — utilisé par /health"""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error("db_connection_failed", error=str(e))
        return False