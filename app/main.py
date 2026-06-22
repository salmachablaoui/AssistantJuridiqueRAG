# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.database import engine, Base
from app.routes import upload_routes, chat_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Failed to create tables: {e}")

app = FastAPI(title="RAG Local Assistant", version="1.0", docs_url="/docs")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_routes.router, prefix="/api", tags=["Upload"])
app.include_router(chat_routes.router, prefix="/api", tags=["Chat"])

@app.get("/")
async def root():
    return {
        "message": "RAG Local Assistant API",
        "status": "running",
        "endpoints": {
            "upload": "POST /api/upload",
            "chat": "POST /api/chat",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}