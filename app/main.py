"""
FastAPI main application for NotebookLM Clone
"""
import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.utils.config import settings
from app.routes import upload, chat
from app.services.vector_service import VectorService
from app.services.gemini_service import GeminiService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global services
vector_service = VectorService()
gemini_service = GeminiService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting NotebookLM Clone API...")

    # Initialize services
    try:
        await gemini_service.initialize()
        logger.info("✅ Gemini service initialized")

        await vector_service.initialize()
        logger.info("✅ Vector service initialized")

        # Store services in app state
        app.state.vector_service = vector_service
        app.state.gemini_service = gemini_service

        logger.info("🎉 All services initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

    yield

    # Cleanup
    logger.info("🛑 Shutting down services...")
    await vector_service.cleanup()
    logger.info("✅ Services cleaned up")


# Create FastAPI app
app = FastAPI(
    title="NotebookLM Clone API",
    description="AI-powered PDF chat application using Google Gemini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Custom middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()

    # Log request
    logger.info(f"📥 {request.method} {request.url.path}")

    response = await call_next(request)

    # Log response time
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")

    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_error"}
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors"""
    logger.error(f"Value error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "value_error"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        # Check Gemini service
        gemini_status = await app.state.gemini_service.health_check()

        # Check vector service
        vector_status = app.state.vector_service.health_check()

        return {
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "gemini": gemini_status,
                "vector": vector_status,
            },
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NotebookLM Clone API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


# Additional endpoints
@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    try:
        session_info = await app.state.vector_service.get_session_info(session_id)
        return {
            "session_id": session_id,
            "valid": session_info is not None,
            "info": session_info
        }
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session info")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data"""
    try:
        await app.state.vector_service.delete_session(session_id)
        return {"message": "Session deleted successfully", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


# Development server
if __name__ == "__main__":
    import time

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )