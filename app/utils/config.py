"""
Simple configuration settings
"""
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Debug: Print all environment variables that start with GOOGLE or GEMINI
print("🔍 DEBUG: Environment variables:")
for key, value in os.environ.items():
    if 'GOOGLE' in key or 'GEMINI' in key or 'API' in key:
        print(f"  {key} = {value[:10]}..." if value else f"  {key} = (empty)")

class Settings:
    """Simple settings class"""

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: List[str] = [os.getenv("CORS_ORIGINS", "http://localhost:5173")]

    # Google Gemini API - Try both variable names
    GEMINI_API_KEY: str = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

    # Debug: Print what we found
    print(f"🔍 DEBUG: GEMINI_API_KEY = {GEMINI_API_KEY[:10]}..." if GEMINI_API_KEY else "🔍 DEBUG: No API key found!")

    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
    ENABLE_REDIS: bool = os.getenv("ENABLE_REDIS", "false").lower() == "true"

    # File Upload Settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB
    ALLOWED_FILE_TYPES: List[str] = ["application/pdf"]
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    # PDF Processing
    PDF_CHUNK_SIZE: int = int(os.getenv("PDF_CHUNK_SIZE", "1000"))
    PDF_CHUNK_OVERLAP: int = int(os.getenv("PDF_CHUNK_OVERLAP", "200"))

    # Vector Search
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "768"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

    # Session Management
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour
    MAX_SESSIONS: int = int(os.getenv("MAX_SESSIONS", "100"))

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Create global settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)