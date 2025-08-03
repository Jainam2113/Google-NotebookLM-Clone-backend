"""
Pydantic models for request/response schemas
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class MessageType(str, Enum):
    """Message types for chat"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"


class CitationType(str, Enum):
    """Citation types"""
    PAGE_REFERENCE = "page_reference"
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"


# ===== Base Models =====

class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error: str
    error_type: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ===== Upload Models =====

class UploadRequest(BaseModel):
    """PDF upload request (handled by FastAPI's File upload)"""
    pass


class UploadResponse(BaseResponse):
    """PDF upload response"""
    session_id: str
    filename: str
    file_size: int
    total_pages: int
    chunks_created: int
    processing_time: float


# ===== Chat Models =====

class ChatRequest(BaseModel):
    """Chat message request"""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1)
    context: Optional[str] = None

    @validator('message')
    def validate_message(cls, v):
        """Validate message content"""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class Citation(BaseModel):
    """Citation model"""
    page: int = Field(..., ge=1)
    text: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    type: CitationType = CitationType.PAGE_REFERENCE
    section: Optional[str] = None


class TokenUsage(BaseModel):
    """Token usage statistics"""
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)

    @validator('total_tokens')
    def validate_total(cls, v, values):
        """Ensure total equals input + output"""
        input_tokens = values.get('input_tokens', 0)
        output_tokens = values.get('output_tokens', 0)
        expected_total = input_tokens + output_tokens
        return expected_total


class ChatResponse(BaseResponse):
    """Chat response"""
    response: str
    citations: List[Citation] = []
    token_usage: Optional[TokenUsage] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time: float = 0.0


# ===== Message Models =====

class ChatMessage(BaseModel):
    """Individual chat message"""
    id: Optional[str] = None
    type: MessageType
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    citations: List[Citation] = []
    token_usage: Optional[TokenUsage] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatHistory(BaseModel):
    """Chat history for a session"""
    session_id: str
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ===== Session Models =====

class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    filename: Optional[str] = None
    file_size: Optional[int] = None
    total_pages: Optional[int] = None
    chunks_count: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    is_active: bool = True


class SessionResponse(BaseResponse):
    """Session response"""
    session_info: SessionInfo


# ===== PDF Models =====

class PDFChunk(BaseModel):
    """PDF chunk model"""
    id: Optional[str] = None
    content: str
    page: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class PDFDocument(BaseModel):
    """PDF document model"""
    filename: str
    file_size: int
    total_pages: int
    chunks: List[PDFChunk] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


# ===== Search Models =====

class SearchResult(BaseModel):
    """Search result model"""
    content: str
    page: int
    chunk_index: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Search request"""
    query: str = Field(..., min_length=1, max_length=500)
    session_id: str
    max_results: int = Field(5, ge=1, le=20)
    threshold: float = Field(0.7, ge=0.0, le=1.0)


class SearchResponse(BaseResponse):
    """Search response"""
    results: List[SearchResult] = []
    query: str
    total_results: int = 0
    processing_time: float = 0.0


# ===== Health Models =====

class ServiceStatus(BaseModel):
    """Individual service status"""
    name: str
    status: str  # "healthy", "unhealthy", "unknown"
    details: Optional[Dict[str, Any]] = None
    last_check: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseResponse):
    """Health check response"""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    environment: str
    services: Dict[str, ServiceStatus] = {}
    uptime: Optional[float] = None


# ===== Statistics Models =====

class SessionStats(BaseModel):
    """Session statistics"""
    total_messages: int = 0
    user_messages: int = 0
    ai_messages: int = 0
    total_tokens: int = 0
    average_response_time: float = 0.0
    first_message: Optional[datetime] = None
    last_message: Optional[datetime] = None


class SystemStats(BaseModel):
    """System-wide statistics"""
    total_sessions: int = 0
    active_sessions: int = 0
    total_files_processed: int = 0
    total_messages: int = 0
    total_tokens_used: int = 0
    average_file_size: float = 0.0
    uptime: float = 0.0


# ===== Configuration Models =====

class APIConfig(BaseModel):
    """API configuration that can be shared with frontend"""
    max_file_size: int
    allowed_file_types: List[str]
    max_message_length: int = 2000
    session_timeout: int
    features: Dict[str, bool] = {}


# ===== Utility Functions =====

def create_error_response(error: str, error_type: str = None, details: Dict = None) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(
        error=error,
        error_type=error_type,
        details=details
    )


def create_success_response(message: str = None, **kwargs) -> BaseResponse:
    """Create standardized success response"""
    return BaseResponse(message=message, **kwargs)


# ===== Response Examples for Documentation =====

class ResponseExamples:
    """Response examples for API documentation"""

    UPLOAD_SUCCESS = {
        "example": {
            "success": True,
            "message": "PDF uploaded and processed successfully",
            "session_id": "abc123",
            "filename": "document.pdf",
            "file_size": 1048576,
            "total_pages": 10,
            "chunks_created": 25,
            "processing_time": 2.5,
            "timestamp": "2023-12-01T12:00:00Z"
        }
    }

    CHAT_SUCCESS = {
        "example": {
            "success": True,
            "response": "Based on the document, the main points are...",
            "citations": [
                {
                    "page": 1,
                    "text": "relevant excerpt",
                    "confidence": 0.95,
                    "type": "page_reference"
                }
            ],
            "token_usage": {
                "input_tokens": 150,
                "output_tokens": 75,
                "total_tokens": 225
            },
            "processing_time": 1.2,
            "timestamp": "2023-12-01T12:00:00Z"
        }
    }

    ERROR_RESPONSE = {
        "example": {
            "success": False,
            "error": "File too large",
            "error_type": "validation_error",
            "details": {"max_size": 10485760, "received_size": 20971520},
            "timestamp": "2023-12-01T12:00:00Z"
        }
    }