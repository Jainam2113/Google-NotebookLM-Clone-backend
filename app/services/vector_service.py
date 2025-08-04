"""
Vector service for embedding storage and similarity search using FAISS
"""
import logging
import asyncio
import json
import pickle
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import numpy as np
import faiss
from redis.asyncio import Redis
from concurrent.futures import ThreadPoolExecutor

from app.models.schemas import (
    PDFDocument,
    PDFChunk,
    SearchResult,
    ChatMessage,
    SessionInfo
)
from app.utils.config import settings

logger = logging.getLogger(__name__)


class VectorService:
    """Service for vector storage and similarity search"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.redis_client: Optional[aioredis.Redis] = None

        # In-memory storage for vectors and metadata
        self.sessions: Dict[str, Dict] = {}
        self.vector_indices: Dict[str, faiss.IndexFlatIP] = {}  # Inner product for similarity
        self.chunk_metadata: Dict[str, List[Dict]] = {}

        # Vector dimension (will be set when first embedding is received)
        self.vector_dimension = settings.VECTOR_DIMENSION

    async def initialize(self):
        """Initialize the vector service"""
        logger.info("🔧 Initializing Vector Service...")

        # Initialize Redis if enabled
        if settings.ENABLE_REDIS:
            try:
                self.redis_client = aioredis.from_url(
                    settings.REDIS_URL,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    max_connections=settings.REDIS_MAX_CONNECTIONS,
                    decode_responses=True
                )

                # Test connection
                await self.redis_client.ping()
                logger.info("✅ Redis connected")

            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                logger.info("📝 Continuing with in-memory storage only")
                self.redis_client = None

        logger.info("✅ Vector Service initialized")

    async def store_document(self, session_id: str, document: PDFDocument):
        """Store document chunks and create vector index"""
        logger.info(f"💾 Storing document for session: {session_id}")

        if not document.chunks:
            raise ValueError("Document has no chunks to store")

        # Collect embeddings and metadata
        embeddings = []
        metadata = []

        for chunk in document.chunks:
            if not chunk.embedding:
                raise ValueError(f"Chunk {chunk.id} has no embedding")

            embeddings.append(chunk.embedding)
            metadata.append({
                "chunk_id": chunk.id,
                "content": chunk.content,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata or {}
            })

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Update vector dimension if needed
        if embeddings_array.shape[1] != self.vector_dimension:
            self.vector_dimension = embeddings_array.shape[1]
            logger.info(f"📏 Updated vector dimension to {self.vector_dimension}")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Create FAISS index
        index = faiss.IndexFlatIP(self.vector_dimension)  # Inner product index
        index.add(embeddings_array)

        # Store in memory
        self.vector_indices[session_id] = index
        self.chunk_metadata[session_id] = metadata

        # Store session info
        self.sessions[session_id] = {
            "document": {
                "filename": document.filename,
                "file_size": document.file_size,
                "total_pages": document.total_pages,
                "chunks_count": len(document.chunks),
                "created_at": datetime.utcnow().isoformat(),
            },
            "chat_history": [],
            "last_activity": datetime.utcnow().isoformat(),
        }

        # Optionally store in Redis for persistence
        if self.redis_client:
            try:
                await self._store_in_redis(session_id, embeddings_array, metadata, document)
            except Exception as e:
                logger.warning(f"Failed to store in Redis: {e}")

        logger.info(f"✅ Stored {len(embeddings)} chunks for session {session_id}")

    async def search_similar(
            self,
            session_id: str,
            query: str,
            query_embedding: List[float] = None,
            max_results: int = 5,
            threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search for similar chunks"""
        if session_id not in self.vector_indices:
            logger.warning(f"Session {session_id} not found")
            return []

        if not query_embedding:
            # This should be provided by the caller (gemini_service)
            raise ValueError("Query embedding is required")

        # Convert to numpy array and normalize
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # Search in FAISS index
        index = self.vector_indices[session_id]
        metadata = self.chunk_metadata[session_id]

        # Search for similar vectors
        scores, indices = index.search(query_vector, min(max_results * 2, len(metadata)))

        # Filter results by threshold and create SearchResult objects
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(metadata):
                chunk_meta = metadata[idx]
                result = SearchResult(
                    content=chunk_meta["content"],
                    page=chunk_meta["page"],
                    chunk_index=chunk_meta["chunk_index"],
                    similarity_score=float(score),
                    metadata=chunk_meta.get("metadata", {})
                )
                results.append(result)

        # Sort by similarity score (descending) and limit results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:max_results]

    async def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        if session_id not in self.sessions:
            # Try to load from Redis
            if self.redis_client:
                try:
                    session_data = await self.redis_client.get(f"session:{session_id}")
                    if session_data:
                        data = json.loads(session_data)
                        return SessionInfo(**data)
                except Exception as e:
                    logger.error(f"Failed to load session from Redis: {e}")
            return None

        session_data = self.sessions[session_id]
        doc_info = session_data.get("document", {})

        return SessionInfo(
            session_id=session_id,
            filename=doc_info.get("filename"),
            file_size=doc_info.get("file_size"),
            total_pages=doc_info.get("total_pages"),
            chunks_count=doc_info.get("chunks_count"),
            created_at=datetime.fromisoformat(doc_info.get("created_at", datetime.utcnow().isoformat())),
            last_activity=datetime.fromisoformat(session_data.get("last_activity", datetime.utcnow().isoformat())),
            message_count=len(session_data.get("chat_history", [])),
            is_active=True
        )

    async def store_session_info(self, session_id: str, info: Dict[str, Any]):
        """Store additional session information"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {"chat_history": []}

        self.sessions[session_id].update(info)
        self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()

        # Store in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"session:{session_id}",
                    settings.SESSION_TIMEOUT,
                    json.dumps(info)
                )
            except Exception as e:
                logger.warning(f"Failed to store session info in Redis: {e}")

    async def store_chat_message(self, session_id: str, message: ChatMessage):
        """Store a chat message"""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found for storing message")
            return

        # Convert message to dict for storage
        message_dict = {
            "id": message.id or str(datetime.utcnow().timestamp()),
            "type": message.type.value,
            "content": message.content,
            "timestamp": message.timestamp.isoformat() if isinstance(message.timestamp,
                                                                     datetime) else message.timestamp,
            "citations": [
                {
                    "page": c.page,
                    "text": c.text,
                    "confidence": c.confidence,
                    "type": c.type.value
                } for c in (message.citations or [])
            ],
            "token_usage": (
                {
                    "input_tokens": message.token_usage.input_tokens,
                    "output_tokens": message.token_usage.output_tokens,
                    "total_tokens": message.token_usage.total_tokens
                } if message.token_usage else None
            ),
            "metadata": message.metadata or {}
        }

        # Store in memory
        self.sessions[session_id]["chat_history"].append(message_dict)
        self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()

        # Store in Redis
        if self.redis_client:
            try:
                chat_key = f"chat:{session_id}"
                await self.redis_client.lpush(chat_key, json.dumps(message_dict))
                await self.redis_client.expire(chat_key, settings.SESSION_TIMEOUT)
            except Exception as e:
                logger.warning(f"Failed to store chat message in Redis: {e}")

    async def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """Get chat history for a session"""
        # Try memory first
        if session_id in self.sessions:
            history = self.sessions[session_id].get("chat_history", [])
        else:
            # Try Redis
            history = []
            if self.redis_client:
                try:
                    messages = await self.redis_client.lrange(f"chat:{session_id}", 0, -1)
                    history = [json.loads(msg) for msg in reversed(messages)]
                except Exception as e:
                    logger.error(f"Failed to get chat history from Redis: {e}")

        # Convert to ChatMessage objects
        chat_messages = []
        for msg_dict in history:
            try:
                # Convert back to proper objects
                message = ChatMessage(
                    id=msg_dict.get("id"),
                    type=msg_dict["type"],
                    content=msg_dict["content"],
                    timestamp=msg_dict["timestamp"],
                    citations=[],  # Will be populated if needed
                    token_usage=None,  # Will be populated if needed
                    metadata=msg_dict.get("metadata", {})
                )
                chat_messages.append(message)
            except Exception as e:
                logger.error(f"Failed to parse chat message: {e}")
                continue

        return chat_messages

    async def clear_chat_history(self, session_id: str):
        """Clear chat history for a session"""
        if session_id in self.sessions:
            self.sessions[session_id]["chat_history"] = []
            self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()

        if self.redis_client:
            try:
                await self.redis_client.delete(f"chat:{session_id}")
            except Exception as e:
                logger.warning(f"Failed to clear chat history in Redis: {e}")

    async def delete_session(self, session_id: str):
        """Delete a session and all its data"""
        # Remove from memory
        self.sessions.pop(session_id, None)
        self.vector_indices.pop(session_id, None)
        self.chunk_metadata.pop(session_id, None)

        # Remove from Redis
        if self.redis_client:
            try:
                keys_to_delete = [
                    f"session:{session_id}",
                    f"chat:{session_id}",
                    f"vectors:{session_id}",
                    f"metadata:{session_id}"
                ]
                await self.redis_client.delete(*keys_to_delete)
            except Exception as e:
                logger.warning(f"Failed to delete session from Redis: {e}")

        logger.info(f"🗑️ Deleted session: {session_id}")

    async def get_document_chunks(self, session_id: str, limit: int = None) -> List[PDFChunk]:
        """Get document chunks for a session"""
        if session_id not in self.chunk_metadata:
            return []

        metadata = self.chunk_metadata[session_id]
        chunks = []

        for i, chunk_meta in enumerate(metadata):
            if limit and i >= limit:
                break

            chunk = PDFChunk(
                id=chunk_meta["chunk_id"],
                content=chunk_meta["content"],
                page=chunk_meta["page"],
                chunk_index=chunk_meta["chunk_index"],
                metadata=chunk_meta.get("metadata", {})
            )
            chunks.append(chunk)

        return chunks

    async def _store_in_redis(self, session_id: str, embeddings: np.ndarray, metadata: List[Dict],
                              document: PDFDocument):
        """Store vectors and metadata in Redis"""
        try:
            # Store vectors
            vectors_data = pickle.dumps(embeddings)
            await self.redis_client.setex(
                f"vectors:{session_id}",
                settings.SESSION_TIMEOUT,
                vectors_data
            )

            # Store metadata
            metadata_data = json.dumps(metadata)
            await self.redis_client.setex(
                f"metadata:{session_id}",
                settings.SESSION_TIMEOUT,
                metadata_data
            )

            logger.info(f"✅ Stored session data in Redis: {session_id}")

        except Exception as e:
            logger.error(f"Failed to store in Redis: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return {
            "status": "healthy",
            "active_sessions": len(self.sessions),
            "total_indices": len(self.vector_indices),
            "redis_connected": self.redis_client is not None,
            "vector_dimension": self.vector_dimension
        }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Vector Service...")

        # Cleanup sessions older than timeout
        current_time = datetime.utcnow()
        expired_sessions = []

        for session_id, session_data in self.sessions.items():
            last_activity = datetime.fromisoformat(session_data.get("last_activity", current_time.isoformat()))
            if current_time - last_activity > timedelta(seconds=settings.SESSION_TIMEOUT):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self.delete_session(session_id)
            logger.info(f"🗑️ Cleaned up expired session: {session_id}")

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        # Shutdown thread pool
        self.executor.shutdown(wait=False)

        logger.info("✅ Vector Service cleanup complete")
