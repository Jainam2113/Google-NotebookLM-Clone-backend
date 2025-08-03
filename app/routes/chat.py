"""
Chat routes for AI conversation with PDF documents
"""
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatHistory,
    ChatMessage,
    Citation,
    TokenUsage,
    MessageType,
    CitationType,
    ErrorResponse,
    ResponseExamples
)
from app.services.vector_service import VectorService
from app.services.gemini_service import GeminiService
from app.utils.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Dependencies
async def get_services(request: Request):
    """Get services from app state"""
    return {
        "vector_service": request.app.state.vector_service,
        "gemini_service": request.app.state.gemini_service,
    }


async def validate_session(session_id: str, vector_service: VectorService):
    """Validate that session exists and has data"""
    session_info = await vector_service.get_session_info(session_id)
    if not session_info:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a PDF first."
        )
    return session_info


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Send chat message",
    description="Send a message and get AI response based on uploaded PDF",
    responses={
        200: {"description": "Chat response generated", **ResponseExamples.CHAT_SUCCESS},
        400: {"description": "Invalid request", "model": ErrorResponse},
        404: {"description": "Session not found", "model": ErrorResponse},
        500: {"description": "AI service error", "model": ErrorResponse},
    }
)
async def chat_message(
        request: ChatRequest,
        services: dict = Depends(get_services)
) -> ChatResponse:
    """
    Send a chat message and get AI response.

    This endpoint:
    1. Validates the session
    2. Searches for relevant document chunks
    3. Generates AI response with context
    4. Returns response with citations
    """
    start_time = time.time()

    try:
        vector_service = services["vector_service"]
        gemini_service = services["gemini_service"]

        # Validate session
        session_info = await validate_session(request.session_id, vector_service)
        logger.info(f"💬 Chat message for session: {request.session_id}")

        # Generate embedding for the query
        logger.info("🔍 Generating query embedding...")
        query_embedding = await gemini_service.generate_embedding(request.message)

        # Search for relevant chunks
        logger.info("🔍 Searching for relevant content...")
        search_results = await vector_service.search_similar(
            session_id=request.session_id,
            query=request.message,
            query_embedding=query_embedding,
            max_results=settings.MAX_SEARCH_RESULTS,
            threshold=settings.SIMILARITY_THRESHOLD
        )

        if not search_results:
            logger.warning("No relevant content found")
            return ChatResponse(
                response="I couldn't find relevant information in the document to answer your question. Could you please rephrase or ask about a different topic?",
                citations=[],
                processing_time=time.time() - start_time,
                message="No relevant context found"
            )

        logger.info(f"📄 Found {len(search_results)} relevant chunks")

        # Prepare context for AI
        context_chunks = []
        citations = []

        for i, result in enumerate(search_results):
            context_chunks.append(f"[Page {result.page}] {result.content}")

            # Create citation
            citation = Citation(
                page=result.page,
                text=result.content[:100] + "..." if len(result.content) > 100 else result.content,
                confidence=result.similarity_score,
                type=CitationType.PAGE_REFERENCE
            )
            citations.append(citation)

        context = "\n\n".join(context_chunks)

        # Generate AI response
        logger.info("🤖 Generating AI response...")
        ai_response = await gemini_service.generate_chat_response(
            message=request.message,
            context=context,
            session_id=request.session_id
        )

        # Extract token usage if available
        token_usage = None
        if hasattr(ai_response, 'usage_metadata') and ai_response.usage_metadata:
            token_usage = TokenUsage(
                input_tokens=getattr(ai_response.usage_metadata, 'prompt_token_count', 0),
                output_tokens=getattr(ai_response.usage_metadata, 'candidates_token_count', 0),
                total_tokens=getattr(ai_response.usage_metadata, 'total_token_count', 0)
            )

        # Store chat message in history
        await vector_service.store_chat_message(
            session_id=request.session_id,
            message=ChatMessage(
                type=MessageType.USER,
                content=request.message,
                metadata={"search_results_count": len(search_results)}
            )
        )

        await vector_service.store_chat_message(
            session_id=request.session_id,
            message=ChatMessage(
                type=MessageType.ASSISTANT,
                content=ai_response.text,
                citations=citations,
                token_usage=token_usage,
                metadata={"context_chunks": len(context_chunks)}
            )
        )

        processing_time = time.time() - start_time
        logger.info(f"✅ Chat response generated in {processing_time:.2f}s")

        return ChatResponse(
            response=ai_response.text,
            citations=citations,
            token_usage=token_usage,
            processing_time=processing_time,
            message="Response generated successfully"
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)

        # Determine error type and message
        error_message = "I apologize, but I encountered an error while processing your message. Please try again."

        if "gemini" in str(e).lower() or "api" in str(e).lower():
            error_message = "The AI service is temporarily unavailable. Please try again in a moment."
        elif "embedding" in str(e).lower():
            error_message = "There was an issue searching the document. Please try rephrasing your question."

        raise HTTPException(
            status_code=500,
            detail=error_message
        )


@router.get(
    "/history/{session_id}",
    response_model=ChatHistory,
    summary="Get chat history",
    description="Retrieve chat history for a session"
)
async def get_chat_history(
        session_id: str,
        services: dict = Depends(get_services)
) -> ChatHistory:
    """Get chat history for a session"""
    try:
        vector_service = services["vector_service"]

        # Validate session
        await validate_session(session_id, vector_service)

        # Get chat history
        messages = await vector_service.get_chat_history(session_id)

        return ChatHistory(
            session_id=session_id,
            messages=messages or []
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve chat history"
        )


@router.delete(
    "/history/{session_id}",
    summary="Clear chat history",
    description="Clear all chat history for a session"
)
async def clear_chat_history(
        session_id: str,
        services: dict = Depends(get_services)
):
    """Clear chat history for a session"""
    try:
        vector_service = services["vector_service"]

        # Validate session
        await validate_session(session_id, vector_service)

        # Clear history
        await vector_service.clear_chat_history(session_id)

        return {
            "success": True,
            "message": "Chat history cleared successfully",
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear chat history"
        )


@router.post(
    "/regenerate",
    response_model=ChatResponse,
    summary="Regenerate last response",
    description="Regenerate the last AI response"
)
async def regenerate_response(
        session_id: str,
        services: dict = Depends(get_services)
) -> ChatResponse:
    """Regenerate the last AI response"""
    try:
        vector_service = services["vector_service"]

        # Validate session
        await validate_session(session_id, vector_service)

        # Get last user message
        chat_history = await vector_service.get_chat_history(session_id)
        if not chat_history:
            raise HTTPException(
                status_code=400,
                detail="No chat history found"
            )

        # Find last user message
        last_user_message = None
        for message in reversed(chat_history):
            if message.type == MessageType.USER:
                last_user_message = message
                break

        if not last_user_message:
            raise HTTPException(
                status_code=400,
                detail="No user message found to regenerate from"
            )

        # Remove last AI response if it exists
        if (chat_history and
                chat_history[-1].type == MessageType.ASSISTANT):
            await vector_service.remove_last_message(session_id)

        # Regenerate response
        request = ChatRequest(
            message=last_user_message.content,
            session_id=session_id
        )

        return await chat_message(request, services)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating response: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to regenerate response"
        )


@router.get(
    "/suggestions/{session_id}",
    summary="Get conversation suggestions",
    description="Get suggested questions based on document content"
)
async def get_suggestions(
        session_id: str,
        services: dict = Depends(get_services)
):
    """Get suggested questions for the document"""
    try:
        vector_service = services["vector_service"]
        gemini_service = services["gemini_service"]

        # Validate session
        session_info = await validate_session(session_id, vector_service)

        # Get document summary or first few chunks for context
        chunks = await vector_service.get_document_chunks(session_id, limit=3)
        if not chunks:
            return {"suggestions": []}

        # Create context from first chunks
        context = "\n".join([chunk.content for chunk in chunks])

        # Generate suggestions using AI
        prompt = f"""Based on this document excerpt, suggest 5 interesting questions a user might ask:

{context[:1500]}...

Provide 5 short, specific questions that would help someone understand the key concepts and information in this document. Format as a simple list."""

        response = await gemini_service.generate_simple_response(prompt)

        # Parse suggestions (simple implementation)
        suggestions = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('1')):
                # Clean up the line
                suggestion = line.lstrip('•-123456789. ').strip()
                if suggestion:
                    suggestions.append(suggestion)

        return {
            "success": True,
            "suggestions": suggestions[:5],  # Limit to 5 suggestions
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return {
            "success": False,
            "suggestions": [
                "What is this document about?",
                "Can you summarize the main points?",
                "What are the key findings or conclusions?"
            ]
        }


@router.get(
    "/stats/{session_id}",
    summary="Get chat statistics",
    description="Get statistics for a chat session"
)
async def get_chat_stats(
        session_id: str,
        services: dict = Depends(get_services)
):
    """Get chat statistics for a session"""
    try:
        vector_service = services["vector_service"]

        # Validate session
        await validate_session(session_id, vector_service)

        # Get chat history
        messages = await vector_service.get_chat_history(session_id)

        if not messages:
            return {
                "session_id": session_id,
                "total_messages": 0,
                "user_messages": 0,
                "ai_messages": 0,
                "total_tokens": 0,
                "conversation_started": None
            }

        # Calculate statistics
        user_messages = [m for m in messages if m.type == MessageType.USER]
        ai_messages = [m for m in messages if m.type == MessageType.ASSISTANT]

        total_tokens = sum(
            msg.token_usage.total_tokens if msg.token_usage else 0
            for msg in ai_messages
        )

        return {
            "session_id": session_id,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "ai_messages": len(ai_messages),
            "total_tokens": total_tokens,
            "conversation_started": messages[0].timestamp if messages else None,
            "last_activity": messages[-1].timestamp if messages else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get chat statistics"
        )