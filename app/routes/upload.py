"""
Upload routes for PDF file handling
"""
import logging
import time
import aiofiles
import shortuuid
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from app.models.schemas import UploadResponse, ErrorResponse, ResponseExamples
from app.utils.config import settings
from app.services.pdf_service import PDFService
from app.services.vector_service import VectorService
from app.services.gemini_service import GeminiService

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


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file type
    if file.content_type not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
        )

    # Check file size (this is a rough check, actual size checked during read)
    if hasattr(file, 'size') and file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
        )

    # Check filename
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Must be a PDF file."
        )


@router.post(
    "/",
    response_model=UploadResponse,
    summary="Upload PDF file",
    description="Upload a PDF file for processing and analysis",
    responses={
        200: {"description": "File uploaded successfully", **ResponseExamples.UPLOAD_SUCCESS},
        400: {"description": "Invalid file", "model": ErrorResponse},
        413: {"description": "File too large", "model": ErrorResponse},
        500: {"description": "Processing error", "model": ErrorResponse},
    }
)
async def upload_pdf(
        file: UploadFile = File(..., description="PDF file to upload"),
        services: dict = Depends(get_services)
) -> UploadResponse:
    """
    Upload and process a PDF file.

    This endpoint:
    1. Validates the uploaded file
    2. Saves it temporarily
    3. Extracts text and creates chunks
    4. Generates embeddings
    5. Stores in vector database
    6. Returns session ID for future interactions
    """
    start_time = time.time()
    session_id = None
    temp_file_path = None

    try:
        # Validate file
        validate_file(file)
        logger.info(f"📤 Uploading file: {file.filename} ({file.content_type})")

        # Generate session ID
        session_id = shortuuid.uuid()

        # Create temporary file path
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(exist_ok=True)
        temp_file_path = upload_dir / f"{session_id}_{file.filename}"

        # Read and save file with size check
        file_size = 0
        async with aiofiles.open(temp_file_path, 'wb') as f:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                file_size += len(chunk)

                # Check size limit during reading
                if file_size > settings.MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
                    )

                await f.write(chunk)

        logger.info(f"📁 File saved: {temp_file_path} ({file_size} bytes)")

        # Initialize PDF service
        pdf_service = PDFService()

        # Extract text from PDF
        logger.info("📖 Extracting text from PDF...")
        pdf_document = await pdf_service.process_pdf(str(temp_file_path))

        if not pdf_document.chunks:
            raise HTTPException(
                status_code=400,
                detail="No text content found in PDF. Please ensure the PDF contains readable text."
            )

        logger.info(f"✂️ Created {len(pdf_document.chunks)} chunks from {pdf_document.total_pages} pages")

        # Generate embeddings
        logger.info("🔮 Generating embeddings...")
        gemini_service = services["gemini_service"]

        for chunk in pdf_document.chunks:
            embedding = await gemini_service.generate_embedding(chunk.content)
            chunk.embedding = embedding

        logger.info(f"✅ Generated embeddings for {len(pdf_document.chunks)} chunks")

        # Store in vector database
        logger.info("💾 Storing in vector database...")
        vector_service = services["vector_service"]
        await vector_service.store_document(session_id, pdf_document)

        # Store session info
        await vector_service.store_session_info(session_id, {
            "filename": file.filename,
            "file_size": file_size,
            "total_pages": pdf_document.total_pages,
            "chunks_count": len(pdf_document.chunks),
            "upload_time": time.time(),
        })

        processing_time = time.time() - start_time
        logger.info(f"🎉 Upload completed in {processing_time:.2f}s")

        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
            logger.info(f"🗑️ Temporary file deleted: {temp_file_path}")

        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            file_size=file_size,
            total_pages=pdf_document.total_pages,
            chunks_created=len(pdf_document.chunks),
            processing_time=processing_time,
            message="PDF uploaded and processed successfully"
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(f"❌ Upload failed: {e}", exc_info=True)

        # Clean up on error
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logger.info(f"🗑️ Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temporary file: {cleanup_error}")

        if session_id:
            try:
                vector_service = services["vector_service"]
                await vector_service.delete_session(session_id)
                logger.info(f"🗑️ Cleaned up session: {session_id}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up session: {cleanup_error}")

        # Return appropriate error
        if "PDF" in str(e) or "text" in str(e).lower():
            raise HTTPException(
                status_code=400,
                detail=f"PDF processing error: {str(e)}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Internal server error during file processing"
            )


@router.get(
    "/status/{session_id}",
    summary="Get upload status",
    description="Get the processing status of an uploaded file"
)
async def get_upload_status(
        session_id: str,
        services: dict = Depends(get_services)
):
    """Get upload/processing status for a session"""
    try:
        vector_service = services["vector_service"]
        session_info = await vector_service.get_session_info(session_id)

        if not session_info:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )

        return {
            "success": True,
            "session_id": session_id,
            "status": "completed",
            "session_info": session_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upload status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get upload status"
        )


@router.delete(
    "/{session_id}",
    summary="Delete uploaded file",
    description="Delete an uploaded file and its associated data"
)
async def delete_upload(
        session_id: str,
        services: dict = Depends(get_services)
):
    """Delete uploaded file and session data"""
    try:
        vector_service = services["vector_service"]

        # Check if session exists
        session_info = await vector_service.get_session_info(session_id)
        if not session_info:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )

        # Delete session data
        await vector_service.delete_session(session_id)

        logger.info(f"🗑️ Deleted session: {session_id}")

        return {
            "success": True,
            "message": "Upload deleted successfully",
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting upload: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete upload"
        )


@router.get(
    "/config",
    summary="Get upload configuration",
    description="Get upload limits and configuration"
)
async def get_upload_config():
    """Get upload configuration and limits"""
    return {
        "max_file_size": settings.MAX_FILE_SIZE,
        "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024),
        "allowed_file_types": settings.ALLOWED_FILE_TYPES,
        "upload_dir": settings.UPLOAD_DIR,
        "chunk_size": settings.PDF_CHUNK_SIZE,
        "chunk_overlap": settings.PDF_CHUNK_OVERLAP,
    }