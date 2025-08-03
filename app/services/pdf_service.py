"""
PDF processing service using PyMuPDF and pdfplumber
"""
import logging
import re
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from concurrent.futures import ThreadPoolExecutor

from app.models.schemas import PDFDocument, PDFChunk
from app.utils.config import settings

logger = logging.getLogger(__name__)


class PDFService:
    """Service for processing PDF documents"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.chunk_size = settings.PDF_CHUNK_SIZE
        self.chunk_overlap = settings.PDF_CHUNK_OVERLAP

    async def process_pdf(self, file_path: str) -> PDFDocument:
        """
        Process a PDF file and extract text chunks

        Args:
            file_path: Path to the PDF file

        Returns:
            PDFDocument with extracted chunks
        """
        logger.info(f"📖 Processing PDF: {file_path}")

        # Run PDF processing in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._process_pdf_sync,
            file_path
        )

        return result

    def _process_pdf_sync(self, file_path: str) -> PDFDocument:
        """Synchronous PDF processing"""
        try:
            file_path = Path(file_path)

            # Extract text using PyMuPDF (faster, primary method)
            logger.info("🔍 Extracting text with PyMuPDF...")
            pages_text = self._extract_text_pymupdf(file_path)

            # If PyMuPDF fails or returns empty, try pdfplumber
            if not any(page.strip() for page in pages_text):
                logger.info("🔄 Falling back to pdfplumber...")
                pages_text = self._extract_text_pdfplumber(file_path)

            # Check if we got any text
            if not any(page.strip() for page in pages_text):
                raise ValueError("No readable text found in PDF")

            # Create chunks
            logger.info("✂️ Creating text chunks...")
            chunks = self._create_chunks(pages_text)

            # Get file size
            file_size = file_path.stat().st_size

            return PDFDocument(
                filename=file_path.name,
                file_size=file_size,
                total_pages=len(pages_text),
                chunks=chunks,
                metadata={
                    "extraction_method": "pymupdf" if pages_text else "pdfplumber",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "total_chunks": len(chunks)
                }
            )

        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            raise ValueError(f"Failed to process PDF: {str(e)}")

    def _extract_text_pymupdf(self, file_path: Path) -> List[str]:
        """Extract text using PyMuPDF (faster)"""
        try:
            pages_text = []

            with fitz.open(str(file_path)) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]

                    # Extract text
                    text = page.get_text()

                    # Clean and process text
                    cleaned_text = self._clean_text(text)
                    pages_text.append(cleaned_text)

                    logger.debug(f"Page {page_num + 1}: {len(cleaned_text)} characters")

            logger.info(f"✅ Extracted {sum(len(p) for p in pages_text)} characters from {len(pages_text)} pages")
            return pages_text

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return []

    def _extract_text_pdfplumber(self, file_path: Path) -> List[str]:
        """Extract text using pdfplumber (more thorough but slower)"""
        try:
            pages_text = []

            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text
                        text = page.extract_text()

                        if text:
                            # Clean and process text
                            cleaned_text = self._clean_text(text)
                            pages_text.append(cleaned_text)
                        else:
                            pages_text.append("")

                        logger.debug(f"Page {page_num + 1}: {len(pages_text[-1])} characters")

                    except Exception as page_error:
                        logger.warning(f"Failed to extract page {page_num + 1}: {page_error}")
                        pages_text.append("")

            logger.info(f"✅ Extracted {sum(len(p) for p in pages_text)} characters from {len(pages_text)} pages")
            return pages_text

        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove or replace problematic characters
        text = text.replace('\x00', '')  # Remove null bytes
        text = text.replace('\ufffd', ' ')  # Replace replacement characters

        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _create_chunks(self, pages_text: List[str]) -> List[PDFChunk]:
        """Create overlapping text chunks from pages"""
        chunks = []
        chunk_id = 0

        for page_num, page_text in enumerate(pages_text, 1):
            if not page_text.strip():
                continue

            # Split page into sentences for better chunking
            sentences = self._split_into_sentences(page_text)

            if not sentences:
                continue

            # Create chunks from sentences
            current_chunk = ""
            current_sentences = []

            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence

                if len(test_chunk) <= self.chunk_size:
                    current_chunk = test_chunk
                    current_sentences.append(sentence)
                else:
                    # Current chunk is full, save it
                    if current_chunk:
                        chunks.append(PDFChunk(
                            id=f"chunk_{chunk_id}",
                            content=current_chunk.strip(),
                            page=page_num,
                            chunk_index=chunk_id,
                            metadata={
                                "sentence_count": len(current_sentences),
                                "char_count": len(current_chunk)
                            }
                        ))
                        chunk_id += 1

                    # Start new chunk with overlap
                    overlap_sentences = current_sentences[-self._calculate_overlap_sentences(current_sentences):]
                    current_chunk = " ".join(overlap_sentences + [sentence])
                    current_sentences = overlap_sentences + [sentence]

            # Add remaining chunk
            if current_chunk.strip():
                chunks.append(PDFChunk(
                    id=f"chunk_{chunk_id}",
                    content=current_chunk.strip(),
                    page=page_num,
                    chunk_index=chunk_id,
                    metadata={
                        "sentence_count": len(current_sentences),
                        "char_count": len(current_chunk)
                    }
                ))
                chunk_id += 1

        logger.info(f"✂️ Created {len(chunks)} chunks")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking"""
        if not text:
            return []

        # Simple sentence splitting (can be improved with nltk/spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out very short or empty sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        return sentences

    def _calculate_overlap_sentences(self, sentences: List[str]) -> int:
        """Calculate how many sentences to include in overlap"""
        if not sentences:
            return 0

        # Target overlap in characters
        target_overlap = min(self.chunk_overlap, len(" ".join(sentences)) // 2)

        overlap_sentences = 0
        overlap_chars = 0

        for i in range(len(sentences) - 1, -1, -1):
            sentence_len = len(sentences[i])
            if overlap_chars + sentence_len <= target_overlap:
                overlap_chars += sentence_len
                overlap_sentences += 1
            else:
                break

        return min(overlap_sentences, len(sentences) // 2)

    async def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._extract_metadata_sync,
            file_path
        )

    def _extract_metadata_sync(self, file_path: str) -> Dict[str, Any]:
        """Synchronously extract PDF metadata"""
        try:
            metadata = {}

            with fitz.open(file_path) as doc:
                # Basic document info
                metadata.update({
                    "page_count": len(doc),
                    "is_pdf": True,
                    "is_encrypted": doc.is_encrypted,
                    "permissions": doc.permissions,
                })

                # Document metadata
                doc_metadata = doc.metadata
                if doc_metadata:
                    metadata.update({
                        "title": doc_metadata.get("title", ""),
                        "author": doc_metadata.get("author", ""),
                        "subject": doc_metadata.get("subject", ""),
                        "creator": doc_metadata.get("creator", ""),
                        "producer": doc_metadata.get("producer", ""),
                        "creation_date": doc_metadata.get("creationDate", ""),
                        "modification_date": doc_metadata.get("modDate", ""),
                    })

                # Page dimensions (first page)
                if len(doc) > 0:
                    first_page = doc[0]
                    rect = first_page.rect
                    metadata.update({
                        "page_width": rect.width,
                        "page_height": rect.height,
                        "page_rotation": first_page.rotation,
                    })

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return {"error": str(e)}

    async def get_page_text(self, file_path: str, page_number: int) -> str:
        """Get text from a specific page"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._get_page_text_sync,
            file_path,
            page_number
        )

    def _get_page_text_sync(self, file_path: str, page_number: int) -> str:
        """Synchronously get text from specific page"""
        try:
            with fitz.open(file_path) as doc:
                if page_number < 1 or page_number > len(doc):
                    raise ValueError(f"Page {page_number} not found")

                page = doc[page_number - 1]  # PyMuPDF uses 0-based indexing
                text = page.get_text()
                return self._clean_text(text)

        except Exception as e:
            logger.error(f"Failed to get page text: {e}")
            return ""

    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)