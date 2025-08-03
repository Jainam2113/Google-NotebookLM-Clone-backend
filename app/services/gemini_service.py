"""
Google Gemini AI service for chat and embeddings
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from concurrent.futures import ThreadPoolExecutor

from app.utils.config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for Google Gemini AI interactions"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.chat_model = None
        self.embedding_model = None
        self.is_initialized = False

        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        # Generation config
        self.generation_config = {
            "temperature": settings.GEMINI_TEMPERATURE,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": settings.GEMINI_MAX_TOKENS,
        }

    async def initialize(self):
        """Initialize Gemini models"""
        logger.info("🤖 Initializing Gemini Service...")

        try:
            # Configure API key
            genai.configure(api_key=settings.GEMINI_API_KEY)

            # Initialize models
            self.chat_model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL_NAME,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            # Test the connection
            await self._test_connection()

            self.is_initialized = True
            logger.info("✅ Gemini Service initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            raise ValueError(f"Gemini initialization failed: {str(e)}")

    async def _test_connection(self):
        """Test Gemini API connection"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.chat_model.generate_content("Hello")
            )
            logger.info("✅ Gemini API connection test successful")
        except Exception as e:
            logger.error(f"❌ Gemini API connection test failed: {e}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not self.is_initialized:
            raise ValueError("Gemini service not initialized")

        if not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            loop = asyncio.get_event_loop()

            # Use the embedding model
            result = await loop.run_in_executor(
                self.executor,
                lambda: genai.embed_content(
                    model=settings.GEMINI_EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
            )

            return result['embedding']

        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise ValueError(f"Failed to generate embedding: {str(e)}")

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        if not self.is_initialized:
            raise ValueError("Gemini service not initialized")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            loop = asyncio.get_event_loop()

            # Use the embedding model with query task type
            result = await loop.run_in_executor(
                self.executor,
                lambda: genai.embed_content(
                    model=settings.GEMINI_EMBEDDING_MODEL,
                    content=query,
                    task_type="retrieval_query"
                )
            )

            return result['embedding']

        except Exception as e:
            logger.error(f"❌ Query embedding generation failed: {e}")
            raise ValueError(f"Failed to generate query embedding: {str(e)}")

    async def generate_chat_response(
            self,
            message: str,
            context: str,
            session_id: str = None
    ) -> Any:
        """Generate chat response with context"""
        if not self.is_initialized:
            raise ValueError("Gemini service not initialized")

        if not message.strip():
            raise ValueError("Message cannot be empty")

        try:
            # Create prompt with context
            prompt = self._create_chat_prompt(message, context)

            logger.info(f"🤖 Generating response for session: {session_id}")
            logger.debug(f"Prompt length: {len(prompt)} characters")

            loop = asyncio.get_event_loop()

            # Generate response
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.chat_model.generate_content(prompt)
            )

            # Check if response was blocked
            if not response.text:
                if response.prompt_feedback:
                    logger.warning(f"Response blocked: {response.prompt_feedback}")
                    raise ValueError("Response was blocked by safety filters")
                else:
                    raise ValueError("Empty response generated")

            logger.info("✅ Chat response generated successfully")
            return response

        except Exception as e:
            logger.error(f"❌ Chat response generation failed: {e}")
            raise ValueError(f"Failed to generate chat response: {str(e)}")

    async def generate_simple_response(self, prompt: str) -> Any:
        """Generate a simple response without context formatting"""
        if not self.is_initialized:
            raise ValueError("Gemini service not initialized")

        try:
            loop = asyncio.get_event_loop()

            response = await loop.run_in_executor(
                self.executor,
                lambda: self.chat_model.generate_content(prompt)
            )

            if not response.text:
                raise ValueError("Empty response generated")

            return response

        except Exception as e:
            logger.error(f"❌ Simple response generation failed: {e}")
            raise ValueError(f"Failed to generate response: {str(e)}")

    def _create_chat_prompt(self, message: str, context: str) -> str:
        """Create a well-formatted prompt for chat"""

        system_prompt = """You are a helpful AI assistant that answers questions based on PDF documents. 

    Key instructions:
    1. Answer questions using ONLY the information provided in the context
    2. Always cite specific page numbers when referencing information
    3. Format your response in clean Markdown with:
       - **Bold** for important points
       - *Italics* for emphasis
       - `code blocks` for technical terms
       - ## Headers for sections
       - - Bullet points for lists
       - > Blockquotes for direct quotes
    4. Always include page references like: **(Page 3)**
    5. If the context doesn't contain enough information, say so clearly
    6. Use a conversational, helpful tone
    7. Structure your response with clear sections when appropriate

    Format example:
    ## Summary
    The document discusses **key topic** which involves...

    ## Key Points
    - First important point **(Page 1)**
    - Second point with *emphasis* **(Page 2)**

    > "Direct quote from document" **(Page 3)**

    ## Technical Details
    For `technical terms`, the document explains... **(Page 4)**"""

        formatted_prompt = f"""{system_prompt}

    DOCUMENT CONTEXT:
    {context}

    USER QUESTION: {message}

    RESPONSE (in clean Markdown format with page citations):"""

        return formatted_prompt

    async def generate_summary(self, text: str, max_length: int = 500) -> str:
        """Generate a summary of the given text"""
        if not self.is_initialized:
            raise ValueError("Gemini service not initialized")

        try:
            prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters or less. Focus on the main points and key information:

{text[:3000]}  # Limit input text to avoid token limits

Summary:"""

            response = await self.generate_simple_response(prompt)
            return response.text

        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            raise ValueError(f"Failed to generate summary: {str(e)}")

    async def generate_questions(self, text: str, num_questions: int = 5) -> List[str]:
        """Generate potential questions based on document content"""
        if not self.is_initialized:
            raise ValueError("Gemini service not initialized")

        try:
            prompt = f"""Based on the following document excerpt, generate {num_questions} thoughtful questions that a reader might ask to better understand the content. Make the questions specific and relevant to the material:

{text[:2000]}

Generate {num_questions} questions (one per line, starting with a number):"""

            response = await self.generate_simple_response(prompt)

            # Parse questions from response
            questions = []
            for line in response.text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                    # Clean up the question
                    question = line.lstrip('0123456789.•- ').strip()
                    if question:
                        questions.append(question)

            return questions[:num_questions]

        except Exception as e:
            logger.error(f"❌ Question generation failed: {e}")
            return [
                "What is the main topic of this document?",
                "What are the key points discussed?",
                "Are there any important conclusions or findings?"
            ]

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        if not self.is_initialized:
            raise ValueError("Gemini service not initialized")

        try:
            prompt = f"""Analyze the sentiment and tone of the following text. Provide:
1. Overall sentiment (positive/negative/neutral)
2. Confidence level (0-1)
3. Key emotional indicators
4. Brief explanation

Text: {text[:1000]}

Analysis:"""

            response = await self.generate_simple_response(prompt)

            # Simple parsing - in production, you might want more structured output
            return {
                "sentiment": "neutral",  # Default
                "confidence": 0.5,
                "analysis": response.text,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "analysis": "Analysis failed",
                "error": str(e)
            }

    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            if not self.is_initialized:
                return {
                    "status": "unhealthy",
                    "initialized": False,
                    "error": "Service not initialized"
                }

            # Quick health check with a simple prompt
            await self.generate_simple_response("Test")

            return {
                "status": "healthy",
                "initialized": True,
                "model": settings.GEMINI_MODEL_NAME,
                "embedding_model": settings.GEMINI_EMBEDDING_MODEL
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "initialized": self.is_initialized,
                "error": str(e)
            }

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        try:
            loop = asyncio.get_event_loop()

            models = await loop.run_in_executor(
                self.executor,
                lambda: list(genai.list_models())
            )

            model_info = []
            for model in models:
                model_info.append({
                    "name": model.name,
                    "display_name": getattr(model, 'display_name', ''),
                    "description": getattr(model, 'description', ''),
                    "input_token_limit": getattr(model, 'input_token_limit', 0),
                    "output_token_limit": getattr(model, 'output_token_limit', 0),
                })

            return {
                "models": model_info,
                "current_chat_model": settings.GEMINI_MODEL_NAME,
                "current_embedding_model": settings.GEMINI_EMBEDDING_MODEL
            }

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}

    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)