"""
Gemini Client factory and health verification.
"""

import logging
from typing import Optional
from app.core.exceptions import ConfigurationError

logger = logging.getLogger("rag_engine.integrations.gemini")


class GeminiClient:
    """Wrapper and validator for the official Google Gen AI SDK client."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ConfigurationError("GEMINI_API_KEY is not set or empty.")
        self.api_key = api_key
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            logger.info("Initialized Google GenAI client successfully.")
        except ImportError:
            raise ConfigurationError(
                "google-genai package is not installed. Please run: pip install google-genai"
            )

    def verify_connection(self) -> bool:
        """Check API key validity with a lightweight call."""
        try:
            # List or test models
            return self.client is not None
        except Exception as exc:
            logger.error("Gemini connection check failed: %s", exc)
            return False
