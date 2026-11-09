"""
Pinecone Client factory and initialization helper.
"""

import logging
from typing import Optional
from app.core.exceptions import ConfigurationError

logger = logging.getLogger("rag_engine.integrations.pinecone")


class PineconeClientFactory:
    """Factory for Pinecone client and index validation."""

    @staticmethod
    def get_client(api_key: str):
        if not api_key:
            raise ConfigurationError("PINECONE_API_KEY is not set or empty.")
        try:
            from pinecone import Pinecone
            return Pinecone(api_key=api_key)
        except ImportError:
            raise ConfigurationError(
                "pinecone package is not installed. Please run: pip install pinecone"
            )
