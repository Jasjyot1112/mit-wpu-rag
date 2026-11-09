"""
Extraction Service — facade for extracting PDF, PPTX, and DOCX educational documents.
"""

import logging
from pathlib import Path
from typing import Dict, List
from app.extractors.base import BaseExtractor, ExtractionResult
from app.core.exceptions import UnsupportedFileTypeError

logger = logging.getLogger("rag_engine.services.extraction")


class ExtractionService:
    """Dispatches document extraction to the appropriate format extractor."""

    def __init__(self, extractors: List[BaseExtractor]):
        self._extractors: Dict[str, BaseExtractor] = {}
        for extractor in extractors:
            for ext in extractor.supported_extensions:
                self._extractors[ext.lower()] = extractor

    def extract(self, file_path: str, document_id: str, filename: str) -> ExtractionResult:
        ext = Path(filename).suffix.lower()
        extractor = self._extractors.get(ext)
        if not extractor:
            raise UnsupportedFileTypeError(
                f"No extractor registered for file extension '{ext}'"
            )
        logger.info("Extracting '%s' using %s...", filename, extractor.__class__.__name__)
        return extractor.extract(file_path=file_path, document_id=document_id, filename=filename)
