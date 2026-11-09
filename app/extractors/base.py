"""Base extractor interface for all document types."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ExtractionMethod(str, Enum):
    DIGITAL = "digital"
    OCR = "ocr"
    MIXED = "mixed"
    SLIDE = "slide"
    DOCX = "docx"


@dataclass
class ExtractedPage:
    """Represents a single extracted page or slide."""
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    heading: Optional[str] = None
    text: str = ""
    extraction_method: ExtractionMethod = ExtractionMethod.DIGITAL
    ocr_confidence: Optional[float] = None
    is_empty: bool = False
    error: Optional[str] = None
    # For PPTX: images OCR'd on this slide
    image_texts: List[str] = field(default_factory=list)
    # For DOCX: section context
    section: Optional[str] = None


@dataclass
class ExtractionResult:
    """Full extraction result from a document."""
    document_id: str
    source_filename: str
    source_type: str  # PDF, PPTX, DOCX
    pages: List[ExtractedPage] = field(default_factory=list)
    total_pages: int = 0
    total_slides: int = 0
    primary_extraction_method: ExtractionMethod = ExtractionMethod.DIGITAL
    errors: List[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_error(self, page_or_slide: Optional[int], error_type: str, message: str) -> None:
        """Record a non-fatal extraction error."""
        self.errors.append({
            "page_or_slide": page_or_slide,
            "error_type": error_type,
            "message": message,
        })

    def has_content(self) -> bool:
        """Return True if at least one page has non-empty text."""
        return any(not p.is_empty and p.text.strip() for p in self.pages)


class BaseExtractor(ABC):
    """Abstract base class for document extractors."""

    @abstractmethod
    def extract(self, file_path: str, document_id: str, filename: str) -> ExtractionResult:
        """
        Extract text and metadata from a document file.

        Args:
            file_path: Absolute path to the document on disk.
            document_id: UUID of the document record.
            filename: Original filename (for metadata).

        Returns:
            ExtractionResult with all extracted pages/slides.
        """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of file extensions this extractor supports."""

    def supports(self, extension: str) -> bool:
        """Return True if this extractor handles the given extension."""
        return extension.lower() in [e.lower() for e in self.supported_extensions]
