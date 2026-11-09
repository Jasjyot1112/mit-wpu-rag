"""
PDF extractor using PyMuPDF with selective OCR for scanned pages.

Strategy:
- Extract digital text first from every page.
- Evaluate text quality (length, character distribution).
- For pages with insufficient digital text, render to image and apply Tesseract OCR.
- Record extraction method per page (digital / ocr / mixed).
- Never silently skip a page; always record errors.
"""
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

from app.extractors.base import BaseExtractor, ExtractionResult, ExtractedPage, ExtractionMethod
from app.core.exceptions import ExtractionError

logger = logging.getLogger("rag_engine.extractors.pdf")

# Minimum meaningful characters to consider a page as having digital text
_MIN_DIGITAL_TEXT_CHARS = 50
# Ratio of printable chars that must be present for digital text to be acceptable
_MIN_PRINTABLE_RATIO = 0.70


class PDFExtractor(BaseExtractor):
    """
    Extract text from PDF files using PyMuPDF.
    Uses selective OCR only for pages lacking sufficient digital text.
    """

    def __init__(self, ocr_service=None, max_pages: int = 500):
        """
        Args:
            ocr_service: An OCRService instance (injected). If None, OCR is skipped.
            max_pages: Hard limit on pages to process.
        """
        self._ocr = ocr_service
        self._max_pages = max_pages

    @property
    def supported_extensions(self):
        return [".pdf"]

    def extract(self, file_path: str, document_id: str, filename: str) -> ExtractionResult:
        """Extract text from every page of a PDF, using OCR where needed."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ExtractionError("PyMuPDF (fitz) is not installed. Run: pip install pymupdf")

        result = ExtractionResult(
            document_id=document_id,
            source_filename=filename,
            source_type="PDF",
        )

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise ExtractionError(f"Cannot open PDF '{filename}': {exc}")

        total_pages = min(len(doc), self._max_pages)
        result.total_pages = total_pages

        if total_pages < len(doc):
            logger.warning(
                "PDF '%s' has %d pages; processing only the first %d (MAX_PDF_PAGES limit).",
                filename, len(doc), total_pages,
            )

        digital_count = 0
        ocr_count = 0

        for page_index in range(total_pages):
            page_num = page_index + 1  # 1-based
            extracted = self._extract_page(doc, page_index, page_num, filename, result)
            result.pages.append(extracted)

            if extracted.extraction_method == ExtractionMethod.DIGITAL:
                digital_count += 1
            elif extracted.extraction_method == ExtractionMethod.OCR:
                ocr_count += 1

        doc.close()

        # Determine overall extraction method
        if ocr_count == 0:
            result.primary_extraction_method = ExtractionMethod.DIGITAL
        elif digital_count == 0:
            result.primary_extraction_method = ExtractionMethod.OCR
        else:
            result.primary_extraction_method = ExtractionMethod.MIXED

        logger.info(
            "PDF '%s': %d pages processed (%d digital, %d OCR, %d errors).",
            filename, total_pages, digital_count, ocr_count, len(result.errors),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_page(
        self,
        doc,
        page_index: int,
        page_num: int,
        filename: str,
        result: ExtractionResult,
    ) -> ExtractedPage:
        """Extract a single page, falling back to OCR if needed."""
        try:
            page = doc.load_page(page_index)
        except Exception as exc:
            result.add_error(page_num, "PAGE_LOAD_FAILURE", str(exc))
            return ExtractedPage(
                page_number=page_num,
                is_empty=True,
                error=f"Cannot load page: {exc}",
                extraction_method=ExtractionMethod.DIGITAL,
            )

        # --- Attempt digital text extraction ---
        try:
            digital_text = page.get_text("text")
        except Exception as exc:
            digital_text = ""
            result.add_error(page_num, "TEXT_EXTRACTION_FAILURE", str(exc))

        digital_text = self._clean_raw_text(digital_text)
        is_scanned = self._is_scanned_page(digital_text)

        if not is_scanned:
            # Digital text is sufficient
            if not digital_text.strip():
                return ExtractedPage(
                    page_number=page_num,
                    text="",
                    is_empty=True,
                    extraction_method=ExtractionMethod.DIGITAL,
                )
            return ExtractedPage(
                page_number=page_num,
                text=digital_text,
                extraction_method=ExtractionMethod.DIGITAL,
            )

        # --- Fallback: OCR ---
        if self._ocr is None:
            logger.warning("Page %d of '%s' appears scanned but no OCR service configured.", page_num, filename)
            result.add_error(page_num, "OCR_UNAVAILABLE", "Page appears scanned; OCR service not configured.")
            return ExtractedPage(
                page_number=page_num,
                text=digital_text,  # use whatever partial digital text we have
                is_empty=not digital_text.strip(),
                extraction_method=ExtractionMethod.DIGITAL,
                error="OCR service unavailable for scanned page",
            )

        try:
            ocr_text, confidence = self._ocr_page(page)
            if not ocr_text.strip():
                return ExtractedPage(
                    page_number=page_num,
                    text="",
                    is_empty=True,
                    extraction_method=ExtractionMethod.OCR,
                    ocr_confidence=confidence,
                )
            return ExtractedPage(
                page_number=page_num,
                text=ocr_text,
                extraction_method=ExtractionMethod.OCR,
                ocr_confidence=confidence,
            )
        except Exception as exc:
            result.add_error(page_num, "OCR_FAILURE", str(exc))
            logger.error("OCR failed for page %d of '%s': %s", page_num, filename, exc)
            # Return partial digital text if available
            return ExtractedPage(
                page_number=page_num,
                text=digital_text,
                is_empty=not digital_text.strip(),
                extraction_method=ExtractionMethod.MIXED if digital_text.strip() else ExtractionMethod.OCR,
                error=f"OCR failed: {exc}",
            )

    def _ocr_page(self, page) -> tuple[str, Optional[float]]:
        """Render a PDF page to image and apply OCR. Returns (text, confidence)."""
        import fitz
        # Render at configured DPI (default 250 for good OCR quality)
        mat = fitz.Matrix(250 / 72, 250 / 72)  # 72 DPI is base PDF resolution
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

        # Convert pixmap to PIL Image
        from PIL import Image
        import io
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        return self._ocr.extract_text_from_image(image)

    @staticmethod
    def _is_scanned_page(text: str) -> bool:
        """Heuristically determine if a page is scanned (image-only)."""
        if not text or len(text.strip()) < _MIN_DIGITAL_TEXT_CHARS:
            return True
        # Count printable vs total characters
        printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
        ratio = printable / len(text) if len(text) > 0 else 0.0
        return ratio < _MIN_PRINTABLE_RATIO

    @staticmethod
    def _clean_raw_text(text: str) -> str:
        """Remove invalid control characters and normalize unicode."""
        if not text:
            return ""
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        # Remove null bytes and other control chars (keep \n, \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text
