"""
PPTX extractor — slide-level extraction with OCR for embedded images.

For each slide, extracts:
- Slide title (from title placeholder or first text shape)
- All text content: paragraphs, bullet points, text boxes
- Tables (converted to readable text)
- Notes (if present)
- Embedded images (OCR applied to extract any text)
- Slide number is preserved throughout

Every extracted passage keeps its slide_number so citations are always accurate.
"""
import io
import logging
import re
import unicodedata
from typing import List, Optional, Tuple

from app.extractors.base import BaseExtractor, ExtractionResult, ExtractedPage, ExtractionMethod
from app.core.exceptions import ExtractionError

logger = logging.getLogger("rag_engine.extractors.pptx")


class PPTXExtractor(BaseExtractor):
    """Extract text from PowerPoint presentations at the slide level."""

    def __init__(self, ocr_service=None):
        """
        Args:
            ocr_service: Optional OCRService for image-based text extraction.
        """
        self._ocr = ocr_service

    @property
    def supported_extensions(self):
        return [".pptx", ".ppt"]

    def extract(self, file_path: str, document_id: str, filename: str) -> ExtractionResult:
        """Extract text from every slide in the presentation."""
        try:
            from pptx import Presentation
            from pptx.util import Pt
        except ImportError:
            raise ExtractionError("python-pptx is not installed. Run: pip install python-pptx")

        result = ExtractionResult(
            document_id=document_id,
            source_filename=filename,
            source_type="PPTX",
            primary_extraction_method=ExtractionMethod.SLIDE,
        )

        try:
            prs = Presentation(file_path)
        except Exception as exc:
            raise ExtractionError(f"Cannot open PPTX '{filename}': {exc}")

        total_slides = len(prs.slides)
        result.total_slides = total_slides
        result.total_pages = total_slides

        for slide_index, slide in enumerate(prs.slides):
            slide_num = slide_index + 1
            extracted = self._extract_slide(slide, slide_num, filename, result)
            result.pages.append(extracted)

        logger.info(
            "PPTX '%s': %d slides extracted (%d errors).",
            filename, total_slides, len(result.errors),
        )
        return result

    # ------------------------------------------------------------------
    # Slide extraction
    # ------------------------------------------------------------------

    def _extract_slide(self, slide, slide_num: int, filename: str, result: ExtractionResult) -> ExtractedPage:
        """Extract all content from a single slide."""
        try:
            title = self._get_slide_title(slide)
            text_parts: List[str] = []
            image_texts: List[str] = []

            for shape in slide.shapes:
                try:
                    shape_text = self._extract_shape(shape, image_texts)
                    if shape_text.strip():
                        # Avoid duplicating the title in body text
                        if shape_text.strip() != (title or "").strip():
                            text_parts.append(shape_text.strip())
                except Exception as exc:
                    result.add_error(slide_num, "SHAPE_EXTRACTION_FAILURE",
                                     f"Shape error on slide {slide_num}: {exc}")

            # Notes
            notes_text = self._get_notes(slide)

            # Build final text: title + body + notes
            combined_parts = []
            if title:
                combined_parts.append(f"[Title] {title}")
            combined_parts.extend(text_parts)
            if notes_text:
                combined_parts.append(f"[Notes] {notes_text}")

            full_text = "\n\n".join(combined_parts)
            full_text = self._clean_text(full_text)

            # Decide extraction method
            method = ExtractionMethod.OCR if (not text_parts and image_texts) else ExtractionMethod.SLIDE

            # Append image texts to the slide content
            if image_texts:
                image_section = "\n".join(f"[Image Text] {t}" for t in image_texts if t.strip())
                if image_section:
                    full_text = (full_text + "\n\n" + image_section).strip()

            return ExtractedPage(
                slide_number=slide_num,
                page_number=slide_num,
                heading=title,
                text=full_text,
                extraction_method=method,
                is_empty=not full_text.strip(),
                image_texts=image_texts,
            )

        except Exception as exc:
            result.add_error(slide_num, "SLIDE_EXTRACTION_FAILURE", str(exc))
            logger.error("Failed to extract slide %d of '%s': %s", slide_num, filename, exc)
            return ExtractedPage(
                slide_number=slide_num,
                page_number=slide_num,
                is_empty=True,
                error=str(exc),
                extraction_method=ExtractionMethod.SLIDE,
            )

    def _get_slide_title(self, slide) -> Optional[str]:
        """Return slide title from title placeholder or first text shape."""
        try:
            if slide.shapes.title and slide.shapes.title.has_text_frame:
                title = slide.shapes.title.text_frame.text.strip()
                if title:
                    return title
        except Exception:
            pass

        # Fallback: first shape with text
        try:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text = shape.text_frame.text.strip()
                    if text:
                        return text[:200]  # Cap title length
        except Exception:
            pass

        return None

    def _extract_shape(self, shape, image_texts: List[str]) -> str:
        """Extract text from a single shape (text frame, table, picture, group)."""
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        # Text frames
        if shape.has_text_frame:
            return self._extract_text_frame(shape.text_frame)

        # Tables
        if shape.has_table:
            return self._extract_table(shape.table)

        # Pictures / images → OCR
        try:
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                ocr_text = self._ocr_shape_image(shape)
                if ocr_text:
                    image_texts.append(ocr_text)
                return ""
        except Exception:
            pass

        # Group shapes (recursive)
        try:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                parts = []
                for child in shape.shapes:
                    child_text = self._extract_shape(child, image_texts)
                    if child_text.strip():
                        parts.append(child_text)
                return "\n".join(parts)
        except Exception:
            pass

        return ""

    def _extract_text_frame(self, text_frame) -> str:
        """Extract all paragraphs and their runs from a text frame."""
        paragraphs = []
        for para in text_frame.paragraphs:
            para_text = para.text.strip()
            if para_text:
                paragraphs.append(para_text)
        return "\n".join(paragraphs)

    def _extract_table(self, table) -> str:
        """Convert a PPTX table to a text representation."""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _get_notes(self, slide) -> str:
        """Extract speaker notes from a slide."""
        try:
            if slide.has_notes_slide:
                notes_slide = slide.notes_slide
                notes_tf = notes_slide.notes_text_frame
                if notes_tf:
                    text = notes_tf.text.strip()
                    # Notes often contain "Click to edit..." placeholder text
                    if text and "click to edit" not in text.lower():
                        return text
        except Exception:
            pass
        return ""

    def _ocr_shape_image(self, shape) -> str:
        """Apply OCR to an image shape in a slide."""
        if self._ocr is None:
            return ""
        try:
            from PIL import Image
            img_bytes = shape.image.blob
            image = Image.open(io.BytesIO(img_bytes))
            text, _ = self._ocr.extract_text_from_image(image)
            return text.strip()
        except Exception as exc:
            logger.debug("Image OCR failed for shape: %s", exc)
            return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize unicode and strip control characters."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Collapse excessive blank lines (max 2 consecutive)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
