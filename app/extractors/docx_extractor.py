"""
DOCX extractor using python-docx.

Extracts paragraphs, headings, tables, numbered and bulleted lists
while preserving document structure and heading context.
"""
import logging
import re
import unicodedata
from typing import List, Optional

from app.extractors.base import BaseExtractor, ExtractionResult, ExtractedPage, ExtractionMethod
from app.core.exceptions import ExtractionError

logger = logging.getLogger("rag_engine.extractors.docx")


class DOCXExtractor(BaseExtractor):
    """Extract text from DOCX files, preserving structure."""

    @property
    def supported_extensions(self):
        return [".docx", ".doc"]

    def extract(self, file_path: str, document_id: str, filename: str) -> ExtractionResult:
        """Extract structured text from a DOCX file."""
        try:
            import docx
        except ImportError:
            raise ExtractionError("python-docx is not installed. Run: pip install python-docx")

        result = ExtractionResult(
            document_id=document_id,
            source_filename=filename,
            source_type="DOCX",
            primary_extraction_method=ExtractionMethod.DOCX,
        )

        try:
            doc = docx.Document(file_path)
        except Exception as exc:
            raise ExtractionError(f"Cannot open DOCX '{filename}': {exc}")

        # Extract document body
        sections = self._extract_sections(doc, result)
        result.total_pages = len(sections)

        for i, section_data in enumerate(sections):
            page = ExtractedPage(
                page_number=i + 1,
                heading=section_data["heading"],
                text=section_data["text"],
                section=section_data["section"],
                extraction_method=ExtractionMethod.DOCX,
                is_empty=not section_data["text"].strip(),
            )
            result.pages.append(page)

        # Also process tables that are not inline
        self._process_tables(doc, result)

        logger.info(
            "DOCX '%s': %d sections extracted, %d errors.",
            filename, len(sections), len(result.errors),
        )
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _extract_sections(self, doc, result: ExtractionResult) -> List[dict]:
        """
        Walk document paragraphs and group them by heading sections.
        Each heading starts a new section.
        """
        sections = []
        current_heading: Optional[str] = None
        current_section_name: Optional[str] = None
        current_parts: List[str] = []

        for para in doc.paragraphs:
            try:
                style_name = para.style.name if para.style else ""
                text = self._clean_text(para.text)

                if not text:
                    continue

                is_heading = style_name.startswith("Heading")

                if is_heading:
                    # Save previous section
                    if current_parts:
                        sections.append({
                            "heading": current_heading,
                            "section": current_section_name,
                            "text": "\n".join(current_parts),
                        })
                    # Start new section
                    current_heading = text
                    current_section_name = text
                    current_parts = []
                else:
                    # Format list items
                    if style_name.startswith("List"):
                        prefix = self._get_list_prefix(para)
                        current_parts.append(f"{prefix}{text}")
                    else:
                        current_parts.append(text)

            except Exception as exc:
                result.add_error(None, "PARAGRAPH_EXTRACTION_FAILURE", str(exc))

        # Flush last section
        if current_parts:
            sections.append({
                "heading": current_heading,
                "section": current_section_name,
                "text": "\n".join(current_parts),
            })

        # If no sections found (no headings), treat whole document as one section
        if not sections:
            all_text = "\n".join(
                self._clean_text(p.text) for p in doc.paragraphs if p.text.strip()
            )
            if all_text.strip():
                sections.append({"heading": None, "section": None, "text": all_text})

        return sections

    def _process_tables(self, doc, result: ExtractionResult) -> None:
        """Extract tables and append them as additional pages."""
        for table_index, table in enumerate(doc.tables):
            try:
                table_text = self._extract_table(table)
                if table_text.strip():
                    page = ExtractedPage(
                        page_number=result.total_pages + table_index + 1,
                        heading=f"Table {table_index + 1}",
                        text=table_text,
                        section="Tables",
                        extraction_method=ExtractionMethod.DOCX,
                    )
                    result.pages.append(page)
            except Exception as exc:
                result.add_error(None, "TABLE_EXTRACTION_FAILURE",
                                 f"Table {table_index + 1}: {exc}")

    def _extract_table(self, table) -> str:
        """Convert a docx table to readable text."""
        rows = []
        for row in table.rows:
            cells = [self._clean_text(cell.text) for cell in row.cells]
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    @staticmethod
    def _get_list_prefix(para) -> str:
        """Return bullet/number prefix based on paragraph style."""
        style_name = para.style.name if para.style else ""
        if "Number" in style_name or "Ordered" in style_name:
            return "• "  # We use bullet for both; number ordering is in original doc
        return "• "

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize unicode and remove control characters."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()
