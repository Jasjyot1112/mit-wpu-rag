"""
TextCleaningService — reusable text normalization for educational content.

Preserves:
- Mathematical expressions, code snippets, technical terms
- Numbered lists, definitions, academic terminology
- Important punctuation

Removes:
- Repeated headers/footers (heuristic)
- Extraction artifacts (ligatures, hyphenation artifacts)
- Invalid control characters, excessive whitespace
"""
import logging
import re
import unicodedata
from typing import List, Optional

logger = logging.getLogger("rag_engine.services.text_cleaning")

# Ligature map for common OCR/extraction artifacts
_LIGATURES = {
    "\uFB00": "ff", "\uFB01": "fi", "\uFB02": "fl",
    "\uFB03": "ffi", "\uFB04": "ffl", "\uFB05": "st", "\uFB06": "st",
    "\u2013": "-", "\u2014": "--",  # en-dash, em-dash
    "\u2018": "'", "\u2019": "'",  # smart quotes
    "\u201C": '"', "\u201D": '"',
    "\u2026": "...",              # ellipsis
    "\u00AD": "",                 # soft hyphen
}


class TextCleaningService:
    """Clean and normalize extracted document text."""

    def __init__(
        self,
        min_line_length: int = 2,
        max_header_repetitions: int = 3,
        remove_page_numbers: bool = True,
    ):
        self._min_line_len = min_line_length
        self._max_header_reps = max_header_repetitions
        self._remove_page_numbers = remove_page_numbers

    def clean(self, text: str, source_type: str = "PDF") -> str:
        """
        Main cleaning pipeline.

        Args:
            text: Raw extracted text.
            source_type: Document type hint (PDF, PPTX, DOCX) for type-specific rules.

        Returns:
            Cleaned text string.
        """
        if not text or not text.strip():
            return ""

        text = self._replace_ligatures(text)
        text = self._normalize_unicode(text)
        text = self._remove_control_chars(text)
        text = self._fix_hyphenation(text)
        text = self._normalize_whitespace(text)

        if source_type == "PDF":
            text = self._remove_pdf_artifacts(text)

        text = self._remove_repeated_short_lines(text)

        if self._remove_page_numbers:
            text = self._strip_standalone_page_numbers(text)

        text = self._normalize_whitespace(text)
        return text.strip()

    def clean_for_embedding(self, text: str, max_chars: int = 8000) -> str:
        """Clean text and truncate to safe embedding input length."""
        cleaned = self.clean(text)
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
            # Don't cut mid-word
            last_space = cleaned.rfind(" ")
            if last_space > max_chars // 2:
                cleaned = cleaned[:last_space]
        return cleaned

    def clean_batch(self, texts: List[str], source_type: str = "PDF") -> List[str]:
        """Clean a list of texts."""
        return [self.clean(t, source_type) for t in texts]

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    @staticmethod
    def _replace_ligatures(text: str) -> str:
        """Replace typographic ligatures and special characters."""
        for char, replacement in _LIGATURES.items():
            text = text.replace(char, replacement)
        return text

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize to NFKC (compatibility decomposition, then canonical composition)."""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """Remove invalid control characters but preserve tab and newline."""
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]", "", text)

    @staticmethod
    def _fix_hyphenation(text: str) -> str:
        """
        Rejoin words broken with a hyphen at end of line (common OCR artifact).
        Example: "distrib-\nuted" → "distributed"
        """
        # Only join if the hyphen is at end of line and next line starts lowercase
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize runs of spaces/tabs; keep single newlines and double newlines."""
        # Collapse multiple spaces/tabs into one
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ consecutive newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove trailing spaces on each line
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return text

    @staticmethod
    def _remove_pdf_artifacts(text: str) -> str:
        """Remove common PDF extraction artifacts."""
        # Remove form feed characters
        text = text.replace("\x0c", "\n")
        # Remove sequences of repeated dashes (table borders)
        text = re.sub(r"-{10,}", "---", text)
        # Remove sequences of repeated dots (table of contents leaders)
        text = re.sub(r"\.{5,}", "...", text)
        # Remove sequences of underscores (form fields)
        text = re.sub(r"_{5,}", "___", text)
        return text

    def _remove_repeated_short_lines(self, text: str) -> str:
        """
        Remove lines that appear verbatim many times (likely repeated headers/footers).
        Only removes lines shorter than 80 chars to avoid removing content.
        """
        lines = text.split("\n")
        # Count occurrences of each short line
        from collections import Counter
        short_lines = [l.strip() for l in lines if 0 < len(l.strip()) < 80]
        counts = Counter(short_lines)

        # Identify lines to remove (appear more than threshold times)
        to_remove = {
            line for line, count in counts.items()
            if count > self._max_header_reps and len(line) < 60
        }

        if not to_remove:
            return text

        filtered = []
        removed_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped in to_remove:
                removed_count += 1
                # Keep the first occurrence
                to_remove.discard(stripped)
            else:
                filtered.append(line)

        if removed_count > 0:
            logger.debug("Removed %d repeated header/footer lines.", removed_count)

        return "\n".join(filtered)

    @staticmethod
    def _strip_standalone_page_numbers(text: str) -> str:
        """Remove lines that contain only a page number."""
        # Patterns: standalone numbers, "Page 5", "- 5 -", "5 of 100"
        patterns = [
            r"^\s*\d+\s*$",             # bare number
            r"^\s*[Pp]age\s+\d+\s*$",  # "Page 5"
            r"^\s*-\s*\d+\s*-\s*$",    # "- 5 -"
            r"^\s*\d+\s+of\s+\d+\s*$", # "5 of 100"
        ]
        combined = re.compile("|".join(patterns))
        lines = text.split("\n")
        return "\n".join(line for line in lines if not combined.match(line))
