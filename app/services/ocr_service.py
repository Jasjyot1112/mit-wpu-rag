"""
OCR Service using Tesseract via pytesseract.

Responsibilities:
- Extract text from PIL Image objects.
- Return text and confidence score.
- Configure Tesseract path and language from settings.
- Handle Tesseract not found gracefully.
"""
import logging
from typing import Optional, Tuple

from app.core.exceptions import ExtractionError

logger = logging.getLogger("rag_engine.services.ocr")

_tesseract_configured = False


def _configure_tesseract(cmd_path: str) -> None:
    """Set pytesseract executable path once."""
    global _tesseract_configured
    if _tesseract_configured:
        return
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd_path
        _tesseract_configured = True
        logger.info("Tesseract configured at: %s", cmd_path)
    except ImportError:
        raise ExtractionError("pytesseract is not installed. Run: pip install pytesseract")


class OCRService:
    """Wrapper around Tesseract OCR for image-to-text extraction."""

    def __init__(self, tesseract_cmd: str, language: str = "eng", dpi: int = 250):
        """
        Args:
            tesseract_cmd: Full path to tesseract.exe (Windows).
            language: Tesseract language code (default "eng").
            dpi: Image DPI used during rendering (affects OCR quality).
        """
        self._cmd = tesseract_cmd
        self._language = language
        self._dpi = dpi
        _configure_tesseract(tesseract_cmd)

    def is_available(self) -> bool:
        """Check whether Tesseract is accessible."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_text_from_image(self, image) -> Tuple[str, Optional[float]]:
        """
        Extract text from a PIL Image.

        Returns:
            (text, confidence) where confidence is 0.0–100.0 or None.
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise ExtractionError(f"OCR dependency missing: {exc}")

        _configure_tesseract(self._cmd)

        if not isinstance(image, Image.Image):
            raise ExtractionError("Image must be a PIL Image object.")

        # Convert to RGB if needed (Tesseract works best with RGB)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        try:
            # Get text with confidence data
            data = pytesseract.image_to_data(
                image,
                lang=self._language,
                output_type=pytesseract.Output.DICT,
                config="--psm 3",  # Fully automatic page segmentation
            )

            # Build text from words with confidence > 0
            words = []
            confidences = []
            for i, word in enumerate(data["text"]):
                conf = data["conf"][i]
                if isinstance(conf, (int, float)) and conf > 0 and word.strip():
                    words.append(word)
                    confidences.append(float(conf))

            text = " ".join(words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else None

            return text, avg_confidence

        except Exception as exc:
            logger.error("Tesseract OCR failed: %s", exc)
            # Fallback: simple text extraction without confidence
            try:
                text = pytesseract.image_to_string(image, lang=self._language)
                return text.strip(), None
            except Exception as exc2:
                raise ExtractionError(f"OCR extraction failed: {exc2}")

    def extract_text_from_image_path(self, image_path: str) -> Tuple[str, Optional[float]]:
        """Extract text from an image file path."""
        try:
            from PIL import Image
            image = Image.open(image_path)
            return self.extract_text_from_image(image)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Cannot open image '{image_path}': {exc}")
