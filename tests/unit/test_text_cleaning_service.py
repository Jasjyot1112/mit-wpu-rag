"""
Unit tests for TextCleaningService.
"""

import pytest
from app.services.text_cleaning_service import TextCleaningService


class TestTextCleaningService:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cleaner = TextCleaningService()

    def test_replaces_ligatures(self):
        raw = "ef\uFB01cient algorithm with \uFB02oating point and \uFB00"
        cleaned = self.cleaner.clean(raw)
        assert "efficient" in cleaned
        assert "floating" in cleaned
        assert "ff" in cleaned

    def test_removes_invalid_control_characters(self):
        raw = "Distributed\x00\x08Systems\x0bUnit\x0c"
        cleaned = self.cleaner.clean(raw)
        assert cleaned == "DistributedSystemsUnit"

    def test_normalizes_whitespace(self):
        raw = "Line 1    has   many    spaces.\n\n\n\nLine 2 after gaps."
        cleaned = self.cleaner.clean(raw)
        assert "Line 1 has many spaces." in cleaned
        assert "\n\nLine 2 after gaps." in cleaned

    def test_fixes_ocr_hyphenation(self):
        raw = "distrib-\nuted computing architecture"
        cleaned = self.cleaner.clean(raw)
        assert "distributed computing architecture" in cleaned

    def test_preserves_technical_terms_and_code(self):
        raw = "Let x = y * 2 + (z / 3); while(x > 0) { x--; }"
        cleaned = self.cleaner.clean(raw)
        assert "Let x = y * 2 + (z / 3);" in cleaned
        assert "while(x > 0)" in cleaned

    def test_strips_standalone_page_numbers(self):
        raw = "Introduction to Algorithms\n\nPage 45\n\nBinary Search Trees"
        cleaned = self.cleaner.clean(raw)
        assert "Page 45" not in cleaned
        assert "Introduction to Algorithms" in cleaned
        assert "Binary Search Trees" in cleaned

    def test_clean_for_embedding_truncates_safely(self):
        raw = "word " * 3000
        truncated = self.cleaner.clean_for_embedding(raw, max_chars=1000)
        assert len(truncated) <= 1000
        assert not truncated.endswith(" ")
