"""
Unit tests for PDF, PPTX, and DOCX Extractors using dynamically generated files.
"""

import os
import tempfile
import pytest

from app.extractors.pdf_extractor import PDFExtractor
from app.extractors.pptx_extractor import PPTXExtractor
from app.extractors.docx_extractor import DOCXExtractor
from app.extractors.base import ExtractionMethod


class TestExtractors:
    def test_pptx_extraction(self):
        from pptx import Presentation
        from pptx.util import Inches

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "sample.pptx")
            prs = Presentation()

            # Slide 1: Title & Content
            slide_layout = prs.slide_layouts[0]
            slide1 = prs.slides.add_slide(slide_layout)
            slide1.shapes.title.text = "Distributed Systems Unit 2"
            slide1.placeholders[1].text = "Bully Election Algorithm Overview"

            # Slide 2: Bullet points
            slide_layout2 = prs.slide_layouts[1]
            slide2 = prs.slides.add_slide(slide_layout2)
            slide2.shapes.title.text = "Bully Algorithm Details"
            slide2.placeholders[1].text = "Coordinator crash detection\nHighest ID election"

            prs.save(file_path)

            extractor = PPTXExtractor()
            result = extractor.extract(
                file_path=file_path,
                document_id="doc_pptx_01",
                filename="sample.pptx",
            )

            assert result.source_type == "PPTX"
            assert result.total_slides == 2
            assert len(result.pages) == 2

            # Verify slide 1
            s1 = result.pages[0]
            assert s1.slide_number == 1
            assert s1.heading == "Distributed Systems Unit 2"
            assert "Bully Election Algorithm Overview" in s1.text

            # Verify slide 2
            s2 = result.pages[1]
            assert s2.slide_number == 2
            assert s2.heading == "Bully Algorithm Details"
            assert "Highest ID election" in s2.text

    def test_pdf_extraction(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "sample.pdf")
            doc = fitz.open()

            # Page 1
            page1 = doc.new_page()
            page1.insert_text(
                fitz.Point(50, 72),
                "Chapter 1: Cloud Computing Fundamentals and Architectures.\n"
                "Cloud computing provides on-demand availability of computer system resources.",
            )

            # Page 2
            page2 = doc.new_page()
            page2.insert_text(
                fitz.Point(50, 72),
                "Chapter 2: Virtualization Technologies.\n"
                "Hypervisors decouple operating systems from underlying physical hardware.",
            )

            doc.save(file_path)
            doc.close()

            extractor = PDFExtractor(ocr_service=None)
            result = extractor.extract(
                file_path=file_path,
                document_id="doc_pdf_01",
                filename="sample.pdf",
            )

            assert result.source_type == "PDF"
            assert result.total_pages == 2
            assert len(result.pages) == 2

            p1 = result.pages[0]
            assert p1.page_number == 1
            assert "Cloud Computing Fundamentals" in p1.text
            assert p1.extraction_method == ExtractionMethod.DIGITAL

            p2 = result.pages[1]
            assert p2.page_number == 2
            assert "Virtualization Technologies" in p2.text

    def test_docx_extraction(self):
        import docx

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "sample.docx")
            doc = docx.Document()

            doc.add_heading("Operating Systems Lecture Notes", level=1)
            doc.add_paragraph("Process scheduling determines which process runs when CPU is free.")

            doc.add_heading("Round Robin Scheduling", level=2)
            doc.add_paragraph("Time quantum allocation allows fair CPU sharing among ready processes.")

            doc.save(file_path)

            extractor = DOCXExtractor()
            result = extractor.extract(
                file_path=file_path,
                document_id="doc_docx_01",
                filename="sample.docx",
            )

            assert result.source_type == "DOCX"
            assert len(result.pages) >= 1
            headings = [p.heading for p in result.pages]
            assert "Round Robin Scheduling" in headings or "Operating Systems Lecture Notes" in headings
            all_text = " ".join(p.text for p in result.pages)
            assert "Process scheduling" in all_text
