"""Tests for arabic-extract.

OCR engine tests are skipped gracefully when Tesseract / EasyOCR are absent.
PDF tests use a synthetic in-memory PDF built with reportlab (optional) or
fall back to testing the pipeline + repair logic directly.
"""
from __future__ import annotations

import io
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import arabic_rt as ar_rt
import arabic_repair as ar_repair
import arabic_extract as aocr
from arabic_extract._pipeline import run_pipeline, contamination_summary


# ---------------------------------------------------------------------------
# Pipeline (no OCR engine needed)
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_repair_fully_baked(self):
        baked = ar_rt.fix("مرحبا بالعالم")
        clean = run_pipeline(baked, normalize=False)
        assert clean == "مرحبا بالعالم"

    def test_repair_partial_shaping(self):
        # Use a longer sentence so arabic-repair's partial detection works cleanly
        shaped_word = ar_rt.shape("السلام")
        partial = shaped_word + " عليكم ورحمة الله"
        clean = run_pipeline(partial, normalize=False)
        assert "السلام" in clean
        assert "عليكم" in clean

    def test_noop_on_clean(self):
        text = "مرحبا بالعالم"
        assert run_pipeline(text, normalize=False) == text

    def test_noop_on_empty(self):
        assert run_pipeline("", normalize=False) == ""
        assert run_pipeline("   ", normalize=False) == "   "

    def test_noop_on_latin(self):
        text = "Hello World 123"
        assert run_pipeline(text, normalize=False) == text

    def test_normalize_removes_presentation_forms(self):
        baked = ar_rt.fix("مرحبا بالعالم")
        clean = run_pipeline(baked, normalize=True)
        pf = [c for c in clean if 0xFB50 <= ord(c) <= 0xFDFF or 0xFE70 <= ord(c) <= 0xFEFF]
        assert pf == []

    def test_contamination_summary_baked(self):
        baked = ar_rt.fix("مرحبا بالعالم")
        stats = contamination_summary([baked])
        assert stats["needed_repair"] is True
        assert stats["contaminated_ratio"] == 1.0

    def test_contamination_summary_clean(self):
        stats = contamination_summary(["مرحبا بالعالم"])
        assert stats["needed_repair"] is False
        assert stats["contaminated_ratio"] == 0.0

    def test_contamination_summary_empty(self):
        stats = contamination_summary([])
        assert stats["needed_repair"] is False


# ---------------------------------------------------------------------------
# extract() routing
# ---------------------------------------------------------------------------

class TestExtractRouting:
    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "file.docx"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported file type"):
            aocr.extract(f)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            aocr.extract(tmp_path / "nonexistent.pdf")

    def test_missing_image_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            aocr.extract(tmp_path / "nonexistent.jpg")


# ---------------------------------------------------------------------------
# ExtractResult
# ---------------------------------------------------------------------------

class TestExtractResult:
    def _make_result(self) -> aocr.ExtractResult:
        page = aocr.PageResult(
            page_number=1,
            text="مرحبا بالعالم",
            raw_text=ar_rt.fix("مرحبا بالعالم"),
            method="text_layer",
        )
        return aocr.ExtractResult(
            text="مرحبا بالعالم",
            pages=[page],
            source="test.pdf",
            contamination={"contaminated_ratio": 1.0, "needed_repair": True},
        )

    def test_repr_contains_pages(self):
        r = self._make_result()
        assert "pages=1" in repr(r)

    def test_text_is_accessible(self):
        r = self._make_result()
        assert r.text == "مرحبا بالعالم"

    def test_pages_list(self):
        r = self._make_result()
        assert len(r.pages) == 1
        assert r.pages[0].method == "text_layer"


# ---------------------------------------------------------------------------
# PDF extraction (mocked pdfplumber)
# ---------------------------------------------------------------------------

class TestPDFExtraction:
    def _make_mock_pdf(self, pages_text: list[str]):
        """Build a mock pdfplumber PDF with given per-page text."""
        mock_pdf = MagicMock()
        mock_pages = []
        for i, text in enumerate(pages_text, start=1):
            page = MagicMock()
            page.page_number = i
            page.extract_text.return_value = text
            mock_pages.append(page)
        mock_pdf.pages = mock_pages
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)
        return mock_pdf

    def test_text_layer_pdf_repaired(self, tmp_path):
        baked = ar_rt.fix("مرحبا بالعالم")
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_pdf = self._make_mock_pdf([baked])

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = aocr.extract_pdf(fake_pdf, normalize=False)

        assert result.pages[0].method == "text_layer"
        assert result.text == "مرحبا بالعالم"

    def test_multipage_pdf(self, tmp_path):
        page1 = ar_rt.fix("السلام عليكم")
        page2 = ar_rt.fix("مرحبا بالعالم")
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_pdf = self._make_mock_pdf([page1, page2])

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = aocr.extract_pdf(fake_pdf, normalize=False)

        assert len(result.pages) == 2
        assert "السلام عليكم" in result.text
        assert "مرحبا بالعالم" in result.text

    def test_empty_page_marked_correctly(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        # Page with no text — scanned, but ocr_scanned=False so we skip
        mock_pdf = self._make_mock_pdf([""])

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = aocr.extract_pdf(fake_pdf, ocr_scanned=False, normalize=False)

        assert result.pages[0].method == "text_layer_empty"

    def test_contamination_stats_populated(self, tmp_path):
        baked = ar_rt.fix("مرحبا بالعالم")
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_pdf = self._make_mock_pdf([baked])

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = aocr.extract_pdf(fake_pdf, normalize=False)

        assert result.contamination["needed_repair"] is True

    def test_clean_pdf_no_repair_needed(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_pdf = self._make_mock_pdf(["مرحبا بالعالم"])

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = aocr.extract_pdf(fake_pdf, normalize=False)

        assert result.contamination["needed_repair"] is False
        assert result.text == "مرحبا بالعالم"


# ---------------------------------------------------------------------------
# Image OCR (mocked OCR engine)
# ---------------------------------------------------------------------------

class TestImageOCR:
    def test_image_ocr_result_repaired(self, tmp_path):
        """Mock the OCR engine returning baked Arabic — repair must fix it."""
        baked = ar_rt.fix("مرحبا بالعالم")
        fake_img = tmp_path / "scan.jpg"
        fake_img.write_bytes(b"fake image bytes")

        with patch("arabic_extract._image.ocr_image", return_value=baked):
            result = aocr.extract_image(fake_img, normalize=False)

        assert result.text == "مرحبا بالعالم"
        assert result.pages[0].method == "image_ocr"

    def test_image_ocr_latin_preserved(self, tmp_path):
        baked = ar_rt.fix("Hello مرحبا World")
        fake_img = tmp_path / "scan.png"
        fake_img.write_bytes(b"fake image bytes")

        with patch("arabic_extract._image.ocr_image", return_value=baked):
            result = aocr.extract_image(fake_img, normalize=False)

        assert "Hello" in result.text
        assert "World" in result.text
        assert "مرحبا" in result.text

    def test_image_contamination_stats(self, tmp_path):
        baked = ar_rt.fix("مرحبا بالعالم")
        fake_img = tmp_path / "scan.jpg"
        fake_img.write_bytes(b"fake image bytes")

        with patch("arabic_extract._image.ocr_image", return_value=baked):
            result = aocr.extract_image(fake_img, normalize=False)

        assert result.contamination["needed_repair"] is True

    def test_extract_routes_to_image(self, tmp_path):
        baked = ar_rt.fix("مرحبا")
        fake_img = tmp_path / "scan.jpeg"
        fake_img.write_bytes(b"fake")

        with patch("arabic_extract._image.ocr_image", return_value=baked):
            result = aocr.extract(fake_img, normalize=False)

        assert result.text == "مرحبا"


# ---------------------------------------------------------------------------
# OCR engine availability (live — skipped if not installed)
# ---------------------------------------------------------------------------

class TestLiveOCREngine:
    @pytest.fixture(autouse=True)
    def require_any_engine(self):
        from arabic_extract._image import available_engine
        if available_engine() is None:
            pytest.skip("No OCR engine installed (tesseract or easyocr)")

    def test_available_engine_returns_string(self):
        from arabic_extract._image import available_engine
        eng = available_engine()
        assert eng in ("tesseract", "easyocr")
