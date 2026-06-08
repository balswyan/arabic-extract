# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""PDF extraction — text layer first, image OCR fallback for scanned pages.

Strategy per page
-----------------
1. Try pdfplumber to extract the text layer.
2. If the page yields enough text (> MIN_CHARS_PER_PAGE) → use it.
   The text may be in visual order with presentation forms; arabic-repair
   handles that downstream.
3. If the page is empty or near-empty → the PDF is scanned.
   Render the page as a PNG image and pass it to the OCR backend.

This means a single PDF can have mixed pages — some with a text layer
and some scanned — and we handle each correctly.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

# Minimum characters extracted from a page to consider it "has a text layer".
# Below this threshold we assume the page is scanned and OCR it instead.
MIN_CHARS_PER_PAGE = 8


@dataclass
class PageResult:
    """Extraction result for one PDF page."""
    page_number: int          # 1-based
    raw_text: str             # text before repair
    method: str               # "text_layer" | "ocr"


def _render_page_to_image(page) -> bytes:
    """Render a pdfplumber page to PNG bytes (requires pdf2image or pymupdf)."""
    # Try pymupdf (fitz) first — faster and no poppler dependency
    try:
        import fitz  # PyMuPDF
        # page.pdf is the parent pdfplumber PDF; we need the page index
        doc = fitz.open(page.pdf.stream.name)
        fitz_page = doc[page.page_number - 1]
        mat = fitz.Matrix(2.0, 2.0)   # 2× zoom = ~150 dpi effective
        pix = fitz_page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    except (ImportError, Exception):
        pass

    # Fall back to pdf2image (requires poppler)
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(
            page.pdf.stream.name,
            first_page=page.page_number,
            last_page=page.page_number,
            dpi=150,
        )
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        return buf.getvalue()
    except (ImportError, Exception):
        pass

    raise ImportError(
        "Cannot render scanned PDF page to image.\n"
        "Install one of:\n"
        "  pip install arabic-ocr[pymupdf]    (recommended, no system deps)\n"
        "  pip install arabic-ocr[pdf2image]  (requires poppler)"
    )


def extract_pdf(
    pdf_path: str | Path,
    ocr_engine: str = "auto",
    ocr_scanned: bool = True,
) -> list[PageResult]:
    """Extract Arabic text from every page of a PDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    ocr_engine:
        OCR engine to use for scanned pages: ``"tesseract"``, ``"easyocr"``,
        or ``"auto"`` (tries tesseract then easyocr).
    ocr_scanned:
        If False, skip OCR for scanned pages and return empty string for them.
        Useful for text-layer-only PDFs where speed matters.

    Returns
    -------
    list[PageResult]
        One PageResult per page in the PDF.

    Raises
    ------
    ImportError
        If pdfplumber is not installed.
    FileNotFoundError
        If the PDF does not exist.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF extraction.\n"
            "Install it with: pip install arabic-ocr[pdf]\n"
            "or: pip install arabic-ocr[all]"
        )

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    from ._image import ocr_image

    results: list[PageResult] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            raw = raw.strip()

            if len(raw) >= MIN_CHARS_PER_PAGE:
                results.append(PageResult(
                    page_number=i,
                    raw_text=raw,
                    method="text_layer",
                ))
            elif ocr_scanned:
                # Render page to image and OCR
                try:
                    img_bytes = _render_page_to_image(page)
                    # Write to a temp file for the OCR engine
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.write(img_bytes)
                        tmp_path = tmp.name
                    try:
                        raw = ocr_image(tmp_path, engine=ocr_engine)
                    finally:
                        os.unlink(tmp_path)
                except Exception as e:
                    raw = ""  # degrade gracefully; caller can inspect method
                    results.append(PageResult(
                        page_number=i,
                        raw_text=raw,
                        method=f"ocr_failed:{e}",
                    ))
                    continue

                results.append(PageResult(
                    page_number=i,
                    raw_text=raw,
                    method="ocr",
                ))
            else:
                results.append(PageResult(
                    page_number=i,
                    raw_text=raw,
                    method="text_layer_empty",
                ))

    return results
