# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""arabic-ocr — clean Arabic text extraction from PDFs and images.

Combines PDF text extraction (pdfplumber), image OCR (Tesseract / EasyOCR),
and arabic-repair into a single pipeline that returns clean logical-order
Arabic text ready for NLP, LLMs, and search.

Quick start::

    import arabic_ocr as aocr

    # From a PDF (text-layer or scanned — auto-detected per page)
    result = aocr.extract("document.pdf")
    print(result.text)           # clean Arabic, all pages joined
    print(result.pages)          # per-page breakdown

    # From a scanned image
    result = aocr.extract("scan.jpg")
    print(result.text)

    # Chain into CAMeL Tools (arabic-ocr already does this by default)
    result = aocr.extract("document.pdf", normalize=True)

Install extras::

    pip install arabic-ocr[pdf]         # PDF text extraction only
    pip install arabic-ocr[tesseract]   # + Tesseract OCR (needs binary)
    pip install arabic-ocr[easyocr]     # + EasyOCR (pure Python)
    pip install arabic-ocr[pymupdf]     # + PyMuPDF for scanned PDF rendering
    pip install arabic-ocr[all]         # everything
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._pipeline import run_pipeline, contamination_summary
from ._image import available_engine


@dataclass
class PageResult:
    """Extraction result for one page / one image."""
    page_number: int     # 1-based; always 1 for images
    text: str            # clean repaired text
    raw_text: str        # text before repair (for debugging)
    method: str          # "text_layer" | "ocr" | "image_ocr"


@dataclass
class ExtractResult:
    """Full extraction result for a document or image.

    Attributes
    ----------
    text:
        All pages joined by double newline — clean, logical-order Arabic.
    pages:
        Per-page breakdown.
    source:
        Path of the input file.
    contamination:
        Dict with repair stats: how many Arabic words needed fixing.
    """
    text: str
    pages: list[PageResult]
    source: str
    contamination: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        ratio = self.contamination.get("contaminated_ratio", 0)
        return (
            f"ExtractResult(pages={len(self.pages)}, "
            f"chars={len(self.text)}, "
            f"repaired={ratio:.0%})"
        )


def extract(
    path: str | Path,
    engine: str = "auto",
    normalize: bool = True,
    ocr_scanned: bool = True,
) -> ExtractResult:
    """Extract clean Arabic text from a PDF or image file.

    Auto-detects the file type. For PDFs, tries the text layer first and
    falls back to OCR for scanned pages. Applies arabic-repair to every
    page before returning.

    Parameters
    ----------
    path:
        Path to a PDF or image file (JPEG, PNG, TIFF, BMP, WEBP).
    engine:
        OCR engine: ``"tesseract"``, ``"easyocr"``, or ``"auto"``.
    normalize:
        Apply Unicode normalization (NFKC / CAMeL Tools) after repair.
    ocr_scanned:
        For PDFs: OCR pages that have no text layer. Set False for
        text-layer-only PDFs (faster).

    Returns
    -------
    ExtractResult

    Raises
    ------
    ValueError
        If the file extension is not a supported PDF or image format.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    PDF_EXTS   = {".pdf"}
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}

    if suffix in PDF_EXTS:
        return _extract_pdf(path, engine=engine, normalize=normalize, ocr_scanned=ocr_scanned)
    elif suffix in IMAGE_EXTS:
        return _extract_image(path, engine=engine, normalize=normalize)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix!r}. "
            f"Supported: PDF ({', '.join(sorted(PDF_EXTS))}) "
            f"and images ({', '.join(sorted(IMAGE_EXTS))})"
        )


def extract_pdf(
    path: str | Path,
    engine: str = "auto",
    normalize: bool = True,
    ocr_scanned: bool = True,
) -> ExtractResult:
    """Extract from a PDF explicitly (same as extract() for .pdf files)."""
    return _extract_pdf(Path(path), engine=engine, normalize=normalize, ocr_scanned=ocr_scanned)


def extract_image(
    path: str | Path,
    engine: str = "auto",
    normalize: bool = True,
) -> ExtractResult:
    """Extract from an image file explicitly."""
    return _extract_image(Path(path), engine=engine, normalize=normalize)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _extract_pdf(
    path: Path,
    engine: str,
    normalize: bool,
    ocr_scanned: bool,
) -> ExtractResult:
    from ._pdf import extract_pdf as _pdf_pages

    raw_pages = _pdf_pages(path, ocr_engine=engine, ocr_scanned=ocr_scanned)

    pages: list[PageResult] = []
    raw_texts: list[str] = []

    for rp in raw_pages:
        clean = run_pipeline(rp.raw_text, normalize=normalize)
        pages.append(PageResult(
            page_number=rp.page_number,
            text=clean,
            raw_text=rp.raw_text,
            method=rp.method,
        ))
        raw_texts.append(rp.raw_text)

    full_text = "\n\n".join(p.text for p in pages if p.text.strip())
    stats = contamination_summary(raw_texts)

    return ExtractResult(
        text=full_text,
        pages=pages,
        source=str(path),
        contamination=stats,
    )


def _extract_image(path: Path, engine: str, normalize: bool) -> ExtractResult:
    from ._image import ocr_image

    raw = ocr_image(path, engine=engine)
    clean = run_pipeline(raw, normalize=normalize)

    page = PageResult(
        page_number=1,
        text=clean,
        raw_text=raw,
        method="image_ocr",
    )
    stats = contamination_summary([raw])

    return ExtractResult(
        text=clean,
        pages=[page],
        source=str(path),
        contamination=stats,
    )


__version__ = "0.1.0"
__author__ = "Bandar AlSwyan"
__license__ = "MPL-2.0"

__all__ = [
    "extract",
    "extract_pdf",
    "extract_image",
    "ExtractResult",
    "PageResult",
    "__version__",
]
