# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Image OCR backend — wraps Tesseract (pytesseract) or EasyOCR.

Both backends output text that may contain presentation forms depending on
their Arabic language packs.  The result always goes through arabic-repair.

Priority order:
  1. pytesseract  — faster, lighter, needs Tesseract binary installed
  2. easyocr      — pure Python, larger download, no binary needed
  3. ImportError raised with install instructions if neither is available
"""
from __future__ import annotations

from pathlib import Path


def _ocr_tesseract(image_path: str | Path) -> str:
    """OCR via pytesseract (requires Tesseract binary + ara language pack)."""
    import pytesseract
    from PIL import Image
    img = Image.open(image_path)
    # lang='ara' for Arabic; oem 3 = LSTM; psm 6 = assume uniform block of text
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(img, lang="ara", config=config)


def _ocr_easyocr(image_path: str | Path) -> str:
    """OCR via EasyOCR (pure Python, downloads model on first use ~200 MB)."""
    import easyocr
    reader = easyocr.Reader(["ar"], gpu=False, verbose=False)
    results = reader.readtext(str(image_path), detail=0, paragraph=True)
    return "\n".join(results)


def ocr_image(image_path: str | Path, engine: str = "auto") -> str:
    """Extract Arabic text from an image file.

    Parameters
    ----------
    image_path:
        Path to the image (JPEG, PNG, TIFF, BMP, etc.).
    engine:
        ``"tesseract"`` — use pytesseract (requires Tesseract binary).
        ``"easyocr"``   — use EasyOCR (no binary, larger download).
        ``"auto"``      — try tesseract first, fall back to easyocr.

    Returns
    -------
    str
        Raw OCR text (not yet repaired — caller applies the pipeline).

    Raises
    ------
    ImportError
        If the requested engine (or both, for "auto") is not installed.
    FileNotFoundError
        If image_path does not exist.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if engine == "tesseract":
        return _ocr_tesseract(path)

    if engine == "easyocr":
        return _ocr_easyocr(path)

    # auto: try tesseract, fall back to easyocr
    try:
        return _ocr_tesseract(path)
    except (ImportError, Exception) as e_tess:
        try:
            return _ocr_easyocr(path)
        except ImportError:
            raise ImportError(
                "No OCR engine available. Install one of:\n"
                "  pip install arabic-extract[tesseract]   (+ Tesseract binary)\n"
                "  pip install arabic-extract[easyocr]\n"
                "  pip install arabic-extract[all]\n\n"
                f"Tesseract error: {e_tess}"
            )


def available_engine() -> str | None:
    """Return the name of the first available OCR engine, or None."""
    try:
        import pytesseract  # noqa: F401
        return "tesseract"
    except ImportError:
        pass
    try:
        import easyocr  # noqa: F401
        return "easyocr"
    except ImportError:
        pass
    return None
