# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Repair + normalize pipeline applied to raw OCR / PDF-extracted text.

This is the shared post-processing step for all extraction methods:
  raw text  →  arabic-repair (de-shape + restore order)  →  optional CAMeL normalization

We do NOT do diacritics, dialect, or morphological analysis — that is
CAMeL Tools' job, and we hand off to it cleanly.
"""
from __future__ import annotations

import unicodedata

from arabic_repair import repair, detect


def run_pipeline(raw: str, normalize: bool = True) -> str:
    """Apply repair + optional normalization to a raw extracted string.

    Parameters
    ----------
    raw:
        Text as returned by a PDF extractor or OCR engine.
    normalize:
        If True, apply Unicode NFKC normalization after repair.
        CAMeL Tools normalize_unicode() is used when available,
        falling back to plain NFKC.

    Returns
    -------
    str
        Clean logical-order Arabic text.
    """
    if not raw or not raw.strip():
        return raw

    text = repair(raw)

    if normalize:
        try:
            from camel_tools.utils.normalize import normalize_unicode
            text = normalize_unicode(text)
        except Exception:
            text = unicodedata.normalize("NFKC", text)

    return text


def contamination_summary(pages: list[str]) -> dict:
    """Return aggregate contamination stats across all pages."""
    total_words = 0
    contaminated_words = 0
    for page in pages:
        info = detect(page)
        total_words += info.total_arabic_words
        contaminated_words += info.contaminated_words
    ratio = contaminated_words / total_words if total_words else 0.0
    return {
        "total_arabic_words": total_words,
        "contaminated_words": contaminated_words,
        "contaminated_ratio": round(ratio, 4),
        "needed_repair": contaminated_words > 0,
    }
