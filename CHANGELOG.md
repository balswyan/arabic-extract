# Changelog

## 0.1.0 — 2026-06-08
- Initial release.
- `extract(path)` — auto-detects PDF vs image, returns `ExtractResult`.
- `extract_pdf(path)` — PDF-specific: text layer first, OCR fallback for scanned pages.
- `extract_image(path)` — image OCR via Tesseract or EasyOCR (auto-detected).
- Per-page `method` tracking: "text_layer" | "ocr" | "text_layer_empty".
- `contamination` stats per result showing how many words needed repair.
- OCR backends optional: `[tesseract]`, `[easyocr]`, or `[all]`.
- PDF rendering backends optional: `[pymupdf]` (recommended) or `[pdf2image]`.
- Requires `arabic-repair>=0.1.0` for the repair step.
