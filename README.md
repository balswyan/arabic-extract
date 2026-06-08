# arabic-ocr

Clean Arabic text extraction from PDFs and scanned images — one call, clean output.

Combines PDF text extraction, image OCR, and [arabic-repair](https://pypi.org/project/arabic-repair/)
into a single pipeline. Handles the visual-order problem that breaks standard Arabic NLP pipelines.

## The problem it solves

Arabic PDFs and scanned documents store text in **visual order** with **presentation-form characters**.
Standard tools (NFKC, CAMeL Tools) remove the presentation forms but cannot restore the reversed
word order — retrieval recall stays broken at ~27%. arabic-ocr applies arabic-repair automatically,
restoring both letter forms and word order before the text reaches your NLP pipeline.

## Install

```bash
pip install arabic-ocr[pdf]          # PDF text-layer extraction
pip install arabic-ocr[tesseract]    # + image OCR via Tesseract (needs binary)
pip install arabic-ocr[easyocr]      # + image OCR via EasyOCR (pure Python, ~200 MB)
pip install arabic-ocr[pymupdf]      # + scanned PDF rendering via PyMuPDF
pip install arabic-ocr[all]          # everything
```

**Tesseract binary** (for the tesseract extra):
- Windows: download from https://github.com/UB-Mannheim/tesseract/wiki — install the Arabic language pack
- Linux: `sudo apt install tesseract-ocr tesseract-ocr-ara`
- macOS: `brew install tesseract && brew install tesseract-lang`

## Quick start

```python
import arabic_ocr as aocr

# PDF — auto-detects text layer vs scanned, repairs each page
result = aocr.extract("document.pdf")
print(result.text)           # clean logical Arabic, all pages joined
print(result.pages)          # per-page breakdown
print(result.contamination)  # how many words needed repair

# Scanned image
result = aocr.extract("scan.jpg")
print(result.text)

# Explicit PDF extraction
result = aocr.extract_pdf("document.pdf", engine="tesseract")

# Explicit image extraction
result = aocr.extract_image("scan.png", engine="easyocr")

# Chain into CAMeL Tools (normalize=True is the default)
result = aocr.extract("document.pdf", normalize=True)
```

## How it works

```
Input PDF or image
    │
    ├─ PDF with text layer  → pdfplumber extracts text (visual order)
    │                                     ↓
    ├─ Scanned PDF          → render page as image → OCR engine
    │                                     ↓
    └─ Image file           → OCR engine (Tesseract or EasyOCR)
                                          ↓
                               arabic-repair (de-shape + restore order)
                                          ↓
                               NFKC / CAMeL Tools normalization
                                          ↓
                               Clean logical Arabic text
```

A single PDF can have mixed pages — some with a text layer, some scanned.
Each page is handled correctly.

## Per-page results

```python
result = aocr.extract("document.pdf")

for page in result.pages:
    print(f"Page {page.page_number} [{page.method}]: {page.text[:80]}")
    # method: "text_layer" | "ocr" | "text_layer_empty"
```

## Ecosystem

| Package | Role |
|---|---|
| [arabic-rt](https://pypi.org/project/arabic-rt/) | Core shaping / fix / unfix engine |
| [arabic-repair](https://pypi.org/project/arabic-repair/) | Detect and repair visual-order contamination |
| [arabic-ocr](https://pypi.org/project/arabic-ocr/) | Full PDF + image extraction pipeline |
| [arabic-benchmark](https://github.com/balswyan/arabic-benchmark) | Benchmark proving the reordering gap |

## License

MPL-2.0
