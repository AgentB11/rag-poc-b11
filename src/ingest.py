from pathlib import Path

import easyocr
import pymupdf
from pypdf import PdfReader

TEXT_EXTENSIONS = {".txt"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Loading the OCR model is expensive, so create the reader once and reuse it.
_ocr_reader = None


def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_text_from_pdf(path: Path) -> str:
    try:
        doc = pymupdf.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if text.strip():
            return text
    except Exception:
        pass

    # Fall back to pypdf if pymupdf produced nothing (e.g. an unusual PDF encoding).
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_image(path: Path) -> str:
    reader = _get_ocr_reader()
    results = reader.readtext(str(path), detail=0)
    return "\n".join(results)


def ingest_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return extract_text_from_txt(path)
    if ext in PDF_EXTENSIONS:
        return extract_text_from_pdf(path)
    if ext in IMAGE_EXTENSIONS:
        return extract_text_from_image(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def ingest_directory(raw_dir: Path) -> list[dict]:
    documents = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        text = ingest_file(path)
        documents.append({"source": path.name, "text": text})
    return documents


def save_documents(documents: list[dict], processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        out_path = processed_dir / f"{Path(doc['source']).stem}.txt"
        out_path.write_text(doc["text"], encoding="utf-8")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    raw_dir = project_dir / "data" / "raw"
    processed_dir = project_dir / "data" / "processed"

    documents = ingest_directory(raw_dir)
    save_documents(documents, processed_dir)

    for doc in documents:
        preview = doc["text"][:200].replace("\n", " ")
        print(f"=== {doc['source']} ({len(doc['text'])} chars) -> {processed_dir / (Path(doc['source']).stem + '.txt')} ===")
        print(preview)
        print()
