from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


class DocumentError(ValueError):
    pass


def extract_chunks(filename: str, content: bytes, size: int, overlap: int) -> list[dict]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise DocumentError("Encrypted PDFs are not supported")
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("Could not read PDF") from exc
    elif suffix in {".txt", ".md", ".csv"}:
        try:
            pages = [(None, content.decode("utf-8-sig"))]
        except UnicodeDecodeError as exc:
            raise DocumentError("Text files must use UTF-8 encoding") from exc
    else:
        raise DocumentError("Supported file types: .pdf, .txt, .md, .csv")
    chunks = []
    for page, text in pages:
        text = text.strip().replace("\x00", "")
        for start in range(0, len(text), size - overlap):
            part = text[start : start + size].strip()
            if part:
                chunks.append({"index": len(chunks), "page": page, "text": part})
            if start + size >= len(text):
                break
    if not chunks:
        raise DocumentError("No readable text found. Scanned PDFs require OCR before upload.")
    return chunks
