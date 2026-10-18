"""Read text out of uploads. PDFs need pypdf; everything else is utf-8."""

import io

from pypdf import PdfReader


def text_from_bytes(content: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        return "\n\n".join(
            page.extract_text()
            for page in PdfReader(io.BytesIO(content)).pages
            if page.extract_text()
        )
    return content.decode("utf-8", errors="ignore")
