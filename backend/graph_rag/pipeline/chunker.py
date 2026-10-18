"""Paragraph-aware chunker with overlap.

Chunk ids are ``{document_id}#{idx:04d}``. Document ids are content hashes, so the
same upload is idempotent and two different documents can never collide.
"""

import re

from graph_rag.models import TextChunk


def chunk_text(
    text: str, document_id: str, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[TextChunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = re.sub(r"\r\n", "\n", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [_chunk(document_id, 0, cleaned)]

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", cleaned):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in re.split(r"(?<=[.?!])\s+", paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(_split_words(sentence, chunk_size))

    chunks: list[TextChunk] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        if current and current_len + len(unit) + 1 > chunk_size:
            chunks.append(_chunk(document_id, len(chunks), "\n\n".join(current)))
            overlap: list[str] = []
            overlap_len = 0
            for previous in reversed(current):
                if overlap_len + len(previous) + 1 > chunk_overlap:
                    break
                overlap.insert(0, previous)
                overlap_len += len(previous) + 1
            current = [*overlap, unit]
            current_len = sum(len(item) + 1 for item in current)
        else:
            current.append(unit)
            current_len += len(unit) + 1

    if current:
        chunks.append(_chunk(document_id, len(chunks), "\n\n".join(current)))
    return chunks


def _split_words(sentence: str, chunk_size: int) -> list[str]:
    parts: list[str] = []
    words: list[str] = []
    length = 0
    for word in sentence.split():
        if words and length + len(word) + 1 > chunk_size:
            parts.append(" ".join(words))
            words, length = [word], len(word)
        else:
            words.append(word)
            length += len(word) + 1
    if words:
        parts.append(" ".join(words))
    return parts


def _chunk(document_id: str, idx: int, text: str) -> TextChunk:
    return TextChunk(
        id=f"{document_id}#{idx:04d}",
        document_id=document_id,
        idx=idx,
        text=text,
        token_count=len(text.split()),
    )
