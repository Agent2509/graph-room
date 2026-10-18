"""Entity normalization and de-duplication across chunks and documents."""

import re

from graph_rag.models import ExtractedEdge, ExtractedEntity

HONORIFICS = (r"Dr\.?", r"Prof\.?", r"Mr\.?", r"Mrs\.?", r"Ms\.")
CORPORATE = (r"Inc\.?", r"LLC\.?", r"Corp\.?", r"Corporation", r"Ltd\.?", r"Co\.?")


def normalize_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = re.sub(r"^[\"']|[\"']$", "", cleaned).strip()
    cleaned = re.sub(r"^[Tt]he\s+", "", cleaned).strip()
    cleaned = re.sub(r"['\u2019]s$", "", cleaned).strip()
    for honorific in HONORIFICS:
        cleaned = re.sub(rf"^{honorific}\s+", "", cleaned, flags=re.IGNORECASE).strip()
    for suffix in CORPORATE:
        cleaned = re.sub(rf"\s+{suffix}\b$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.rstrip(",.")


def merge_descriptions(first: str, second: str, max_chars: int = 600) -> str:
    """Join two descriptions sentence by sentence, dropping near-duplicates."""
    left, right = (first or "").strip(), (second or "").strip()
    if not left:
        return right[:max_chars]
    if not right:
        return left[:max_chars]
    if right.lower() in left.lower():
        return left[:max_chars]
    if left.lower() in right.lower():
        return right[:max_chars]

    kept = [s.strip() for s in re.split(r"(?<=[.?!])\s+", left) if s.strip()]
    for sentence in re.split(r"(?<=[.?!])\s+", right):
        sentence = sentence.strip()
        words = set(re.findall(r"\w+", sentence.lower()))
        if not words:
            continue
        duplicate = any(
            len(words & set(re.findall(r"\w+", existing.lower()))) / len(words) > 0.7
            for existing in kept
        )
        if not duplicate and len(" ".join([*kept, sentence])) <= max_chars:
            kept.append(sentence)
    return " ".join(kept)[:max_chars].strip()


def resolve_entities(entities: list[ExtractedEntity]) -> dict[str, ExtractedEntity]:
    resolved: dict[str, ExtractedEntity] = {}
    for entity in entities:
        name = normalize_name(entity.name)
        if not name:
            continue
        key = name.lower()
        existing = resolved.get(key)
        if existing is None:
            resolved[key] = ExtractedEntity(
                name=name,
                type=entity.type,
                description=entity.description[:600],
                chunk_ids=sorted(set(entity.chunk_ids)),
            )
            continue
        if entity.description:
            existing.description = merge_descriptions(existing.description, entity.description)
        existing.chunk_ids = sorted(set(existing.chunk_ids) | set(entity.chunk_ids))
        if existing.type == "CONCEPT" and entity.type != "CONCEPT":
            existing.type = entity.type
    return resolved


def resolve_edges(
    edges: list[ExtractedEdge], entities: dict[str, ExtractedEntity]
) -> list[ExtractedEdge]:
    """Drop edges whose endpoints disappeared during resolution; de-duplicate triples."""
    merged: dict[tuple[str, str, str], ExtractedEdge] = {}
    for edge in edges:
        source = entities.get(normalize_name(edge.src).lower())
        target = entities.get(normalize_name(edge.dst).lower())
        if source is None or target is None or source.name == target.name:
            continue
        key = (source.name, target.name, edge.rel)
        if key not in merged:
            merged[key] = ExtractedEdge(
                src=source.name,
                dst=target.name,
                rel=edge.rel,
                description=edge.description[:400],
                weight=edge.weight,
                chunk_ids=sorted(set(edge.chunk_ids)),
            )
        else:
            current = merged[key]
            if edge.description:
                current.description = merge_descriptions(
                    current.description, edge.description, max_chars=400
                )
            current.weight += edge.weight
            current.chunk_ids = sorted(set(current.chunk_ids) | set(edge.chunk_ids))
    return list(merged.values())
