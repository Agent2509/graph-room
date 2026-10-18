"""Rule-based extraction: entity spans + verb-phrase labelling with embeddings.

No LLM, no network, no new dependency. The BGE model that already powers retrieval
scores a between-entity phrase against a small label ontology; anything below the
threshold is skipped rather than guessed.

Deterministic: same text in, same triples out.
"""

import re
from dataclasses import dataclass

import numpy as np

from graph_rag.models import ExtractedEdge, ExtractedEntity, TextChunk

MIN_MARGIN = 0.03
MAX_ENTITY_SENTENCES = 2
MAX_SPANS_PER_SENTENCE = 6

HONORIFICS = {"dr", "prof", "mr", "mrs", "ms"}
STOP_PREFIX = {
    "the", "this", "that", "these", "those", "in", "on", "at", "for", "with", "by",
    "from", "and", "or", "an", "a", "following", "under", "recognizing", "based",
    "mid", "early", "late", "during", "after", "before", "its", "their", "his", "her",
    "our", "your", "when", "while", "as", "to", "of", "also", "however", "then", "now",
    "here", "there", "such", "each", "both", "all", "more", "most", "other", "another",
}
ORG_SUFFIXES = {
    "labs", "inc", "corp", "corporation", "llc", "ltd", "co", "bio", "ai", "systems",
    "cloud", "group", "agency", "university", "institute", "hospital", "foundation",
    "partners", "technologies", "therapeutics", "biosciences", "pharma", "capital",
    "bank", "media", "network", "networks", "software", "studios", "ventures",
}
MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
ROLE_WORDS = {
    "chief", "officer", "director", "head", "professor", "ceo", "cto", "founder",
    "president", "manager", "senior", "vice", "lead",
}
LOCATIVE_PREPOSITIONS = {"in", "at", "from"}
LOCATIVE_CUES = ("based in", "headquartered in", "located in", "operates from", "based out of")
VERB_HINTS = {
    "founded", "co-founded", "established", "started", "acquired", "bought",
    "purchased", "developed", "built", "created", "engineered", "designed",
    "entered", "partnered", "collaborated", "teamed", "signed", "provides",
    "supplies", "offers", "hosts", "serves", "based", "headquartered", "located",
    "works", "joined", "appointed", "promoted", "employed", "granted", "approved",
    "authorized", "gave", "uses", "used", "deployed", "adopted", "implemented",
    "invested", "backed", "owns", "operates", "supports", "enables",
    "part", "subsidiary", "division",
}
AUXILIARIES = {"was", "were", "is", "are", "been", "being", "has", "have", "had", "will", "would"}
PASSIVE = re.compile(r"\b(was|were|is|are|been|being)\b[^,;]{0,40}\bby\b", re.IGNORECASE)
AGENTLESS_PASSIVE = re.compile(r"\b\w+ed by\b", re.IGNORECASE)
ABBREVIATIONS = {"dr", "prof", "mr", "mrs", "ms", "inc", "corp", "ltd", "co", "vs", "e.g", "i.e", "u.s", "st"}

CODE_BLOCK = re.compile(r"```[\s\S]*?```")
INLINE_CODE = re.compile(r"`[^`\n]{1,80}`")
IDENTIFIER = re.compile(r"^[A-Z][A-Z0-9_]*$")
CAP_RE = re.compile(
    r"\b[A-Z][\w&.'-]*(?:\s+(?:of|the|de|van|von)\s+[A-Z][\w&.'-]*"
    r"|\s+[A-Z][\w&.'-]*){0,5}"
)

# One label per row of intent. Phrases are what the between-text looks like.
LABEL_PROTOTYPES: dict[str, list[str]] = {
    "FOUNDED": [
        "founded the company", "co-founded the company", "established the organization",
        "started the business", "the founding of",
    ],
    "ACQUIRED": [
        "acquired the company", "bought the company", "purchased the business",
        "the acquisition of",
    ],
    "DEVELOPED": [
        "developed the product", "built the system", "created the technology",
        "engineered the treatment", "designed the platform",
    ],
    "PARTNERS_WITH": [
        "entered a partnership with", "collaborated with", "teamed up with",
        "signed an agreement with", "the partnership between",
    ],
    "PROVIDES": [
        "provides infrastructure to", "supplies services to", "offers cloud capacity to",
        "hosts data for",
    ],
    "BASED_IN": [
        "is headquartered in", "is based in", "is located in", "operates from",
    ],
    "WORKS_FOR": [
        "works for", "joined the company", "employed by", "member of the team at",
    ],
    "PROMOTED": [
        "promoted the employee to", "appointed the executive as", "named the chief officer",
    ],
    "GRANTED": [
        "granted approval to", "approved the treatment", "authorized the drug",
        "gave priority review to",
    ],
    "USES": [
        "uses the technology", "deployed the system", "adopted the platform",
        "implemented the framework",
    ],
    "PART_OF": [
        "is a subsidiary of", "is part of", "is a division of",
    ],
}


@dataclass
class Span:
    start: int
    end: int
    name: str
    type: str


def extract(
    chunks: list[TextChunk], embed, threshold: float = 0.62
) -> tuple[list[ExtractedEntity], list[ExtractedEdge]]:
    """Extract entities and relations using spans + embedding-labelled verb phrases."""
    entities: dict[str, ExtractedEntity] = {}
    candidates: list[tuple[str, str, str, str, str, str]] = []  # chunk, sentence, src, mid, dst, types

    for chunk in chunks:
        for _, _, sentence in sentences(strip_code(chunk.text)):
            spans = find_spans(sentence)
            if not spans:
                continue
            for span in spans:
                _remember_entity(entities, span, sentence, chunk.id)
            if len(spans) > MAX_SPANS_PER_SENTENCE:
                continue
            for left, right in zip(spans, spans[1:], strict=False):
                raw_mid = sentence[left.end : right.start]
                source = _appositive_subject(sentence, spans, left, raw_mid)
                mid = raw_mid.strip(" ,;:()[]\"'")
                if not _mid_is_relation(mid, right.type):
                    continue
                if source.name == right.name:
                    continue
                candidates.append(
                    (chunk.id, sentence, source.name, mid, right.name, spans[0].name)
                )

    if not candidates:
        return list(entities.values()), []

    labels, _ = classify([c[3] for c in candidates], embed, threshold)
    edges: dict[tuple[str, str, str], ExtractedEdge] = {}
    for (chunk_id, sentence, source, mid, target, first_span), label in zip(
        candidates, labels, strict=False
    ):
        if label is None:
            continue
        if PASSIVE.search(mid):
            source, target = target, source
        elif AGENTLESS_PASSIVE.search(mid):
            # "DeepBio ... founded by Marcus Smith": the agent is the subject.
            source, target = target, first_span
        if source == target:
            continue
        key = (source, target, label)
        existing = edges.get(key)
        if existing is None:
            edges[key] = ExtractedEdge(
                src=source, dst=target, rel=label,
                description=sentence[:400], weight=1.0, chunk_ids=[chunk_id],
            )
        else:
            existing.weight += 1.0
            if chunk_id not in existing.chunk_ids:
                existing.chunk_ids.append(chunk_id)

    return list(entities.values()), list(edges.values())


def classify(
    phrases: list[str], embed, threshold: float
) -> tuple[list[str | None], list[float]]:
    """Map between-texts to a label; None when below threshold or ambiguous."""
    prototypes: list[str] = []
    owners: list[int] = []
    for index, phrases_for_label in enumerate(LABEL_PROTOTYPES.values()):
        prototypes.extend(phrases_for_label)
        owners.extend([index] * len(phrases_for_label))

    vectors = _normalize(np.asarray(embed(prototypes), dtype=np.float32))
    phrase_vectors = _normalize(np.asarray(embed(phrases), dtype=np.float32))
    sims = phrase_vectors @ vectors.T

    per_label = np.full((len(phrases), len(LABEL_PROTOTYPES)), -1.0, dtype=np.float32)
    for column, label_index in enumerate(owners):
        np.maximum(per_label[:, label_index], sims[:, column], out=per_label[:, label_index])

    names = list(LABEL_PROTOTYPES)
    best = per_label.argmax(axis=1)
    best_score = per_label[np.arange(len(phrases)), best]
    if per_label.shape[1] > 1:
        second = np.partition(per_label, -2, axis=1)[:, -2]
    else:  # pragma: no cover - ontology always has several labels
        second = np.zeros_like(best_score)

    out: list[str | None] = []
    for row in range(len(phrases)):
        if best_score[row] >= threshold and best_score[row] - second[row] >= MIN_MARGIN:
            out.append(names[int(best[row])])
        else:
            out.append(None)
    return out, [float(value) for value in best_score]


def strip_code(text: str) -> str:
    """Code blocks and inline code are not prose; identifiers make junk entities."""
    return INLINE_CODE.sub(" ", CODE_BLOCK.sub(" ", text))


def sentences(text: str) -> list[tuple[int, int, str]]:
    """Sentence boundaries with offsets, tolerating Dr./Inc./U.S. abbreviations."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[.?!])\s+|\n\s*\n", text):
        if match.start() > start:
            bounds.append((start, match.start()))
        start = match.end()
    if start < len(text):
        bounds.append((start, len(text)))

    merged: list[tuple[int, int]] = []
    for begin, end in bounds:
        if merged and _ends_with_abbreviation(text[merged[-1][0] : merged[-1][1]]):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((begin, end))
    return [(begin, end, text[begin:end].strip()) for begin, end in merged if text[begin:end].strip()]


def find_spans(sentence: str) -> list[Span]:
    raw: list[Span] = []
    for match in CAP_RE.finditer(sentence):
        piece = match.group(0)
        lead = len(piece) - len(piece.lstrip(" .,"))
        name = piece.strip(" .,")
        if not name or not re.search(r"[A-Za-z]", name):
            continue
        tokens = name.split()
        while tokens and tokens[0].lower().strip(".") in STOP_PREFIX:
            tokens = tokens[1:]
        if not tokens:
            continue
        name = " ".join(tokens)
        start = sentence.find(name, match.start() + lead)
        if start < 0:
            start = match.start() + lead
        raw.append(Span(start=start, end=start + len(name), name=name, type="CONCEPT"))

    spans: list[Span] = []
    index = 0
    while index < len(raw):
        current = raw[index]
        follow = raw[index + 1] if index + 1 < len(raw) else None
        if (
            current.name.lower().strip(".") in HONORIFICS
            and follow is not None
            and follow.start - current.end <= 2
        ):
            spans.append(
                Span(start=current.start, end=follow.end,
                     name=f"{current.name} {follow.name}", type="PERSON")
            )
            index += 2
            continue
        spans.append(current)
        index += 1

    typed: list[Span] = []
    for span in spans:
        tokens = span.name.split()
        lowered = [token.lower().strip(".") for token in tokens]
        if all(token in MONTHS or token.isdigit() for token in lowered):
            continue
        if lowered[0] in ROLE_WORDS:
            continue
        if IDENTIFIER.match(span.name) and ("_" in span.name or any(c.isdigit() for c in span.name)):
            continue
        if lowered[0] in HONORIFICS:
            span.type = "PERSON"
        elif _looks_like_organization(lowered[-1]):
            span.type = "ORGANIZATION"
        elif _follows_locative_cue(sentence, span.start):
            span.type = "LOCATION"
        typed.append(span)
    return typed


def _looks_like_organization(token: str) -> bool:
    if token in ORG_SUFFIXES:
        return True
    return any(len(suffix) >= 3 and token.endswith(suffix) for suffix in ORG_SUFFIXES)


def _follows_locative_cue(sentence: str, start: int) -> bool:
    before = sentence[max(0, start - 24) : start].lower().rstrip()
    return any(before.endswith(cue) for cue in LOCATIVE_CUES)


def _appositive_subject(sentence: str, spans: list[Span], left: Span, raw_mid: str) -> Span:
    """'Alice Zhao, a professor at MIT, founded X' must credit Alice, not MIT."""
    if not raw_mid.lstrip().startswith(","):
        return left
    for span in spans:
        if span.type == "PERSON" and span.start < left.start:
            return span
    return left


def _mid_is_relation(mid: str, target_type: str) -> bool:
    if not mid or len(mid) > 48 or not re.search(r"[a-z]", mid):
        return False
    if mid.lower() in LOCATIVE_PREPOSITIONS:
        return target_type == "LOCATION"
    tokens = re.split(r"[\s-]", mid.lower())
    first = tokens[0]
    if first in AUXILIARIES:
        return any(token in VERB_HINTS or token.endswith("ed") for token in tokens[1:])
    return first in VERB_HINTS or first.endswith("ed")


def _remember_entity(
    entities: dict[str, ExtractedEntity], span: Span, sentence: str, chunk_id: str
) -> None:
    key = span.name.lower()
    existing = entities.get(key)
    if existing is None:
        entities[key] = ExtractedEntity(
            name=span.name, type=span.type, description=sentence[:600], chunk_ids=[chunk_id]
        )
        return
    if chunk_id not in existing.chunk_ids:
        existing.chunk_ids.append(chunk_id)
    if existing.type == "CONCEPT" and span.type != "CONCEPT":
        existing.type = span.type
    sentences_seen = existing.description.count(". ") + 1
    if sentences_seen < MAX_ENTITY_SENTENCES and sentence not in existing.description:
        existing.description = (existing.description + " " + sentence)[:600]


def _ends_with_abbreviation(piece: str) -> bool:
    tokens = piece.strip().split()
    if not tokens:
        return False
    return tokens[-1].rstrip(".").lower() in ABBREVIATIONS


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
