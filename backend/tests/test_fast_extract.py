"""The rule-based extractor: spans, relations, direction, determinism."""

from graph_rag.models import TextChunk
from graph_rag.pipeline import fast_extract
from graph_rag.pipeline.fast_extract import find_spans, sentences

TEXT = (
    "In 2021, Dr. Alice Zhao, a former machine learning professor at MIT, "
    "founded NexusLabs in Boston. NexusLabs developed CortexGraph for graph reasoning. "
    "NexusLabs acquired DeepBio in November 2023 for $320 million."
)


def _chunk(text: str = TEXT) -> TextChunk:
    return TextChunk(id="d#0000", document_id="d", idx=0, text=text, token_count=len(text.split()))


def test_sentences_do_not_split_abbreviations():
    parts = [sentence for _, _, sentence in sentences("Dr. Zhao met Prof. Smith. Inc. filed later.")]
    assert parts[0].startswith("Dr. Zhao met Prof. Smith")
    assert parts[-1].startswith("Inc.")


def test_span_types_and_date_filtering():
    spans = find_spans("In 2021, Dr. Alice Zhao founded NexusLabs in Boston.")
    by_name = {span.name: span.type for span in spans}
    assert by_name["Dr. Alice Zhao"] == "PERSON"
    assert by_name["NexusLabs"] == "ORGANIZATION"
    assert "2021" not in by_name


def test_location_after_locative_cue():
    spans = find_spans("DeepBio is headquartered in Zurich.")
    assert {span.name: span.type for span in spans}["Zurich"] == "LOCATION"


def test_extracts_expected_relations(semantic_embed):
    entities, edges = fast_extract.extract([_chunk()], semantic_embed)
    triples = {(edge.src, edge.rel, edge.dst) for edge in edges}
    names = {entity.name for entity in entities}

    assert "NexusLabs" in names and "CortexGraph" in names
    assert ("NexusLabs", "DEVELOPED", "CortexGraph") in triples
    assert ("NexusLabs", "ACQUIRED", "DeepBio") in triples
    # appositive: the founder is Alice, not MIT
    assert ("Dr. Alice Zhao", "FOUNDED", "NexusLabs") in triples
    assert not any(triple[0] == "MIT" and triple[1] == "FOUNDED" for triple in triples)


def test_passive_voice_flips_direction(semantic_embed):
    _, edges = fast_extract.extract(
        [_chunk("DeepBio was acquired by NexusLabs in 2023.")], semantic_embed
    )
    assert [(edge.src, edge.rel, edge.dst) for edge in edges] == [
        ("NexusLabs", "ACQUIRED", "DeepBio")
    ]


def test_fragment_between_entities_is_skipped(semantic_embed):
    _, edges = fast_extract.extract(
        [_chunk("NexusLabs signed a deal with ApexCloud and high-bandwidth H100 clusters.")],
        semantic_embed,
    )
    assert not any(edge.src == "H100" or edge.dst == "H100" for edge in edges)


def test_off_vocabulary_phrases_are_dropped(semantic_embed):
    _, edges = fast_extract.extract(
        [_chunk("NexusLabs complied with HIPAA and GDPR requirements.")], semantic_embed
    )
    assert edges == []


def test_extraction_is_deterministic(semantic_embed):
    first = fast_extract.extract([_chunk()], semantic_embed)
    second = fast_extract.extract([_chunk()], semantic_embed)
    assert [(e.src, e.rel, e.dst) for e in first[1]] == [(e.src, e.rel, e.dst) for e in second[1]]
    assert [e.name for e in first[0]] == [e.name for e in second[0]]


def test_indexing_in_fast_mode_needs_no_llm(db, semantic_embed):
    from graph_rag.pipeline.indexer import add_document, index_document

    document_id, _ = add_document(db, "a.txt", TEXT, 500, 50)
    result = index_document(db, None, semantic_embed, "none", document_id, mode="fast")
    assert result == "done"
    assert db.conn().execute("SELECT COUNT(*) AS c FROM entities").fetchone()["c"] > 0
    assert db.conn().execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"] > 0
