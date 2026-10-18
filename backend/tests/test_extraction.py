"""Extraction tolerance and resolution."""

from graph_rag.models import ExtractedEdge, ExtractedEntity, TextChunk
from graph_rag.pipeline.extractor import parse
from graph_rag.pipeline.resolver import (
    merge_descriptions,
    normalize_name,
    resolve_edges,
    resolve_entities,
)

BATCH = [
    TextChunk(id="d#0000", document_id="d", idx=0, text="Alice founded NexusLabs.", token_count=3),
    TextChunk(id="d#0001", document_id="d", idx=1, text="NexusLabs builds CortexGraph.", token_count=3),
]


def test_bad_weight_cannot_kill_a_batch():
    payload = {
        "entities": [{"name": "Alice"}, {"name": "NexusLabs"}],
        "relationships": [
            {"source": "Alice", "target": "NexusLabs", "relation_type": "founded",
             "weight": "1.0 (high)"},
            {"source": "Alice", "target": "NexusLabs", "relation_type": "FOUNDED", "weight": None},
        ],
    }
    entities, edges = parse(payload, BATCH)
    assert len(entities) == 2
    assert len(edges) == 2
    assert all(edge.rel == "FOUNDED" for edge in edges)
    assert all(edge.weight == 1.0 for edge in edges)


def test_malformed_items_are_skipped_not_fatal():
    payload = {
        "entities": [{"name": ""}, "nonsense", {"name": "Alice"}],
        "relationships": [{"source": "Alice"}, {"source": "Alice", "target": "Alice"}],
    }
    entities, edges = parse(payload, BATCH)
    assert [entity.name for entity in entities] == ["Alice"]
    assert edges == []


def test_payload_shapes_never_raise():
    assert parse([], BATCH) == ([], [])
    assert parse({"entities": "nope"}, BATCH) == ([], [])


def test_entity_resolution_dedupes_and_promotes_type():
    resolved = resolve_entities(
        [
            ExtractedEntity(name="The OpenAI", type="CONCEPT", description="AI lab."),
            ExtractedEntity(name="openai", type="ORGANIZATION", description="Creates GPT."),
        ]
    )
    assert len(resolved) == 1
    entity = next(iter(resolved.values()))
    assert entity.name == "OpenAI"
    assert entity.type == "ORGANIZATION"
    assert "AI lab" in entity.description and "Creates GPT" in entity.description


def test_edges_with_unknown_endpoints_are_dropped():
    entities = resolve_entities([ExtractedEntity(name="Alice")])
    edges = resolve_edges(
        [ExtractedEdge(src="Alice", dst="Ghost", rel="KNOWS")], entities
    )
    assert edges == []


def test_normalize_and_merge():
    assert normalize_name("  The Microsoft ") == "Microsoft"
    assert normalize_name("Dr. Alice Zhao") == "Alice Zhao"
    merged = merge_descriptions("Alice founded NexusLabs.", "Alice founded NexusLabs.")
    assert merged == "Alice founded NexusLabs."
