"""Tests for Graph-RAG 2.0 features: Router, Hierarchical Chunking, Hybrid Search."""

from graph_rag.llm.mock import MockLLM
from graph_rag.retrieval.router import QueryRouter
from graph_rag.chunking.hierarchical import HierarchicalChunker
from graph_rag.storage.vector_store import VectorStore


def test_query_router_heuristic():
    router = QueryRouter(MockLLM())
    decision = router.route("Can you summarize the main themes of the entire project?")
    assert decision.mode == "global"


def test_hierarchical_chunker():
    chunker = HierarchicalChunker(parent_size=300, child_size=100, child_overlap=20)
    text = (
        "Alice Zhao founded NexusLabs in Boston. It develops CortexGraph algorithms. "
        "NexusLabs later partnered with DeepBio in Zurich to work on oncology therapeutics. "
        "The CureX program showed substantial clinical response rates across 3 phases."
    )
    children, parents = chunker.chunk_document(text, doc_id="doc1")
    assert len(children) >= 2
    assert len(parents) >= 1
    for c in children:
        assert "parent_id" in c.metadata
        assert c.metadata["parent_id"] in parents


def test_hybrid_search_bm25_rrf():
    vs = VectorStore()
    vs.add("c1", [1.0, 0.0, 0.0], metadata={"item_type": "chunk", "text": "Clinical trial for CX-901 in Boston."})
    vs.add("c2", [0.0, 1.0, 0.0], metadata={"item_type": "chunk", "text": "Financial quarterly earnings report."})

    # Exact keyword match on CX-901 via hybrid search
    results = vs.search_hybrid(query_text="CX-901", query_vector=[0.1, 0.9, 0.0], top_k=2)
    assert len(results) == 2
    # Even if vector was close to c2, BM25 boost should rank c1 high
    assert results[0][0] == "c1"


def test_community_ids_match_detector_order():
    from graph_rag.models import Entity
    from graph_rag.storage.graph_store import GraphStore
    from graph_rag.storage.community_store import CommunityStore
    from graph_rag.clustering.summarizer import CommunitySummarizer

    gs = GraphStore()
    for name in ["Alice", "NexusLabs", "DeepBio", "CureX", "ApexCloud"]:
        gs.add_entity(Entity(name=name, type="CONCEPT", description=name))

    # Detector order: small cluster first, larger cluster second.
    communities = [["Alice", "NexusLabs"], ["DeepBio", "CureX", "ApexCloud"]]
    for cid, members in enumerate(communities):
        for member in members:
            gs.graph.nodes[member]["community_id"] = cid

    store = CommunityStore()
    CommunitySummarizer(MockLLM()).summarize_all(communities, gs, store)

    for community in store.get_all_communities():
        for name in community.entity_names:
            assert gs.graph.nodes[name]["community_id"] == community.id


def test_extraction_cache_scoped_by_model(tmp_path):
    from graph_rag.engine import GraphRAG
    from graph_rag.llm.mock import MockEmbedding

    class CountingLLM(MockLLM):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.calls = 0

        def generate_json(self, *args, **kwargs):
            self.calls += 1
            return super().generate_json(*args, **kwargs)

    def build(model):
        llm = CountingLLM(model)
        rag = GraphRAG(llm=llm, embedding_model=MockEmbedding(dimension=32), chunk_size=300, chunk_overlap=50)
        rag.add_text("Alice founded NexusLabs. NexusLabs builds CortexGraph.", doc_id="d")
        rag.build_index(cache_dir=str(tmp_path))
        return llm

    first = build("model-a")
    second = build("model-b")

    assert first.calls > 0
    assert second.calls > 0  # must not be served from model-a's cache
    files = sorted(p.name for p in tmp_path.glob("extraction_cache_*.json"))
    assert len(files) == 2
