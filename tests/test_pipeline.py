"""End-to-end integration test for GraphRAG pipeline."""

import tempfile
from pathlib import Path
from graph_rag.engine import GraphRAG
from graph_rag.llm.mock import MockLLM, MockEmbedding


def test_pipeline_end_to_end():
    # 1. Initialize with deterministic mock providers
    llm = MockLLM()
    embedding = MockEmbedding(dimension=32)
    rag = GraphRAG(llm=llm, embedding_model=embedding, chunk_size=300, chunk_overlap=50)

    # 2. Add text documents
    doc1 = "Dr. Alice Zhao founded NexusLabs in Boston. NexusLabs builds CortexGraph algorithms."
    doc2 = "NexusLabs acquired DeepBio in Zurich. DeepBio is developing CureX for oncology patients."
    
    chunks1 = rag.add_text(doc1, doc_id="doc1")
    chunks2 = rag.add_text(doc2, doc_id="doc2")

    assert len(chunks1) > 0
    assert len(chunks2) > 0
    assert len(rag.chunks_map) >= 2

    # 3. Build index
    progress_updates = []
    result = rag.build_index(progress_callback=lambda msg, p: progress_updates.append(p), cache_dir=None)

    assert result["status"] == "success"
    assert len(progress_updates) > 0

    stats = rag.get_stats()
    assert stats.node_count > 0
    assert stats.edge_count > 0
    assert stats.community_count >= 1

    # 4. Local Query
    local_res = rag.query("How is Alice connected to CureX?", mode="local")
    assert local_res.search_mode == "local"
    assert len(local_res.answer) > 0
    assert len(local_res.cited_entities) > 0

    # 5. Global Query
    global_res = rag.query("What are the key themes?", mode="global")
    assert global_res.search_mode == "global"
    assert len(global_res.answer) > 0

    # 6. HTML Visualization Generation
    html = rag.get_graph_html()
    assert len(html) > 500
    assert "<html>" in html.lower() or "<div" in html.lower() or "<script" in html.lower()

    # 7. Persistence: Save & Load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = str(Path(tmpdir) / "test_index")
        rag.save(save_path)

        new_rag = GraphRAG(llm=llm, embedding_model=embedding)
        new_rag.load(save_path)

        new_stats = new_rag.get_stats()
        assert new_stats.node_count == stats.node_count
        assert new_stats.edge_count == stats.edge_count
        assert len(new_rag.chunks_map) == len(rag.chunks_map)

        # Query loaded engine
        reloaded_res = new_rag.query("Alice Zhao", mode="local")
        assert len(reloaded_res.answer) > 0
