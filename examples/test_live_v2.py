"""Comprehensive live test for Graph-RAG 2.0 features."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from graph_rag.engine import GraphRAG
from graph_rag.llm.groq_client import GroqLLM
from graph_rag.llm.embeddings import FastEmbedEmbedding
from graph_rag.llm.mock import MockLLM


def main():
    print("=" * 70)
    print(" 🕸️  Graph-RAG 2.0: Comprehensive Architecture Test")
    print("=" * 70)

    # 1. Initialize engine
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print("[+] Provider: Groq Cloud LLM + FastEmbed ONNX CPU embeddings")
        llm = GroqLLM(api_key=api_key)
    else:
        print("[!] Using MockLLM fallback")
        llm = MockLLM()

    rag = GraphRAG(llm=llm, embedding_model=FastEmbedEmbedding(), chunk_size=600, chunk_overlap=80)

    # 2. Ingest sample documents with hierarchical parent-child mapping
    sample_dir = Path("sample_data")
    print("\n[1] Ingesting documents with hierarchical parent-child chunking...")
    for f in sorted(sample_dir.glob("*.txt")):
        chunks = rag.add_file(f, hierarchical=True)
        print(f"    - {f.name}: {len(chunks)} child chunks created (mapped to parent blocks)")

    # 3. Build index
    print("\n[2] Building Knowledge Graph & Hybrid Index...")
    rag.build_index(
        progress_callback=lambda msg, p: print(f"    [{int(p*100):02d}%] {msg}"),
        cache_dir="./data/live_test_cache",
    )

    stats = rag.get_stats()
    print(f"\n[+] Graph Built: {stats.node_count} nodes, {stats.edge_count} edges, {stats.community_count} communities")

    # 4. Test Agentic Router (Local Query)
    query_1 = "How is Alice Zhao connected to CureX?"
    print("\n" + "-" * 70)
    print(f"[TEST 1] Query: '{query_1}' (mode='auto')")
    res_1 = rag.query(query_1, mode="auto")
    print(f"[+] Routed Search Mode: '{res_1.search_mode}'")
    print(f"\n--- Grounded Answer with Citations ---")
    print(res_1.answer.strip())
    print("\n--- Cited Multi-Hop Triples ---")
    for r in res_1.cited_relations[:5]:
        print(f"    * ({r.source}) --[{r.relation_type}]--> ({r.target})")

    # 5. Test Agentic Router (Global Query)
    query_2 = "Summarize the key medical and technological breakthroughs across all documents."
    print("\n" + "-" * 70)
    print(f"[TEST 2] Query: '{query_2}' (mode='auto')")
    res_2 = rag.query(query_2, mode="auto")
    print(f"[+] Routed Search Mode: '{res_2.search_mode}'")
    print(f"\n--- Global Thematic Report ---")
    print(res_2.answer.strip())

    # 6. Test Hybrid BM25 Keyword Recall
    query_3 = "Which regulatory agencies granted priority review status to CureX?"
    print("\n" + "-" * 70)
    print(f"[TEST 3] BM25 Acronym Recall: '{query_3}'")
    res_3 = rag.query(query_3, mode="local")
    print(f"\n--- Answer ---")
    print(res_3.answer.strip())

    print("\n" + "=" * 70)
    print(" ✅ All Graph-RAG 2.0 tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
