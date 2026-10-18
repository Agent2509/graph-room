"""End-to-end runnable demo script for Graph RAG."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from graph_rag.engine import GraphRAG
from graph_rag.llm.groq_client import GroqLLM
from graph_rag.llm.embeddings import FastEmbedEmbedding
from graph_rag.llm.mock import MockLLM, MockEmbedding


def main():
    print("=" * 60)
    print(" 🕸️  Graph RAG: End-to-End Demonstration")
    print("=" * 60)

    # 1. Determine provider
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print("[+] Using Groq LLM (openai/gpt-oss-20b) + FastEmbed ONNX embeddings")
        llm = GroqLLM(api_key=api_key)
    else:
        print("[!] No GROQ_API_KEY found in environment. Using deterministic MockLLM + FastEmbed.")
        print("[!] Set GROQ_API_KEY in your .env file to enable live Groq cloud inference.")
        llm = MockLLM()

    embedding = FastEmbedEmbedding()

    # 2. Initialize Graph RAG
    rag = GraphRAG(llm=llm, embedding_model=embedding, chunk_size=800, chunk_overlap=100)

    # 3. Ingest sample documents
    sample_dir = Path(__file__).parent.parent / "sample_data"
    print(f"\n[1] Ingesting documents from {sample_dir.name}/...")
    for file_path in sorted(sample_dir.glob("*.txt")):
        chunks = rag.add_file(file_path)
        print(f"    - Ingested '{file_path.name}': {len(chunks)} chunks staged.")

    # 4. Build Knowledge Graph index
    print("\n[2] Building Knowledge Graph & Vector Index...")
    result = rag.build_index(
        progress_callback=lambda msg, progress: print(f"    [{int(progress*100):02d}%] {msg}"),
        cache_dir="./data/demo_cache",
    )
    print(f"[+] Indexing result: {result}")

    # 5. Graph Analytics
    stats = rag.get_stats()
    print("\n[3] Knowledge Graph Summary:")
    print(f"    - Extracted Entities (Nodes): {stats.node_count}")
    print(f"    - Relationships (Edges):      {stats.edge_count}")
    print(f"    - Detected Communities:       {stats.community_count}")
    print(f"    - Network Density:            {stats.density:.4f}")

    print("\n[+] Top Entities by Degree Centrality:")
    for ent in stats.top_entities[:5]:
        print(f"    * {ent['name']} ({ent['type']}): degree {ent['degree']}")

    # 6. Local Multi-hop Search
    local_query = "How is Alice Zhao connected to CureX?"
    print(f"\n[4] Executing Local Multi-hop Search: '{local_query}'")
    local_res = rag.query(local_query, mode="local", max_hops=2)
    print("\n--- Answer ---")
    print(local_res.answer.strip())
    print("\n--- Cited Graph Triples ---")
    for r in local_res.cited_relations[:6]:
        print(f"    ({r.source}) --[{r.relation_type}]--> ({r.target}): {r.description}")

    # 7. Global Thematic Search
    global_query = "What are the major organizational relationships and medical breakthroughs across the documents?"
    print(f"\n[5] Executing Global Thematic Search: '{global_query}'")
    global_res = rag.query(global_query, mode="global")
    print("\n--- Executive Summary ---")
    print(global_res.answer.strip())

    print("\n" + "=" * 60)
    print(" ✅ Demonstration completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
