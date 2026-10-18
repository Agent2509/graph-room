"""Community summary generation using LLMs."""

from typing import List
from graph_rag.models import Community
from graph_rag.storage.graph_store import GraphStore
from graph_rag.storage.community_store import CommunityStore
from graph_rag.llm.base import BaseLLM

COMMUNITY_SUMMARY_SYSTEM = """You are an expert analyst. You are given a cluster of related entities and their relationships from a Knowledge Graph.
Your task is to write a comprehensive executive summary for this community.

Return a valid JSON object matching this schema:
{
  "title": "A concise, thematic title for this community",
  "summary": "A detailed 2-3 paragraph explanation of the key themes, relationships, and context represented by these entities.",
  "findings": [
    "Key insight or finding 1",
    "Key insight or finding 2",
    "Key insight or finding 3"
  ],
  "rating": 8.5
}
"""

COMMUNITY_SUMMARY_USER = """Analyze the following knowledge graph community:

--- ENTITIES ---
{entities_str}

--- RELATIONSHIPS ---
{relations_str}
--- END ---
"""


class CommunitySummarizer:
    """Generates analytical reports and summaries for knowledge graph communities."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def summarize_community(
        self,
        community_id: int,
        entity_names: List[str],
        graph_store: GraphStore,
    ) -> Community:
        """Generate summary report for a single community."""
        entities_lines = []
        for name in entity_names:
            ent = graph_store.get_entity(name)
            if ent:
                desc = ent.description or "No description provided."
                entities_lines.append(f"- {ent.name} ({ent.type}): {desc}")

        entities_str = "\n".join(entities_lines) or "None"

        # Find relationships within this community
        entity_set = set(entity_names)
        relations_lines = []
        for u, v, data in graph_store.graph.edges(data=True):
            if u in entity_set and v in entity_set:
                rel_type = data.get("relation_type", "RELATED_TO")
                desc = data.get("description", "")
                relations_lines.append(f"- {u} -[{rel_type}]-> {v}: {desc}")

        relations_str = "\n".join(relations_lines[:50]) or "None"

        prompt = COMMUNITY_SUMMARY_USER.format(
            entities_str=entities_str,
            relations_str=relations_str,
        )

        data = self.llm.generate_json(prompt=prompt, system_prompt=COMMUNITY_SUMMARY_SYSTEM)

        title = f"Community {community_id}: " + ", ".join(entity_names[:3])
        summary = "No summary generated."
        findings = []
        rating = 5.0

        if isinstance(data, dict):
            title = data.get("title", title)
            summary = data.get("summary", summary)
            findings = data.get("findings", findings)
            try:
                rating = float(data.get("rating", rating))
            except Exception:
                pass

        return Community(
            id=community_id,
            level=0,
            entity_names=entity_names,
            title=title,
            summary=summary,
            findings=findings,
            rating=rating,
        )

    def summarize_all(
        self,
        communities: List[List[str]],
        graph_store: GraphStore,
        community_store: CommunityStore,
        max_summaries: int = 15,
    ) -> None:
        """Summarize communities with LLM and heuristic fallbacks to ensure full coverage."""
        # Keep each community's original index (matches CommunityDetector's node
        # annotation) while prioritizing the largest clusters for LLM calls.
        indexed = list(enumerate(communities))
        indexed.sort(key=lambda pair: len(pair[1]), reverse=True)

        for position, (comm_id, node_list) in enumerate(indexed):
            if position < max_summaries and len(node_list) >= 2:
                try:
                    community_obj = self.summarize_community(
                        community_id=comm_id,
                        entity_names=node_list,
                        graph_store=graph_store,
                    )
                    community_store.add_community(community_obj)
                    continue
                except Exception:
                    pass

            # Fast heuristic summary without consuming LLM calls or failing on rate limits
            top_ents = node_list[:5]
            community_obj = Community(
                id=comm_id,
                level=0,
                entity_names=node_list,
                title=f"Cluster {comm_id + 1}: " + ", ".join(top_ents[:3]),
                summary=f"Community of {len(node_list)} related entities centered around {', '.join(top_ents)}.",
                findings=[f"Covers relationships and concepts associated with {', '.join(top_ents[:3])}."],
                rating=7.5,
            )
            community_store.add_community(community_obj)
