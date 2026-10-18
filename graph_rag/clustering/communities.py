"""Community detection algorithms for Knowledge Graphs."""

from typing import Dict, List, Set
import networkx as nx
from graph_rag.storage.graph_store import GraphStore


class CommunityDetector:
    """Detects modular entity communities in a knowledge graph."""

    def __init__(self, resolution: float = 1.0, min_community_size: int = 1):
        self.resolution = resolution
        self.min_community_size = min_community_size

    def detect_communities(self, graph_store: GraphStore) -> List[List[str]]:
        """Detect entity clusters using Louvain modularity optimization.

        The returned list order is authoritative: a cluster's index equals the
        ``community_id`` annotated on its member nodes in ``graph_store``.
        """
        if graph_store.graph.number_of_nodes() == 0:
            return []

        # Convert to undirected graph for community detection
        undirected = graph_store.graph.to_undirected()

        # If isolated nodes or small graph without edges:
        if undirected.number_of_edges() == 0:
            return [[node] for node in undirected.nodes()]

        try:
            communities_set: List[Set[str]] = list(
                nx.community.louvain_communities(
                    undirected,
                    resolution=self.resolution,
                    seed=42,
                )
            )
        except Exception:
            # Fallback to connected components
            communities_set = list(nx.connected_components(undirected))

        # Convert sets to sorted lists, filter by min_size
        results: List[List[str]] = []
        for comm in communities_set:
            node_list = sorted(list(comm))
            if len(node_list) >= self.min_community_size:
                results.append(node_list)

        # Annotate nodes in the graph store with their assigned community_id
        for comm_id, node_list in enumerate(results):
            for node in node_list:
                if graph_store.graph.has_node(node):
                    graph_store.graph.nodes[node]["community_id"] = comm_id

        return results
