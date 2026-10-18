"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useState } from "react";

import { getGraph, getStats } from "@/lib/api";
import type { GraphLink, GraphNode, Stats } from "@/lib/types";
import { CHANGED } from "@/components/StatusBar";

const GraphView = dynamic(() => import("@/components/GraphView"), {
  ssr: false,
  loading: () => <div className="canvas" />,
});

export default function GraphPage() {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [links, setLinks] = useState<GraphLink[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [minDegree, setMinDegree] = useState(true);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const params = new URLSearchParams({ limit: "600" });
      if (minDegree) params.set("min_degree", "1");
      const [graph, overview] = await Promise.all([getGraph(params), getStats()]);
      setNodes(graph.nodes);
      setLinks(graph.edges);
      setStats(overview);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load the graph");
    }
  }, [minDegree]);

  useEffect(() => {
    load();
    const handler = () => load();
    window.addEventListener(CHANGED, handler);
    return () => window.removeEventListener(CHANGED, handler);
  }, [load]);

  return (
    <div className="graph-wrap">
      <section className="canvas" aria-label="Knowledge graph">
        <div className="graph-toolbar">
          <label>
            <input
              type="checkbox"
              checked={minDegree}
              onChange={(event) => setMinDegree(event.target.checked)}
            />
            hide isolated entities
          </label>
          <span>{nodes.length} shown</span>
        </div>
        {nodes.length === 0 ? (
          <div className="empty" style={{ padding: "25vh 24px 0" }}>
            <h1>The graph is empty.</h1>
            <p>Add a document in the workspace. Entities and relations appear here once indexing finishes.</p>
          </div>
        ) : (
          <GraphView nodes={nodes} links={links} onSelect={setSelected} />
        )}
      </section>

      <aside className="inspector">
        {error && <p className="error">{error}</p>}
        {selected ? (
          <>
            <h3>{selected.id}</h3>
            <p className="kind">{selected.type.toLowerCase()}</p>
            <p>{selected.description || "No description recorded."}</p>
            <p className="doc-meta">{selected.degree} connections</p>
          </>
        ) : (
          <>
            <h3>Reading the map</h3>
            <p>
              Pick an entity to read its description. Hover a relation to see what the source text
              says about it.
            </p>
            <div className="legend">
              {["PERSON", "ORGANIZATION", "TECHNOLOGY", "CONCEPT", "LOCATION"].map((type) => (
                <span key={type}>
                  <i
                    style={{
                      background:
                        {
                          PERSON: "#e0a458",
                          ORGANIZATION: "#8fbf9b",
                          TECHNOLOGY: "#7fa6c9",
                          CONCEPT: "#b9b3a3",
                          LOCATION: "#c4693f",
                        }[type] ?? "#b9b3a3",
                    }}
                  />
                  {type.charAt(0) + type.slice(1).toLowerCase()}
                </span>
              ))}
            </div>
          </>
        )}

        {stats && stats.community_reports.length > 0 && (
          <div style={{ marginTop: 26 }}>
            <h2 className="panel-title">Communities</h2>
            {stats.community_reports.slice(0, 6).map((community) => (
              <article key={community.id} className="community">
                <h4>{community.title}</h4>
                <p>{community.summary}</p>
              </article>
            ))}
          </div>
        )}
      </aside>
    </div>
  );
}
