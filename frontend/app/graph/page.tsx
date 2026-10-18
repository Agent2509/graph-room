"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";

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
  const [minDegree, setMinDegree] = useState(false);
  const [showMentions, setShowMentions] = useState(true);
  const [showCooccur, setShowCooccur] = useState(true);
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      // Always load every node; the "hide isolated" toggle filters here so a
      // graph with zero edges can never render as empty by accident.
      const params = new URLSearchParams({ limit: "600" });
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

  const degreeById = useMemo(() => {
    const degrees = new Map<string, number>();
    for (const link of links) {
      degrees.set(link.src, (degrees.get(link.src) ?? 0) + 1);
      degrees.set(link.dst, (degrees.get(link.dst) ?? 0) + 1);
    }
    return degrees;
  }, [links]);

  const visibleNodes = useMemo(() => {
    const base = nodes
      .filter((node) => showMentions || node.type !== "DOCUMENT")
      .filter((node) => !hiddenTypes.has(node.type));
    if (!minDegree || links.length === 0) return base;

    const connected = base.filter(
      (node) => node.type !== "DOCUMENT" && (degreeById.get(node.id) ?? 0) > 0,
    );
    const ids = new Set(connected.map((node) => node.id));
    const documents = base.filter(
      (node) =>
        node.type === "DOCUMENT" &&
        links.some((link) => link.src === node.id && ids.has(link.dst)),
    );
    return [...connected, ...documents];
  }, [degreeById, hiddenTypes, links, minDegree, nodes, showMentions]);

  const visibleLinks = useMemo(() => {
    const keep = new Set(visibleNodes.map((node) => node.id));
    return links.filter(
      (link) =>
        keep.has(link.src) &&
        keep.has(link.dst) &&
        (showMentions || link.kind !== "mention") &&
        (showCooccur || link.kind !== "cooccur"),
    );
  }, [links, showCooccur, showMentions, visibleNodes]);

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
          <label>
            <input
              type="checkbox"
              checked={showMentions}
              onChange={(event) => setShowMentions(event.target.checked)}
            />
            document links
          </label>
          <label>
            <input
              type="checkbox"
              checked={showCooccur}
              onChange={(event) => setShowCooccur(event.target.checked)}
            />
            co-occurrence
          </label>
          <span>
            {visibleNodes.length}
            {visibleNodes.length === nodes.length ? " shown" : ` of ${nodes.length}`}
          </span>
        </div>
        {visibleNodes.length === 0 ? (
          <div className="empty" style={{ padding: "25vh 24px 0" }}>
            <h1>The graph is empty.</h1>
            <p>Add a document in the workspace. Entities and relations appear here once indexing finishes.</p>
          </div>
        ) : (
          <GraphView nodes={visibleNodes} links={visibleLinks} onSelect={setSelected} />
        )}
      </section>

      <aside className="inspector">
        {error && <p className="error">{error}</p>}
        {selected ? (
          <>
            <h3>{selected.name ?? selected.id}</h3>
            <p className="kind">{selected.type.toLowerCase()}</p>
            {selected.type === "DOCUMENT" ? (
              <p>
                {selected.mentions ?? 0} entity mentions. Every line from this document to
                an entity is a structural link, not an extracted relation.
              </p>
            ) : (
              <p>{selected.description || "No description recorded."}</p>
            )}
            <p className="doc-meta">
              {selected.type === "DOCUMENT"
                ? `${selected.mentions ?? 0} mentions`
                : `${selected.degree} relations`}
            </p>
          </>
        ) : (
          <>
            <h3>Reading the map</h3>
            <p>
              Pick an entity to read its description. Hover a relation to see what the source text
              says about it.
            </p>
            <div className="legend">
              {["PERSON", "ORGANIZATION", "TECHNOLOGY", "CONCEPT", "LOCATION", "DOCUMENT"].map((type) => (
                <button
                  key={type}
                  type="button"
                  aria-pressed={!hiddenTypes.has(type)}
                  title={`Toggle ${type.toLowerCase()}`}
                  onClick={() =>
                    setHiddenTypes((current) => {
                      const next = new Set(current);
                      if (next.has(type)) next.delete(type);
                      else next.add(type);
                      return next;
                    })
                  }
                >
                  <i
                    style={{
                      background:
                        {
                          PERSON: "#e0a458",
                          ORGANIZATION: "#8fbf9b",
                          TECHNOLOGY: "#7fa6c9",
                          CONCEPT: "#b9b3a3",
                          LOCATION: "#c4693f",
                          DOCUMENT: "#5f7f8c",
                        }[type] ?? "#b9b3a3",
                    }}
                  />
                  {type.charAt(0) + type.slice(1).toLowerCase()}
                </button>
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
