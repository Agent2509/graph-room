"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

import type { GraphLink, GraphNode } from "@/lib/types";

const COLORS: Record<string, string> = {
  PERSON: "#e0a458",
  ORGANIZATION: "#8fbf9b",
  TECHNOLOGY: "#7fa6c9",
  PRODUCT: "#7fa6c9",
  CONCEPT: "#b9b3a3",
  LOCATION: "#c4693f",
  EVENT: "#c58fb5",
};

export default function GraphView({
  nodes,
  links,
  onSelect,
}: {
  nodes: GraphNode[];
  links: GraphLink[];
  onSelect: (node: GraphNode) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const graphRef = useRef<{ zoomToFit: (ms?: number, padding?: number) => void } | null>(null);
  const [size, setSize] = useState({ width: 900, height: 620 });

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const measure = () =>
      setSize({ width: element.clientWidth, height: element.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const data = useMemo(
    () => ({
      nodes: nodes.map((node) => ({ ...node })),
      links: links.map((link) => ({ ...link, source: link.src, target: link.dst })),
    }),
    [nodes, links],
  );

  return (
    <div ref={container} style={{ position: "absolute", inset: 0 }}>
      <ForceGraph2D
        ref={graphRef as never}
        graphData={data}
        width={size.width}
        height={size.height}
        backgroundColor="#0b1310"
        nodeRelSize={4}
        nodeVal={(node: GraphNode) => 1 + Math.sqrt(node.degree)}
        nodeColor={(node: GraphNode) => COLORS[node.type] ?? COLORS.CONCEPT}
        nodeLabel={(node: GraphNode) => `${node.id} · ${node.type.toLowerCase()}`}
        linkColor={() => "rgba(237,231,218,0.16)"}
        linkWidth={0.6}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
        linkLabel={(link: GraphLink) => link.rel}
        onNodeClick={(node: GraphNode) => onSelect(node)}
        cooldownTicks={160}
        warmupTicks={40}
        onEngineStop={() => graphRef.current?.zoomToFit(500, 70)}
        d3VelocityDecay={0.32}
      />
    </div>
  );
}
