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
  DOCUMENT: "#5f7f8c",
};

const COMMUNITY_COLORS = [
  "#7fa6c9", "#8fbf9b", "#e0a458", "#c58fb5", "#8aa4d6",
  "#d69a7a", "#a3c9a8", "#c9b47f", "#9bb8c4", "#b9a3c9",
];

const colorOf = (node: GraphNode, colorBy: "type" | "community") => {
  if (colorBy === "community" && node.community != null) {
    return COMMUNITY_COLORS[node.community % COMMUNITY_COLORS.length];
  }
  return COLORS[node.type] ?? COLORS.CONCEPT;
};

export default function GraphView({
  nodes,
  links,
  colorBy,
  fitSignal,
  onSelect,
}: {
  nodes: GraphNode[];
  links: GraphLink[];
  colorBy: "type" | "community";
  fitSignal: number;
  onSelect: (node: GraphNode) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const graphRef = useRef<{ zoomToFit: (ms?: number, padding?: number) => void } | null>(null);
  const [size, setSize] = useState({ width: 900, height: 620 });
  const [focus, setFocus] = useState<string | null>(null);

  const active = useMemo(() => {
    if (!focus) return null;
    const set = new Set<string>([focus]);
    for (const link of links) {
      if (link.src === focus) set.add(link.dst);
      if (link.dst === focus) set.add(link.src);
    }
    return set;
  }, [focus, links]);

  // The canvas painter runs every frame; keep the current focus in a ref.
  const activeRef = useRef<Set<string> | null>(active);
  activeRef.current = active;
  const colorByRef = useRef(colorBy);
  colorByRef.current = colorBy;

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const measure = () => setSize({ width: element.clientWidth, height: element.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!nodes.some((node) => node.id === focus)) setFocus(null);
  }, [nodes, focus]);

  const data = useMemo(
    () => ({
      nodes: nodes.map((node) => ({ ...node })),
      links: links.map((link) => ({ ...link, source: link.src, target: link.dst })),
    }),
    [nodes, links],
  );

  // Pull disconnected components toward the middle. Without this, two documents
  // with no relation between them drift to opposite corners of the canvas.
  useEffect(() => {
    const graph = graphRef.current as {
      d3Force?: (name: string, force?: unknown) => unknown;
    } | null;
    if (!graph?.d3Force) return;
    (graph.d3Force("charge") as { strength?: (value: number) => void } | undefined)?.strength?.(-140);
    (graph.d3Force("link") as { distance?: (value: number) => void } | undefined)?.distance?.(80);
    graph.d3Force("centering", (alpha: number) => {
      const cx = size.width / 2;
      const cy = size.height / 2;
      for (const node of data.nodes as (GraphNode & { x?: number; y?: number })[]) {
        node.x = (node.x ?? cx) + (cx - (node.x ?? cx)) * 0.15 * alpha;
        node.y = (node.y ?? cy) + (cy - (node.y ?? cy)) * 0.15 * alpha;
      }
    });
  }, [data, size]);


  useEffect(() => {
    if (fitSignal > 0) graphRef.current?.zoomToFit(400, 90);
  }, [fitSignal]);

  const dim = (id: string) => {
    const set = activeRef.current;
    return set ? (set.has(id) ? 1 : 0.1) : 1;
  };

  return (
    <div ref={container} style={{ position: "absolute", inset: 0 }}>
      <ForceGraph2D
        ref={graphRef as never}
        graphData={data as never}
        width={size.width}
        height={size.height}
        backgroundColor="#0b1310"
        cooldownTicks={160}
        warmupTicks={40}
        d3VelocityDecay={0.32}
        onBackgroundClick={() => setFocus(null)}
        onNodeClick={(node: any) => {
          setFocus((current) => (current === node.id ? null : node.id));
          onSelect(node);
        }}
        linkColor={(link: GraphLink) => {
          if (link.kind === "mention") return "rgba(95,127,140,0.28)";
          if (link.kind === "cooccur") return "rgba(237,231,218,0.1)";
          const set = activeRef.current;
          if (!set) return "rgba(237,231,218,0.16)";
          return set.has(link.src) && set.has(link.dst)
            ? "rgba(143,191,155,0.6)"
            : "rgba(237,231,218,0.04)";
        }}
        linkWidth={(link: GraphLink) => (link.kind === "mention" ? 0.4 : 0.8)}
        linkDirectionalArrowLength={(link: GraphLink) => (link.kind === "mention" ? 0 : 3)}
        linkDirectionalArrowRelPos={1}
        linkLabel={(link: GraphLink) => link.rel}
        nodeLabel={(node: GraphNode) => `${node.name ?? node.id} · ${node.type.toLowerCase()}`}
        nodeCanvasObject={(node: any, ctx, scale) => {
          const alpha = dim(node.id);
          const radius =
            node.type === "DOCUMENT" ? 5.5 : 3.2 + Math.sqrt(Math.max(node.degree, 0));

          ctx.globalAlpha = alpha;
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = colorOf(node, colorByRef.current);
          ctx.fill();
          if (node.type === "DOCUMENT") {
            ctx.lineWidth = 1.2 / scale;
            ctx.strokeStyle = "rgba(237,231,218,0.55)";
            ctx.stroke();
          }

          const focused = activeRef.current;
          if (
            node.type === "DOCUMENT" ||
            scale > 0.7 ||
            (focused !== null && focused.has(node.id))
          ) {
            const fontSize = Math.max(7, 11 / scale);
            ctx.font = `${fontSize}px system-ui, -apple-system, sans-serif`;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillStyle =
              alpha < 1
                ? "rgba(237,231,218,0.18)"
                : node.type === "DOCUMENT"
                  ? "rgba(237,231,218,0.62)"
                  : "rgba(237,231,218,0.9)";
            ctx.fillText(node.name ?? node.id, node.x + radius + 2 / scale, node.y);
          }
          ctx.globalAlpha = 1;
        }}
        nodePointerAreaPaint={(node: any, color, ctx) => {
          const radius =
            node.type === "DOCUMENT" ? 7 : 5 + Math.sqrt(Math.max(node.degree, 0));
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
          ctx.fill();
        }}
      />
    </div>
  );
}
