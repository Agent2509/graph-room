export type Job = {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "failed";
  phase: string;
  done: number;
  total: number;
  message: string;
  error: string;
};

export type DocumentRow = {
  id: string;
  filename: string;
  char_count: number;
  created_at: string;
  chunks: number;
};

export type Citation = {
  mode: string;
  entities: { name: string; type: string; description: string }[];
  edges: { src: string; dst: string; rel: string; description: string }[];
  chunks: { id: string; document_id: string }[];
};

export type GraphNode = {
  id: string;
  type: string;
  description: string;
  degree: number;
  name?: string;
  mentions?: number;
};

export type GraphLink = {
  src: string;
  dst: string;
  rel: string;
  description: string;
  weight: number;
  kind?: "relation" | "mention";
};

export type GraphPayload = { nodes: GraphNode[]; edges: GraphLink[] };

export type Stats = {
  documents: number;
  chunks: number;
  entities: number;
  edges: number;
  communities: number;
  density: number;
  top_entities: { name: string; type: string; degree: number }[];
  community_reports: {
    id: number;
    title: string;
    summary: string;
    findings: string[];
    rating: number;
  }[];
};
