import type { Citation, DocumentRow, GraphPayload, Job, Stats } from "./types";

export const API = process.env.NEXT_PUBLIC_API_BASE ?? "";
const TOKEN_KEY = "graphrag.token";

export function getToken(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(TOKEN_KEY) ?? "";
}

export function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token.trim());
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const token = getToken();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const listDocuments = () =>
  fetch(`${API}/api/documents`, { headers: headers() }).then(json<DocumentRow[]>);

export const addText = (text: string, filename: string) =>
  fetch(`${API}/api/documents`, {
    method: "POST",
    headers: headers(),
    body: new URLSearchParams({ text, filename }),
  }).then(json<{ document_id: string; chunks: number; job_id: string | null; duplicate: boolean }>);

export const addFile = (file: File) => {
  const body = new FormData();
  body.append("file", file);
  return fetch(`${API}/api/documents`, {
    method: "POST",
    headers: headers(),
    body,
  }).then(json<{ document_id: string; chunks: number; job_id: string | null; duplicate: boolean }>);
};

export const removeDocument = (id: string) =>
  fetch(`${API}/api/documents/${id}`, { method: "DELETE", headers: headers() }).then(
    json<{ chunks_removed: number; job_id: string }>,
  );

export const getJob = (id: string) => fetch(`${API}/api/jobs/${id}`, { headers: headers() }).then(json<Job>);

export const getStats = () => fetch(`${API}/api/stats`, { headers: headers() }).then(json<Stats>);

export const getGraph = (params: URLSearchParams) =>
  fetch(`${API}/api/graph?${params.toString()}`, { headers: headers() }).then(json<GraphPayload>);

export type StreamEvent =
  | { type: "meta"; data: Citation }
  | { type: "delta"; data: { text: string } }
  | { type: "error"; data: { message: string } }
  | { type: "done"; data: Record<string, never> };

export async function* askStream(
  question: string,
  mode: string,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API}/api/query?stream=true`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({ question, mode }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`Query failed: ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = frame.match(/^event: (.+)$/m)?.[1];
      const data = frame.match(/^data: (.+)$/m)?.[1];
      if (event && data) {
        yield { type: event as StreamEvent["type"], data: JSON.parse(data) } as StreamEvent;
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}
