"use client";

import { useEffect, useRef, useState } from "react";

import { askStream, getLLMKey, listDocuments } from "@/lib/api";
import { toHtml } from "@/lib/markdown";
import { CHANGED } from "./StatusBar";
import { KEY_CHANGED } from "./KeyPanel";
import type { Citation } from "@/lib/types";

type Turn = {
  role: "user" | "assistant";
  text: string;
  meta?: Citation;
  streaming?: boolean;
  failed?: boolean;
};

const EXAMPLES = [
  "How is Alice Zhao connected to CureX?",
  "What did NexusLabs acquire, and why?",
  "What are the main themes across this corpus?",
];

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState("auto");
  const [busy, setBusy] = useState(false);
  const [hasKey, setHasKey] = useState(false);
  const [documents, setDocuments] = useState<{ id: string; filename: string }[]>([]);
  const [documentId, setDocumentId] = useState("");
  const abort = useRef<AbortController | null>(null);
  const transcript = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcript.current?.scrollTo({ top: transcript.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  useEffect(() => {
    const syncKey = () => setHasKey(Boolean(getLLMKey()));
    syncKey();
    window.addEventListener(KEY_CHANGED, syncKey);
    return () => {
      window.removeEventListener(KEY_CHANGED, syncKey);
      abort.current?.abort();
    };
  }, []);

  useEffect(() => {
    const load = () =>
      listDocuments()
        .then((rows) => {
          setDocuments(rows);
          setDocumentId((current) =>
            current && rows.some((row) => row.id === current) ? current : "",
          );
        })
        .catch(() => setDocuments([]));
    load();
    window.addEventListener(CHANGED, load);
    return () => window.removeEventListener(CHANGED, load);
  }, []);

  const send = async (question: string) => {
    if (!question.trim() || busy) return;
    setBusy(true);
    setDraft("");
    setTurns((previous) => [
      ...previous,
      { role: "user", text: question },
      { role: "assistant", text: "", streaming: true },
    ]);

    const controller = new AbortController();
    abort.current = controller;
    let answer = "";

    try {
      const scope = mode === "global" ? "" : documentId;
      for await (const event of askStream(question, mode, scope, controller.signal)) {
        if (event.type === "meta") {
          setTurns((previous) => {
            const next = [...previous];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, meta: event.data };
            return next;
          });
        } else if (event.type === "error") {
          answer = event.data.message;
          setTurns((previous) => {
            const next = [...previous];
            next[next.length - 1] = {
              ...next[next.length - 1],
              text: event.data.message,
              failed: true,
            };
            return next;
          });
        } else if (event.type === "delta") {
          answer += event.data.text;
          setTurns((previous) => {
            const next = [...previous];
            next[next.length - 1] = { ...next[next.length - 1], text: answer };
            return next;
          });
        }
      }
    } catch (cause) {
      if (!controller.signal.aborted) {
        setTurns((previous) => {
          const next = [...previous];
          next[next.length - 1] = {
            ...next[next.length - 1],
            text: cause instanceof Error ? cause.message : "The query failed.",
            failed: true,
          };
          return next;
        });
      }
    } finally {
      setTurns((previous) => {
        const next = [...previous];
        const last = next[next.length - 1];
        if (last?.role === "assistant") next[next.length - 1] = { ...last, streaming: false };
        return next;
      });
      setBusy(false);
    }
  };

  return (
    <>
      <div className="transcript" ref={transcript}>
        {turns.length === 0 ? (
          <div className="empty">
            <h1>Ask the corpus.</h1>
            <p>
              Answers are grounded in the knowledge graph and the source chunks behind it. Every
              claim carries the entity, relation or chunk it came from.
            </p>
            {!hasKey && (
              <p className="notice" style={{ marginBottom: 16 }}>
                Add your Groq key in the sidebar to ask questions.
              </p>
            )}
            <div className="examples">
              {EXAMPLES.map((example) => (
                <button key={example} className="example" onClick={() => send(example)}>
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          turns.map((turn, index) =>
            turn.role === "user" ? (
              <div key={index} className="turn turn-user">
                {turn.text}
              </div>
            ) : (
              <article key={index} className="turn">
                <div
                  className={`turn-answer${turn.failed ? " error" : ""}`}
                  aria-live="polite"
                  dangerouslySetInnerHTML={{ __html: toHtml(turn.text) }}
                />
                {turn.streaming && <span className="caret" aria-hidden="true" />}
                {turn.meta && (
                  <div className="citations">
                    <span className="tag tag-mode">{turn.meta.mode}</span>
                    {turn.meta.entities.slice(0, 8).map((entity) => (
                      <span key={entity.name} className="tag" title={entity.description}>
                        {entity.name}
                      </span>
                    ))}
                    {turn.meta.chunks.slice(0, 3).map((chunk) => (
                      <span key={chunk.id} className="chunk-id">
                        {chunk.id}
                      </span>
                    ))}
                  </div>
                )}
              </article>
            ),
          )
        )}
      </div>

      <div className="composer">
        <select
          value={documentId}
          onChange={(event) => setDocumentId(event.target.value)}
          aria-label="Document scope"
          disabled={mode === "global"}
          title={mode === "global" ? "Global search covers the whole corpus" : "Limit retrieval to one document"}
        >
          <option value="">all documents</option>
          {documents.map((document) => (
            <option key={document.id} value={document.id}>
              {document.filename}
            </option>
          ))}
        </select>
        <select value={mode} onChange={(event) => setMode(event.target.value)} aria-label="Search mode">
          <option value="auto">auto</option>
          <option value="local">local</option>
          <option value="global">global</option>
        </select>
        <textarea
          value={draft}
          placeholder="Ask a question about the indexed documents"
          aria-label="Question"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send(draft);
            }
          }}
        />
        <button
          className="btn btn-primary"
          onClick={() => send(draft)}
          disabled={busy || !draft.trim() || !hasKey}
        >
          {busy ? "Thinking…" : "Ask"}
        </button>
      </div>
    </>
  );
}
