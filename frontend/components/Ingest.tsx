"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { addFile, addText, getJob, getLLMKey, listDocuments, removeDocument } from "@/lib/api";
import type { DocumentRow, Job } from "@/lib/types";

import { KEY_CHANGED } from "./KeyPanel";
import { CHANGED } from "./StatusBar";

export default function Ingest() {
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [text, setText] = useState("");
  const [name, setName] = useState("pasted-notes");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [hasKey, setHasKey] = useState(false);
  const timers = useRef<number[]>([]);

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load documents");
    }
  }, []);

  useEffect(() => {
    refresh();
    const syncKey = () => setHasKey(Boolean(getLLMKey()));
    syncKey();
    window.addEventListener(KEY_CHANGED, syncKey);
    return () => {
      window.removeEventListener(KEY_CHANGED, syncKey);
      timers.current.forEach((timer) => window.clearTimeout(timer));
    };
  }, [refresh]);

  const track = useCallback(
    (jobId: string) => {
      const poll = async () => {
        try {
          const job = await getJob(jobId);
          setJobs((previous) => [job, ...previous.filter((item) => item.id !== job.id)].slice(0, 3));
          if (job.status === "done" || job.status === "failed") {
            window.dispatchEvent(new Event(CHANGED));
            refresh();
            return;
          }
        } catch {
          // transient errors while the job runs: keep polling
        }
        timers.current.push(window.setTimeout(poll, 800));
      };
      poll();
    },
    [refresh],
  );

  const submitText = async () => {
    setError("");
    setNotice("");
    if (!text.trim()) {
      setError("Paste some text first.");
      return;
    }
    try {
      const result = await addText(text, name || "pasted-notes");
      if (result.duplicate) {
        setNotice("That text is already indexed.");
      } else {
        setText("");
        if (result.job_id) track(result.job_id);
      }
      refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload failed");
    }
  };

  const submitFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setError("");
    setNotice("");
    for (const file of Array.from(files)) {
      try {
        const result = await addFile(file);
        if (result.duplicate) setNotice(`${file.name} is already indexed.`);
        else if (result.job_id) track(result.job_id);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : `${file.name} failed`);
      }
    }
    refresh();
  };

  const drop = async (id: string) => {
    setError("");
    try {
      const result = await removeDocument(id);
      if (result.job_id) track(result.job_id);
      refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Delete failed");
    }
  };

  return (
    <>
      <section className="stack" aria-labelledby="add-title">
        <h2 id="add-title">Add a document</h2>
        <p className="hint">Text, markdown or PDF. Indexing extracts entities, relations and communities.</p>
        {!hasKey && <p className="notice">Add your Groq key above to ingest or ask.</p>}

        <label className="file-drop">
          <span>Choose a file</span>
          <span aria-hidden="true">＋</span>
          <input
            type="file"
            multiple
            accept=".txt,.md,.markdown,.pdf,.csv,.json"
            onChange={(event) => {
              submitFiles(event.target.files);
              event.target.value = "";
            }}
          />
        </label>

        <input
          className="field"
          value={name}
          aria-label="Document name"
          onChange={(event) => setName(event.target.value)}
        />
        <textarea
          className="field"
          rows={5}
          placeholder="…or paste text here"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <button className="btn" onClick={submitText} disabled={!text.trim() || !hasKey}>
          Add to graph
        </button>

        {notice && <p className="notice">{notice}</p>}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </section>

      {jobs.length > 0 && (
        <section aria-labelledby="jobs-title">
          <h2 id="jobs-title">Indexing</h2>
          <ul className="job-list">
            {jobs.map((job) => (
              <li key={job.id} className="job" data-status={job.status}>
                <div className="job-head">
                  <span>
                    {job.status === "failed" ? job.error : job.message || job.phase || job.status}
                  </span>
                  <span>{job.total ? `${job.done}/${job.total}` : ""}</span>
                </div>
                <div className="bar">
                  <i style={{ width: `${job.total ? (100 * job.done) / job.total : job.status === "done" ? 100 : 8}%` }} />
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section aria-labelledby="docs-title">
        <h2 id="docs-title">Documents</h2>
        {documents.length === 0 ? (
          <p className="hint">Nothing indexed yet. Add a document to build the graph.</p>
        ) : (
          <ul className="doc-list">
            {documents.map((document) => (
              <li key={document.id} className="doc-item">
                <span className="doc-name" title={document.filename}>
                  {document.filename}
                </span>
                <span className="doc-meta">{document.chunks} ch</span>
                <button
                  className="btn btn-quiet"
                  aria-label={`Delete ${document.filename}`}
                  onClick={() => drop(document.id)}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
