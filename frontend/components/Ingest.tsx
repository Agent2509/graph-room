"use client";

import { useCallback, useEffect, useState } from "react";

import { addFile, addText, listDocuments, removeDocument } from "@/lib/api";
import type { DocumentRow, Job } from "@/lib/types";

import { JOBS_EVENT, KICK } from "./JobWatcher";
import { CHANGED } from "./StatusBar";

export default function Ingest() {
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [text, setText] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load documents");
    }
  }, []);

  useEffect(() => {
    if (!jobs.length) return;
    const active = jobs.some((job) => job.status === "queued" || job.status === "running");
    if (active) return;
    const timer = window.setTimeout(() => setJobs([]), 8000);
    return () => window.clearTimeout(timer);
  }, [jobs]);

  useEffect(() => {
    refresh();
    const onJobs = (event: Event) => {
      setJobs((event as CustomEvent).detail as Job[]);
    };
    window.addEventListener(JOBS_EVENT, onJobs);
    window.addEventListener(CHANGED, refresh);
    return () => {
      window.removeEventListener(JOBS_EVENT, onJobs);
      window.removeEventListener(CHANGED, refresh);
    };
  }, [refresh]);

  const submitText = async () => {
    setError("");
    setNotice("");
    if (!text.trim()) {
      setError("Paste some text first.");
      return;
    }
    try {
      const result = await addText(text, `pasted ${new Date().toLocaleString()}`);
      if (result.duplicate) {
        setNotice("That text is already indexed.");
      } else {
        setText("");
        if (result.job_id) window.dispatchEvent(new Event(KICK));
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
        else if (result.job_id) window.dispatchEvent(new Event(KICK));
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : `${file.name} failed`);
      }
    }
    refresh();
  };

  const drop = async (id: string, filename: string) => {
    setError("");
    if (!window.confirm(`Delete ${filename}? Its entities and links go with it.`)) return;
    try {
      const result = await removeDocument(id);
      if (result.job_id) window.dispatchEvent(new Event(KICK));
      // Counts live in the header and the graph page: refresh them even when the
      // delete did not queue a job (fast extraction mode).
      window.dispatchEvent(new Event(CHANGED));
      refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Delete failed");
    }
  };

  // Active work plus the last finished job; a long history is noise.
  const running = jobs.filter((job) => job.status === "queued" || job.status === "running");
  const lastDone = jobs.find((job) => job.status === "done" || job.status === "failed");
  const shownJobs = running.length ? running : lastDone ? [lastDone] : [];

  return (
    <>
      <section className="stack" aria-labelledby="add-title">
        <h2 id="add-title">Add a document</h2>
        <p className="hint">Text, markdown or PDF. Indexing extracts entities, relations and communities.</p>

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

        <textarea
          className="field"
          rows={5}
          placeholder="…or paste text here"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <button className="btn" onClick={submitText} disabled={!text.trim()}>
          Add to graph
        </button>

        {notice && <p className="notice">{notice}</p>}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </section>

      {shownJobs.length > 0 && (
        <section aria-labelledby="jobs-title">
          <h2 id="jobs-title">Indexing</h2>
          <ul className="job-list">
            {shownJobs.map((job) => (
              <li key={job.id} className="job" data-status={job.status}>
                <div className="job-head">
                  <span>
                    {job.status === "failed"
                      ? job.error
                      : `${job.kind} · ${job.message || job.phase || job.status}`}
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
                <button
                  className="btn btn-quiet"
                  aria-label={`Delete ${document.filename}`}
                  onClick={() => drop(document.id, document.filename)}
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
