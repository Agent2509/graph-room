"use client";

import { useEffect, useState } from "react";

import { getLLMKey, setLLMKey } from "@/lib/api";

export const KEY_CHANGED = "graphrag:key";

export default function KeyPanel() {
  const [key, setKey] = useState("");
  const [editing, setEditing] = useState(false);

  useEffect(() => setKey(getLLMKey()), []);

  const hasKey = Boolean(key);

  return (
    <section className="stack" aria-labelledby="key-title">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2 id="key-title" style={{ margin: 0 }}>Groq key</h2>
        {hasKey && !editing && (
          <button className="btn btn-quiet" onClick={() => setEditing(true)}>
            change
          </button>
        )}
      </div>

      {hasKey && !editing ? (
        <p className="hint">Key set for this browser. Answers spend your quota.</p>
      ) : (
        <>
          <p className="hint">Your key, your quota. Stored in this browser only.</p>
          <input
            className="field"
            type="password"
            value={key}
            placeholder="gsk_..."
            aria-label="Groq API key"
            autoFocus={editing}
            onChange={(event) => {
              setKey(event.target.value);
              setLLMKey(event.target.value);
              window.dispatchEvent(new Event(KEY_CHANGED));
            }}
            onBlur={() => setEditing(false)}
          />
          <a className="hint" href="https://console.groq.com/keys" target="_blank" rel="noreferrer">
            Get a free key
          </a>
        </>
      )}
    </section>
  );
}
