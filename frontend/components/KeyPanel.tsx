"use client";

import { useEffect, useState } from "react";

import { getLLMKey, setLLMKey } from "@/lib/api";

export const KEY_CHANGED = "graphrag:key";

export default function KeyPanel() {
  const [key, setKey] = useState("");

  useEffect(() => setKey(getLLMKey()), []);

  return (
    <section className="stack" aria-labelledby="key-title">
      <h2 id="key-title">Groq key</h2>
      <p className="hint">
        Your key, your quota. It stays in this browser and travels with each request. The
        server keeps it in memory for one run and never writes it down.
      </p>
      <input
        className="field"
        type="password"
        value={key}
        placeholder="gsk_..."
        aria-label="Groq API key"
        onChange={(event) => {
          setKey(event.target.value);
          setLLMKey(event.target.value);
          window.dispatchEvent(new Event(KEY_CHANGED));
        }}
      />
      <a className="hint" href="https://console.groq.com/keys" target="_blank" rel="noreferrer">
        Get a free key
      </a>
    </section>
  );
}
