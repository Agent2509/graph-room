"use client";

import { useCallback, useEffect, useState } from "react";

import { getStats, getToken, setToken } from "@/lib/api";
import type { Stats } from "@/lib/types";

export const CHANGED = "graphrag:changed";

export default function StatusBar() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [token, setTokenValue] = useState("");

  const load = useCallback(() => {
    getStats()
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  useEffect(() => {
    setTokenValue(getToken());
    load();
    const handler = () => load();
    window.addEventListener(CHANGED, handler);
    return () => window.removeEventListener(CHANGED, handler);
  }, [load]);

  return (
    <div className="topbar-right">
      <div className="counts" aria-live="polite">
        <span>
          <b>{stats?.entities ?? "–"}</b> entities
        </span>
        <span>
          <b>{stats?.edges ?? "–"}</b> relations
        </span>
        <span>
          <b>{stats?.cooccurrences ?? "–"}</b> co-occurrences
        </span>
        <span>
          <b>{stats?.chunks ?? "–"}</b> chunks
        </span>
      </div>
      <label className="token">
        token
        <input
          type="password"
          value={token}
          placeholder="not set"
          aria-label="API token"
          onChange={(event) => {
            setTokenValue(event.target.value);
            setToken(event.target.value);
            load();
          }}
        />
      </label>
    </div>
  );
}
