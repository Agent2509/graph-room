"use client";

import { useCallback, useEffect, useState } from "react";

import { API, getStats, getToken, setToken } from "@/lib/api";
import type { Stats } from "@/lib/types";

export const CHANGED = "graphrag:changed";

export default function StatusBar() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [token, setTokenValue] = useState("");
  const [authRequired, setAuthRequired] = useState(false);

  const load = useCallback(() => {
    getStats()
      .then(setStats)
      .catch(() => setStats(null));
    fetch(`${API}/api/health`)
      .then((response) => response.json())
      .then((body) => setAuthRequired(Boolean(body?.features?.auth_required)))
      .catch(() => setAuthRequired(false));
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
          <b>{stats?.documents ?? "–"}</b> documents
        </span>
      </div>
      {authRequired && (
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
      )}
    </div>
  );
}
