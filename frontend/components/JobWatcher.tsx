"use client";

import { useEffect, useRef } from "react";

import { listJobs } from "@/lib/api";
import type { Job } from "@/lib/types";

import { CHANGED } from "./StatusBar";

export const JOBS_EVENT = "graphrag:jobs";
export const KICK = "graphrag:kick";

/**
 * Lives in the layout, so polling survives page navigation and reloads.
 * Emits the job list; emits CHANGED when running work finishes.
 */
export default function JobWatcher() {
  const wasActive = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const tick = async () => {
      let jobs: Job[] = [];
      try {
        jobs = await listJobs();
      } catch {
        jobs = [];
      }
      if (cancelled) return;

      const running = jobs.some((job) => job.status === "queued" || job.status === "running");
      window.dispatchEvent(new CustomEvent(JOBS_EVENT, { detail: jobs }));
      if (wasActive.current && !running) window.dispatchEvent(new Event(CHANGED));
      wasActive.current = running;
      timer = window.setTimeout(tick, running ? 1200 : 5000);
    };

    const kick = () => {
      window.clearTimeout(timer);
      tick();
    };
    window.addEventListener(KICK, kick);
    tick();

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      window.removeEventListener(KICK, kick);
    };
  }, []);

  return null;
}
