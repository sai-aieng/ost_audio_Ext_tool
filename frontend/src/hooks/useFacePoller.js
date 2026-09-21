import { useEffect, useState } from "react";
import { getFaceResults, getFaceStatus } from "../api/client";

export function useFacePoller(jobId) {
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    let cancelled = false;
    let timer;
    setStatus(null);
    setResults(null);
    setError("");
    setIsPolling(Boolean(jobId));
    async function poll() {
      try {
        const next = await getFaceStatus(jobId);
        if (cancelled) return;
        setStatus(next);
        setError("");
        if (next.status === "failed") {
          setError(next.error || "Face extraction failed.");
          setIsPolling(false);
          return;
        }
        if (next.status === "completed") {
          const output = await getFaceResults(jobId);
          if (cancelled) return;
          setResults(output);
          setIsPolling(false);
          return;
        }
        timer = window.setTimeout(poll, 1500);
      } catch (failure) {
        if (cancelled) return;
        if (failure?.status === 404) {
          setError("Face job is no longer available. Select a video to start again.");
          setIsPolling(false);
          return;
        }
        setError("Face update temporarily unavailable; retrying automatically. " +
          (failure instanceof Error ? failure.message : "Could not retrieve progress."));
        timer = window.setTimeout(poll, 3000);
      }
    }
    if (jobId) void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [jobId]);
  return { status, results, isPolling, error };
}
