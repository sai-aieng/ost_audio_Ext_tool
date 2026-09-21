import { useEffect, useState } from "react";

import { getResults, getStatus } from "../api/client";

const POLL_INTERVAL_MS = 1500;
const RETRY_INTERVAL_MS = 3000;

export function useJobPoller(jobId) {
  const [jobStatus, setJobStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timerId;

    function stopPolling() {
      window.clearTimeout(timerId);
      if (!cancelled) {
        setIsPolling(false);
      }
    }

    async function poll() {
      if (!jobId || cancelled) {
        return;
      }
      try {
        const nextStatus = await getStatus(jobId);
        if (cancelled) {
          return;
        }
        setJobStatus(nextStatus);
        setError("");
        if (nextStatus.status === "completed") {
          const nextResults = await getResults(jobId);
          if (cancelled) return;
          setResults(nextResults);
          stopPolling();
        } else if (nextStatus.status === "failed") {
          setError(nextStatus.error || "Video processing failed.");
          stopPolling();
        } else {
          // Schedule after the request finishes so polls cannot overlap.
          timerId = window.setTimeout(() => void poll(), POLL_INTERVAL_MS);
        }
      } catch (pollError) {
        if (cancelled) return;
        if (pollError?.status === 404) {
          setError("OCR job is no longer available. The backend may have restarted. Reset and upload again.");
          stopPolling();
          return;
        }
        setError("OCR update temporarily unavailable; retrying automatically. " +
          (pollError instanceof Error ? pollError.message : "Could not retrieve job status."));
        // Keep the last progress/results and remain in the polling state.
        timerId = window.setTimeout(() => void poll(), RETRY_INTERVAL_MS);
      }
    }

    if (!jobId) {
      setJobStatus(null);
      setResults(null);
      setError("");
      setIsPolling(false);
      return () => {
        cancelled = true;
      };
    }

    setJobStatus(null);
    setResults(null);
    setError("");
    setIsPolling(true);
    void poll();

    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [jobId]);

  return { jobStatus, results, isPolling, error };
}
