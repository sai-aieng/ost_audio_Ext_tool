import { useEffect, useRef, useState } from "react";

import { getResults, getStatus } from "../api/client";

const POLL_INTERVAL_MS = 1500;

export function useJobPoller(jobId) {
  const [jobStatus, setJobStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState("");
  const intervalRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    function stopPolling() {
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
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
        if (nextStatus.status === "completed") {
          const nextResults = await getResults(jobId);
          if (!cancelled) {
            setResults(nextResults);
          }
          stopPolling();
        } else if (nextStatus.status === "failed") {
          setError(nextStatus.error || "Video processing failed.");
          stopPolling();
        }
      } catch (pollError) {
        if (!cancelled) {
          setError(
            pollError instanceof Error
              ? pollError.message
              : "Could not retrieve job status.",
          );
        }
        stopPolling();
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
    intervalRef.current = window.setInterval(() => {
      void poll();
    }, POLL_INTERVAL_MS);
    void poll();

    return () => {
      cancelled = true;
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [jobId]);

  return { jobStatus, results, isPolling, error };
}
