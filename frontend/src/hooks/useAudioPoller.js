import { useEffect, useState } from "react";

import { getAudioResults, getStatus } from "../api/client";

export function useAudioPoller(jobId) {
  const [status, setStatus] = useState(null);
  const [transcript, setTranscript] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!jobId) {
      setStatus(null);
      setTranscript(null);
      setError("");
      return undefined;
    }
    let cancelled = false;
    let timerId;
    async function poll() {
      try {
        const nextStatus = await getStatus(jobId);
        if (cancelled) return;
        setStatus(nextStatus);
        if (nextStatus.status === "completed") {
          setTranscript(await getAudioResults(jobId));
          return;
        }
        if (nextStatus.status === "failed") {
          setError(nextStatus.error || "Audio transcription failed.");
          return;
        }
        timerId = window.setTimeout(() => void poll(), 1500);
      } catch (pollError) {
        if (!cancelled) setError(pollError instanceof Error ? pollError.message : "Could not retrieve audio transcription.");
      }
    }
    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [jobId]);

  return { status, transcript, error };
}
