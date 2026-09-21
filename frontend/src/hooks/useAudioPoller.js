import { useEffect, useState } from "react";

import { getAudioResults, getStatus } from "../api/client";

export function useAudioPoller(jobId) {
  const [status, setStatus] = useState(null);
  const [transcript, setTranscript] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setStatus(null);
    setTranscript(null);
    setError("");
    if (!jobId) {
      return undefined;
    }
    let cancelled = false;
    let timerId;
    async function poll() {
      try {
        const nextStatus = await getStatus(jobId);
        if (cancelled) return;
        setStatus(nextStatus);
        setError("");
        if (nextStatus.status === "completed") {
          const result = await getAudioResults(jobId);
          if (!cancelled) setTranscript(result);
          return;
        }
        if (nextStatus.status === "failed") {
          setError(nextStatus.error || "Audio transcription failed.");
          return;
        }
        timerId = window.setTimeout(() => void poll(), 1500);
      } catch (pollError) {
        if (cancelled) return;
        if (pollError?.status === 404) {
          setStatus(null);
          setError("Audio job is no longer available. The backend may have restarted. Reset and upload the video again.");
          return;
        }
        setError("Audio status unavailable; retrying. " +
          (pollError instanceof Error ? pollError.message : "Could not reach the backend."));
        timerId = window.setTimeout(() => void poll(), 3000);
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
