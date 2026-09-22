const API_BASE = (import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.PROD ? "/api/v1" : "http://127.0.0.1:8001/api/v1")).replace(/\/$/, "");

export function uploadFaceVideo(file) {
  const body = new FormData();
  body.append("file", file);
  return request("/faces/extract", { method: "POST", body });
}

export function getFaceStatus(id) {
  return request("/faces/status/" + encodeURIComponent(id), { signal: AbortSignal.timeout(10000) });
}

export function getFaceResults(id) {
  return request("/faces/results/" + encodeURIComponent(id), { signal: AbortSignal.timeout(10000) });
}

export function getFaceImageUrl(id, track) {
  return API_BASE + "/faces/images/" + encodeURIComponent(id) + "/" + encodeURIComponent(track);
}

export function getFaceDownloadUrl(id) {
  return API_BASE + "/faces/download/" + encodeURIComponent(id);
}

async function readResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(API_BASE + path, options);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error("Could not reach the Video OCR API: " + message);
  }

  const body = await readResponse(response);
  if (!response.ok) {
    let detail = "";
    if (body && typeof body === "object" && "detail" in body) {
      detail = String(body.detail);
    } else if (typeof body === "string" && body.trim()) {
      detail = body.trim();
    }
    const suffix = detail ? ": " + detail : "";
    const error = new Error(
      "API request failed with " + response.status + " " + response.statusText + suffix,
    );
    error.status = response.status;
    throw error;
  }
  return body;
}

export async function uploadVideo(file) {
  const formData = new FormData();
  formData.append("file", file);
  return request("/upload", {
    method: "POST",
    body: formData,
  });
}

export async function startProcessing(jobId, opts) {
  return request("/process/" + encodeURIComponent(jobId), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
}

export async function getStatus(jobId) {
  return request("/status/" + encodeURIComponent(jobId), { signal: AbortSignal.timeout(10000) });
}

export async function getResults(jobId) {
  return request("/results/" + encodeURIComponent(jobId), { signal: AbortSignal.timeout(10000) });
}

export async function getSavedExtractions() {
  return request("/saved-extractions");
}

export async function getSavedExtraction(jobId) {
  return request("/saved-extractions/" + encodeURIComponent(jobId));
}

export async function getAudioResults(jobId) {
  return request("/audio/results/" + encodeURIComponent(jobId), { signal: AbortSignal.timeout(10000) });
}

export function getDownloadUrl(jobId, format) {
  return (
    API_BASE +
    "/results/" +
    encodeURIComponent(jobId) +
    "/download?format=" +
    encodeURIComponent(format)
  );
}

export async function deleteJob(jobId) {
  return request("/jobs/" + encodeURIComponent(jobId), {
    method: "DELETE",
  });
}

// Gemini has independent uploads and job state; credentials stay on the server.
export function getGeminiConfig() { return request('/gemini/config'); }
export function analyzeWithGemini(file, options) {
  const body = new FormData();
  body.append('file', file);
  body.append('include_dialogue', String(options.dialogue));
  body.append('include_faces', String(options.faces));
  return request('/gemini/analyze', { method: 'POST', body });
}
export function getGeminiStatus(id) { return request('/gemini/status/' + encodeURIComponent(id), { signal: AbortSignal.timeout(15000) }); }
export function getGeminiResults(id) { return request('/gemini/results/' + encodeURIComponent(id), { signal: AbortSignal.timeout(15000) }); }
export function geminiDownloadUrl(id, format) { return API_BASE + '/gemini/download/' + encodeURIComponent(id) + '/' + format; }
export function geminiFaceUrl(id, face) { return API_BASE + '/gemini/images/' + encodeURIComponent(id) + '/' + encodeURIComponent(face); }
