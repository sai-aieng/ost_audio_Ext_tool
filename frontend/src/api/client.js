const API_BASE = "http://127.0.0.1:8001/api/v1";

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
    throw new Error(
      "API request failed with " + response.status + " " + response.statusText + suffix,
    );
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
  return request("/status/" + encodeURIComponent(jobId));
}

export async function getResults(jobId) {
  return request("/results/" + encodeURIComponent(jobId));
}

export async function getSavedExtractions() {
  return request("/saved-extractions");
}

export async function getSavedExtraction(jobId) {
  return request("/saved-extractions/" + encodeURIComponent(jobId));
}

export async function getAudioResults(jobId) {
  return request("/audio/results/" + encodeURIComponent(jobId));
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
