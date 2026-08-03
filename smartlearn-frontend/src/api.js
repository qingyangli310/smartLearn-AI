export const API =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

export const CHAT_ID = "day2-demo";

async function readJSON(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail =
      data.detail || `Request failed (${response.status})`
    throw new Error(detail);
  }

  return data
}

export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(
    `${API}/upload?chat_id=${encodeURIComponent(CHAT_ID)}`,
    { method: "POST", body: formData },
  );

  return readJSON(res);
}

export async function askQuestion(message) {
  const res = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, chat_id: CHAT_ID }),
  });

  return readJSON(res);
}

export function getDocumentFileURL(page = 1, previewKey = null) {
  const safePage = Number.isInteger(page) && page > 0 ? page : 1;
  const refreshQuery = previewKey === null
    ? ""
    : `?preview=${encodeURIComponent(previewKey)}`;
  return `${API}/documents/${encodeURIComponent(CHAT_ID)}/file${refreshQuery}#page=${safePage}`;
}
