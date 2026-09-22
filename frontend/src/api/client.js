/**
 * API Client for interacting with the FastAPI Grounded RAG backend.
 */

const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  (typeof window !== 'undefined' && window.location.hostname !== 'localhost'
    ? 'https://rag-5djx.onrender.com'
    : 'http://localhost:8000');
export const DEFAULT_API_KEY = 'rag-secret-key-prod-2026';

/**
 * Execute query through RAG pipeline.
 */
export async function queryAPI({
  query,
  conversationId,
  metadataFilter = null,
  thresholdOverride = null,
  apiKey = DEFAULT_API_KEY,
}) {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey,
    },
    body: JSON.stringify({
      query,
      conversation_id: conversationId,
      metadata_filter: metadataFilter,
      threshold_override: thresholdOverride,
    }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `Server error: ${response.status}`);
  }

  return response.json();
}

/**
 * Upload and ingest document file (.pdf, .txt, .md).
 */
export async function ingestFile(file, apiKey = DEFAULT_API_KEY) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: 'POST',
    headers: {
      'X-API-Key': apiKey,
    },
    body: formData,
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `Upload failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * List all ingested documents.
 */
export async function fetchDocuments() {
  const response = await fetch(`${API_BASE_URL}/documents`);
  if (!response.ok) {
    throw new Error(`Failed to load documents (${response.status})`);
  }
  return response.json();
}

/**
 * Health check endpoint.
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed (${response.status})`);
  }
  return response.json();
}
