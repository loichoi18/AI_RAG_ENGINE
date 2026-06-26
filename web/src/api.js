// Thin API client for the RAG Engine. The base URL is configurable so the same
// build runs against localhost in dev and the deployed API in production.
export const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(API_BASE + path, {
      headers: { 'content-type': 'application/json' },
      ...options,
    })
  } catch {
    throw new Error(`Cannot reach the API at ${API_BASE}. Is it running?`)
  }
  let data = null
  try { data = await res.json() } catch { /* empty body */ }
  if (!res.ok) {
    const msg = data?.detail || data?.error || `Request failed (HTTP ${res.status})`
    throw new Error(msg)
  }
  return data
}

export const api = {
  health: () => request('/v1/health'),
  query: (body) => request('/v1/query', { method: 'POST', body: JSON.stringify(body) }),
  documents: () => request('/v1/documents'),
  ingest: (documents) => request('/v1/ingest', { method: 'POST', body: JSON.stringify({ documents }) }),
  metrics: () => request('/v1/metrics'),
}
