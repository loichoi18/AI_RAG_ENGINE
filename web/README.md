# RAG Engine — Web Frontend

A professional React + Vite + Tailwind UI for the RAG Engine API. Ask questions
over the knowledge base and see grounded answers with citations, confidence,
sources, and a retrieval inspector — plus live knowledge-base and metrics panels.

## Develop

```bash
npm install
cp .env.example .env          # point VITE_API_URL at your API (default localhost:8000)
npm run dev                   # http://localhost:5173
```

## Build

```bash
npm run build                 # static output in dist/
```

## Configuration

`VITE_API_URL` — base URL of the RAG Engine API. Set it in `.env` for local dev
and as an environment variable in your host (e.g. Vercel) for production.

The UI talks to these API endpoints: `POST /v1/query`, `GET /v1/documents`,
`POST /v1/ingest`, `GET /v1/metrics`, `GET /v1/health`.
