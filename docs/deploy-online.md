# Deploying the RAG Engine online

This guide deploys a publicly accessible version of the system:

- **Frontend** (React, `web/`) → **Vercel** (free)
- **Vector store** → **Qdrant Cloud** (free 1 GB cluster)
- **LLM** → **OpenAI or Anthropic** (hosted API, pay-as-you-go)
- **API** (FastAPI) → a host with enough memory (see the note below)

> For purely local testing with Docker + Ollama, see
> [`deployment.md`](deployment.md) instead — that path needs no cloud accounts.

## Important: the API needs real memory

The API loads two transformer models at runtime — the embedding model
(`bge-base`) and the cross-encoder reranker (`bge-reranker`) — on top of PyTorch.
In practice this needs about **1.5–2 GB of RAM**. Free 512 MB tiers (Render Free,
Fly shared-cpu-256) will run out of memory.

Realistic options for the API host:

| Host | Memory | Cost | Notes |
|---|---|---|---|
| **Oracle Cloud Always Free** | up to 24 GB (ARM) | Free | Best truly-free option; run `docker compose` on the VM. More setup. |
| **Render** (Standard) | 2 GB | ~paid | Simplest managed Docker deploy. |
| **Railway** | configurable | trial credit then paid | Easy Docker deploy. |
| **Any small VPS** (Hetzner, etc.) | 2–4 GB | ~cheap | Full control via Docker. |

The **frontend (Vercel), vector store (Qdrant Cloud), and LLM key** stay cheap or
free regardless. Only the model-serving API needs the memory.

---

## Step 1 — Qdrant Cloud (vector store)

1. Create a free cluster at <https://cloud.qdrant.io>.
2. Copy the **cluster URL host** (e.g. `xxxx-xxxx.aws.cloud.qdrant.io`) and an
   **API key**.

You'll pass these to the API as:

```
RAG_QDRANT__HOST=xxxx-xxxx.aws.cloud.qdrant.io
RAG_QDRANT__HTTPS=true
RAG_QDRANT__API_KEY=<your-qdrant-api-key>
```

## Step 2 — An LLM API key

Create a key from OpenAI or Anthropic. You'll set:

```
RAG_LLM__PROVIDER=openai            # or: anthropic
RAG_LLM__API_KEY=<your-llm-api-key>
```

## Step 3 — Deploy the API (managed example: Render)

1. Push the repo to GitHub.
2. Render → **New → Web Service** → connect the repo.
3. Environment: **Docker**. Dockerfile path: `docker/Dockerfile`. Set the
   instance type to one with **≥ 2 GB RAM**, and the service **Port** to `8000`.
4. Add environment variables:

   ```
   RAG_QDRANT__HOST=xxxx-xxxx.aws.cloud.qdrant.io
   RAG_QDRANT__HTTPS=true
   RAG_QDRANT__API_KEY=<qdrant-key>
   RAG_LLM__PROVIDER=openai
   RAG_LLM__API_KEY=<llm-key>
   RAG_CORS_ORIGINS=https://<your-app>.vercel.app
   ```

   (You'll know the exact Vercel URL after Step 4 — you can set a placeholder now
   and update it after.)
5. Deploy. First boot is slow: the models download on startup. When it's up,
   check `https://<your-api>.onrender.com/v1/health` → `{"status":"ok"}` and the
   interactive docs at `/docs`.

**Truly-free alternative (Oracle Cloud VM):** create an Always-Free ARM VM, install
Docker, clone the repo, and run the API + Qdrant with
`docker compose -f docker/docker-compose.yml up -d api qdrant` (drop the `ollama`
service and set the OpenAI env vars instead). Point the frontend at the VM's
public IP/domain.

## Step 4 — Deploy the frontend (Vercel)

1. Vercel → **New Project** → import the repo.
2. **Root Directory:** `web`  ·  Framework preset: **Vite** (auto-detected).
3. Add an environment variable:

   ```
   VITE_API_URL=https://<your-api>.onrender.com
   ```
4. Deploy. You'll get a URL like `https://rag-engine.vercel.app`.

## Step 5 — Connect CORS + seed data

1. Set `RAG_CORS_ORIGINS` on the API to your exact Vercel URL and redeploy the API.
2. Seed some documents so there's something to search. Either:
   - use the **+** button in the frontend's *Knowledge base* panel, or
   - call the API directly:

     ```bash
     curl -s https://<your-api>.onrender.com/v1/ingest \
       -H 'content-type: application/json' \
       -d '{"documents":[{"document_id":"runbook","text":"Deploys use docker compose ...","acl":[]}]}'
     ```

Open the Vercel URL and ask a question — you should get a grounded answer with
citations, sources, and confidence.

## Environment variables reference

| Variable | Purpose |
|---|---|
| `RAG_QDRANT__HOST` | Qdrant Cloud cluster host |
| `RAG_QDRANT__HTTPS` | `true` for Qdrant Cloud (TLS) |
| `RAG_QDRANT__API_KEY` | Qdrant Cloud API key |
| `RAG_LLM__PROVIDER` | `openai` or `anthropic` for online |
| `RAG_LLM__API_KEY` | LLM provider key |
| `RAG_CORS_ORIGINS` | Allowed frontend origin(s), comma-separated |
| `VITE_API_URL` | (frontend) Base URL of the deployed API |

All API settings use the `RAG_` prefix with `__` for nesting; see
[`.env.example`](../.env.example).
