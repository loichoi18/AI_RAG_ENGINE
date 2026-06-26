import { useState } from 'react'
import { Loader2, MessageSquareText } from 'lucide-react'
import TopBar from './components/TopBar'
import QueryPanel from './components/QueryPanel'
import AnswerCard from './components/AnswerCard'
import KnowledgePanel from './components/KnowledgePanel'
import MetricsPanel from './components/MetricsPanel'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)

  function handleResult(r) {
    setResult(r)
    setRefreshKey((k) => k + 1) // refresh metrics after each query
  }

  return (
    <div className="min-h-screen">
      <TopBar />
      <main className="max-w-6xl mx-auto px-5 py-7 grid lg:grid-cols-[1fr_320px] gap-5 items-start">
        {/* Primary column */}
        <div className="space-y-5">
          <QueryPanel onResult={handleResult} onLoading={setLoading} />

          {loading && (
            <div className="rounded-xl border border-line bg-panel p-5 flex items-center gap-3 text-slate-400 text-sm">
              <Loader2 size={16} className="animate-spin text-accent" />
              Retrieving, reranking, and generating a grounded answer…
            </div>
          )}

          {!loading && result && <AnswerCard result={result} />}

          {!loading && !result && (
            <div className="rounded-xl border border-dashed border-line bg-panel/40 p-10 text-center">
              <MessageSquareText size={28} className="mx-auto text-slate-600 mb-3" />
              <h2 className="text-slate-300 font-medium">Ask the knowledge base</h2>
              <p className="text-sm text-slate-500 mt-1 max-w-sm mx-auto">
                Hybrid retrieval (dense + BM25) with reranking, a confidence gate, and
                grounded answers with verifiable citations.
              </p>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <aside className="space-y-5">
          <KnowledgePanel refreshKey={refreshKey} />
          <MetricsPanel refreshKey={refreshKey} />
        </aside>
      </main>

      <footer className="max-w-6xl mx-auto px-5 py-6 text-center">
        <p className="font-mono text-[11px] text-slate-600">
          RAG Engine · FastAPI + Qdrant + hybrid retrieval · React frontend
        </p>
      </footer>
    </div>
  )
}
