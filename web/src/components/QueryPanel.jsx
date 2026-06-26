import { useState } from 'react'
import { Search, Loader2, Sparkles } from 'lucide-react'
import { api } from '../api'

const EXAMPLES = [
  'How do we deploy services?',
  'What is our on-call process?',
  'How is customer data secured?',
]
const USERS = [
  { v: '', label: 'admin (all)' },
  { v: 'engineering', label: 'engineering' },
  { v: 'hr', label: 'hr' },
]

export default function QueryPanel({ onResult, onLoading }) {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState('hybrid')
  const [topK, setTopK] = useState(5)
  const [user, setUser] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function run(q) {
    const text = (q ?? query).trim()
    if (!text || loading) return
    setQuery(text)
    setLoading(true); setError(null); onLoading?.(true)
    try {
      const body = { query: text, mode, top_k: topK }
      if (user) body.user = user
      const result = await api.query(body)
      onResult?.(result)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false); onLoading?.(false)
    }
  }

  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <div className="relative">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run() }}
          rows={3}
          placeholder="Ask a question over the knowledge base…"
          className="w-full resize-none rounded-lg bg-panel2 border border-line focus:border-accent/60 outline-none
                     text-slate-100 placeholder-slate-600 text-sm p-3 pr-12"
        />
        <button
          onClick={() => run()}
          disabled={loading || !query.trim()}
          className="absolute bottom-3 right-3 h-8 w-8 grid place-items-center rounded-lg
                     bg-accent text-bg disabled:opacity-40 disabled:cursor-not-allowed hover:bg-sky-300 transition-colors"
          aria-label="Ask"
        >
          {loading ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
        </button>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
        <div className="inline-flex rounded-lg border border-line overflow-hidden">
          {['hybrid', 'dense'].map((m) => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-3 py-1.5 font-medium capitalize transition-colors ${
                mode === m ? 'bg-accent/15 text-accent' : 'text-slate-400 hover:text-slate-200'}`}>
              {m}
            </button>
          ))}
        </div>

        <label className="flex items-center gap-2 text-slate-400">
          top_k
          <input type="range" min="1" max="20" value={topK}
            onChange={(e) => setTopK(Number(e.target.value))} className="accent-accent w-24" />
          <span className="font-mono text-slate-300 w-5">{topK}</span>
        </label>

        <label className="flex items-center gap-2 text-slate-400">
          user
          <select value={user} onChange={(e) => setUser(e.target.value)}
            className="bg-panel2 border border-line rounded-md px-2 py-1 text-slate-200 outline-none focus:border-accent/60">
            {USERS.map((u) => <option key={u.v} value={u.v}>{u.label}</option>)}
          </select>
        </label>

        <span className="text-slate-600 ml-auto hidden sm:block">⌘/Ctrl + Enter</span>
      </div>

      {error && (
        <div className="mt-3 text-xs text-bad bg-bad/10 border border-bad/30 rounded-lg px-3 py-2">{error}</div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1 text-[11px] text-slate-500"><Sparkles size={12} /> Try:</span>
        {EXAMPLES.map((ex) => (
          <button key={ex} onClick={() => run(ex)} disabled={loading}
            className="text-[11px] text-slate-400 bg-panel2 border border-line rounded-full px-2.5 py-1
                       hover:border-accent/40 hover:text-slate-200 transition-colors disabled:opacity-50">
            {ex}
          </button>
        ))}
      </div>
    </div>
  )
}
