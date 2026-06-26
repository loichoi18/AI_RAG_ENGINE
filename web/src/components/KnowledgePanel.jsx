import { useEffect, useState } from 'react'
import { Database, RefreshCw, Plus, Loader2 } from 'lucide-react'
import { api } from '../api'

export default function KnowledgePanel({ refreshKey }) {
  const [docs, setDocs] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState(null)
  const [adding, setAdding] = useState(false)
  const [form, setForm] = useState({ document_id: '', text: '' })
  const [busy, setBusy] = useState(false)

  async function load() {
    setLoading(true); setErr(null)
    try {
      const d = await api.documents()
      setDocs(d.documents || []); setTotal(d.total || 0)
    } catch (e) { setErr(e.message) } finally { setLoading(false) }
  }

  useEffect(() => { load() }, [refreshKey])

  async function add() {
    if (!form.document_id.trim() || !form.text.trim() || busy) return
    setBusy(true); setErr(null)
    try {
      await api.ingest([{ document_id: form.document_id.trim(), text: form.text.trim(), acl: [] }])
      setForm({ document_id: '', text: '' }); setAdding(false); load()
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
          <Database size={15} className="text-accent2" /> Knowledge base
          <span className="font-mono text-[11px] text-slate-500">{total} docs</span>
        </h3>
        <div className="flex items-center gap-1">
          <button onClick={() => setAdding((v) => !v)} className="text-slate-400 hover:text-white p-1" aria-label="Add document">
            <Plus size={15} />
          </button>
          <button onClick={load} className="text-slate-400 hover:text-white p-1" aria-label="Refresh">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {adding && (
        <div className="mb-3 space-y-2 rounded-lg border border-line bg-panel2 p-3">
          <input value={form.document_id} onChange={(e) => setForm({ ...form, document_id: e.target.value })}
            placeholder="document_id" className="w-full text-xs bg-bg border border-line rounded px-2 py-1.5 outline-none focus:border-accent/60 text-slate-200" />
          <textarea value={form.text} onChange={(e) => setForm({ ...form, text: e.target.value })} rows={3}
            placeholder="Document text to index…" className="w-full text-xs bg-bg border border-line rounded px-2 py-1.5 outline-none focus:border-accent/60 resize-none text-slate-200" />
          <button onClick={add} disabled={busy}
            className="w-full text-xs font-medium bg-accent text-bg rounded py-1.5 hover:bg-sky-300 disabled:opacity-50 inline-flex items-center justify-center gap-1.5">
            {busy ? <Loader2 size={13} className="animate-spin" /> : <Plus size={13} />} Ingest
          </button>
        </div>
      )}

      {err && <div className="text-[11px] text-bad mb-2">{err}</div>}

      {docs.length === 0 && !loading ? (
        <p className="text-xs text-slate-500">No documents indexed yet.</p>
      ) : (
        <ul className="space-y-1.5 max-h-72 overflow-auto pr-1">
          {docs.map((d) => (
            <li key={d.document_id} className="flex items-center justify-between text-xs rounded-md bg-panel2 border border-line px-2.5 py-1.5">
              <span className="text-slate-300 truncate">{d.document_id}</span>
              <span className="font-mono text-[11px] text-slate-500 shrink-0 ml-2">{d.chunk_count} chunks</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
