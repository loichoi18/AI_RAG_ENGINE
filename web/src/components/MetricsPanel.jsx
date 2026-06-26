import { useEffect, useState } from 'react'
import { Activity, RefreshCw } from 'lucide-react'
import { api } from '../api'

export default function MetricsPanel({ refreshKey }) {
  const [m, setM] = useState(null)
  const [loading, setLoading] = useState(false)

  async function load() {
    setLoading(true)
    try { setM(await api.metrics()) } catch { /* ignore */ } finally { setLoading(false) }
  }
  useEffect(() => { load() }, [refreshKey])

  const ops = m ? Object.entries(m.latency_ms || {}) : []
  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
          <Activity size={15} className="text-ok" /> Metrics
        </h3>
        <button onClick={load} className="text-slate-400 hover:text-white p-1" aria-label="Refresh metrics">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {!m ? (
        <p className="text-xs text-slate-500">No metrics yet.</p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-2 mb-3">
            <Stat label="Success" value={`${Math.round((m.success_rate || 0) * 100)}%`} tone="text-ok" />
            <Stat label="Error" value={`${Math.round((m.error_rate || 0) * 100)}%`} tone="text-bad" />
          </div>
          {ops.length > 0 && (
            <div className="space-y-1.5">
              {ops.map(([op, stats]) => (
                <div key={op} className="flex items-center justify-between text-[11px] rounded-md bg-panel2 border border-line px-2.5 py-1.5">
                  <span className="text-slate-300 capitalize">{op}</span>
                  <span className="font-mono text-slate-500">
                    p50 {fmt(stats.p50)} · p95 {fmt(stats.p95)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

function fmt(v) { return v == null ? '—' : `${Math.round(v)}ms` }
function Stat({ label, value, tone }) {
  return (
    <div className="rounded-lg bg-panel2 border border-line px-3 py-2">
      <div className={`text-lg font-semibold ${tone}`}>{value}</div>
      <div className="text-[11px] text-slate-500">{label} rate</div>
    </div>
  )
}
