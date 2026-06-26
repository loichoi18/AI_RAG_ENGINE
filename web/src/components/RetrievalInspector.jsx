import { useState } from 'react'
import { ChevronDown, Layers } from 'lucide-react'

// Collapsible panel exposing the raw retrieved chunks + scores — the
// "show your work" view that makes the retrieval pipeline transparent.
export default function RetrievalInspector({ chunks = [] }) {
  const [open, setOpen] = useState(false)
  if (!chunks.length) return null
  return (
    <div className="mt-4 border border-line rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 bg-panel2 hover:bg-line/40 transition-colors"
      >
        <span className="flex items-center gap-2 text-xs font-medium text-slate-300">
          <Layers size={14} className="text-accent2" />
          Retrieval inspector — {chunks.length} chunks
        </span>
        <ChevronDown size={15} className={`text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="divide-y divide-line">
          {chunks.map((c, i) => (
            <div key={c.chunk_id || i} className="px-3 py-2.5">
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-[11px] text-slate-500">{c.chunk_id}</span>
                <span className="font-mono text-[11px] text-accent">score {c.score?.toFixed(4)}</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed line-clamp-4">{c.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
