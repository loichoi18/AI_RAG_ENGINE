import { FileText } from 'lucide-react'

export default function Sources({ sources = [] }) {
  if (!sources.length) return null
  return (
    <div className="mt-5">
      <h4 className="text-xs uppercase tracking-wider text-slate-500 font-semibold mb-2">
        Sources ({sources.length})
      </h4>
      <ul className="space-y-2">
        {sources.map((s, i) => (
          <li key={i} className="flex items-center gap-3 rounded-lg border border-line bg-panel2 px-3 py-2">
            <span className="font-mono text-[11px] text-accent bg-accent/10 border border-accent/30 rounded px-1.5 py-0.5">
              {i + 1}
            </span>
            <FileText size={15} className="text-slate-500 shrink-0" />
            <div className="min-w-0 flex-1">
              <div className="text-sm text-slate-200 truncate">{s.document_id}</div>
              {(s.section_title || s.page_number != null) && (
                <div className="text-[11px] text-slate-500 truncate">
                  {s.section_title}{s.page_number != null ? ` · p.${s.page_number}` : ''}
                </div>
              )}
            </div>
            <span className="font-mono text-[11px] text-slate-400 shrink-0">{s.score?.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
