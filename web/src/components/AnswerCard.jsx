import { Quote, Clock, AlertTriangle } from 'lucide-react'
import ConfidenceBadge from './ConfidenceBadge'
import Sources from './Sources'
import RetrievalInspector from './RetrievalInspector'

// Replace [1], [2] citation markers with styled chips.
function renderAnswer(text) {
  const parts = String(text).split(/(\[\d+\])/g)
  return parts.map((p, i) =>
    /^\[\d+\]$/.test(p) ? <sup key={i} className="cite">{p.replace(/[[\]]/g, '')}</sup> : <span key={i}>{p}</span>
  )
}

export default function AnswerCard({ result }) {
  const refused = result.refused
  return (
    <div className="fade-up rounded-xl border border-line bg-panel p-5">
      <div className="flex items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 text-slate-400">
          {refused ? <AlertTriangle size={16} className="text-bad" /> : <Quote size={16} className="text-accent" />}
          <span className="text-xs uppercase tracking-wider font-semibold">
            {refused ? 'No confident answer' : 'Answer'}
          </span>
        </div>
        <ConfidenceBadge value={result.confidence} refused={refused} />
      </div>

      <p className="text-[15px] leading-relaxed text-slate-100 whitespace-pre-wrap">
        {renderAnswer(result.answer)}
      </p>

      {result.citations?.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {result.citations.map((c, i) => (
            <span key={i} className="font-mono text-[11px] text-slate-400 bg-panel2 border border-line rounded px-2 py-0.5">
              {c}
            </span>
          ))}
        </div>
      )}

      <Sources sources={result.sources} />
      <RetrievalInspector chunks={result.retrieved_chunks} />

      <div className="mt-4 flex items-center gap-1.5 text-[11px] text-slate-500 font-mono">
        <Clock size={12} /> {result.latency_ms?.toFixed(0)} ms
      </div>
    </div>
  )
}
