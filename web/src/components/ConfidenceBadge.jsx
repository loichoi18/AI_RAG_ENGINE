// Maps a 0..1 confidence to a labeled, colored badge with a mini bar.
export default function ConfidenceBadge({ value = 0, refused = false }) {
  const pct = Math.round(value * 100)
  let tone = 'ok', label = 'High confidence'
  if (refused || value < 0.4) { tone = 'bad'; label = refused ? 'Refused' : 'Low confidence' }
  else if (value < 0.7) { tone = 'warn'; label = 'Medium confidence' }
  const color = { ok: 'text-ok', warn: 'text-warn', bad: 'text-bad' }[tone]
  const bar = { ok: 'bg-ok', warn: 'bg-warn', bad: 'bg-bad' }[tone]
  return (
    <div className="flex items-center gap-2">
      <span className={`text-xs font-medium ${color}`}>{label}</span>
      <div className="w-20 h-1.5 rounded-full bg-line overflow-hidden">
        <div className={`h-full ${bar}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-xs text-slate-500">{pct}%</span>
    </div>
  )
}
