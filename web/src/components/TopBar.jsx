import { useEffect, useState } from 'react'
import { Boxes, Github, Circle } from 'lucide-react'
import { api, API_BASE } from '../api'

export default function TopBar() {
  const [status, setStatus] = useState('checking') // checking | online | offline

  useEffect(() => {
    let alive = true
    const ping = () =>
      api.health()
        .then(() => alive && setStatus('online'))
        .catch(() => alive && setStatus('offline'))
    ping()
    const t = setInterval(ping, 15000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  const meta = {
    checking: { c: 'text-slate-500', t: 'Checking API…' },
    online: { c: 'text-ok', t: 'API online' },
    offline: { c: 'text-bad', t: 'API offline' },
  }[status]

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/80 backdrop-blur-md">
      <div className="max-w-6xl mx-auto px-5 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Boxes size={20} className="text-accent" />
          <div className="leading-none">
            <div className="font-semibold text-white text-sm">RAG Engine</div>
            <div className="text-[11px] text-slate-500 font-mono">hybrid knowledge search</div>
          </div>
        </div>
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-1.5 text-xs" title={API_BASE}>
            <Circle size={9} className={`${meta.c} ${status === 'checking' ? 'pulse2' : ''}`} fill="currentColor" />
            <span className={meta.c}>{meta.t}</span>
          </div>
          <a href="https://github.com/loichoi18/rag-engine" target="_blank" rel="noreferrer"
             className="text-slate-400 hover:text-white" aria-label="GitHub repository">
            <Github size={18} />
          </a>
        </div>
      </div>
    </header>
  )
}
