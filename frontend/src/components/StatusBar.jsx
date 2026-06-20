import { useEffect } from 'react'
import useStore from '../store/useStore'

export default function StatusBar() {
  const isConnected    = useStore(s => s.isConnected)
  const fps            = useStore(s => s.fps)
  const inferenceMs    = useStore(s => s.inferenceMs)
  const activeModel    = useStore(s => s.activeModel)
  const sttLoaded      = useStore(s => s.sttLoaded)
  const ttsLoaded      = useStore(s => s.ttsLoaded)
  const setSystemStatus = useStore(s => s.setSystemStatus)

  // Poll /health every 10s to keep sttLoaded / ttsLoaded / activeModel in sync
  useEffect(() => {
    const poll = async () => {
      try {
        const res  = await fetch('/health')
        const data = await res.json()
        setSystemStatus({
          sttLoaded:   data.stt,
          ttsLoaded:   data.tts,
        })
      } catch { /* server not ready yet */ }
    }
    poll()
    const id = setInterval(poll, 10000)
    return () => clearInterval(id)
  }, [setSystemStatus])

  return (
    <header className="glass h-12 flex items-center px-4 gap-4 border-b border-iris-border shrink-0"
            style={{ borderBottomColor: isConnected ? 'rgba(124,58,237,0.4)' : undefined }}>

      {/* Logo */}
      <span className="text-iris-text font-semibold text-[15px] tracking-wide">👁️ IRIS</span>

      <div className="w-px h-5 bg-iris-border" />

      {/* Live indicator */}
      <div className="flex items-center gap-1.5">
        <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-iris-success animate-pulse' : 'bg-iris-danger'}`} />
        <span className={`text-[11px] font-medium ${isConnected ? 'text-iris-success' : 'text-iris-danger'}`}>
          {isConnected ? 'LIVE' : 'DISCONNECTED'}
        </span>
      </div>

      <div className="w-px h-5 bg-iris-border" />

      {/* Model badge */}
      <span className="text-[11px] bg-iris-accent/20 text-iris-accent px-2 py-0.5 rounded-full font-medium">
        {activeModel}
      </span>

      <div className="w-px h-5 bg-iris-border" />

      {/* Detection stats */}
      <span className="text-[11px] text-iris-cyan font-mono">{fps.toFixed(1)} fps</span>
      <span className="text-[11px] text-iris-muted font-mono">{inferenceMs.toFixed(0)}ms</span>

      <div className="ml-auto flex items-center gap-3">
        <div className="w-px h-5 bg-iris-border" />
        <StatusPill label="STT" active={sttLoaded} />
        <StatusPill label="TTS" active={ttsLoaded} />
      </div>
    </header>
  )
}

function StatusPill({ label, active }) {
  return (
    <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${
      active ? 'text-iris-success bg-iris-success/10' : 'text-iris-muted bg-iris-muted/10'
    }`}>
      {label}
    </span>
  )
}
