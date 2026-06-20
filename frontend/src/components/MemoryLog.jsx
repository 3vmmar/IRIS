import useStore from '../store/useStore'

function toRelativeTime(isoString) {
  const diff = Math.floor((Date.now() - new Date(isoString).getTime()) / 1000)
  if (diff < 10)  return 'just now'
  if (diff < 60)  return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return `${Math.floor(diff / 3600)}h ago`
}

export default function MemoryLog() {
  const memoryLog = useStore(s => s.memoryLog)
  const displayed = memoryLog.slice(0, 20)

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-[11px] font-medium uppercase tracking-widest text-iris-muted">
          🗃 Visual Memory
        </span>
      </div>

      {displayed.length === 0 ? (
        <p className="text-iris-muted text-[12px] text-center mt-4">No objects recorded yet</p>
      ) : (
        <div className="flex flex-col gap-1.5 overflow-y-auto">
          {displayed.map((entry) => (
            <div
              key={entry.id}
              className="flex items-center gap-2 py-1.5 border-b border-iris-border/50 animate-slide-up"
            >
              <span className="text-iris-text text-[13px] font-semibold flex-1 truncate">
                {entry.object}
              </span>
              <span className="text-[11px] text-iris-cyan bg-iris-cyan/10 px-1.5 py-0.5 rounded-full shrink-0">
                {entry.zone}
              </span>
              <span className="text-[11px] text-iris-muted shrink-0">
                {toRelativeTime(entry.timestamp)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
