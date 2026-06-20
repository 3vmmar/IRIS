import { useState, useEffect, useRef } from 'react'
import useStore from '../store/useStore'

export default function ChatPanel({ sendMessage }) {
  const messages        = useStore(s => s.messages)
  const isConnected     = useStore(s => s.isConnected)
  const addUserMessage  = useStore(s => s.addUserMessage)

  const [input,   setInput]   = useState('')
  const bottomRef             = useRef(null)

  // Auto-scroll on every message or token update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    const text = input.trim()
    if (!text || !isConnected) return
    addUserMessage(text)
    sendMessage({ type: 'text_query', text })
    setInput('')
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-iris-border shrink-0">
        <span className="text-[11px] font-medium uppercase tracking-widest text-iris-muted">
          💬 Chat
        </span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3 flex flex-col gap-3">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-center">
            <span className="text-3xl opacity-30">👁️</span>
            <p className="text-iris-muted text-[13px]">IRIS is watching. Ask anything.</p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
            {msg.role === 'iris' && (
              <span className="text-[10px] text-iris-accent font-medium mb-1 ml-1">IRIS</span>
            )}
            <div className={`${msg.role === 'user' ? 'bubble-user' : 'bubble-iris'} ${msg.streaming ? 'streaming-cursor' : ''}`}>
              {msg.text || (msg.streaming ? '' : '—')}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-3 border-t border-iris-border flex gap-2 shrink-0">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask IRIS what it sees..."
          className="flex-1 bg-iris-elevated border border-iris-border rounded-lg px-3 py-2 text-[13px] text-iris-text placeholder:text-iris-muted outline-none focus:border-iris-accent transition-colors"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || !isConnected}
          className="bg-iris-accent hover:bg-violet-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-[13px] font-medium px-4 py-2 rounded-lg transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  )
}
