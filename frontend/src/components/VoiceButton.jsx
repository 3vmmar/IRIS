import useStore    from '../store/useStore'
import useAudio    from '../hooks/useAudio'

export default function VoiceButton({ sendMessage }) {
  const isConnected  = useStore(s => s.isConnected)
  const isListening  = useStore(s => s.isListening)
  const { startRecording, stopRecording } = useAudio({ sendMessage })

  return (
    <div className="flex flex-col items-center gap-1.5">
      <button
        onMouseDown={startRecording}
        onMouseUp={stopRecording}
        onTouchStart={startRecording}
        onTouchEnd={stopRecording}
        disabled={!isConnected}
        aria-label="Hold to speak"
        role="button"
        className={`w-14 h-14 rounded-full flex items-center justify-center text-xl transition-all
          ${isListening
            ? 'bg-iris-accent glow-accent scale-110'
            : 'bg-iris-elevated border border-iris-border hover:border-iris-accent'
          }
          disabled:opacity-40 disabled:cursor-not-allowed`}
      >
        🎤
      </button>
      <span className={`text-[11px] transition-colors ${isListening ? 'text-iris-accent' : 'text-iris-muted'}`}>
        {isListening ? 'Listening...' : 'Hold to speak'}
      </span>
    </div>
  )
}
