import StatusBar  from './components/StatusBar'
import CameraFeed from './components/CameraFeed'
import ChatPanel  from './components/ChatPanel'
import VoiceButton from './components/VoiceButton'
import MemoryLog  from './components/MemoryLog'
import useWebSocket from './hooks/useWebSocket'

export default function App() {
  const { sendMessage } = useWebSocket()

  return (
    <div className="h-screen w-screen bg-iris-bg flex flex-col overflow-hidden font-inter">

      <StatusBar />

      <main className="flex flex-1 overflow-hidden">

        {/* Left — Camera feed (65%) */}
        <div className="flex-[65] relative border-r border-iris-border">
          <CameraFeed />
        </div>

        {/* Right — Chat + Voice + Memory (35%) */}
        <div className="flex-[35] flex flex-col bg-iris-surface min-w-0">

          {/* Chat */}
          <div className="flex-1 overflow-hidden flex flex-col min-h-0">
            <ChatPanel sendMessage={sendMessage} />
          </div>

          {/* Voice button */}
          <div className="flex justify-center items-center py-3 border-t border-iris-border shrink-0">
            <VoiceButton sendMessage={sendMessage} />
          </div>

          {/* Memory log */}
          <div className="h-52 border-t border-iris-border overflow-y-auto p-3 shrink-0">
            <MemoryLog />
          </div>

        </div>
      </main>
    </div>
  )
}
