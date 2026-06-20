import { useEffect, useRef, useCallback } from 'react'
import useStore from '../store/useStore'

const MAX_RECONNECT_ATTEMPTS = 10
const BASE_DELAY_MS = 3000

export default function useWebSocket() {
  const wsRef             = useRef(null)
  const reconnectTimer    = useRef(null)
  const attemptRef        = useRef(0)
  const isStreamingRef    = useRef(false)   // tracks whether startIrisMessage was called
  const audioCtxRef       = useRef(null)

  const store = useStore()

  const playAudio = useCallback(async (base64Data) => {
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext()
      }
      const binary     = atob(base64Data)
      const bytes      = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
      const audioBuffer = await audioCtxRef.current.decodeAudioData(bytes.buffer)
      const source      = audioCtxRef.current.createBufferSource()
      source.buffer     = audioBuffer
      source.connect(audioCtxRef.current.destination)
      source.start()
    } catch (err) {
      console.error('[IRIS] Audio playback error:', err)
    }
  }, [])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(`ws://${window.location.host}/ws`)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[IRIS] WebSocket connected')
      attemptRef.current = 0
      useStore.getState().setConnected(true)
    }

    ws.onclose = () => {
      useStore.getState().setConnected(false)
      if (attemptRef.current >= MAX_RECONNECT_ATTEMPTS) return
      const delay = Math.min(BASE_DELAY_MS * 2 ** attemptRef.current, 30000)
      attemptRef.current += 1
      reconnectTimer.current = setTimeout(connect, delay)
    }

    ws.onerror = (err) => console.error('[IRIS] WebSocket error:', err)

    ws.onmessage = async (event) => {
      let msg
      try { msg = JSON.parse(event.data) } catch { return }

      const s = useStore.getState()

      switch (msg.type) {
        case 'detections':
          s.setDetections(msg)
          break
        case 'text_token':
          if (!isStreamingRef.current) {
            s.startIrisMessage()
            isStreamingRef.current = true
          }
          s.appendToken(msg.token)
          break
        case 'text_done':
          s.finalizeIrisMessage(msg.full)
          isStreamingRef.current = false
          break
        case 'memory_update':
          s.addMemoryEntry(msg.object, msg.zone)
          break
        case 'audio_chunk':
          await playAudio(msg.data)
          break
        case 'error':
          console.error('[IRIS] Server error:', msg.message)
          break
        default:
          console.warn('[IRIS] Unknown message type:', msg.type)
      }
    }
  }, [playAudio])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  const sendMessage = useCallback((payload) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload))
    }
  }, [])

  return { sendMessage, isConnected: store.isConnected }
}
