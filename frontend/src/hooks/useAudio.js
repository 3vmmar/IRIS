import { useRef, useCallback } from 'react'
import useStore from '../store/useStore'

export default function useAudio({ sendMessage }) {
  const streamRef   = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef   = useRef([])

  const initStream = useCallback(async () => {
    if (streamRef.current) return
    streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true })
  }, [])

  const startRecording = useCallback(async () => {
    try {
      await initStream()
      chunksRef.current = []

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'

      const recorder  = new MediaRecorder(streamRef.current, { mimeType })
      recorderRef.current = recorder

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const blob   = new Blob(chunksRef.current, { type: mimeType })
        const reader = new FileReader()
        reader.onloadend = () => {
          const base64 = reader.result.split(',')[1]
          sendMessage({ type: 'audio_chunk', data: base64 })
        }
        reader.readAsDataURL(blob)
        useStore.getState().setListening(false)
      }

      recorder.start()
      useStore.getState().setListening(true)
    } catch (err) {
      console.error('[IRIS] Recording error:', err)
    }
  }, [initStream, sendMessage])

  const stopRecording = useCallback(() => {
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop()
    }
  }, [])

  return { startRecording, stopRecording }
}
