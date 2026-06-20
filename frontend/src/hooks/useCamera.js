import { useEffect, useRef, useState } from 'react'

export default function useCamera() {
  const videoRef  = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const [isReady, setIsReady] = useState(false)
  const [error,   setError]   = useState(null)

  useEffect(() => {
    let cancelled = false

    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720 } })
      .then((stream) => {
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.onloadedmetadata = () => {
            if (canvasRef.current && videoRef.current) {
              canvasRef.current.width  = videoRef.current.videoWidth
              canvasRef.current.height = videoRef.current.videoHeight
            }
            setIsReady(true)
          }
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err.message ?? 'Camera access denied')
      })

    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [])

  return { videoRef, canvasRef, isReady, error }
}
