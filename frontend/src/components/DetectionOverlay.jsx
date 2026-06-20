import { useEffect } from 'react'
import useStore from '../store/useStore'

export default function DetectionOverlay({ canvasRef }) {
  const detections  = useStore(s => s.detections)
  const frameWidth  = useStore(s => s.frameWidth)
  const frameHeight = useStore(s => s.frameHeight)

  useEffect(() => {
    const canvas = canvasRef?.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!detections.length) return

    const scaleX = canvas.width  / (frameWidth  || canvas.width)
    const scaleY = canvas.height / (frameHeight || canvas.height)

    detections.forEach(({ label, confidence, bbox, zone }) => {
      const [x1, y1, x2, y2] = bbox
      const x = x1 * scaleX
      const y = y1 * scaleY
      const w = (x2 - x1) * scaleX
      const h = (y2 - y1) * scaleY

      // Box stroke
      ctx.strokeStyle = '#7c3aed'
      ctx.lineWidth   = 1.5
      ctx.beginPath()
      ctx.roundRect(x, y, w, h, 2)
      ctx.stroke()

      // Label badge background
      const text  = `${label} ${(confidence * 100).toFixed(0)}%`
      ctx.font    = '11px Inter, system-ui, sans-serif'
      const tw    = ctx.measureText(text).width
      const bh    = 18
      const bx    = x
      const by    = y - bh
      ctx.fillStyle = 'rgba(124,58,237,0.85)'
      ctx.beginPath()
      ctx.roundRect(bx, by, tw + 8, bh, 2)
      ctx.fill()

      // Label text
      ctx.fillStyle = '#ffffff'
      ctx.fillText(text, bx + 4, by + 12)

      // Zone hint in corner
      ctx.fillStyle = 'rgba(255,255,255,0.35)'
      ctx.font      = '10px Inter, system-ui, sans-serif'
      ctx.fillText(zone, x + 4, y + h - 5)
    })
  }, [detections, frameWidth, frameHeight, canvasRef])

  return null
}
