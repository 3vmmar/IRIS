import useCamera from '../hooks/useCamera'
import DetectionOverlay from './DetectionOverlay'

export default function CameraFeed() {
  const { videoRef, canvasRef, isReady, error } = useCamera()

  if (error) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center bg-iris-bg gap-3">
        <span className="text-iris-danger text-2xl">⚠️</span>
        <p className="text-iris-secondary text-sm text-center px-6">{error}</p>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full bg-black">
      {!isReady && (
        <div className="absolute inset-0 flex flex-col items-center justify-center z-10 gap-3">
          <div className="w-16 h-16 rounded-full border-2 border-iris-accent animate-pulse" />
          <p className="text-iris-secondary text-sm">Requesting camera access...</p>
        </div>
      )}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-contain"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />
      <DetectionOverlay canvasRef={canvasRef} />
    </div>
  )
}
