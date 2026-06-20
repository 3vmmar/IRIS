import React, { useRef, useEffect, useState } from 'react';

export default function CameraView({ detections }) {
  const containerRef = useRef(null);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });

  // Update video display size to calculate correct scaling for bounding boxes
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setVideoSize({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Assuming original camera resolution from backend is 1280x720 (as set in OpenCV)
  const ORIGINAL_WIDTH = 1280;
  const ORIGINAL_HEIGHT = 720;

  const scaleX = videoSize.width / ORIGINAL_WIDTH;
  const scaleY = videoSize.height / ORIGINAL_HEIGHT;

  return (
    <div className="relative w-full h-full bg-black rounded-lg overflow-hidden flex items-center justify-center border border-border" ref={containerRef}>
      {/* MJPEG Stream */}
      <img 
        src="/video_feed" 
        alt="Live Camera Feed"
        className="w-full h-full object-contain pointer-events-none"
      />
      
      {/* Bounding Boxes Overlay */}
      {videoSize.width > 0 && detections.map((det, idx) => {
        // det has x1, y1, x2, y2, label, confidence
        const left = det.x1 * scaleX;
        const top = det.y1 * scaleY;
        const width = (det.x2 - det.x1) * scaleX;
        const height = (det.y2 - det.y1) * scaleY;

        return (
          <div
            key={idx}
            className="absolute border-2 border-primary bg-primary/10 rounded-sm pointer-events-none transition-all duration-75"
            style={{
              left: `${left}px`,
              top: `${top}px`,
              width: `${width}px`,
              height: `${height}px`,
            }}
          >
            <div className="absolute -top-6 left-0 bg-primary text-white text-xs px-2 py-0.5 rounded-sm whitespace-nowrap shadow-sm">
              {det.label} {Math.round(det.confidence * 100)}%
            </div>
          </div>
        );
      })}
    </div>
  );
}
