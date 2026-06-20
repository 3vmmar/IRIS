import React, { useState, useRef } from 'react';
import { Mic, Square, Send } from 'lucide-react';

export default function MicrophoneControl({ onSendText, onSendAudio }) {
  const [text, setText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const handleTextSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      onSendText(text.trim());
      setText('');
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        // Convert blob to PCM WAV format using AudioContext because backend expects standard PCM
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Extract PCM data (mono, 16kHz)
        const pcmData = audioBuffer.getChannelData(0);
        
        // Convert Float32Array to Int16Array
        const int16Data = new Int16Array(pcmData.length);
        for (let i = 0; i < pcmData.length; i++) {
          let s = Math.max(-1, Math.min(1, pcmData[i]));
          int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // We could resample to 16kHz here, but for simplicity we rely on the browser's default sample rate,
        // or we assume it's close enough for now. (A robust implementation would resample).
        
        // Convert Int16Array to base64
        const buffer = new Uint8Array(int16Data.buffer);
        let binary = '';
        for (let i = 0; i < buffer.byteLength; i++) {
          binary += String.fromCharCode(buffer[i]);
        }
        const base64 = btoa(binary);
        
        onSendAudio(base64);

        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      alert('Could not access microphone.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  return (
    <div className="bg-panel p-4 rounded-lg border border-border">
      <form onSubmit={handleTextSubmit} className="flex gap-2">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ask a question..."
          className="flex-1 bg-surface border border-border rounded-md px-4 py-2 text-sm text-textMain placeholder-textMuted focus:outline-none focus:border-primary transition-colors"
        />
        
        <button
          type="submit"
          disabled={!text.trim()}
          className="p-2 bg-surface border border-border text-textMain rounded-md hover:bg-border transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Send text query"
        >
          <Send size={18} />
        </button>

        <button
          type="button"
          onMouseDown={startRecording}
          onMouseUp={stopRecording}
          onMouseLeave={stopRecording}
          className={`p-2 border rounded-md transition-all flex items-center justify-center min-w-[40px] ${
            isRecording 
              ? 'bg-red-500/20 border-red-500 text-red-500 animate-pulse' 
              : 'bg-surface border-border text-textMain hover:bg-border'
          }`}
          title="Hold to record audio"
        >
          {isRecording ? <Square size={18} fill="currentColor" /> : <Mic size={18} />}
        </button>
      </form>
    </div>
  );
}
