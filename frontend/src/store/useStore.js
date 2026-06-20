import { create } from 'zustand'

const useStore = create((set, get) => ({

  // ── Connection ─────────────────────────────────
  isConnected: false,
  wsReadyState: 3,

  // ── Detection ──────────────────────────────────
  detections: [],
  frameId: 0,
  frameWidth: 1280,
  frameHeight: 720,
  inferenceMs: 0,
  timeSinceAnalysis: Infinity,
  fps: 0,

  // ── Chat ───────────────────────────────────────
  messages: [],
  _streamingMessageId: null,   // tracks id of the in-progress IRIS message

  // ── Voice ──────────────────────────────────────
  isListening: false,
  isSpeaking:  false,

  // ── Memory ─────────────────────────────────────
  memoryLog: [],

  // ── System ─────────────────────────────────────
  sttLoaded:   false,
  ttsLoaded:   false,
  activeModel: 'claude-haiku-4-5',

  // ── Actions ────────────────────────────────────

  setConnected: (bool) => set({
    isConnected:  bool,
    wsReadyState: bool ? 1 : 3,
  }),

  setDetections: (payload) => set({
    detections:       payload.boxes          ?? [],
    frameId:          payload.frame_id       ?? 0,
    frameWidth:       payload.frame_width    ?? 1280,
    frameHeight:      payload.frame_height   ?? 720,
    inferenceMs:      payload.inference_ms   ?? 0,
    timeSinceAnalysis: payload.time_since_analysis ?? Infinity,
  }),

  addUserMessage: (text) => set((state) => ({
    messages: [
      ...state.messages,
      { id: Date.now(), role: 'user', text, timestamp: new Date().toISOString(), streaming: false },
    ],
  })),

  startIrisMessage: () => {
    const id = Date.now()
    set((state) => ({
      _streamingMessageId: id,
      messages: [
        ...state.messages,
        { id, role: 'iris', text: '', timestamp: new Date().toISOString(), streaming: true },
      ],
    }))
  },

  appendToken: (token) => set((state) => ({
    messages: state.messages.map((m) =>
      m.id === state._streamingMessageId ? { ...m, text: m.text + token } : m
    ),
  })),

  finalizeIrisMessage: (fullText) => set((state) => ({
    _streamingMessageId: null,
    messages: state.messages.map((m) =>
      m.id === state._streamingMessageId
        ? { ...m, text: fullText, streaming: false }
        : m.streaming
        ? { ...m, streaming: false }   // fallback: close any open streaming message
        : m
    ),
  })),

  addMemoryEntry: (object, zone) => set((state) => ({
    memoryLog: [
      { id: Date.now(), object, zone, timestamp: new Date().toISOString() },
      ...state.memoryLog,
    ].slice(0, 50),
  })),

  setListening: (bool) => set({ isListening: bool }),
  setSpeaking:  (bool) => set({ isSpeaking: bool }),
  setFps:       (fps)  => set({ fps }),

  setSystemStatus: ({ sttLoaded, ttsLoaded, activeModel }) => set({
    sttLoaded:   sttLoaded   ?? get().sttLoaded,
    ttsLoaded:   ttsLoaded   ?? get().ttsLoaded,
    activeModel: activeModel ?? get().activeModel,
  }),

  clearMessages: () => set({ messages: [], _streamingMessageId: null }),
}))

export default useStore
