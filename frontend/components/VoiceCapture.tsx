"use client";

/**
 * VoiceCapture — captures mic audio as PCM linear16 16kHz mono and
 * uploads to the backend's /api/calls/{call_id}/audio endpoint.
 *
 * Uses the Web Audio API's ScriptProcessorNode for synchronous PCM
 * capture (deprecated but universally supported; the AudioWorklet
 * equivalent would need a separate worklet file). For each audio frame
 * the processor receives Float32 samples in [-1, 1]; we convert to
 * Int16 little-endian and accumulate. When the user stops, we
 * concatenate and POST the raw PCM bytes.
 *
 * Browser-side conversion to 16kHz mono is handled by passing the
 * sample rate via AudioContext's constructor — the browser resamples
 * automatically.
 */

import { useRef, useState } from "react";
import { pushAudio, startCall } from "../lib/api";

const TARGET_SAMPLE_RATE = 16000;

type Props = {
  onCallStart: (call_id: string) => void;
  busy: boolean;
};

export function VoiceCapture({ onCallStart, busy }: Props) {
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState<string>("idle");
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const chunksRef = useRef<Int16Array[]>([]);

  const start = async () => {
    if (busy) return;
    setStatus("requesting mic...");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: TARGET_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (err) {
      setStatus(`mic permission denied: ${(err as Error).message}`);
      return;
    }

    streamRef.current = stream;
    const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    audioContextRef.current = ctx;
    const source = ctx.createMediaStreamSource(stream);
    sourceRef.current = source;

    // 4096-sample buffer at 16kHz = ~256ms per onaudioprocess callback.
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;
    chunksRef.current = [];

    processor.onaudioprocess = (e) => {
      const float32 = e.inputBuffer.getChannelData(0);
      const int16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      chunksRef.current.push(int16);
    };

    source.connect(processor);
    processor.connect(ctx.destination);
    setRecording(true);
    setStatus("🔴 recording — speak now");
  };

  const stop = async () => {
    setStatus("processing...");
    sourceRef.current?.disconnect();
    processorRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    await audioContextRef.current?.close();
    audioContextRef.current = null;
    sourceRef.current = null;
    processorRef.current = null;
    streamRef.current = null;

    // Concatenate Int16 chunks into a single buffer.
    const total = chunksRef.current.reduce((sum, c) => sum + c.length, 0);
    if (total === 0) {
      setRecording(false);
      setStatus("no audio captured");
      return;
    }
    const merged = new Int16Array(total);
    let offset = 0;
    for (const c of chunksRef.current) {
      merged.set(c, offset);
      offset += c.length;
    }
    chunksRef.current = [];

    const blob = new Blob([merged.buffer], { type: "application/octet-stream" });
    const seconds = total / TARGET_SAMPLE_RATE;
    setStatus(`captured ${seconds.toFixed(1)}s of PCM — starting voice call...`);

    try {
      // Start a voice-mode call; then upload audio. Backend transcribes
      // and kicks off the pipeline; SSE will surface stages to the UI.
      const { call_id } = await startCall({ mode: "voice" });
      onCallStart(call_id);
      setStatus(`uploading audio to ${call_id}...`);
      const result = await pushAudio(call_id, blob);
      if (!result.voice_available) {
        setStatus(
          "backend voice not available — got stub response (set DEEPGRAM_API_KEY + ELEVENLABS_API_KEY in backend .env)"
        );
      } else {
        setStatus("audio uploaded — watch the pipeline above");
      }
    } catch (err) {
      setStatus(`upload failed: ${(err as Error).message}`);
    }

    setRecording(false);
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-5 shadow-xl backdrop-blur">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-wider text-zinc-500">Voice intake</div>
          <div className="text-sm text-zinc-300">Speak into your mic; the agent will transcribe + classify in real time</div>
        </div>
        <button
          onClick={recording ? stop : start}
          disabled={busy && !recording}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
            recording
              ? "bg-red-500 text-white hover:bg-red-600"
              : "bg-violet-500 text-white hover:bg-violet-600 disabled:opacity-50"
          }`}
        >
          {recording ? "⏹ Stop & send" : "🎤 Start voice call"}
        </button>
      </div>
      <div className="text-xs text-zinc-500 font-mono">{status}</div>
    </div>
  );
}
