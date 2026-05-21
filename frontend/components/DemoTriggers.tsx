"use client";

import { useEffect, useState } from "react";
import { listTranscripts } from "@/lib/api";
import type { TranscriptInfo } from "@/lib/types";

type Props = {
  onSelect: (transcript_id: string) => void;
  disabled: boolean;
  activeCallId: string | null;
};

const TONE: Record<string, { dot: string; ring: string; label: string }> = {
  "fake-routine": {
    dot: "bg-emerald-500",
    ring: "hover:border-emerald-500/60",
    label: "Routine",
  },
  "fake-trap": {
    dot: "bg-amber-500",
    ring: "hover:border-amber-500/60",
    label: "Over-escalation trap",
  },
  "fake-emergency": {
    dot: "bg-rose-500",
    ring: "hover:border-rose-500/60",
    label: "True emergency",
  },
};

function toneFor(t: TranscriptInfo, index: number) {
  if (TONE[t.transcript_id]) return TONE[t.transcript_id];
  const fallback = [TONE["fake-routine"], TONE["fake-trap"], TONE["fake-emergency"]];
  return fallback[index % fallback.length];
}

export default function DemoTriggers({ onSelect, disabled, activeCallId }: Props) {
  const [transcripts, setTranscripts] = useState<TranscriptInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listTranscripts()
      .then((list) => {
        if (!cancelled) setTranscripts(list);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="panel p-5 text-sm text-rose-400">
        Could not reach the backend on{" "}
        <code className="font-mono">
          {process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"}
        </code>
        . Start it with <code className="font-mono">make backend-only</code>, or set{" "}
        <code className="font-mono">NEXT_PUBLIC_USE_FAKE_BACKEND=1</code> for the
        in-browser scripted demo.
      </div>
    );
  }

  if (!transcripts) {
    return <div className="panel p-5 muted text-sm">Loading transcripts…</div>;
  }

  return (
    <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {transcripts.map((t, idx) => {
        const tone = toneFor(t, idx);
        return (
          <button
            key={t.transcript_id}
            onClick={() => onSelect(t.transcript_id)}
            disabled={disabled}
            className={`panel text-left p-5 transition-colors group ${tone.ring} disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.15em] muted">
              <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
              {tone.label}
            </div>
            <div className="mt-3 text-lg font-medium">{t.label}</div>
            <div className="mt-2 text-sm muted">{t.expected_outcome}</div>
            <div className="mt-4 text-[11px] font-mono muted">
              transcript_id: {t.transcript_id}
            </div>
          </button>
        );
      })}
      {activeCallId && (
        <div className="md:col-span-3 text-xs muted font-mono">
          Active call: {activeCallId}
        </div>
      )}
    </section>
  );
}
