"use client";

import { useEffect, useRef } from "react";

type Turn = { speaker: string; text: string };

export default function TranscriptTicker({ turns }: { turns: Turn[] }) {
  const scrollerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [turns.length]);

  return (
    <section className="panel flex flex-col h-[340px]">
      <header className="flex items-center justify-between border-b border-default px-5 py-3">
        <h2 className="text-sm font-medium tracking-tight">Caller transcript</h2>
        <span className="text-[10px] uppercase tracking-[0.15em] muted">
          {turns.length} turn{turns.length === 1 ? "" : "s"}
        </span>
      </header>
      <div
        ref={scrollerRef}
        className="flex-1 overflow-y-auto px-5 py-4 space-y-3 text-sm"
      >
        {turns.length === 0 && (
          <div className="muted italic">
            Trigger a demo to start the call.
          </div>
        )}
        {turns.map((turn, i) => (
          <div key={i} className="font-mono leading-relaxed">
            <span
              className={`mr-2 text-[10px] uppercase tracking-[0.15em] ${
                turn.speaker === "agent" ? "text-accent-400" : "text-emerald-400"
              }`}
            >
              {turn.speaker}
            </span>
            <span>{turn.text}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
