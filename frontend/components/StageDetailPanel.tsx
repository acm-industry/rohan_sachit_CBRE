"use client";

import type { Stage, StageNodeState } from "@/lib/types";

type Props = {
  stage: Stage | null;
  stageState: StageNodeState | undefined;
};

export default function StageDetailPanel({ stage, stageState }: Props) {
  return (
    <section className="panel flex flex-col h-[340px]">
      <header className="flex items-center justify-between border-b border-default px-5 py-3">
        <h2 className="text-sm font-medium tracking-tight">
          {stage ? `Stage payload · ${stage}` : "Stage payload"}
        </h2>
        {stageState && (
          <span className="font-mono text-[10px] uppercase tracking-[0.12em] muted">
            {stageState.status} · {stageState.timing_ms}ms
          </span>
        )}
      </header>
      <div className="flex-1 overflow-auto px-5 py-4 text-xs">
        {!stage && (
          <div className="muted italic text-sm">
            Click any lit pipeline stage to inspect its raw payload.
          </div>
        )}
        {stage && !stageState && (
          <div className="muted italic text-sm">
            No data yet for {stage}.
          </div>
        )}
        {stage && stageState && (
          <pre className="font-mono leading-relaxed whitespace-pre-wrap break-words">
            {JSON.stringify(stageState.payload, null, 2)}
          </pre>
        )}
      </div>
    </section>
  );
}
