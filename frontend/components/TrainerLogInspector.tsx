"use client";

import { useState } from "react";

type Props = {
  trainerLog: Record<string, unknown> | undefined;
};

export default function TrainerLogInspector({ trainerLog }: Props) {
  const [open, setOpen] = useState(false);
  const ready = !!trainerLog;

  return (
    <section className="panel">
      <button
        type="button"
        onClick={() => ready && setOpen((v) => !v)}
        disabled={!ready}
        className="w-full flex items-center justify-between px-5 py-4 text-left disabled:cursor-not-allowed"
      >
        <div className="flex items-center gap-3">
          <span
            className={`h-2 w-2 rounded-full ${
              ready ? "bg-emerald-500" : "bg-zinc-600"
            }`}
          />
          <h2 className="text-sm font-medium tracking-tight">
            Trainer log
          </h2>
          <span className="text-[10px] uppercase tracking-[0.15em] muted">
            {ready ? "complete" : "pending"}
          </span>
        </div>
        <span className="text-xs muted">
          {ready ? (open ? "Hide ▲" : "Expand ▼") : "fills when the call completes"}
        </span>
      </button>
      {ready && open && (
        <div className="border-t border-default px-5 py-4">
          <pre className="font-mono text-xs leading-relaxed whitespace-pre-wrap break-words overflow-auto max-h-96">
            {colorize(JSON.stringify(trainerLog, null, 2))}
          </pre>
        </div>
      )}
    </section>
  );
}

// Light-touch JSON syntax highlighting — string/number/bool/null tinting only.
function colorize(json: string) {
  const pattern =
    /("(?:\\.|[^"\\])*"(?:\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)/g;
  const parts: (string | JSX.Element)[] = [];
  let lastIndex = 0;
  let i = 0;
  for (const m of json.matchAll(pattern)) {
    if (m.index === undefined) continue;
    if (m.index > lastIndex) parts.push(json.slice(lastIndex, m.index));
    const tok = m[0];
    let cls = "";
    if (tok.startsWith('"')) {
      cls = tok.trimEnd().endsWith(":") ? "text-accent-400" : "text-emerald-300";
    } else if (tok === "true" || tok === "false") {
      cls = "text-amber-300";
    } else if (tok === "null") {
      cls = "muted";
    } else {
      cls = "text-sky-300";
    }
    parts.push(
      <span key={`t-${i++}`} className={cls}>
        {tok}
      </span>,
    );
    lastIndex = m.index + tok.length;
  }
  if (lastIndex < json.length) parts.push(json.slice(lastIndex));
  return parts;
}
