"use client";

import { PIPELINE_STAGES, type CallState, type PipelineStage, type Stage, type StageNodeState } from "@/lib/types";

type Props = {
  stages: CallState["stages"];
  selectedStage: Stage | null;
  onSelectStage: (stage: Stage | null) => void;
};

const STAGE_LABEL: Record<PipelineStage, string> = {
  extract: "Extract",
  retrieve: "Retrieve",
  classify: "Classify",
  location: "Location",
  risk: "Risk",
  validate: "Validate",
  vendor: "Vendor",
  clarify: "Clarify",
  summary: "Summary",
};

function colorsFor(node: StageNodeState | undefined, isSelected: boolean) {
  if (!node) {
    return {
      bg: "bg-zinc-900/40",
      border: isSelected ? "border-zinc-400" : "border-zinc-800",
      dot: "bg-zinc-600",
      label: "muted",
      animate: "",
    };
  }
  switch (node.status) {
    case "started":
      return {
        bg: "bg-accent-500/10",
        border: "border-accent-500/60",
        dot: "bg-accent-500",
        label: "",
        animate: "animate-pulseSoft animate-glow",
      };
    case "complete":
      return {
        bg: "bg-emerald-500/10",
        border: isSelected ? "border-emerald-300" : "border-emerald-500/50",
        dot: "bg-emerald-500",
        label: "",
        animate: "",
      };
    case "failed":
      return {
        bg: "bg-rose-500/10",
        border: "border-rose-500/60",
        dot: "bg-rose-500",
        label: "text-rose-400",
        animate: "",
      };
    case "gate_open":
      return {
        bg: "bg-amber-500/15",
        border: "border-amber-400/70",
        dot: "bg-amber-400",
        label: "text-amber-300",
        animate: "animate-pulseSoft animate-glow",
      };
    case "gate_resumed":
      return {
        bg: "bg-emerald-500/10",
        border: "border-emerald-500/50",
        dot: "bg-emerald-500",
        label: "",
        animate: "",
      };
  }
}

export default function PipelineDiagram({
  stages,
  selectedStage,
  onSelectStage,
}: Props) {
  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium tracking-tight">Pipeline</h2>
        <span className="text-[10px] uppercase tracking-[0.15em] muted">
          {PIPELINE_STAGES.length} stages · click to inspect
        </span>
      </div>
      <div className="overflow-x-auto">
        <ol className="flex items-stretch gap-2 min-w-max pb-2">
          {PIPELINE_STAGES.map((stage, i) => {
            const node = stages[stage];
            const isSelected = selectedStage === stage;
            const colors = colorsFor(node, isSelected);
            const clickable = !!node;
            return (
              <li key={stage} className="flex items-center">
                <button
                  type="button"
                  disabled={!clickable}
                  onClick={() => onSelectStage(isSelected ? null : stage)}
                  className={`group relative flex flex-col items-start gap-2 rounded-xl border ${colors.border} ${colors.bg} ${colors.animate} px-4 py-3 min-w-[8.5rem] text-left transition-all ${
                    clickable ? "cursor-pointer hover:scale-[1.02]" : "cursor-default"
                  } ${isSelected ? "ring-2 ring-offset-2 ring-offset-transparent ring-accent-500/50" : ""}`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`h-2 w-2 rounded-full ${colors.dot}`} />
                    <span className={`text-sm font-medium ${colors.label}`}>
                      {STAGE_LABEL[stage]}
                    </span>
                  </div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.12em] muted">
                    {node ? node.status.replace("_", " ") : "pending"}
                  </div>
                  {node && node.status !== "started" && (
                    <div className="font-mono text-[10px] muted">
                      {node.timing_ms}ms
                    </div>
                  )}
                </button>
                {i < PIPELINE_STAGES.length - 1 && (
                  <div
                    className={`mx-1 h-px w-4 ${
                      stages[PIPELINE_STAGES[i + 1]] || node ? "bg-accent-500/40" : "bg-zinc-700"
                    }`}
                  />
                )}
              </li>
            );
          })}
        </ol>
      </div>
    </section>
  );
}
