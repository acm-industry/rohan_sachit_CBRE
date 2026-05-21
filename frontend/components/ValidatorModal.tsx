"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";

type Props = {
  open: boolean;
  payload: Record<string, unknown>;
  onApprove: () => Promise<void> | void;
  onOverride: (override: Record<string, unknown>) => Promise<void> | void;
};

export default function ValidatorModal({ open, payload, onApprove, onOverride }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [draftError, setDraftError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const aiPrediction = (payload.ai_prediction ?? {}) as Record<string, unknown>;
  const reasons = (payload.validator_reasons ?? []) as string[];

  useEffect(() => {
    if (open) {
      setEditing(false);
      setDraft(JSON.stringify(aiPrediction, null, 2));
      setDraftError(null);
    }
  }, [open, payload]);

  async function handleApprove() {
    setSubmitting(true);
    try {
      await onApprove();
    } finally {
      setSubmitting(false);
    }
  }

  async function handleOverride() {
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(draft);
    } catch (err) {
      setDraftError(`Invalid JSON: ${(err as Error).message}`);
      return;
    }
    setDraftError(null);
    setSubmitting(true);
    try {
      await onOverride(parsed);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog.Root open={open}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm data-[state=open]:animate-in data-[state=open]:fade-in-0" />
        <Dialog.Content
          onPointerDownOutside={(e) => e.preventDefault()}
          onEscapeKeyDown={(e) => e.preventDefault()}
          className="fixed left-1/2 top-1/2 z-50 w-[min(640px,90vw)] -translate-x-1/2 -translate-y-1/2 panel p-0 shadow-2xl"
        >
          <div className="border-b border-default px-6 py-4">
            <div className="text-[10px] uppercase tracking-[0.18em] text-amber-300">
              Validator gate — human review required
            </div>
            <Dialog.Title className="mt-1 text-lg font-semibold tracking-tight">
              Approve the AI's call, or override it
            </Dialog.Title>
            <Dialog.Description className="muted mt-1 text-sm">
              The validator flagged this prediction. The pipeline is paused until
              you decide.
            </Dialog.Description>
          </div>

          <div className="px-6 py-5 space-y-5">
            <section>
              <div className="text-[10px] uppercase tracking-[0.15em] muted mb-2">
                AI prediction
              </div>
              {editing ? (
                <textarea
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  spellCheck={false}
                  className="w-full font-mono text-xs bg-black/30 border border-default rounded-md px-3 py-2 h-44 focus:outline-none focus:border-accent-500"
                />
              ) : (
                <pre className="font-mono text-xs bg-black/30 border border-default rounded-md px-3 py-2 max-h-44 overflow-auto whitespace-pre-wrap">
                  {JSON.stringify(aiPrediction, null, 2)}
                </pre>
              )}
              {draftError && (
                <div className="mt-2 text-xs text-rose-400">{draftError}</div>
              )}
            </section>

            <section>
              <div className="text-[10px] uppercase tracking-[0.15em] muted mb-2">
                Validator reasons
              </div>
              <ul className="space-y-1.5">
                {reasons.length === 0 && (
                  <li className="text-xs muted italic">(no reasons supplied)</li>
                )}
                {reasons.map((r, i) => (
                  <li key={i} className="text-sm flex gap-2">
                    <span className="text-amber-400 mt-0.5">›</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </section>
          </div>

          <div className="flex flex-col-reverse sm:flex-row sm:justify-between gap-3 border-t border-default px-6 py-4">
            {editing ? (
              <button
                type="button"
                disabled={submitting}
                onClick={() => setEditing(false)}
                className="text-sm muted hover:text-white transition-colors disabled:opacity-50"
              >
                Cancel edit
              </button>
            ) : (
              <button
                type="button"
                disabled={submitting}
                onClick={() => setEditing(true)}
                className="text-sm muted hover:text-white transition-colors disabled:opacity-50"
              >
                Edit prediction →
              </button>
            )}
            <div className="flex gap-2">
              {editing && (
                <button
                  type="button"
                  disabled={submitting}
                  onClick={handleOverride}
                  className="px-4 py-2 rounded-md bg-amber-500/90 hover:bg-amber-400 text-black text-sm font-medium transition-colors disabled:opacity-50"
                >
                  Submit override
                </button>
              )}
              {!editing && (
                <button
                  type="button"
                  disabled={submitting}
                  onClick={handleApprove}
                  className="px-4 py-2 rounded-md bg-accent-600 hover:bg-accent-500 text-white text-sm font-medium transition-colors disabled:opacity-50"
                >
                  Approve
                </button>
              )}
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
