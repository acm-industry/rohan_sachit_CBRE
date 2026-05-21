"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import DemoHeader from "@/components/DemoHeader";
import DemoTriggers from "@/components/DemoTriggers";
import TranscriptTicker from "@/components/TranscriptTicker";
import PipelineDiagram from "@/components/PipelineDiagram";
import StageDetailPanel from "@/components/StageDetailPanel";
import ValidatorModal from "@/components/ValidatorModal";
import TrainerLogInspector from "@/components/TrainerLogInspector";
import { startCall, submitReview, subscribeToCall } from "@/lib/api";
import { callReducer, initialCallState } from "@/lib/reducer";
import type { Stage } from "@/lib/types";

export default function HomePage() {
  const [state, dispatch] = useReducer(callReducer, initialCallState(""));
  const [selectedStage, setSelectedStage] = useState<Stage | null>(null);
  const [busy, setBusy] = useState(false);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  useEffect(
    () => () => {
      unsubscribeRef.current?.();
    },
    [],
  );

  const onStart = useCallback(async (transcript_id: string) => {
    if (busy) return;
    setBusy(true);
    try {
      unsubscribeRef.current?.();
      const { call_id } = await startCall({ mode: "canned", transcript_id });
      dispatch({ type: "reset", call_id });
      setSelectedStage(null);
      unsubscribeRef.current = subscribeToCall(call_id, {
        onEvent: (event) => dispatch({ type: "event", event }),
        onTranscript: ({ speaker, text }) =>
          dispatch({ type: "transcript_chunk", speaker, text }),
        onError: (err) => {
          console.error("SSE error", err);
        },
        onClose: () => {
          // stream closed; nothing else to do
        },
      });
    } finally {
      setBusy(false);
    }
  }, [busy]);

  const onApprove = useCallback(async () => {
    if (!state.call_id) return;
    await submitReview(state.call_id, { decision: "approve" });
    dispatch({ type: "close_validator_gate" });
  }, [state.call_id]);

  const onOverride = useCallback(
    async (override: Record<string, unknown>) => {
      if (!state.call_id) return;
      await submitReview(state.call_id, { decision: "override", override });
      dispatch({ type: "close_validator_gate" });
    },
    [state.call_id],
  );

  const validatorOpen = state.validator_gate?.open ?? false;
  const validatorPayload = state.validator_gate?.payload ?? {};

  return (
    <main className="mx-auto max-w-6xl px-6 py-10 space-y-10">
      <DemoHeader />

      <DemoTriggers
        onSelect={onStart}
        disabled={busy}
        activeCallId={state.call_id || null}
      />

      <section className="grid grid-cols-1 lg:grid-cols-[1.1fr_1fr] gap-6">
        <TranscriptTicker turns={state.transcript} />
        <StageDetailPanel
          stage={selectedStage}
          stageState={selectedStage ? state.stages[selectedStage] : undefined}
        />
      </section>

      <PipelineDiagram
        stages={state.stages}
        selectedStage={selectedStage}
        onSelectStage={setSelectedStage}
      />

      <TrainerLogInspector trainerLog={state.trainer_log} />

      <ValidatorModal
        open={validatorOpen}
        payload={validatorPayload}
        onApprove={onApprove}
        onOverride={onOverride}
      />
    </main>
  );
}
