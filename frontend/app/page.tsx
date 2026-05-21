"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import DemoHeader from "@/components/DemoHeader";
import DemoTriggers from "@/components/DemoTriggers";
import TranscriptTicker from "@/components/TranscriptTicker";
import PipelineDiagram from "@/components/PipelineDiagram";
import StageDetailPanel from "@/components/StageDetailPanel";
import ValidatorModal from "@/components/ValidatorModal";
import TrainerLogInspector from "@/components/TrainerLogInspector";
import { VoiceCapture } from "@/components/VoiceCapture";
import { startCall, submitReview, subscribeToCall } from "@/lib/api";
import { callReducer, initialCallState } from "@/lib/reducer";
import type { Stage } from "@/lib/types";

export default function HomePage() {
  const [state, dispatch] = useReducer(callReducer, initialCallState(""));
  const [selectedStage, setSelectedStage] = useState<Stage | null>(null);
  const [busy, setBusy] = useState(false);
  const unsubscribeRef = useRef<(() => void) | null>(null);
  // VoiceCapture registers callbacks here, one per phase of the call.
  // We invoke them when the matching voice_response audio finishes
  // playing — so the mic only opens once the speakers are silent
  // (otherwise echo feedback contaminates the next caller turn).
  //   - greetingDone: fires after the opening greeting → first mic open
  //   - clarificationDone: fires after the agent's clarifying question
  //     → re-opens the mic for the follow-up answer
  const greetingDoneCallbackRef = useRef<(() => void) | null>(null);
  const clarificationDoneCallbackRef = useRef<(() => void) | null>(null);

  useEffect(
    () => () => {
      unsubscribeRef.current?.();
    },
    [],
  );

  const registerGreetingDoneCallback = useCallback((cb: () => void) => {
    greetingDoneCallbackRef.current = cb;
  }, []);

  const registerClarificationDoneCallback = useCallback((cb: () => void) => {
    clarificationDoneCallbackRef.current = cb;
  }, []);

  const subscribeToActiveCall = useCallback((call_id: string) => {
    unsubscribeRef.current?.();
    dispatch({ type: "reset", call_id });
    setSelectedStage(null);
    unsubscribeRef.current = subscribeToCall(call_id, {
      onEvent: (event) => {
        // Voice-mode: the backend emits each conversation turn as a
        // stage="transcript" event. Route it into the transcript ticker
        // via the dedicated reducer action.
        if (
          event.stage === "transcript" &&
          event.status === "complete" &&
          event.payload &&
          typeof (event.payload as Record<string, unknown>).text === "string"
        ) {
          const p = event.payload as Record<string, string>;
          dispatch({
            type: "transcript_chunk",
            speaker: p.speaker ?? "agent",
            text: p.text,
          });
          return;
        }

        dispatch({ type: "event", event });
        // Voice-mode: the backend emits voice_response events with a
        // base64 mp3 payload + a `kind` field marking what the audio
        // is for. We decode and play, then fire the matching "done"
        // callback so VoiceCapture can re-arm the mic without echo.
        //   kind="greeting"      → fires greetingDoneCallbackRef
        //   kind="clarification" → fires clarificationDoneCallbackRef
        //   kind="summary"       → no callback; call has ended
        if (
          event.stage === "voice_response" &&
          event.status === "complete" &&
          event.payload &&
          typeof (event.payload as Record<string, unknown>).audio_base64 === "string"
        ) {
          const p = event.payload as Record<string, string>;
          const b64 = p.audio_base64;
          const mime = p.audio_mime || "audio/mpeg";
          const kind = p.kind ?? "greeting"; // legacy fallback
          try {
            const bin = atob(b64);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            const blob = new Blob([bytes], { type: mime });
            const audio = new Audio(URL.createObjectURL(blob));

            const callbackRef =
              kind === "clarification"
                ? clarificationDoneCallbackRef
                : kind === "greeting"
                  ? greetingDoneCallbackRef
                  : null; // summary: no re-arm
            const pending = callbackRef?.current ?? null;
            if (pending && callbackRef) {
              callbackRef.current = null;
              audio.onended = () => pending();
              audio.onerror = () => pending();
            }
            void audio.play();
          } catch (err) {
            console.error("failed to play voice_response audio", err);
            const callbackRef =
              kind === "clarification"
                ? clarificationDoneCallbackRef
                : kind === "greeting"
                  ? greetingDoneCallbackRef
                  : null;
            const pending = callbackRef?.current ?? null;
            if (pending && callbackRef) {
              callbackRef.current = null;
              pending();
            }
          }
        }
      },
      onTranscript: ({ speaker, text }) =>
        dispatch({ type: "transcript_chunk", speaker, text }),
      onError: (err) => {
        console.error("SSE error", err);
      },
      onClose: () => {
        // stream closed; nothing else to do
      },
    });
  }, []);

  const onStart = useCallback(async (transcript_id: string) => {
    if (busy) return;
    setBusy(true);
    try {
      const { call_id } = await startCall({ mode: "canned", transcript_id });
      subscribeToActiveCall(call_id);
    } finally {
      setBusy(false);
    }
  }, [busy, subscribeToActiveCall]);

  const onVoiceCallStart = useCallback(
    (call_id: string) => {
      subscribeToActiveCall(call_id);
    },
    [subscribeToActiveCall],
  );

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

      <VoiceCapture
        onCallStart={onVoiceCallStart}
        onRegisterGreetingDone={registerGreetingDoneCallback}
        onRegisterClarificationDone={registerClarificationDoneCallback}
        busy={busy}
      />

      <PipelineDiagram
        stages={state.stages}
        selectedStage={selectedStage}
        onSelectStage={setSelectedStage}
      />

      <section className="grid grid-cols-1 lg:grid-cols-[1.1fr_1fr] gap-6">
        <TranscriptTicker turns={state.transcript} />
        <StageDetailPanel
          stage={selectedStage}
          stageState={selectedStage ? state.stages[selectedStage] : undefined}
        />
      </section>

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
