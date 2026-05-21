import type { CallState, StageEvent } from "./types";

export type CallAction =
  | { type: "reset"; call_id: string }
  | { type: "event"; event: StageEvent }
  | { type: "transcript_chunk"; speaker: string; text: string }
  | { type: "close_validator_gate" };

export function initialCallState(call_id: string): CallState {
  return {
    call_id,
    transcript: [],
    stages: {},
    finished: false,
  };
}

export function callReducer(state: CallState, action: CallAction): CallState {
  switch (action.type) {
    case "reset":
      return initialCallState(action.call_id);

    case "transcript_chunk":
      return {
        ...state,
        transcript: [
          ...state.transcript,
          { speaker: action.speaker, text: action.text },
        ],
      };

    case "close_validator_gate":
      if (!state.validator_gate) return state;
      return { ...state, validator_gate: { ...state.validator_gate, open: false } };

    case "event": {
      const { stage, status, timing_ms, payload } = action.event;

      // Some backends emit transcript chunks as a side-channel via `extract`
      // started events. Keep the dedicated transcript action for the wire
      // shape and only fold stage state here.

      const nextStages = {
        ...state.stages,
        [stage]: { status, timing_ms, payload },
      };

      let validator_gate = state.validator_gate;
      if (stage === "validate") {
        if (status === "gate_open") {
          validator_gate = { open: true, payload };
        } else if (status === "gate_resumed") {
          validator_gate = validator_gate
            ? { ...validator_gate, open: false }
            : { open: false, payload };
        }
      }

      let trainer_log = state.trainer_log;
      let finished = state.finished;
      if (stage === "trainer_log" && status === "complete") {
        trainer_log = payload;
        finished = true;
      }

      return {
        ...state,
        stages: nextStages,
        validator_gate,
        trainer_log,
        finished,
      };
    }
  }
}
