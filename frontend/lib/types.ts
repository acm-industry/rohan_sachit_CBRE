export const PIPELINE_STAGES = [
  "extract",
  "retrieve",
  "classify",
  "location",
  "risk",
  "validate",
  "vendor",
  "clarify",
  "summary",
] as const;

export type PipelineStage = (typeof PIPELINE_STAGES)[number];
// "voice" + "voice_response" are emitted by the backend's voice-mode path
// (one at intake with transcribed turns, one at end of pipeline with TTS audio).
export type Stage =
  | PipelineStage
  | "trainer_log"
  | "voice"
  | "voice_response"
  | "pipeline";

export type StageStatus =
  | "started"
  | "complete"
  | "failed"
  | "gate_open"
  | "gate_resumed";

export type StageEvent = {
  call_id: string;
  stage: Stage;
  status: StageStatus;
  timing_ms: number;
  payload: Record<string, unknown>;
};

export type TranscriptInfo = {
  transcript_id: string;
  label: string;
  expected_outcome: string;
};

export type StageNodeState = {
  status: StageStatus;
  timing_ms: number;
  payload: Record<string, unknown>;
};

export type CallState = {
  call_id: string;
  transcript: { speaker: string; text: string }[];
  stages: Partial<Record<Stage, StageNodeState>>;
  validator_gate?: { open: boolean; payload: Record<string, unknown> };
  trainer_log?: Record<string, unknown>;
  finished: boolean;
};

export type StartCallResponse = { call_id: string };
export type ReviewResponse = { accepted: boolean };
export type ReviewDecision = "approve" | "override";
