import { subscribe as sseSubscribe, type SseHandlers } from "./sse";
import type {
  ReviewDecision,
  ReviewResponse,
  StageEvent,
  StartCallResponse,
  TranscriptInfo,
} from "./types";

export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export const USE_FAKE_BACKEND =
  process.env.NEXT_PUBLIC_USE_FAKE_BACKEND === "1";

// ───────────────────────── Real backend ─────────────────────────

async function realListTranscripts(): Promise<TranscriptInfo[]> {
  const res = await fetch(`${BACKEND_URL}/api/transcripts`);
  if (!res.ok) throw new Error(`GET /api/transcripts -> ${res.status}`);
  return res.json();
}

async function realStartCall(body: {
  mode: "canned" | "voice";
  transcript_id?: string;
  caller_phone?: string;
}): Promise<StartCallResponse> {
  const res = await fetch(`${BACKEND_URL}/api/calls/start`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST /api/calls/start -> ${res.status}`);
  return res.json();
}

async function realSubmitReview(
  call_id: string,
  body: { decision: ReviewDecision; override?: Record<string, unknown> },
): Promise<ReviewResponse> {
  const res = await fetch(`${BACKEND_URL}/api/calls/${call_id}/review`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST review -> ${res.status}`);
  return res.json();
}

function realSubscribeToCall(call_id: string, handlers: SseHandlers) {
  return sseSubscribe(
    `${BACKEND_URL}/api/calls/${call_id}/events`,
    handlers,
  );
}

// ───────────────────────── Fake backend ─────────────────────────
// Set NEXT_PUBLIC_USE_FAKE_BACKEND=1 to use in-browser scripted demos.

type Step =
  | { kind: "transcript"; delay_ms: number; speaker: string; text: string }
  | { kind: "event"; delay_ms: number; event: Omit<StageEvent, "call_id"> };

const FAKE_TRANSCRIPTS: TranscriptInfo[] = [
  {
    transcript_id: "fake-routine",
    label: "Routine work order",
    expected_outcome: "Route to HVAC vendor, no human review needed",
  },
  {
    transcript_id: "fake-trap",
    label: "Over-escalation trap",
    expected_outcome:
      "Sounds urgent but is not 911-worthy — validator catches it, human approves the vendor route",
  },
  {
    transcript_id: "fake-emergency",
    label: "True emergency",
    expected_outcome:
      "Active fire — validator confirms emergency dispatch, no false-911 risk",
  },
];

const FAKE_SCRIPTS: Record<string, Step[]> = {
  "fake-routine": [
    { kind: "transcript", delay_ms: 0, speaker: "caller", text: "Hi, the AC in conference room 4B has stopped working since this morning." },
    { kind: "transcript", delay_ms: 350, speaker: "agent", text: "Got it — can I confirm your building?" },
    { kind: "transcript", delay_ms: 250, speaker: "caller", text: "Building 12, floor 3. It's getting really warm in here." },
    { kind: "event", delay_ms: 200, event: { stage: "extract", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 420, event: { stage: "extract", status: "complete", timing_ms: 420, payload: { issue: "AC unit non-functional", building: "12", floor: "3", room: "4B" } } },
    { kind: "event", delay_ms: 120, event: { stage: "retrieve", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 380, event: { stage: "retrieve", status: "complete", timing_ms: 380, payload: { retrieved: [{ ticket_id: "T-1182", summary: "AC failure - Building 12 floor 3 (Aug 2024)" }, { ticket_id: "T-0931", summary: "HVAC compressor replacement - Building 12 (May 2024)" }] } } },
    { kind: "event", delay_ms: 80, event: { stage: "classify", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 540, event: { stage: "classify", status: "complete", timing_ms: 540, payload: { category: "facilities", subcategory: "hvac", confidence_category: 0.96, confidence_subcategory: 0.91 } } },
    { kind: "event", delay_ms: 90, event: { stage: "location", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 220, event: { stage: "location", status: "complete", timing_ms: 220, payload: { building: "12", floor: "3", room: "4B" } } },
    { kind: "event", delay_ms: 80, event: { stage: "risk", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 260, event: { stage: "risk", status: "complete", timing_ms: 260, payload: { risk_level: "low", risk_reasons: ["No safety hazard", "Single-room comfort issue"] } } },
    { kind: "event", delay_ms: 80, event: { stage: "validate", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 310, event: { stage: "validate", status: "complete", timing_ms: 310, payload: { ai_prediction: { category: "facilities", subcategory: "hvac" }, validator_reasons: ["Confidence above 0.9 threshold", "Risk level matches category baseline"] } } },
    { kind: "event", delay_ms: 80, event: { stage: "vendor", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 410, event: { stage: "vendor", status: "complete", timing_ms: 410, payload: { vendor_id: "V-HVAC-04", vendor_name: "Acme HVAC Services", reasoning: "Building 12 HVAC contract holder; mean response time 38min" } } },
    { kind: "event", delay_ms: 80, event: { stage: "clarify", status: "complete", timing_ms: 90, payload: { clarifying_questions: [] } } },
    { kind: "event", delay_ms: 80, event: { stage: "summary", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 240, event: { stage: "summary", status: "complete", timing_ms: 240, payload: { call_summary: "AC outage reported in B12 / floor 3 / room 4B. Dispatching Acme HVAC. Caller informed of ~40min ETA." } } },
    { kind: "event", delay_ms: 60, event: { stage: "trainer_log", status: "complete", timing_ms: 60, payload: { call_id: "fake-routine", final_decision: "dispatch_vendor", vendor: "V-HVAC-04", confidence: 0.96, human_override: false, total_latency_ms: 3120 } } },
  ],

  "fake-trap": [
    { kind: "transcript", delay_ms: 0, speaker: "caller", text: "There's a strong burning smell coming from the kitchen on floor 2." },
    { kind: "transcript", delay_ms: 300, speaker: "agent", text: "Is anyone in danger right now? Do you see flames or smoke?" },
    { kind: "transcript", delay_ms: 260, speaker: "caller", text: "No flames, no smoke. Just the smell. The microwave shorted out maybe twenty minutes ago." },
    { kind: "event", delay_ms: 200, event: { stage: "extract", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 480, event: { stage: "extract", status: "complete", timing_ms: 480, payload: { issue: "Burning smell, source identified (failed microwave)", building: "7", floor: "2", room: "kitchen" } } },
    { kind: "event", delay_ms: 120, event: { stage: "retrieve", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 360, event: { stage: "retrieve", status: "complete", timing_ms: 360, payload: { retrieved: [{ ticket_id: "T-2210", summary: "Appliance failure with smoke - building 7 (Jan 2024)" }, { ticket_id: "T-1944", summary: "Kitchen microwave replacement (Nov 2023)" }] } } },
    { kind: "event", delay_ms: 90, event: { stage: "classify", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 620, event: { stage: "classify", status: "complete", timing_ms: 620, payload: { category: "facilities", subcategory: "electrical", confidence_category: 0.74, confidence_subcategory: 0.61 } } },
    { kind: "event", delay_ms: 90, event: { stage: "location", status: "complete", timing_ms: 210, payload: { building: "7", floor: "2", room: "kitchen" } } },
    { kind: "event", delay_ms: 80, event: { stage: "risk", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 410, event: { stage: "risk", status: "complete", timing_ms: 410, payload: { risk_level: "medium", risk_reasons: ["Smell of burning", "No active fire reported", "Appliance source isolated"] } } },
    { kind: "event", delay_ms: 80, event: { stage: "validate", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 360, event: { stage: "validate", status: "gate_open", timing_ms: 360, payload: { ai_prediction: { category: "emergency", subcategory: "fire_risk", action: "dispatch_911", confidence: 0.62 }, validator_reasons: ["Confidence below 0.80 emergency-dispatch threshold", "Caller confirmed no flames / no smoke", "Source (microwave) already isolated", "Risk classified medium, not high"] } } },
    // gate stays open until the user clicks Approve or Override; the controller resumes from there.
  ],

  // continuation appended after validator decision; injected dynamically.
  "fake-emergency": [
    { kind: "transcript", delay_ms: 0, speaker: "caller", text: "There's a fire on floor 4 of building 3 — I can see flames in the server room hallway!" },
    { kind: "transcript", delay_ms: 280, speaker: "agent", text: "Has the fire alarm been pulled? Is everyone evacuating?" },
    { kind: "transcript", delay_ms: 220, speaker: "caller", text: "Yes the alarm is going off, people are leaving, but the flames are spreading." },
    { kind: "event", delay_ms: 200, event: { stage: "extract", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 430, event: { stage: "extract", status: "complete", timing_ms: 430, payload: { issue: "Active fire with visible flames, evacuation in progress", building: "3", floor: "4", area: "server room hallway" } } },
    { kind: "event", delay_ms: 80, event: { stage: "retrieve", status: "complete", timing_ms: 200, payload: { retrieved: [{ ticket_id: "T-0009", summary: "Active fire emergency protocol - all buildings" }] } } },
    { kind: "event", delay_ms: 80, event: { stage: "classify", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 380, event: { stage: "classify", status: "complete", timing_ms: 380, payload: { category: "emergency", subcategory: "fire_active", confidence_category: 0.99, confidence_subcategory: 0.97 } } },
    { kind: "event", delay_ms: 80, event: { stage: "location", status: "complete", timing_ms: 180, payload: { building: "3", floor: "4", area: "server room hallway" } } },
    { kind: "event", delay_ms: 60, event: { stage: "risk", status: "complete", timing_ms: 220, payload: { risk_level: "high", risk_reasons: ["Visible flames", "Active evacuation", "Critical infrastructure (server room) adjacent"] } } },
    { kind: "event", delay_ms: 80, event: { stage: "validate", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 360, event: { stage: "validate", status: "gate_open", timing_ms: 360, payload: { ai_prediction: { category: "emergency", subcategory: "fire_active", action: "dispatch_911", confidence: 0.99 }, validator_reasons: ["High-risk emergency dispatch — human confirmation required by policy", "Confidence 0.99, risk_level high, classification consistent across stages"] } } },
  ],
};

// Continuation for fake-trap after the operator clicks Approve (route to vendor) or Override.
function fakeTrapContinuation(decision: ReviewDecision, override?: Record<string, unknown>): Step[] {
  const overrideApplied = decision === "override";
  return [
    { kind: "event", delay_ms: 50, event: { stage: "validate", status: "gate_resumed", timing_ms: 0, payload: { decision, override: override ?? null } } },
    { kind: "event", delay_ms: 80, event: { stage: "vendor", status: "started", timing_ms: 0, payload: {} } },
    { kind: "event", delay_ms: 430, event: { stage: "vendor", status: "complete", timing_ms: 430, payload: { vendor_id: "V-ELEC-02", vendor_name: "Northside Electrical", reasoning: "Building 7 electrical contractor; isolated appliance failure requires inspection, not emergency dispatch" } } },
    { kind: "event", delay_ms: 80, event: { stage: "clarify", status: "complete", timing_ms: 90, payload: { clarifying_questions: ["Should the kitchen be roped off until inspection?"] } } },
    { kind: "event", delay_ms: 80, event: { stage: "summary", status: "complete", timing_ms: 230, payload: { call_summary: "Burning smell isolated to a failed microwave in B7 floor 2 kitchen. Human reviewer confirmed non-emergency; dispatching Northside Electrical. Kitchen closed pending inspection." } } },
    { kind: "event", delay_ms: 60, event: { stage: "trainer_log", status: "complete", timing_ms: 60, payload: { call_id: "fake-trap", final_decision: "dispatch_vendor", vendor: "V-ELEC-02", human_override: overrideApplied, ai_prediction_overridden: { from: "dispatch_911", to: "dispatch_vendor" }, total_latency_ms: 3940 } } },
  ];
}

// Continuation for fake-emergency after the operator confirms the 911 dispatch.
function fakeEmergencyContinuation(decision: ReviewDecision, override?: Record<string, unknown>): Step[] {
  return [
    { kind: "event", delay_ms: 40, event: { stage: "validate", status: "gate_resumed", timing_ms: 0, payload: { decision, override: override ?? null } } },
    { kind: "event", delay_ms: 80, event: { stage: "vendor", status: "complete", timing_ms: 220, payload: { vendor_id: "V-911", vendor_name: "Local Fire Dept dispatch", reasoning: "Active fire emergency confirmed by human reviewer; 911 dispatch initiated" } } },
    { kind: "event", delay_ms: 60, event: { stage: "clarify", status: "complete", timing_ms: 60, payload: { clarifying_questions: [] } } },
    { kind: "event", delay_ms: 60, event: { stage: "summary", status: "complete", timing_ms: 180, payload: { call_summary: "Active fire in B3 floor 4 server-room hallway. 911 dispatched, evacuation in progress, building ops notified." } } },
    { kind: "event", delay_ms: 60, event: { stage: "trainer_log", status: "complete", timing_ms: 60, payload: { call_id: "fake-emergency", final_decision: "dispatch_911", human_override: decision === "override", confidence: 0.99, total_latency_ms: 2380 } } },
  ];
}

class FakeCallController {
  private timers: ReturnType<typeof setTimeout>[] = [];
  private cursor = 0;
  private steps: Step[];
  private handlers: SseHandlers;
  private call_id: string;
  private transcript_id: string;
  private paused = false;
  private closed = false;

  constructor(call_id: string, transcript_id: string, handlers: SseHandlers) {
    this.call_id = call_id;
    this.transcript_id = transcript_id;
    this.handlers = handlers;
    this.steps = FAKE_SCRIPTS[transcript_id] ?? [];
  }

  start() {
    this.scheduleNext();
  }

  close() {
    this.closed = true;
    this.timers.forEach(clearTimeout);
    this.timers = [];
    this.handlers.onClose?.();
  }

  resume(decision: ReviewDecision, override?: Record<string, unknown>) {
    if (!this.paused) return;
    let continuation: Step[] = [];
    if (this.transcript_id === "fake-trap") {
      continuation = fakeTrapContinuation(decision, override);
    } else if (this.transcript_id === "fake-emergency") {
      continuation = fakeEmergencyContinuation(decision, override);
    }
    this.steps = continuation;
    this.cursor = 0;
    this.paused = false;
    this.scheduleNext();
  }

  private scheduleNext() {
    if (this.closed) return;
    if (this.cursor >= this.steps.length) {
      if (!this.paused) this.handlers.onClose?.();
      return;
    }
    const step = this.steps[this.cursor];
    const timer = setTimeout(() => {
      this.fire(step);
      this.cursor += 1;
      if (this.paused) return;
      this.scheduleNext();
    }, step.delay_ms);
    this.timers.push(timer);
  }

  private fire(step: Step) {
    if (this.closed) return;
    if (step.kind === "transcript") {
      this.handlers.onTranscript?.({ speaker: step.speaker, text: step.text });
      return;
    }
    const ev: StageEvent = { call_id: this.call_id, ...step.event };
    this.handlers.onEvent(ev);
    if (ev.stage === "validate" && ev.status === "gate_open") {
      this.paused = true;
    }
  }
}

const fakeControllers = new Map<string, FakeCallController>();
let fakeCallCounter = 0;

async function fakeListTranscripts(): Promise<TranscriptInfo[]> {
  await tinyDelay();
  return FAKE_TRANSCRIPTS;
}

async function fakeStartCall(body: {
  mode: "canned" | "voice";
  transcript_id?: string;
}): Promise<StartCallResponse> {
  await tinyDelay();
  fakeCallCounter += 1;
  return { call_id: `fake-call-${fakeCallCounter}-${body.transcript_id ?? "voice"}` };
}

async function fakeSubmitReview(
  call_id: string,
  body: { decision: ReviewDecision; override?: Record<string, unknown> },
): Promise<ReviewResponse> {
  await tinyDelay();
  const controller = fakeControllers.get(call_id);
  controller?.resume(body.decision, body.override);
  return { accepted: true };
}

function fakeSubscribeToCall(call_id: string, handlers: SseHandlers) {
  // call_id is `fake-call-{n}-{transcript_id}`; transcript_id is everything after the second hyphen.
  const transcript_id = call_id.split("-").slice(3).join("-") || "fake-routine";
  const controller = new FakeCallController(call_id, transcript_id, handlers);
  fakeControllers.set(call_id, controller);
  controller.start();
  return () => {
    controller.close();
    fakeControllers.delete(call_id);
  };
}

function tinyDelay(ms = 80) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

// ───────────────────────── Public API ─────────────────────────

export const listTranscripts = USE_FAKE_BACKEND ? fakeListTranscripts : realListTranscripts;
export const startCall = USE_FAKE_BACKEND ? fakeStartCall : realStartCall;
export const submitReview = USE_FAKE_BACKEND ? fakeSubmitReview : realSubmitReview;
export const subscribeToCall = USE_FAKE_BACKEND ? fakeSubscribeToCall : realSubscribeToCall;
