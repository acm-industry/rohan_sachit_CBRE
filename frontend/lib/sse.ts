import type { StageEvent } from "./types";

export type SseHandlers = {
  onEvent: (event: StageEvent) => void;
  onTranscript?: (chunk: { speaker: string; text: string }) => void;
  onError?: (err: unknown) => void;
  onClose?: () => void;
};

/**
 * Subscribe to a backend SSE stream and route events to the supplied handlers.
 * Returns an unsubscribe function that closes the underlying EventSource.
 */
export function subscribe(url: string, handlers: SseHandlers): () => void {
  const es = new EventSource(url);

  // Default unnamed events: stage events from the backend.
  es.onmessage = (msg) => {
    try {
      const parsed = JSON.parse(msg.data) as StageEvent;
      handlers.onEvent(parsed);
    } catch (err) {
      handlers.onError?.(err);
    }
  };

  // Named event: transcript chunks (optional; backend may bundle into stage payloads).
  es.addEventListener("transcript", (msg) => {
    try {
      const parsed = JSON.parse((msg as MessageEvent).data) as {
        speaker: string;
        text: string;
      };
      handlers.onTranscript?.(parsed);
    } catch (err) {
      handlers.onError?.(err);
    }
  });

  es.addEventListener("done", () => {
    handlers.onClose?.();
    es.close();
  });

  es.onerror = (err) => {
    handlers.onError?.(err);
    // EventSource auto-reconnects on transient errors; we only close on done.
  };

  return () => es.close();
}
