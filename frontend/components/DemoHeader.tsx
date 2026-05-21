const SUBMISSION_TAG_URL =
  "https://github.com/acm-industry/rohan_sachit_CBRE/releases/tag/submission-v1";

export default function DemoHeader() {
  return (
    <header className="flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between border-b border-default pb-8">
      <div className="space-y-2">
        <div className="text-xs uppercase tracking-[0.18em] muted">CBRE — Final Project</div>
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">
          Human-in-the-loop call routing pipeline
        </h1>
        <p className="muted max-w-2xl">
          Live demo. A caller transcript flows through ten stages; the validator
          opens a gate when its confidence drops, so a human reviewer can
          approve or override before any vendor or 911 dispatch.
        </p>
      </div>

      <div className="flex items-stretch gap-3 text-sm">
        <Metric label="Composite score" value="91.94" accent />
        <Metric label="False 911s" value="0" />
        <a
          href={SUBMISSION_TAG_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="panel px-4 py-3 hover:border-accent-500 transition-colors flex flex-col justify-between min-w-[9rem]"
        >
          <span className="text-[10px] uppercase tracking-[0.15em] muted">Anchor</span>
          <span className="font-mono text-sm">submission-v1 ↗</span>
        </a>
      </div>
    </header>
  );
}

function Metric({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="panel px-4 py-3 flex flex-col justify-between min-w-[7.5rem]">
      <span className="text-[10px] uppercase tracking-[0.15em] muted">{label}</span>
      <span
        className={`font-mono text-xl ${
          accent ? "text-accent-400" : ""
        }`}
      >
        {value}
      </span>
    </div>
  );
}
