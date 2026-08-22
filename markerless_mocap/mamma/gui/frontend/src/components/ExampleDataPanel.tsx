import { useEffect, useRef, useState } from 'react';
import { Download, CheckCircle2, AlertCircle, Loader2, Play } from 'lucide-react';

type ExampleState = 'missing' | 'downloading' | 'ready' | 'error';

interface StatusResponse {
  state: ExampleState;
  tail: string[];
  error: string | null;
  started_at: string | null;
}

interface Props {
  onRunDemo: () => void;
}

const POLL_INTERVAL_MS = 1500;

/**
 * Quickstart panel for the Home page. Pulls /api/example/status on mount
 * (and while a download is in flight), shows the right CTA for the
 * current state:
 *
 *   missing      → "Download example data" primary button
 *   downloading  → spinner + tail of the script's stdout, "Run demo" disabled
 *   ready        → green "Example data ready" pill + "Run demo" primary button
 *   error        → red banner + "Retry" button
 *
 * The Run demo path opens the existing QuickstartWizard via onRunDemo —
 * the wizard handles the missing-data case itself, so the button stays
 * usable even before the download finishes (it just won't have anything
 * to run yet).
 */
export function ExampleDataPanel({ onRunDemo }: Props) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [requesting, setRequesting] = useState(false);
  const pollRef = useRef<number | null>(null);

  const fetchStatus = async () => {
    try {
      const r = await fetch('/api/example/status');
      if (!r.ok) return;
      const data: StatusResponse = await r.json();
      setStatus(data);
    } catch {
      // Network blip — keep the previous state.
    }
  };

  useEffect(() => {
    fetchStatus();
    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  // Poll only while a download is in flight.
  useEffect(() => {
    if (status?.state === 'downloading') {
      if (pollRef.current === null) {
        pollRef.current = window.setInterval(fetchStatus, POLL_INTERVAL_MS);
      }
    } else if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, [status?.state]);

  const triggerDownload = async () => {
    if (requesting) return;
    setRequesting(true);
    try {
      const r = await fetch('/api/example/download', { method: 'POST' });
      const data: StatusResponse = await r.json();
      setStatus(data);
    } catch {
      // No-op; the next poll will surface the state.
    } finally {
      setRequesting(false);
    }
  };

  const state = status?.state ?? 'missing';
  const tail = status?.tail ?? [];

  return (
    <div className="bg-surface-1 border border-border-subtle rounded-xl p-6 shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02]">
      <div className="flex flex-wrap items-start gap-6 justify-between">
        <div className="flex-1 min-w-[280px]">
          <div className="flex items-center gap-3">
            <h2 className="text-foreground text-xl font-medium tracking-tight">Quickstart</h2>
            <StatusPill state={state} />
          </div>
          <p className="text-foreground-muted text-sm mt-2 leading-relaxed">
            Launch the MAMMA workflow demo.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 flex-shrink-0">
          {state !== 'ready' && (
            <button
              onClick={triggerDownload}
              disabled={state === 'downloading' || requesting}
              title="Fetches the ~56 MB demo sequence — no account required."
              className="mamma-cta inline-flex items-center gap-2 px-4 py-2.5 bg-surface-2 border border-border hover:border-border-strong text-foreground rounded-md text-sm font-medium disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
            >
              {state === 'downloading' ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Downloading…
                </>
              ) : state === 'error' ? (
                <>
                  <Download className="w-4 h-4" />
                  Retry download
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  Download example data
                </>
              )}
            </button>
          )}
          <button
            onClick={onRunDemo}
            title="Open the demo wizard — submits a full pipeline run on the example sequence."
            className="mamma-cta mamma-cta-primary relative group inline-flex items-center gap-2 px-4 py-2.5 bg-primary text-primary-foreground rounded-md text-sm font-medium shadow-sm shadow-black/30"
          >
            {state === 'ready' && (
              <span aria-hidden className="pointer-events-none absolute inset-0 rounded-md bg-primary opacity-40 animate-ping" />
            )}
            <Play className="relative w-4 h-4 transition-transform duration-200 group-hover:scale-125" />
            <span className="relative">Run demo</span>
          </button>
        </div>
      </div>

      {state === 'downloading' && tail.length > 0 && (
        <pre className="mt-4 max-h-32 overflow-y-auto bg-surface-2/50 border border-border-subtle rounded-md p-3 text-[11.5px] leading-relaxed font-mono text-foreground-muted whitespace-pre-wrap">
          {tail.join('\n')}
        </pre>
      )}

      {state === 'error' && status?.error && (
        <div className="mt-4 flex items-start gap-2 bg-status-failed-bg/60 border border-status-failed/30 rounded-md p-3 text-sm text-foreground-muted">
          <AlertCircle className="w-4 h-4 text-status-failed flex-shrink-0 mt-0.5" />
          <div className="min-w-0">
            <div className="text-foreground">Download failed</div>
            <div className="text-xs mt-1 break-words">{status.error}</div>
            {tail.length > 0 && (
              <pre className="mt-2 max-h-32 overflow-y-auto text-[11.5px] font-mono whitespace-pre-wrap">
                {tail.join('\n')}
              </pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function StatusPill({ state }: { state: ExampleState }) {
  if (state === 'ready') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-status-completed-bg text-status-completed text-[11px] font-medium ring-1 ring-inset ring-status-completed/25">
        <CheckCircle2 className="w-3 h-3" />
        Example ready
      </span>
    );
  }
  if (state === 'downloading') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-status-running-bg text-status-running text-[11px] font-medium ring-1 ring-inset ring-status-running/25">
        <Loader2 className="w-3 h-3 animate-spin" />
        Downloading
      </span>
    );
  }
  if (state === 'error') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-status-failed-bg text-status-failed text-[11px] font-medium ring-1 ring-inset ring-status-failed/25">
        <AlertCircle className="w-3 h-3" />
        Failed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-surface-2 text-foreground-subtle text-[11px] font-medium ring-1 ring-inset ring-border-subtle/40">
      Example data missing
    </span>
  );
}
