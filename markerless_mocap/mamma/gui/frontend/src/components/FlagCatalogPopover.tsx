import { useEffect, useRef, useState } from 'react';
import { Info, X, RotateCcw, Loader2, AlertCircle, Plus } from 'lucide-react';

interface FlagEntry {
  name: string;            // e.g. "--export_gt" or "--start"
  valueHint: string | null; // e.g. "START" or null for bool/store_true flags
  help: string | null;
}

interface FlagsResponse {
  step: string;
  scriptPath: string;
  rawHelp: string;
  flags: FlagEntry[];
  warning: string | null;
}

interface Props {
  /** Pipeline step name, e.g. "ma_cap". Passed to /api/steps/<step>/flags. */
  stepName: string;
  /** Called when the user clicks "+" on a flag. Receives the string to
   *  append to the parent's flags array — e.g. "--start <START>" for
   *  flags that take a value, or just "--export_gt" for store_true.
   *  The parent is responsible for placing it; this component does not
   *  mutate the flag list directly. */
  onInsert: (flagSnippet: string) => void;
  /** Set of flag names (without prefix) already in the parent's list.
   *  Used to dim "+ Add" for flags that are already in use. */
  alreadyPresent?: Set<string>;
}

/**
 * Lazy-loaded catalogue of available CLI flags for a pipeline step.
 *
 * Renders as a small "ⓘ Available flags" link. Clicking opens a
 * popover that fetches /api/steps/<step>/flags (cached argparse parse
 * of the script's `--help`). Each row has a "+" button that emits the
 * flag back to the parent via onInsert.
 *
 * Self-contained: lives next to the per-step Flags editor in
 * PresetDigest. Doesn't share state with anything else. Removing it
 * is a one-line revert in PresetDigest.
 */
export function FlagCatalogPopover({ stepName, onInsert, alreadyPresent }: Props) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<FlagsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showRaw, setShowRaw] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const fetchFlags = (refresh = false) => {
    setLoading(true);
    setError(null);
    const url = `/api/steps/${encodeURIComponent(stepName)}/flags${refresh ? '?refresh=true' : ''}`;
    fetch(url)
      .then(async r => {
        if (!r.ok) {
          const e = await r.json().catch(() => ({}));
          throw new Error(e.error || `HTTP ${r.status}`);
        }
        return r.json();
      })
      .then((d: FlagsResponse) => setData(d))
      .catch(e => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  };

  // Lazy-fetch on first open. Subsequent opens reuse the in-memory
  // result unless the user hits Refresh.
  useEffect(() => {
    if (open && !data && !loading && !error) fetchFlags(false);
  }, [open]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Escape closes the modal. (Click-outside is handled by the backdrop
  // onClick below — the modal is a portal-style overlay now, not a
  // popover, so click-outside in the DOM tree doesn't apply.)
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open]);

  return (
    <div ref={containerRef} className="inline-block">
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        className="inline-flex items-center gap-1 text-[11px] text-foreground-subtle hover:text-primary transition-colors"
        title={`Show all flags accepted by ${stepName}'s script (parsed from --help)`}
      >
        <Info className="w-3 h-3" />
        View available flags
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-sm"
          onClick={() => setOpen(false)}
        >
        <div
          onClick={(e) => e.stopPropagation()}
          className="w-full max-w-3xl max-h-[85vh] bg-surface-1 border border-border rounded-xl shadow-2xl shadow-black/50 ring-1 ring-inset ring-white/[0.03] overflow-hidden flex flex-col"
        >
          <div className="flex items-center justify-between px-5 py-3.5 border-b border-border-subtle">
            <div>
              <div className="text-foreground text-base font-medium">
                Available flags · <span className="font-mono">{stepName}</span>
              </div>
              {data && (
                <div className="text-foreground-subtle text-[11px] mt-0.5 font-mono">
                  {data.scriptPath}
                </div>
              )}
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => fetchFlags(true)}
                disabled={loading}
                className="p-1.5 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-2 disabled:opacity-50 transition-colors"
                title="Force re-parse (skip cache)"
              >
                {loading
                  ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  : <RotateCcw className="w-3.5 h-3.5" />}
              </button>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="p-1.5 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-2 transition-colors"
                aria-label="Close"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-2 py-2">
            {error && (
              <div className="flex items-start gap-2 m-2 p-2.5 text-status-failed text-xs bg-status-failed-bg border border-status-failed/35 rounded">
                <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                <div>
                  <div className="font-medium mb-0.5">Couldn't load flags</div>
                  <div className="text-foreground-muted">{error}</div>
                </div>
              </div>
            )}
            {!error && loading && !data && (
              <div className="text-foreground-subtle text-xs px-2 py-4 flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Running <span className="font-mono">--help</span>…
              </div>
            )}
            {data && !showRaw && data.flags.length > 0 && (
              <>
                <div className="text-foreground-subtle text-[11px] px-2 py-1.5 leading-snug">
                  Click <Plus className="inline w-3 h-3" /> to add a flag to this step. Flags
                  taking a value are inserted with a <span className="font-mono">{'<PLACEHOLDER>'}</span>
                  {' '}you should replace.
                </div>
                <table className="w-full text-xs">
                  <tbody>
                    {data.flags.filter(f => f.name !== '--help').map(flag => {
                      const stripped = flag.name.replace(/^-+/, '');
                      const dim = !!alreadyPresent?.has(stripped);
                      return (
                        <tr key={flag.name} className="hover:bg-surface-2 transition-colors">
                          <td className="px-2 py-1.5 align-top">
                            <button
                              type="button"
                              onClick={() => {
                                const snippet = flag.valueHint
                                  ? `${flag.name} <${flag.valueHint}>`
                                  : flag.name;
                                onInsert(snippet);
                              }}
                              disabled={dim}
                              className="inline-flex items-center justify-center w-5 h-5 rounded text-foreground-muted hover:text-primary hover:bg-primary-muted disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                              title={dim
                                ? `'${stripped}' is already in this step's flags`
                                : `Add ${flag.name} to flags`}
                            >
                              <Plus className="w-3.5 h-3.5" />
                            </button>
                          </td>
                          <td className="px-2 py-1.5 align-top font-mono text-foreground whitespace-nowrap">
                            {flag.name}
                            {flag.valueHint && (
                              <span className="text-foreground-subtle ml-1">{flag.valueHint}</span>
                            )}
                          </td>
                          <td className="px-2 py-1.5 align-top text-foreground-muted leading-snug">
                            {flag.help || <span className="italic opacity-50">no description</span>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </>
            )}
            {data && !showRaw && data.flags.length === 0 && (
              <div className="text-foreground-muted text-xs px-2 py-3">
                Couldn't extract structured flags from this script's output.
                Try the raw view below.
              </div>
            )}
            {data && showRaw && (
              <pre className="text-[11px] text-foreground-muted whitespace-pre-wrap font-mono px-2 py-2 leading-snug">
                {data.rawHelp}
              </pre>
            )}
          </div>

          {data && (
            <div className="flex items-center justify-end gap-2 px-4 py-2 border-t border-border-subtle bg-surface-1/80 text-[11px]">
              <button
                type="button"
                onClick={() => setShowRaw(v => !v)}
                className="text-foreground-muted hover:text-foreground transition-colors"
              >
                {showRaw ? 'Hide raw --help' : 'View raw --help'}
              </button>
            </div>
          )}
        </div>
        </div>
      )}
    </div>
  );
}
