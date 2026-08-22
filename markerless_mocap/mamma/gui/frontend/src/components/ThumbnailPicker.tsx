import { useEffect, useState } from 'react';
import { X, Check, RotateCcw, ImageOff, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import { Thumbnail } from './shared/Thumbnail';

interface Candidate {
  cam: string;
  path: string;
}

interface CandidatesResponse {
  candidates: Candidate[];
  userOverride: string | null;
  autoDetected: string | null;
}

interface Props {
  /** Absolute path to the capture.json being edited. */
  jsonPath: string;
  captureName: string;
  onClose: () => void;
  /** Called after a successful save so the parent can refresh the listing. */
  onSaved: () => void;
}

/**
 * Discreet thumbnail-picker modal opened from the Captures table.
 *
 * Resolution policy in the picker:
 *   - "Auto" (default): no `thumbnail` field in capture.json; the backend
 *     picks the first cam's first frame on each list refresh.
 *   - One-per-camera candidate: explicit override pointing at that frame.
 *   - Custom path: explicit override pointing anywhere on disk.
 *
 * Save = GET current capture.json + write the chosen value into the
 * `thumbnail` field (or remove it for "Auto") + PUT back via the existing
 * /api/captures/json route. Keeps the API surface tiny — no bespoke patch
 * endpoint.
 */
export function ThumbnailPicker({ jsonPath, captureName, onClose, onSaved }: Props) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CandidatesResponse | null>(null);
  /** Selected absolute path, or '' for "Auto". null = nothing chosen yet
   *  (the modal seeds it from the current effective thumbnail on load). */
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [customPath, setCustomPath] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`/api/captures/thumbnail-candidates?path=${encodeURIComponent(jsonPath)}`)
      .then(async r => {
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.error || `${r.status} ${r.statusText}`);
        }
        return r.json() as Promise<CandidatesResponse>;
      })
      .then(d => {
        if (cancelled) return;
        setData(d);
        // Seed selection: explicit override wins, else "Auto".
        setSelectedPath(d.userOverride ?? '');
      })
      .catch(e => { if (!cancelled) setError(String(e.message ?? e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [jsonPath]);

  const isCustomSelected =
    !!selectedPath && data != null && !data.candidates.some(c => c.path === selectedPath);

  const save = async () => {
    setSaving(true);
    try {
      // Round-trip through the existing read+write endpoints. Two requests
      // for a one-off action keeps the backend API surface unchanged.
      const readRes = await fetch(`/api/captures/json?path=${encodeURIComponent(jsonPath)}`);
      if (!readRes.ok) {
        const err = await readRes.json().catch(() => ({}));
        toast.error(err.error || 'Failed to read capture');
        setSaving(false);
        return;
      }
      const readData = await readRes.json();
      const content = { ...(readData.content ?? {}) };

      if (selectedPath) {
        content.thumbnail = selectedPath;
      } else {
        delete content.thumbnail; // "Auto" — strip the field entirely
      }

      const writeRes = await fetch('/api/captures/json', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: readData.absolutePath, content }),
      });
      if (writeRes.ok) {
        toast.success(selectedPath ? 'Thumbnail updated' : 'Thumbnail reset to auto');
        onSaved();
      } else {
        const err = await writeRes.json().catch(() => ({}));
        toast.error(err.error || `Failed to save (${writeRes.status})`);
        setSaving(false);
      }
    } catch (e) {
      console.error(e);
      toast.error('Failed to save. See console.');
      setSaving(false);
    }
  };

  const reset = () => {
    setSelectedPath('');
    setCustomPath('');
  };

  return (
    <div
      className="fixed inset-0 bg-black/65 backdrop-blur-sm flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-surface-1 border border-border rounded-xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl shadow-black/60 ring-1 ring-inset ring-white/[0.03]"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-3 p-4 border-b border-border-subtle bg-surface-2/40">
          <div className="min-w-0">
            <div className="text-foreground text-base font-medium truncate" title={captureName}>
              Thumbnail · {captureName}
            </div>
            <div className="text-foreground-subtle text-xs mt-0.5">
              Choose a frame from any camera, paste a custom path, or fall back to auto.
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-foreground-subtle hover:text-foreground p-1.5 -m-1.5 rounded-md hover:bg-surface-3 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {loading && (
            <div className="flex items-center justify-center py-12 text-foreground-muted text-sm">
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Scanning input frames…
            </div>
          )}
          {error && !loading && (
            <div className="bg-status-failed-bg border border-status-failed/35 rounded-md p-3 text-status-failed text-sm">
              {error}
            </div>
          )}

          {data && !loading && !error && (
            <>
              {/* Auto option */}
              <CandidateTile
                label="Auto"
                sublabel={data.autoDetected
                  ? 'Use the first available input frame.'
                  : 'No input frame detected — placeholder will show.'}
                path={data.autoDetected}
                selected={selectedPath === ''}
                onClick={() => setSelectedPath('')}
                wide
              />

              {/* Per-camera grid */}
              {data.candidates.length > 0 && (
                <div>
                  <div className="text-foreground-subtle text-[11px] uppercase tracking-wider font-medium mb-2">
                    Pick a camera
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                    {data.candidates.map(c => (
                      <CandidateTile
                        key={c.path}
                        label={c.cam}
                        path={c.path}
                        selected={selectedPath === c.path}
                        onClick={() => setSelectedPath(c.path)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Custom path */}
              <div>
                <div className="text-foreground-subtle text-[11px] uppercase tracking-wider font-medium mb-2">
                  Or paste a custom path
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={customPath}
                    onChange={e => setCustomPath(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter' && customPath.trim()) setSelectedPath(customPath.trim());
                    }}
                    placeholder="/absolute/path/to/image.png"
                    className="flex-1 bg-surface-2 border border-border rounded-md px-2.5 py-1.5 text-foreground text-xs font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
                  />
                  <button
                    type="button"
                    onClick={() => customPath.trim() && setSelectedPath(customPath.trim())}
                    disabled={!customPath.trim()}
                    className="px-3 py-1.5 text-xs text-foreground-muted hover:text-foreground rounded-md border border-border hover:border-border-strong disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    Use
                  </button>
                </div>
                {isCustomSelected && (
                  <div className="text-foreground-faint text-[11px] mt-1.5 truncate" title={selectedPath ?? ''}>
                    Selected custom: <span className="font-mono text-foreground-subtle">{selectedPath}</span>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        <footer className="flex items-center justify-between gap-2 p-4 border-t border-border-subtle bg-surface-2/30">
          <button
            onClick={reset}
            disabled={selectedPath === ''}
            className="inline-flex items-center gap-1.5 text-xs text-foreground-muted hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed px-2.5 py-1.5 rounded-md border border-border hover:border-border-strong transition-colors"
            title="Clear override and use auto-detection"
          >
            <RotateCcw className="w-3 h-3" />
            Reset to auto
          </button>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              disabled={saving}
              className="px-3 py-1.5 text-xs text-foreground-muted hover:text-foreground rounded-md border border-border hover:border-border-strong transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={save}
              disabled={saving || selectedPath === null || (data?.userOverride ?? '') === selectedPath}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-medium rounded-md transition-opacity"
            >
              <Check className="w-3 h-3" />
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

/**
 * One pickable tile in the picker. `wide` doubles the height for the
 * "Auto" hero option at the top.
 */
function CandidateTile({
  label, sublabel, path, selected, onClick, wide,
}: {
  label: string;
  sublabel?: string;
  path: string | null;
  selected: boolean;
  onClick: () => void;
  wide?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`group relative bg-surface-2 border ${selected ? 'border-primary/60 ring-2 ring-primary/40' : 'border-border-subtle hover:border-border-strong'} rounded-md overflow-hidden text-left transition-all`}
    >
      <div className={`${wide ? 'aspect-[16/6]' : 'aspect-video'} w-full bg-surface-3 flex items-center justify-center`}>
        {path ? (
          <Thumbnail path={path} alt={label} className="w-full h-full" loading="eager" fit="cover" />
        ) : (
          <ImageOff className="w-5 h-5 text-foreground-faint" />
        )}
      </div>
      <div className="px-2.5 py-1.5">
        <div className={`text-xs font-medium ${selected ? 'text-primary' : 'text-foreground'}`}>{label}</div>
        {sublabel && <div className="text-foreground-subtle text-[11px]">{sublabel}</div>}
      </div>
      {selected && (
        <span className="absolute top-1.5 right-1.5 inline-flex items-center justify-center w-5 h-5 rounded-full bg-primary text-primary-foreground shadow-sm">
          <Check className="w-3 h-3" />
        </span>
      )}
    </button>
  );
}
