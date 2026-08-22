import { Camera, Pencil, Plus, Search, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { InlineCaptureJsonForm } from './InlineCaptureJsonForm';
import { Thumbnail } from './shared/Thumbnail';
import { PathWithCopy } from './shared/PathWithCopy';
import { ThumbnailPicker } from './ThumbnailPicker';

interface Capture {
  id: string;
  captureName: string;
  jsonPath: string;
  seqNames: string[];
  cams: string[];
  outputDir: string;
  taskCount: number;
  thumbnailPath: string | null;
  createdAt: string;
  /** "user" (DB-backed, writable) or "example" (shipped under configs/captures/, read-only). */
  source?: 'user' | 'example';
  // Status / lastTaskAt / processes are intentionally ignored on this
  // page — Captures is about INPUTS. Output activity lives under Results
  // and Tasks. Showing a rolled-up status here misled users (one Failed
  // among ten Completed reads as Failed at the capture level).
}

interface Props {
  /** Click on a row → caller decides where to navigate (CaptureManage). */
  onOpen?: (captureName: string, jsonPath: string) => void;
}

/** Cap on inline chips per chip-list (sequences, cameras). Anything beyond
 *  collapses into a `+N more` chip whose tooltip lists the rest. */
const CHIP_CAP = 5;
const CAM_CHIP_CAP = 3;

/**
 * Captures management table — input-side view. The user lands here to
 * understand "what data do I have to work with?" and to add / edit /
 * remove capture.json files. Optimised for scanning input metadata at a
 * glance: thumbnail, identity (name + path), sequences, cameras.
 */
export function CapturesList({ onOpen }: Props) {
  const [captures, setCaptures] = useState<Capture[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [search, setSearch] = useState('');
  /** When set, shows the ThumbnailPicker modal for this capture. */
  const [pickerFor, setPickerFor] = useState<Capture | null>(null);

  const refresh = () => {
    setError(null);
    fetch('/api/captures')
      .then(res => res.ok ? res.json() : Promise.reject(new Error('fetch failed')))
      .then(data => setCaptures(data))
      .catch(err => { console.error('Error fetching captures:', err); setError(String(err.message ?? err)); setCaptures([]); });
  };

  useEffect(() => { refresh(); }, []);

  const filtered = useMemo(() => {
    if (!captures) return null;
    const q = search.trim().toLowerCase();
    if (!q) return captures;
    return captures.filter(c =>
      c.captureName.toLowerCase().includes(q) ||
      (c.jsonPath?.toLowerCase().includes(q) ?? false)
    );
  }, [captures, search]);

  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-end justify-between gap-4 mb-8 flex-wrap">
        <div>
          <h2 className="text-3xl text-foreground tracking-tight font-medium mb-1 flex items-center gap-2">
            <Camera className="w-6 h-6 text-primary" />
            Captures
          </h2>
          <p className="text-foreground-muted text-sm">
            Pipeline inputs per dataset: footage paths and camera configuration.
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {captures && captures.length > 0 && <SearchBox value={search} onChange={setSearch} />}
          <button
            onClick={() => setShowAdd(o => !o)}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-primary text-primary-foreground hover:opacity-90 rounded-md text-sm font-medium transition-opacity shadow-sm shadow-black/30"
          >
            <Plus className="w-4 h-4" />
            New capture
          </button>
        </div>
      </div>

      {showAdd && (
        <div className="mb-6 bg-surface-1 border border-primary/30 rounded-xl p-5 shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02]">
          <div className="text-foreground text-sm font-medium mb-3">Create a new capture</div>
          <InlineCaptureJsonForm onCreated={() => { setShowAdd(false); refresh(); }} />
        </div>
      )}

      {error && (
        <div className="bg-status-failed-bg border border-status-failed/35 rounded-lg p-4 text-status-failed text-sm mb-6">
          {error}
        </div>
      )}

      {pickerFor && (
        <ThumbnailPicker
          jsonPath={pickerFor.jsonPath}
          captureName={pickerFor.captureName}
          onClose={() => setPickerFor(null)}
          onSaved={() => { setPickerFor(null); refresh(); }}
        />
      )}

      {captures === null ? (
        <SkeletonTable />
      ) : captures.length === 0 ? (
        <div className="text-center py-16 bg-surface-1 border border-border-subtle rounded-xl shadow-sm shadow-black/30">
          <div className="text-foreground text-base mb-1">No captures yet</div>
          <div className="text-foreground-subtle text-sm">
            Click "New capture" to create one from a footage root (images or videos) and calibration.
          </div>
        </div>
      ) : filtered && filtered.length === 0 ? (
        <div className="text-center py-12 bg-surface-1 border border-border-subtle rounded-xl">
          <div className="text-foreground-subtle text-sm">
            No captures match “{search}”.{' '}
            <button onClick={() => setSearch('')} className="text-primary hover:underline ml-1">Clear</button>
          </div>
        </div>
      ) : (
        <div className="bg-surface-1 border border-border-subtle rounded-xl overflow-hidden shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02]">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-surface-2/60 border-b border-border-subtle">
                  <th className="w-[88px]"></th>
                  <Th>Capture</Th>
                  <Th>Sequences</Th>
                  <Th>Cameras</Th>
                  <th className="w-px"></th>
                </tr>
              </thead>
              <tbody>
                {filtered!.map((capture, idx) => {
                  const stripeBg = idx % 2 === 0 ? 'bg-surface-1' : 'bg-surface-1/60';
                  return (
                    <tr
                      key={capture.id}
                      onClick={() => onOpen?.(capture.captureName, capture.jsonPath)}
                      className={`mamma-row-clickable group border-b border-border-subtle/60 ${stripeBg}`}
                    >
                      <td className="pl-6 pr-2 py-3">
                        <ThumbnailWithEdit
                          capture={capture}
                          onEdit={(c) => setPickerFor(c)}
                        />
                      </td>
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-2">
                          <span className="text-foreground">{capture.captureName}</span>
                          {capture.source === 'example' && (
                            <span
                              className="px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide bg-surface-2 text-foreground-muted border border-border-subtle"
                              title="Shipped example (read-only). Use 'Save a writable copy' to edit."
                            >
                              example
                            </span>
                          )}
                        </div>
                        <PathWithCopy path={capture.jsonPath} />
                      </td>
                      <td className="px-4 py-4">
                        <ChipList items={capture.seqNames} cap={CHIP_CAP} unitSingular="sequence" unitPlural="sequences" />
                      </td>
                      <td className="px-4 py-4">
                        <ChipList items={capture.cams} cap={CAM_CHIP_CAP} unitSingular="camera" unitPlural="cameras" />
                      </td>
                      <td className="px-6 py-4 text-right">
                        <button
                          onClick={(e) => { e.stopPropagation(); onOpen?.(capture.captureName, capture.jsonPath); }}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-primary-muted hover:bg-primary-muted-strong border border-primary/35 hover:border-primary/55 text-primary rounded-md text-sm transition-colors"
                          title="Edit this capture"
                        >
                          <Pencil className="w-3.5 h-3.5" />
                          Edit
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ====================================================================
// Local primitives
// ====================================================================

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="text-left px-4 py-3 text-foreground-muted text-[11px] font-semibold uppercase tracking-wider">
      {children}
    </th>
  );
}

/**
 * Compact chip list. Renders a count line + up to `cap` chips, with a
 * `+N more` chip whose `title` tooltip lists the rest.
 */
function ChipList({
  items, cap, unitSingular, unitPlural,
}: {
  items: string[];
  cap: number;
  unitSingular: string;
  unitPlural: string;
}) {
  if (!items || items.length === 0) {
    return <div className="text-foreground-faint text-xs italic">none</div>;
  }
  const visible = items.slice(0, cap);
  const overflow = items.slice(cap);
  return (
    <div>
      <div className="text-foreground-muted text-xs mb-1.5 tabular-nums">
        {items.length} {items.length === 1 ? unitSingular : unitPlural}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {visible.map(name => (
          <span key={name} className="bg-surface-2 border border-border-subtle text-foreground-muted px-2 py-0.5 rounded-md text-xs font-mono">
            {name}
          </span>
        ))}
        {overflow.length > 0 && (
          <span
            title={overflow.join(', ')}
            className="bg-surface-2 border border-border-subtle text-foreground-subtle px-2 py-0.5 rounded-md text-xs cursor-help"
          >
            +{overflow.length} more
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Thumbnail wrapper that surfaces an edit affordance on hover.
 * - The pencil icon is invisible until the user hovers (`opacity-0 → 100`).
 * - Click stops propagation so the row's navigation handler doesn't fire.
 * - Group-hover ring brightens the thumbnail edge to telegraph "this is interactive".
 */
function ThumbnailWithEdit({
  capture, onEdit,
}: {
  capture: Capture;
  onEdit: (capture: Capture) => void;
}) {
  return (
    <div className="relative group/thumb">
      <Thumbnail path={capture.thumbnailPath} alt={capture.captureName} />
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onEdit(capture); }}
        title="Edit thumbnail"
        aria-label="Edit thumbnail"
        className="absolute inset-0 flex items-center justify-center bg-black/55 text-foreground opacity-0 group-hover/thumb:opacity-100 focus-visible:opacity-100 transition-opacity rounded-md"
      >
        <Pencil className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

function SearchBox({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="relative">
      <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-foreground-subtle pointer-events-none" />
      <input
        type="text"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder="Filter by name or path…"
        className="bg-surface-2 border border-border rounded-md pl-7 pr-7 py-1.5 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors w-60 placeholder:text-foreground-faint"
      />
      {value && (
        <button
          onClick={() => onChange('')}
          className="absolute right-1.5 top-1/2 -translate-y-1/2 text-foreground-subtle hover:text-foreground transition-colors"
          aria-label="Clear search"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}

/**
 * Table-shaped skeleton — matches the column widths of the live table so
 * the swap is visually quiet when data lands.
 */
function SkeletonTable() {
  return (
    <div className="bg-surface-1 border border-border-subtle rounded-xl overflow-hidden shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02]">
      <div className="overflow-x-auto">
        <table className="w-full">
          <tbody>
            {Array.from({ length: 4 }).map((_, idx) => (
              <tr key={idx} className={`border-b border-border-subtle/60 ${idx % 2 === 0 ? 'bg-surface-1' : 'bg-surface-1/60'}`}>
                <td className="pl-6 pr-2 py-3 w-[88px]">
                  <div className="w-16 h-10 rounded-md bg-surface-2 animate-pulse" />
                </td>
                <td className="px-4 py-4">
                  <div className="h-3.5 w-40 rounded bg-surface-2 animate-pulse mb-2" />
                  <div className="h-3 w-64 rounded bg-surface-2 animate-pulse" />
                </td>
                <td className="px-4 py-4">
                  <div className="h-3 w-16 rounded bg-surface-2 animate-pulse mb-2" />
                  <div className="flex flex-wrap gap-1.5">
                    {Array.from({ length: 3 }).map((_, j) => (
                      <div key={j} className="h-5 w-16 rounded bg-surface-2 animate-pulse" />
                    ))}
                  </div>
                </td>
                <td className="px-4 py-4">
                  <div className="h-3 w-16 rounded bg-surface-2 animate-pulse mb-2" />
                  <div className="flex flex-wrap gap-1.5">
                    {Array.from({ length: 2 }).map((_, j) => (
                      <div key={j} className="h-5 w-12 rounded bg-surface-2 animate-pulse" />
                    ))}
                  </div>
                </td>
                <td className="px-6 py-4 text-right">
                  <div className="h-7 w-16 rounded-md bg-surface-2 animate-pulse inline-block" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
