import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  ChevronDown,
  FileJson,
  FolderInput,
  FolderOpen,
  Loader2,
  RefreshCw,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import { toast } from 'sonner';
import { stepLabel } from './shared/stepLabels';
import { formatTaskId } from './shared/formatTaskId';

/*
 * "Import previous run" modal — opens from the Tasks page header.
 *
 * Replaces the standalone Database tab. Two panels:
 *
 *   1. Runs on disk not yet in the database. Each row offers an
 *      inline import form whose primary input is the capture.json
 *      path (not a task.json — runners don't persist one).
 *      Capture-first → POST /api/sync/import-task body B.
 *
 *   2. Orphaned DB rows (collapsed by default). Tasks whose output
 *      directories no longer exist; each row has an inline delete
 *      confirmation that calls DELETE /api/sync/task/<id>.
 *
 * On any successful import or delete we re-fetch /api/sync/audit and
 * fire `onImported` so the Tasks table behind the modal can refresh.
 */

interface FsOnlyRun {
  outputId: string;
  outputDir: string;
  dataset: string;
  steps: string[];
  sequences: string[];
  sizeBytes: number;
  sizeHuman: string;
  guessedTaskJsonPath: string;
}

interface DbOnlyTask {
  taskId: string;
  captureName: string;
  outputId: string | null;
  outputPath: string | null;
  createdAt: string | null;
  expectedDir: string;
  steps: string[];
}

interface AuditResponse {
  outputRoot: string;
  summary: { dbTasks: number; fsRuns: number; fsOnly: number; dbOnly: number };
  filesystemOnly: FsOnlyRun[];
  databaseOnly: DbOnlyTask[];
}

/** Mirrors /api/captures. We only use captureName + jsonPath here. */
interface CaptureSummary {
  id: string;
  captureName: string;
  jsonPath: string;
  seqNames: string[];
}

// ───────────────────────── helpers ─────────────────────────────────────

function humanSize(b: number): string {
  if (!b) return '0';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0, n = b;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n >= 100 || i === 0 ? n.toFixed(0) : n.toFixed(1)} ${units[i]}`;
}

interface Props {
  open: boolean;
  onClose: () => void;
  /** Fires after a successful import or orphan delete — the Tasks
   *  table on the page behind this modal uses it to re-query. */
  onMutated?: () => void;
}

export function ImportPreviousRunModal({ open, onClose, onMutated }: Props) {
  const [audit, setAudit] = useState<AuditResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outputRoot, setOutputRoot] = useState<string>('');
  const [outputRootDraft, setOutputRootDraft] = useState<string>('');
  const [captures, setCaptures] = useState<CaptureSummary[]>([]);
  const [activeImport, setActiveImport] = useState<string | null>(null);   // outputId
  const [activeDelete, setActiveDelete] = useState<string | null>(null);   // taskId
  const [orphansOpen, setOrphansOpen] = useState(false);

  const fetchAudit = useCallback(async (root?: string) => {
    setError(null);
    if (audit === null) setLoading(true); else setRefreshing(true);
    try {
      const url = root ? `/api/sync/audit?root=${encodeURIComponent(root)}` : '/api/sync/audit';
      const res = await fetch(url);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `${res.status} ${res.statusText}`);
      }
      const data: AuditResponse = await res.json();
      setAudit(data);
      // Sync the visible output-root input on first fetch so the user
      // can see the resolved default; subsequent fetches use whatever
      // they typed.
      if (!outputRoot) {
        setOutputRoot(data.outputRoot);
        setOutputRootDraft(data.outputRoot);
      }
    } catch (e: any) {
      setError(e.message || 'Failed to load audit.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchCaptures = useCallback(async () => {
    try {
      const res = await fetch('/api/captures');
      if (res.ok) setCaptures(await res.json());
    } catch { /* non-fatal */ }
  }, []);

  // Fetch on open. Reset transient state on close so a re-open is clean.
  useEffect(() => {
    if (open) {
      fetchAudit();
      fetchCaptures();
    } else {
      setActiveImport(null);
      setActiveDelete(null);
      setError(null);
    }
  }, [open, fetchAudit, fetchCaptures]);

  // Esc-to-close, attached only while the modal is open.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const handleRescan = () => {
    const root = outputRootDraft.trim();
    setOutputRoot(root);
    fetchAudit(root || undefined);
  };

  const handleImported = () => {
    setActiveImport(null);
    fetchAudit(outputRoot || undefined);
    onMutated?.();
  };

  const handleDeleted = () => {
    setActiveDelete(null);
    fetchAudit(outputRoot || undefined);
    onMutated?.();
  };

  if (!open) return null;

  const fsOnly = audit?.filesystemOnly ?? [];
  const dbOnly = audit?.databaseOnly ?? [];

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center p-4 sm:p-8 bg-black/65 backdrop-blur-sm overflow-y-auto"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-5xl bg-surface-1 border border-border rounded-xl shadow-2xl shadow-black/50 ring-1 ring-inset ring-white/[0.03] my-auto"
      >
        {/* Header */}
        <div className="flex items-start justify-between px-6 py-4 border-b border-border-subtle">
          <div>
            <div className="text-foreground-subtle text-[11px] uppercase tracking-[0.18em] flex items-center gap-1.5">
              <Upload className="w-3 h-3" aria-hidden /> Import
            </div>
            <h2 className="text-foreground text-xl tracking-tight font-medium mt-1">
              Import previous run
            </h2>
            <p className="text-foreground-muted text-[12px] mt-1 max-w-xl leading-relaxed">
              Register pipeline runs that completed outside the GUI
              (e.g. via the terminal, or before a database reset)
              so they appear in Tasks and Results.
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 -mt-1 -mr-1 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-2 transition-colors"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Output root + rescan */}
        <div className="px-6 py-3 border-b border-border-subtle flex items-center gap-2 flex-wrap">
          <label className="text-foreground-faint text-[10.5px] uppercase tracking-[0.16em]">Output root</label>
          <div className="relative flex-1 min-w-[280px]">
            <FolderOpen className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-foreground-subtle pointer-events-none" aria-hidden />
            <input
              type="text"
              value={outputRootDraft}
              onChange={(e) => setOutputRootDraft(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleRescan(); }}
              placeholder={audit ? audit.outputRoot : 'path/to/output'}
              className="w-full bg-surface-2 border border-border rounded-md pl-8 pr-3 py-1.5 text-foreground text-[12px] font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors"
            />
          </div>
          <button
            onClick={handleRescan}
            disabled={refreshing || loading}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] rounded border border-border text-foreground-muted hover:text-foreground hover:border-border-strong transition-colors disabled:opacity-60"
          >
            <RefreshCw className={'w-3 h-3 ' + (refreshing ? 'animate-spin' : '')} />
            {refreshing ? 'Scanning…' : 'Re-scan'}
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-4 space-y-5">
          {loading && (
            <div className="flex items-center gap-2 text-foreground-muted text-[12px]">
              <Loader2 className="w-4 h-4 animate-spin" /> Scanning {outputRoot || '…'}
            </div>
          )}
          {error && (
            <div className="flex items-start gap-2 p-3 rounded-md border border-status-failed/30 bg-status-failed-bg/40 text-status-failed text-[12px]">
              <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* Panel 1 — fs-only runs */}
          {audit && (
            <section>
              <div className="flex items-baseline justify-between mb-2">
                <h3 className="text-foreground text-[11px] uppercase tracking-[0.16em] font-medium">
                  Runs on disk not yet in the database
                </h3>
                <span className="text-foreground-faint text-[11px] tabular-nums">
                  {fsOnly.length} unimported · {audit.summary.fsRuns} total on disk
                </span>
              </div>
              {fsOnly.length === 0 ? (
                <div className="rounded-md border border-border-subtle bg-surface-2/40 p-4 text-[12px] text-foreground-muted text-center">
                  No unimported runs found under{' '}
                  <span className="font-mono">{audit.outputRoot}</span>.
                  Edit the output root above and re-scan to point at a
                  different tree.
                </div>
              ) : (
                <ul className="space-y-1">
                  {fsOnly.map(run => (
                    <FsOnlyRow
                      key={run.outputId}
                      run={run}
                      captures={captures}
                      outputRoot={audit.outputRoot}
                      expanded={activeImport === run.outputId}
                      onExpand={() => setActiveImport(activeImport === run.outputId ? null : run.outputId)}
                      onImported={handleImported}
                    />
                  ))}
                </ul>
              )}
            </section>
          )}

          {/* Panel 2 — orphans (collapsed by default) */}
          {audit && dbOnly.length > 0 && (
            <section>
              <button
                type="button"
                onClick={() => setOrphansOpen(o => !o)}
                aria-expanded={orphansOpen}
                className="w-full flex items-center justify-between px-3 py-2 rounded-md border border-border-subtle hover:bg-surface-2/40 transition-colors"
              >
                <div className="flex items-center gap-2 text-[12px] text-foreground">
                  <ChevronDown
                    className={'w-3.5 h-3.5 text-foreground-faint transition-transform ' + (orphansOpen ? '' : '-rotate-90')}
                    aria-hidden
                  />
                  <span>
                    Orphaned database rows
                    <span className="text-foreground-faint ml-2">
                      {dbOnly.length} task{dbOnly.length === 1 ? '' : 's'} whose output dir is missing
                    </span>
                  </span>
                </div>
              </button>
              {orphansOpen && (
                <ul className="mt-2 space-y-1">
                  {dbOnly.map(task => (
                    <OrphanRow
                      key={task.taskId}
                      task={task}
                      expanded={activeDelete === task.taskId}
                      onExpand={() => setActiveDelete(activeDelete === task.taskId ? null : task.taskId)}
                      onDeleted={handleDeleted}
                    />
                  ))}
                </ul>
              )}
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Filesystem-only row + inline import form ─────────────────────────

function FsOnlyRow({
  run, captures, outputRoot, expanded, onExpand, onImported,
}: {
  run: FsOnlyRun;
  captures: CaptureSummary[];
  outputRoot: string;
  expanded: boolean;
  onExpand: () => void;
  onImported: () => void;
}) {
  return (
    <li className="rounded-md border border-border-subtle bg-surface-1 hover:bg-surface-2/30 transition-colors">
      <div className="flex items-center gap-3 px-3 py-2">
        <div className="flex-1 min-w-0 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[12px]">
          <span className="text-foreground-faint">Output ID</span>
          <span className="font-mono text-foreground truncate">{run.outputId}</span>
          <span className="text-foreground-faint">Dataset</span>
          <span className="font-mono text-foreground-muted truncate">{run.dataset || '—'}</span>
          <span className="text-foreground-faint">Pipeline</span>
          <span className="text-foreground-muted">
            {run.steps.map(s => stepLabel(s)).join(' · ')}
            <span className="text-foreground-faint ml-2 tabular-nums">{run.sequences.length} seq{run.sequences.length === 1 ? '' : 's'} · {run.sizeHuman}</span>
          </span>
        </div>
        <button
          type="button"
          onClick={onExpand}
          aria-expanded={expanded}
          className={
            'inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] rounded border transition-colors flex-shrink-0 ' +
            (expanded
              ? 'border-primary/60 text-primary bg-primary-muted'
              : 'border-border text-foreground-muted hover:border-primary/60 hover:text-primary')
          }
        >
          {expanded ? <X className="w-3 h-3" /> : <FolderInput className="w-3 h-3" />}
          {expanded ? 'Cancel' : 'Import'}
        </button>
      </div>
      {expanded && (
        <ImportInlineForm
          run={run}
          captures={captures}
          outputRoot={outputRoot}
          onImported={onImported}
        />
      )}
    </li>
  );
}

function ImportInlineForm({
  run, captures, outputRoot, onImported,
}: {
  run: FsOnlyRun;
  captures: CaptureSummary[];
  outputRoot: string;
  onImported: () => void;
}) {
  // Capture-first: the only required user input. We sort captures so
  // an exact name-match floats to the top and pre-select it.
  const sortedCaptures = useMemo(() => {
    const score = (c: CaptureSummary) =>
      c.captureName === run.dataset ? 0
      : (run.dataset && c.captureName.includes(run.dataset)) ? 1
      : 2;
    return [...captures].sort((a, b) => score(a) - score(b) || a.captureName.localeCompare(b.captureName));
  }, [captures, run.dataset]);

  const exactMatch = sortedCaptures.find(c => c.captureName === run.dataset);
  const [captureMode, setCaptureMode] = useState<'pick' | 'custom'>(exactMatch ? 'pick' : sortedCaptures.length > 0 ? 'pick' : 'custom');
  const [capturePath, setCapturePath] = useState<string>(exactMatch?.jsonPath || sortedCaptures[0]?.jsonPath || '');
  const [customPath, setCustomPath] = useState('');
  const [statusPolicy, setStatusPolicy] = useState<'infer' | 'completed'>('infer');
  const [presetPath, setPresetPath] = useState('');
  const [taskJsonPath, setTaskJsonPath] = useState('');
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const effectiveCapture = captureMode === 'pick' ? capturePath : customPath.trim();
  const canSubmit = !!effectiveCapture && !submitting;

  const onSubmit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch('/api/sync/import-task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          captureJsonPath: effectiveCapture,
          outputDir: outputRoot,
          outputId: run.outputId,
          datasetName: run.dataset,
          steps: run.steps,
          statusPolicy,
          presetPath: presetPath.trim() || null,
          taskJsonPath: taskJsonPath.trim() || null,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        if (res.status === 409 && err.code === 'already_imported') {
          setError(`Already imported as task ${formatTaskId(String(err.existingTaskId))}. Refresh to clear this row.`);
        } else {
          setError(err.error || `${res.status} ${res.statusText}`);
        }
        return;
      }
      const data = await res.json();
      toast.success(`Imported run ${run.outputId} as task ${formatTaskId(String(data.taskId))} (${run.sequences.length} seq${run.sequences.length === 1 ? '' : 's'} × ${run.steps.length} step${run.steps.length === 1 ? '' : 's'})`);
      onImported();
    } catch (e: any) {
      setError(e?.message || 'Network error');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="border-t border-border-subtle bg-surface-2/30 p-3 space-y-3">
      {/* Capture picker */}
      <div className="space-y-1.5">
        <div className="flex items-baseline gap-2">
          <span className="text-foreground-faint text-[10.5px] uppercase tracking-[0.14em]">Capture JSON</span>
          <span className="text-status-failed text-[10.5px]">required</span>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <div className="inline-flex rounded-md border border-border bg-surface-2 p-0.5 text-[11px]">
            <button
              type="button"
              onClick={() => setCaptureMode('pick')}
              disabled={sortedCaptures.length === 0}
              className={
                'px-2 py-1 rounded transition-colors disabled:opacity-40 ' +
                (captureMode === 'pick' ? 'bg-primary-muted-strong text-primary' : 'text-foreground-muted hover:text-foreground')
              }
            >
              Registered ({sortedCaptures.length})
            </button>
            <button
              type="button"
              onClick={() => setCaptureMode('custom')}
              className={
                'px-2 py-1 rounded transition-colors ' +
                (captureMode === 'custom' ? 'bg-primary-muted-strong text-primary' : 'text-foreground-muted hover:text-foreground')
              }
            >
              Custom path
            </button>
          </div>
        </div>
        {captureMode === 'pick' ? (
          <select
            value={capturePath}
            onChange={(e) => setCapturePath(e.target.value)}
            className="w-full bg-surface-2 border border-border rounded-md px-2.5 py-1.5 text-foreground text-[12px] focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors"
          >
            {sortedCaptures.length === 0 && <option value="">No registered captures — switch to Custom path</option>}
            {sortedCaptures.map((c, i) => {
              const isSuggested = c.captureName === run.dataset;
              return (
                <option key={c.id} value={c.jsonPath} title={c.jsonPath} className="bg-surface-2">
                  {isSuggested ? '★ ' : (i === 0 && !exactMatch && run.dataset && c.captureName.includes(run.dataset) ? '~ ' : '')}
                  {c.captureName}
                  {c.seqNames.length ? ` · ${c.seqNames.length} seq${c.seqNames.length === 1 ? '' : 's'}` : ''}
                </option>
              );
            })}
          </select>
        ) : (
          <div className="relative">
            <FileJson className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-foreground-subtle pointer-events-none" aria-hidden />
            <input
              type="text"
              value={customPath}
              onChange={(e) => setCustomPath(e.target.value)}
              placeholder="/absolute/path/to/capture.json"
              className="w-full bg-surface-2 border border-border rounded-md pl-8 pr-3 py-1.5 text-foreground text-[12px] font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors"
            />
          </div>
        )}
        {captureMode === 'pick' && exactMatch && (
          <div className="text-foreground-faint text-[10.5px]">
            ★ pre-selected because it matches the run's dataset name.
          </div>
        )}
      </div>

      {/* Status policy */}
      <div className="space-y-1.5">
        <div className="text-foreground-faint text-[10.5px] uppercase tracking-[0.14em]">Status policy</div>
        <div className="flex flex-wrap gap-2 text-[11.5px]">
          {(['infer', 'completed'] as const).map(p => (
            <label
              key={p}
              className={
                'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border cursor-pointer transition-colors ' +
                (statusPolicy === p
                  ? 'border-primary/60 text-primary bg-primary-muted'
                  : 'border-border text-foreground-muted hover:border-border-strong')
              }
            >
              <input
                type="radio"
                name={`policy-${run.outputId}`}
                checked={statusPolicy === p}
                onChange={() => setStatusPolicy(p)}
                className="sr-only"
              />
              <span
                className={'inline-flex w-2.5 h-2.5 rounded-full ' + (statusPolicy === p ? 'bg-primary' : 'border border-border-strong')}
                aria-hidden
              />
              {p === 'infer' ? 'Infer from filesystem' : 'All completed'}
            </label>
          ))}
        </div>
        <div className="text-foreground-faint text-[10.5px] leading-relaxed">
          {statusPolicy === 'infer'
            ? '"Completed" if the per-step output dir contains any files; "Failed" otherwise.'
            : 'Force-mark every step as Completed regardless of disk state.'}
        </div>
      </div>

      {/* Advanced */}
      <div>
        <button
          type="button"
          onClick={() => setAdvancedOpen(v => !v)}
          className="inline-flex items-center gap-1 text-[11px] text-foreground-muted hover:text-foreground transition-colors"
        >
          <ChevronDown className={'w-3 h-3 transition-transform ' + (advancedOpen ? '' : '-rotate-90')} aria-hidden />
          Advanced (audit-only)
        </button>
        {advancedOpen && (
          <div className="mt-2 space-y-2 text-[11.5px]">
            <div className="space-y-1">
              <label className="text-foreground-faint text-[10.5px] uppercase tracking-[0.14em]">Pipeline Configuration Preset path</label>
              <input
                type="text"
                value={presetPath}
                onChange={(e) => setPresetPath(e.target.value)}
                placeholder="optional"
                className="w-full bg-surface-2 border border-border rounded-md px-2.5 py-1.5 text-foreground text-[12px] font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
              />
            </div>
            <div className="space-y-1">
              <label className="text-foreground-faint text-[10.5px] uppercase tracking-[0.14em]">Task.json path</label>
              <input
                type="text"
                value={taskJsonPath}
                onChange={(e) => setTaskJsonPath(e.target.value)}
                placeholder="optional — stored as a string pointer for audit"
                className="w-full bg-surface-2 border border-border rounded-md px-2.5 py-1.5 text-foreground text-[12px] font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
              />
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-start gap-1.5 text-status-failed text-[11px]">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      <div className="flex justify-end">
        <button
          type="button"
          onClick={onSubmit}
          disabled={!canSubmit}
          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] rounded bg-primary text-primary-foreground font-medium disabled:opacity-50 disabled:cursor-not-allowed shadow-sm shadow-black/30"
        >
          {submitting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Upload className="w-3 h-3" />}
          {submitting ? 'Importing…' : 'Import to database'}
        </button>
      </div>
    </div>
  );
}

// ─── Orphan row + inline delete confirm ───────────────────────────────

function OrphanRow({
  task, expanded, onExpand, onDeleted,
}: {
  task: DbOnlyTask;
  expanded: boolean;
  onExpand: () => void;
  onDeleted: () => void;
}) {
  return (
    <li className="rounded-md border border-border-subtle bg-surface-1 hover:bg-surface-2/30 transition-colors">
      <div className="flex items-center gap-3 px-3 py-2">
        <div className="flex-1 min-w-0 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[12px]">
          <span className="text-foreground-faint">Task</span>
          <span className="font-mono text-foreground">{formatTaskId(task.taskId)} · {task.captureName}</span>
          <span className="text-foreground-faint">Output ID</span>
          <span className="font-mono text-foreground-muted truncate">{task.outputId || '—'}</span>
          <span className="text-foreground-faint">Expected</span>
          <span className="font-mono text-foreground-subtle text-[11px] truncate">{task.expectedDir}</span>
        </div>
        <button
          type="button"
          onClick={onExpand}
          aria-expanded={expanded}
          className={
            'inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] rounded border transition-colors flex-shrink-0 ' +
            (expanded
              ? 'border-status-failed/60 text-status-failed bg-status-failed-bg'
              : 'border-border text-foreground-muted hover:border-status-failed/40 hover:text-status-failed')
          }
        >
          {expanded ? <X className="w-3 h-3" /> : <Trash2 className="w-3 h-3" />}
          {expanded ? 'Cancel' : 'Delete'}
        </button>
      </div>
      {expanded && <DeleteInlineConfirm task={task} onDeleted={onDeleted} />}
    </li>
  );
}

function DeleteInlineConfirm({
  task, onDeleted,
}: { task: DbOnlyTask; onDeleted: () => void }) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`/api/sync/task/${encodeURIComponent(task.taskId)}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setError(err.error || `${res.status} ${res.statusText}`);
        return;
      }
      toast.success(`Removed orphaned task ${formatTaskId(task.taskId)}`);
      onDeleted();
    } catch (e: any) {
      setError(e?.message || 'Network error');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="border-t border-border-subtle bg-status-failed-bg/30 p-3 space-y-2">
      <div className="flex items-start gap-2 text-[11.5px] text-foreground">
        <AlertTriangle className="w-3.5 h-3.5 text-status-failed flex-shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          Drop the database row for task <span className="font-mono">{formatTaskId(task.taskId)}</span>{' '}
          and all of its process records ({task.steps.length} step{task.steps.length === 1 ? '' : 's'}).
          {' '}This is a one-way operation; the run_configs file on disk
          (if any) is not removed.
        </div>
      </div>
      {error && (
        <div className="flex items-start gap-1.5 text-status-failed text-[11px]">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}
      <div className="flex justify-end gap-2">
        <button
          type="button"
          onClick={onSubmit}
          disabled={submitting}
          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] rounded bg-status-failed/80 hover:bg-status-failed text-background font-medium disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
        >
          {submitting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
          {submitting ? 'Removing…' : 'Yes, remove the row'}
        </button>
      </div>
    </div>
  );
}

export default ImportPreviousRunModal;
