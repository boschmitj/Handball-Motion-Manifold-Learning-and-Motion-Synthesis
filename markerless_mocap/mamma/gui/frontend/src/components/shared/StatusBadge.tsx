import { Play, XCircle, Clock, Check, Hourglass } from 'lucide-react';

export type StatusKind = 'Running' | 'Pending' | 'Completed' | 'Failed' | 'Queued';

/** Map any backend status string to one of five broad categories. */
export function statusKind(status: string): StatusKind {
  const s = (status || '').toLowerCase();
  if (s.includes('failed') || s.includes('cancelled')) return 'Failed';
  if (s.includes('running') || s.includes('retrying')) return 'Running';
  if (s.includes('completed') || s.includes('done')) return 'Completed';
  if (s.includes('queued')) return 'Queued';
  return 'Pending';
}

interface BadgeStyle {
  icon: typeof Play;
  pill: string; // bg + border + text colour for the pill
  dot: string;  // bg-* for the leading dot indicator
}

const styles: Record<StatusKind, BadgeStyle> = {
  Running:   {
    icon: Play,
    pill: 'bg-status-running-bg    border-status-running/35    text-status-running',
    dot:  'bg-status-running',
  },
  Failed:    {
    icon: XCircle,
    pill: 'bg-status-failed-bg     border-status-failed/35     text-status-failed',
    dot:  'bg-status-failed',
  },
  Pending:   {
    icon: Clock,
    pill: 'bg-status-pending-bg    border-status-pending/35    text-status-pending',
    dot:  'bg-status-pending',
  },
  Completed: {
    icon: Check,
    pill: 'bg-status-completed-bg  border-status-completed/35  text-status-completed',
    dot:  'bg-status-completed',
  },
  Queued: {
    // Re-uses Pending's amber palette but with an hourglass icon so the
    // task-coordinator's "not yet started" state reads distinctly from
    // the runner-side "Waiting" / "Pending" states.
    icon: Hourglass,
    pill: 'bg-status-pending-bg    border-status-pending/35    text-status-pending',
    dot:  'bg-status-pending',
  },
};

/** Style tokens exposed for components that build their own pill variants. */
export function statusStyle(kind: StatusKind) {
  return styles[kind];
}

interface BadgeProps {
  status: string;
  /** Compact = small dot + label, no icon. Used in dense tables. */
  compact?: boolean;
}

export function StatusBadge({ status, compact = false }: BadgeProps) {
  const kind = statusKind(status);
  const s = styles[kind];
  const pulsing = kind === 'Running';

  if (compact) {
    return (
      <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs border ${s.pill} whitespace-nowrap`}>
        <Dot className={s.dot} pulsing={pulsing} />
        {status}
      </span>
    );
  }

  const Icon = s.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border ${s.pill}`}>
      <Icon className={`w-3.5 h-3.5 ${pulsing ? 'animate-pulse' : ''}`} />
      <span className="text-sm whitespace-nowrap">{status}</span>
    </span>
  );
}

/** Coloured leading dot. Pulses when the parent passes pulsing=true. */
export function Dot({ className, pulsing = false }: { className: string; pulsing?: boolean }) {
  return (
    <span className="relative inline-flex w-2 h-2 flex-shrink-0">
      {pulsing && (
        <span className={`absolute inset-0 rounded-full opacity-60 animate-ping ${className}`} />
      )}
      <span className={`relative inline-block w-full h-full rounded-full ${className}`} />
    </span>
  );
}
