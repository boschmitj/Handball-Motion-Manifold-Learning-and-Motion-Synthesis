/**
 * Format an ISO/parseable timestamp as "X ago" using the browser's
 * Intl.RelativeTimeFormat. Falls back to a locale date string for very
 * old timestamps or environments without the Intl API.
 */
const RTF = typeof Intl !== 'undefined' && (Intl as any).RelativeTimeFormat
  ? new (Intl as any).RelativeTimeFormat('en', { numeric: 'auto' })
  : null;

export function formatRelativeTime(iso: string | null | undefined): string {
  if (!iso) return '';
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const diffMs = t - Date.now();
  const absMin = Math.abs(diffMs) / 60_000;
  if (!RTF) return new Date(t).toLocaleDateString();
  if (absMin < 1) return 'just now';
  if (absMin < 60) return RTF.format(Math.round(diffMs / 60_000), 'minute');
  if (absMin < 60 * 24) return RTF.format(Math.round(diffMs / 3_600_000), 'hour');
  if (absMin < 60 * 24 * 30) return RTF.format(Math.round(diffMs / 86_400_000), 'day');
  return new Date(t).toLocaleDateString();
}
