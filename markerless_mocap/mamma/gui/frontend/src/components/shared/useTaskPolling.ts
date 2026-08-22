import { useEffect, useRef, useState, useCallback } from 'react';

interface Options {
  /** ms between polls. Set to 0 to disable auto-polling (manual refresh only). */
  intervalMs?: number;
  /** When false, polling is paused and no fetches happen. Used to stop while a modal is open or the tab is hidden. */
  enabled?: boolean;
}

/**
 * Generic polling hook. Fetches `url` immediately on mount and at the
 * given interval. Returns the latest decoded JSON, an error if any,
 * and a manual `refresh` callback for explicit user-driven updates.
 *
 * Used by the matrix view for live status updates, and (for later phases)
 * by anywhere else that wants to mirror /api/processes/active in real time.
 */
export function useTaskPolling<T>(url: string, options: Options = {}) {
  const { intervalMs = 2000, enabled = true } = options;
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const abortRef = useRef<AbortController | null>(null);

  const fetchOnce = useCallback(async () => {
    abortRef.current?.abort();
    const ctl = new AbortController();
    abortRef.current = ctl;
    try {
      const res = await fetch(url, { signal: ctl.signal });
      if (ctl.signal.aborted) return;
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        setError(err.error || res.statusText);
        return;
      }
      const json = (await res.json()) as T;
      if (!ctl.signal.aborted) {
        setData(json);
        setError(null);
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError(String(e));
    } finally {
      if (!ctl.signal.aborted) setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    if (!enabled) return;
    fetchOnce();
    if (intervalMs > 0) {
      const id = setInterval(fetchOnce, intervalMs);
      return () => { clearInterval(id); abortRef.current?.abort(); };
    }
    return () => { abortRef.current?.abort(); };
  }, [enabled, intervalMs, fetchOnce]);

  return { data, error, loading, refresh: fetchOnce };
}
