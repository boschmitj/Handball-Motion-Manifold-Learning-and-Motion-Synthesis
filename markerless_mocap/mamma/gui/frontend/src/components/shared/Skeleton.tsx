import { ReactNode } from 'react';

interface SkeletonProps {
  /** Tailwind classes for size + shape (e.g. "w-32 h-4 rounded-md"). */
  className?: string;
}

/**
 * A single shimmering placeholder bar. Pairs the project's
 * `animate-shimmer` keyframe with whatever shape (width / height /
 * rounding) the caller supplies via `className`.
 *
 * Use this as the building block for content-shaped skeletons
 * (rows / cards / table cells) — much more readable while loading
 * than a bare "Loading…" string, and the gradient sweep makes the
 * pause feel intentional rather than stuck.
 */
export function Skeleton({ className = '' }: SkeletonProps) {
  return (
    <div
      role="status"
      aria-label="Loading"
      className={`animate-shimmer rounded-md ${className}`}
    />
  );
}

/**
 * One row in a "loading rows" group — used by the Outputs explorer's
 * file listing while a directory is being fetched. Mirrors the real
 * file-row layout (icon + name + size on the right).
 */
export function FileRowSkeleton({ widthClass = 'w-40' }: { widthClass?: string }) {
  return (
    <div className="w-full flex items-center gap-3 px-4 py-1.5">
      <Skeleton className="w-4 h-4 rounded-sm" />
      <Skeleton className={`h-3 ${widthClass}`} />
      <div className="flex-1" />
      <Skeleton className="h-3 w-12" />
    </div>
  );
}

/**
 * Stacks `count` FileRowSkeletons with varying widths so the placeholder
 * doesn't read as a perfect grid. Easier on the eye and feels more like
 * "real content arriving" than identical bars.
 */
export function FileRowsSkeleton({ count = 4 }: { count?: number }): ReactNode {
  const widths = ['w-40', 'w-56', 'w-32', 'w-48', 'w-64', 'w-36'];
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <FileRowSkeleton key={i} widthClass={widths[i % widths.length]} />
      ))}
    </>
  );
}
