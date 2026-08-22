import { useState } from 'react';
import { ImageOff } from 'lucide-react';

interface Props {
  /** Absolute path on disk; rendered via /api/files/image. null/undefined
   *  shows the placeholder. */
  path: string | null | undefined;
  alt: string;
  /** Tailwind utility classes for size + aspect. Defaults to 64×40 (16:10),
   *  the table-row size. Cards override with `w-full h-full aspect-video`. */
  className?: string;
  /** Native lazy-loading. On for off-screen rows; off if you want eager
   *  loading (e.g., the only thumbnail in a hero). */
  loading?: 'eager' | 'lazy';
  /** Object-fit policy. `cover` is the default (crops if needed). `contain`
   *  preserves the full image at the cost of letterboxing. */
  fit?: 'cover' | 'contain';
}

/**
 * Capture thumbnail. Falls back to a subtle "no image" placeholder when
 * the path is missing, the file fails to load, or the image errors out.
 * Backed by the existing `/api/files/image` endpoint (no auth gate; same
 * trust model as the rest of the file APIs).
 */
export function Thumbnail({ path, alt, className = 'w-16 h-10', loading = 'lazy', fit = 'cover' }: Props) {
  const [errored, setErrored] = useState(false);
  const base = `${className} rounded-md border border-border-subtle bg-surface-2`;
  if (!path || errored) {
    return (
      <div className={`${base} flex items-center justify-center text-foreground-faint`} aria-label="No preview">
        <ImageOff className="w-4 h-4" />
      </div>
    );
  }
  return (
    <img
      src={`/api/files/image?path=${encodeURIComponent(path)}`}
      alt={alt}
      loading={loading}
      onError={() => setErrored(true)}
      className={`${base} ${fit === 'cover' ? 'object-cover' : 'object-contain'}`}
    />
  );
}
