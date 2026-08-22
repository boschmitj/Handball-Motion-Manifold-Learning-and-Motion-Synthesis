import { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { toast } from 'sonner';

interface Props {
  path: string;
  /** Tailwind max-width class for the truncated text. Defaults to xs. */
  maxWidthClass?: string;
}

/**
 * Path display that keeps the *end* of the string visible (filename, the
 * most identifying part) and ellipsises the prefix at the left, plus a
 * click-to-copy button.
 *
 * Implementation note: CSS truncates with text-overflow: ellipsis only on
 * the right. To get a left-side ellipsis we render the path in a
 * direction:rtl container with a leading left-to-right mark (‎) so the
 * path itself stays visually LTR. The click on the copy button stops
 * propagation so it doesn't trigger any row-level click handler.
 */
export function PathWithCopy({ path, maxWidthClass = 'max-w-xs' }: Props) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(path);
      setCopied(true);
      toast.success('Path copied');
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error('Failed to copy. Path: ' + path);
    }
  };
  return (
    <div className={`flex items-center gap-1.5 ${maxWidthClass}`}>
      <span
        title={path}
        className="text-foreground-subtle text-xs font-mono overflow-hidden whitespace-nowrap"
        style={{ direction: 'rtl', textOverflow: 'ellipsis', textAlign: 'left' }}
      >
        &lrm;{path}
      </span>
      <button
        onClick={handleCopy}
        className="flex-shrink-0 text-foreground-subtle hover:text-foreground p-1 -my-1 rounded hover:bg-surface-3 transition-colors"
        title="Copy full path"
      >
        {copied ? <Check className="w-3 h-3 text-status-completed" /> : <Copy className="w-3 h-3" />}
      </button>
    </div>
  );
}
