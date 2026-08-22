import { FileText } from 'lucide-react';
import { statusKind } from './StatusBadge';

interface Props {
  /** Status determines whether log files are likely to exist yet. */
  status: string;
  /** Identifier used to label the file in the viewer (e.g. "task_11_horse_test_ma_cap"). */
  labelPrefix: string;
  outFile?: string | null;
  errFile?: string | null;
  onView: (fileName: string, filePath: string) => void;
}

/**
 * The .err / .out button pair that Active.tsx and Logs.tsx both render
 * next to each (step, sequence) row. Hidden when the process hasn't
 * started yet (Pending).
 */
export function ProcessLogButtons({ status, labelPrefix, outFile, errFile, onView }: Props) {
  // Skip rendering anything if the process is still queued — the runner
  // hasn't written to the file paths yet.
  if (statusKind(status) === 'Pending') return null;

  return (
    <div className="flex gap-2">
      {errFile && (
        <button
          onClick={(e) => { e.stopPropagation(); onView(`${labelPrefix}.err`, errFile); }}
          className="text-red-400 hover:text-red-300 transition-colors"
          title="View .err file"
        >
          <FileText className="w-4 h-4" />
        </button>
      )}
      {outFile && (
        <button
          onClick={(e) => { e.stopPropagation(); onView(`${labelPrefix}.out`, outFile); }}
          className="text-green-400 hover:text-green-300 transition-colors"
          title="View .out file"
        >
          <FileText className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}
