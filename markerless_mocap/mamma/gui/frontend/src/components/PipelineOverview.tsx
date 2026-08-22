import { FolderInput, ScissorsLineDashed, Crosshair, Move3d, PlayCircle } from 'lucide-react';

interface Props {
  /** Reduce icon + padding scale for placements above dense tables
   *  (e.g. above the Tasks page's ProcessTable). Default false is
   *  the original Home-page scale. */
  compact?: boolean;
}

/**
 * Five-card overview of the MAMMA pipeline: ma_cap → ma_masks → ma_2d
 * → ma_3d → ma_vis. No props, no state — pure visual reference. Lives
 * at the top of the Tasks page so users see "what the pipeline does"
 * at the moment they're about to start a run.
 */
export function PipelineOverview({ compact = false }: Props) {
  return (
    <section className={compact ? 'space-y-2' : 'space-y-4'}>
      <div>
        <h2 className={`text-foreground font-medium tracking-tight ${compact ? 'text-base' : 'text-xl'}`}>
          Pipeline
        </h2>
        <p className={`text-foreground-muted mt-1 ${compact ? 'text-xs' : 'text-sm'}`}>
          Five sequential stages. Each step consumes the previous step's outputs and writes its
          own under{' '}
          <code className="font-mono text-foreground bg-surface-2 px-1 rounded">
            {`<output>/<step>/<output_id>/<dataset>/<sequence>/`}
          </code>
          .
        </p>
      </div>
      <div className={`bg-surface-1 border border-border-subtle rounded-xl shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02] ${compact ? 'p-3' : 'p-5'}`}>
        <div className={`grid grid-cols-1 md:grid-cols-5 ${compact ? 'gap-2' : 'gap-3'}`}>
          <PipelineStep compact={compact}
            num="1"
            icon={<FolderInput className={compact ? 'w-4 h-4' : 'w-5 h-5'} />}
            label="Input"
            ident="ma_cap"
            desc="Read multi-view frames and camera calibration; package per-sequence inputs."
          />
          <PipelineStep compact={compact}
            num="2"
            icon={<ScissorsLineDashed className={compact ? 'w-4 h-4' : 'w-5 h-5'} />}
            label="Segmentation Masks"
            ident="ma_masks"
            desc="Per-frame, per-view person segmentation, used to condition the landmark stage."
          />
          <PipelineStep compact={compact}
            num="3"
            icon={<Crosshair className={compact ? 'w-4 h-4' : 'w-5 h-5'} />}
            label="2D Landmarks"
            ident="ma_2d"
            desc="Dense 2D contact-aware & visibility-aware surface landmarks predicted by MammaNet which has per-landmark learnable queries."
          />
          <PipelineStep compact={compact}
            num="4"
            icon={<Move3d className={compact ? 'w-4 h-4' : 'w-5 h-5'} />}
            label="3D Fitting"
            ident="ma_3d"
            desc="Cross-view matching and SMPL-X fitting (optimization) to the multi-view 2D landmarks."
          />
          <PipelineStep compact={compact}
            num="5"
            icon={<PlayCircle className={compact ? 'w-4 h-4' : 'w-5 h-5'} />}
            label="Visualization"
            ident="ma_vis"
            desc="Render preview videos and an interactive 3D scene for review of fitted meshes and landmarks."
          />
        </div>
      </div>
    </section>
  );
}

function PipelineStep({
  num, icon, label, ident, desc, compact,
}: {
  num: string;
  icon: React.ReactNode;
  label: string;
  ident: string;
  desc: string;
  compact: boolean;
}) {
  return (
    <div className={`bg-surface-2/50 border border-border-subtle rounded-lg flex flex-col gap-1.5 ${compact ? 'p-2.5' : 'p-3'}`}>
      <div className="flex items-center justify-between">
        <span className={`inline-flex items-center justify-center rounded-md bg-primary-muted border border-primary/30 text-primary font-mono tabular-nums ${compact ? 'w-5 h-5 text-[10px]' : 'w-6 h-6 text-xs'}`}>
          {num}
        </span>
        <span className="text-primary">{icon}</span>
      </div>
      <div>
        <div className={`text-foreground font-medium leading-tight ${compact ? 'text-xs' : 'text-sm'}`}>{label}</div>
        <div className={`text-foreground-faint font-mono mt-0.5 ${compact ? 'text-[9px]' : 'text-[10px]'}`}>{ident}</div>
      </div>
      <div className={`text-foreground-subtle leading-snug ${compact ? 'text-[11px]' : 'text-xs'}`}>{desc}</div>
    </div>
  );
}
