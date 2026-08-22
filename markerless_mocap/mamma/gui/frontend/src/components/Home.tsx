import { useState } from 'react';
import { ArrowRight, Globe, FileText, Database } from 'lucide-react';
import { QuickstartWizard } from './QuickstartWizard';
import { ExampleDataPanel } from './ExampleDataPanel';
import { DataReadinessPanel } from './DataReadinessPanel';
import { DatasetLibraryPanel } from './DatasetLibraryPanel';
import { SignInCenter } from './SignInCenter';

interface Props {
  /** "Submit a task" CTA — opens the New Task form. */
  onSubmitTask: () => void;
  /** Fired after the QuickstartWizard successfully creates a task —
   *  navigate to the Tasks list so the user can watch the run.
   *  Distinct from onSubmitTask, which lands on the empty form. */
  onTaskSubmitted: () => void;
  onBrowseCaptures: () => void;
}

const DATASET_URL = 'https://mamma.is.tue.mpg.de/download.php';

/**
 * Landing page for the MAMMA webtool. Positioned as a thin wrapper around
 * the published method (CVPR '26 oral) — points users at the paper, the
 * project page, and the dataset for the science, and explains how this
 * web UI is organised for day-to-day operators.
 *
 * Authoritative sources for copy on this page:
 *   - https://mamma.is.tue.mpg.de/                  (project page)
 *   - https://arxiv.org/abs/2506.13040              (arXiv abstract)
 *   - https://arxiv.org/pdf/2506.13040              (paper PDF)
 */
export function Home({ onSubmitTask, onTaskSubmitted, onBrowseCaptures }: Props) {
  const [showWizard, setShowWizard] = useState(false);
  return (
    <div className="w-full max-w-5xl mx-auto px-6 py-10 space-y-12">
      {/* Hero */}
      <section className="space-y-4">
        <div className="text-foreground-subtle text-xs uppercase tracking-[0.2em] flex items-center gap-2">
          <span>CVPR 2026 · Oral</span>
        </div>
        <h1 className="text-4xl sm:text-5xl text-foreground tracking-tight font-medium leading-tight">
          MAMMA
        </h1>
        <div className="text-foreground-muted text-lg">
          Markerless Accurate Multi-person Motion Acquisition
        </div>
        <div className="text-foreground-subtle text-sm pt-1">
          Hanz Cuevas-Velasquez*, Anastasios Yiannakidis*, Soyong Shin, Giorgio Becherini, Markus Höschle,
          Joachim Tesch, Taylor Obersat, Tsvetelina Alexiadis, Eni Halilaj, Michael J. Black ·
          MPI for Intelligent Systems &amp; Carnegie Mellon University · *equal contribution
        </div>
        <div className="flex flex-wrap items-center gap-3 pt-3">
          <button
            onClick={onSubmitTask}
            className="mamma-cta mamma-cta-primary group inline-flex items-center gap-2 px-4 py-2.5 bg-primary text-primary-foreground rounded-md text-sm font-medium shadow-sm shadow-black/30"
          >
            Submit a task
            <ArrowRight className="w-4 h-4 transition-transform duration-200 group-hover:translate-x-1" />
          </button>
          <button
            onClick={onBrowseCaptures}
            className="mamma-cta mamma-cta-secondary inline-flex items-center gap-2 px-4 py-2.5 bg-surface-2 border border-border text-foreground rounded-md text-sm"
          >
            Browse captures
          </button>
        </div>
      </section>

      {/* Quickstart — data-aware panel: shows a "Download example data"
          button when the demo sequence isn't on disk yet, streams the
          script's progress while it runs, and surfaces the "Run demo"
          CTA prominently once the data lands. The wizard still handles
          the missing-data case itself if the user clicks Run demo early. */}
      <section>
        <ExampleDataPanel onRunDemo={() => setShowWizard(true)} />
      </section>

      {/* Session sign-in surface — one card, two sub-cards (MAMMA + SMPL-X).
          Credentials are held in CredentialsContext (browser memory only,
          wiped on refresh). Both download panels below skip their inline
          sign-in prompt when the matching domain is signed in here. */}
      <section>
        <SignInCenter />
      </section>

      {/* Data readiness — compact systems-readout for the pipeline's
          installation assets (model weights + body models). One-click
          downloads for public files, an inline credential form for
          MPI-account-gated files (creds never persisted — see
          gui/backend/data_readiness.py for the contract), and short
          step-lists for files that have to be obtained by hand. */}
      <section className="space-y-4">
        <div>
          <h2 className="text-foreground text-xl font-medium tracking-tight">Pipeline assets</h2>
          <p className="text-foreground-muted text-sm mt-2 leading-relaxed">
            Assets that MAMMA needs at runtime. Some are open downloads, others require a free research account.
          </p>
        </div>
        <DataReadinessPanel />
      </section>

      {/* MAMMA dataset library — a per-family form that wraps the
          data/download_mamma_*.sh scripts behind a single MAMMA sign-in.
          Same credential contract as the readiness panel above. */}
      <section className="space-y-4">
        <div>
          <h2 className="text-foreground text-xl font-medium tracking-tight">MAMMA datasets</h2>
          <p className="text-foreground-muted text-sm mt-2 leading-relaxed">
            The multi-view captures released with the paper. Sign in with your MAMMA account and pick
            only the cameras, sequences, and video variants you need.
          </p>
        </div>
        <DatasetLibraryPanel />
      </section>

      {/* Resources */}
      <section className="space-y-4">
        <div>
          <h2 className="text-foreground text-xl font-medium tracking-tight">Resources</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <ResourceLink
            icon={<Globe className="w-4 h-4 text-primary" />}
            title="Project page"
            desc="The official MAMMA landing page."
            href="https://mamma.is.tue.mpg.de/"
          />
          <ResourceLink
            icon={<FileText className="w-4 h-4 text-primary" />}
            title="Paper (PDF)"
            desc="Read the paper (CVPR 2026 oral)"
            href="https://arxiv.org/pdf/2506.13040"
          />
          <ResourceLink
            icon={<Database className="w-4 h-4 text-primary" />}
            title="Dataset Page"
            desc="Browse all datasets released with the paper."
            href="https://mamma.is.tue.mpg.de/download.php"
          />
        </div>
      </section>

      {/* License — non-commercial research notice. Full terms link to the
          MPI MAMMA license page so users can read them in full. */}
      <section className="pt-6 border-t border-border-subtle">
        <h2 className="text-foreground text-xl font-medium tracking-tight">License</h2>
        <p className="text-foreground-muted text-sm mt-2 leading-relaxed">
          Copyright license for non-commercial scientific research purposes. See the{' '}
          <a
            href="https://mamma.is.tue.mpg.de/license.html"
            target="_blank"
            rel="noreferrer"
            className="text-primary hover:underline"
          >
            full license terms
          </a>
          {' '}for details.
        </p>
      </section>

      <QuickstartWizard
        open={showWizard}
        onClose={() => setShowWizard(false)}
        onSubmitted={() => {
          setShowWizard(false);
          onTaskSubmitted();
        }}
      />
    </div>
  );
}

function ResourceLink({
  icon, title, desc, href, secondary,
}: {
  icon: React.ReactNode;
  title: string;
  desc: string;
  href: string;
  secondary?: { label: string; href: string };
}) {
  return (
    <div className="bg-surface-1 border border-border-subtle rounded-lg p-4 transition-colors hover:bg-surface-2 hover:border-border">
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className="block group"
      >
        <div className="flex items-center gap-2 mb-1">
          {icon}
          <div className="text-foreground text-sm font-medium group-hover:underline">{title}</div>
        </div>
        <div className="text-foreground-muted text-xs leading-relaxed">{desc}</div>
      </a>
      {secondary && (
        <a
          href={secondary.href}
          target="_blank"
          rel="noreferrer"
          className="inline-block mt-2 text-[11px] text-primary hover:underline"
        >
          {secondary.label} ↗
        </a>
      )}
    </div>
  );
}
