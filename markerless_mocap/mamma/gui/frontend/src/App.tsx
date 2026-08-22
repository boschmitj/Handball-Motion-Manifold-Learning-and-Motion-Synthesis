import image_dde8493fc3c936ece0e21797e939c09c940edb80 from 'figma:asset/dde8493fc3c936ece0e21797e939c09c940edb80.png';
import { useState } from 'react';
import { ImageWithFallback } from './components/figma/ImageWithFallback';
import { Home } from './components/Home';
import { CapturesList } from './components/CapturesList';
import { ResultsList } from './components/ResultsList';
import { CaptureDetail } from './components/CaptureDetail';
import { CaptureManage } from './components/CaptureManage';
import { Tasks } from './components/Tasks';
import { Docs } from './components/Docs';
import { Toaster } from './components/ui/sonner';
import { CredentialsProvider } from './components/CredentialsContext';

/**
 * Top-level navigation:
 *   Home — landing/documentation page
 *   Captures — manage capture.json files (list, add, full edit, delete) → opens CaptureManage
 *   Tasks — submit + monitor pipeline runs (the matrix lives here);
 *           the Import-previous-run modal also lives here (replaces
 *           the old standalone Database tab)
 *   Results — browse outputs of past runs → opens CaptureDetail
 *   Docs — distilled README reference (static)
 *
 * Both detail screens are reachable from a list-tab click. `detailOrigin`
 * records which tab opened the detail so "Back" returns there.
 */
type Tab = 'Home' | 'Captures' | 'Tasks' | 'Results' | 'Docs';
/** A "capture detail" view appears under both Captures (manage) and Results
 *  (browse outputs) — distinct components, single shared origin tracker. */
type View = Tab | 'CaptureManage' | 'CaptureResults';

export default function App() {
  const [view, setView] = useState<View>('Home');
  const [detailOrigin, setDetailOrigin] = useState<Tab>('Results');
  const [selectedCapture, setSelectedCapture] = useState<{ name: string; jsonPath: string } | null>(null);
  /** Deep-link state for cross-tab Tasks→Results navigation. Cleared on
   *  any tab change so it doesn't leak into a fresh capture-results visit. */
  const [outputsInitial, setOutputsInitial] = useState<{ taskId?: string; sequence?: string; process?: string } | null>(null);
  /** Which sub-view Tasks should open in on its next mount. Home's
   *  "Submit a task" CTA sets this to 'submit' so the user lands on
   *  the new-task form directly; every other path (nav bar, completion
   *  redirect, etc.) defaults back to 'list'. */
  const [tasksInitialSubView, setTasksInitialSubView] = useState<'list' | 'submit'>('list');

  const goToTab = (tab: Tab, opts?: { tasksSubView?: 'list' | 'submit' }) => {
    setView(tab);
    setSelectedCapture(null);
    setOutputsInitial(null);
    setTasksInitialSubView(opts?.tasksSubView ?? 'list');
  };

  const openCaptureForManage = (name: string, jsonPath: string) => {
    setSelectedCapture({ name, jsonPath });
    setDetailOrigin('Captures');
    setOutputsInitial(null);
    setView('CaptureManage');
  };
  const openCaptureForResults = (name: string, _jsonPath: string) => {
    setSelectedCapture({ name, jsonPath: _jsonPath });
    setDetailOrigin('Results');
    setOutputsInitial(null);
    setView('CaptureResults');
  };

  /** Fired from a Tasks-tab cell's "Browse outputs" — jumps to the Results
   *  detail page with the explorer pre-scoped to the cell's coordinates. */
  const browseOutputsFromCell = (
    captureName: string,
    captureJsonPath: string,
    taskId: string,
    seqName: string,
    stepName: string,
  ) => {
    setSelectedCapture({ name: captureName, jsonPath: captureJsonPath });
    setDetailOrigin('Tasks');
    setOutputsInitial({ taskId, sequence: seqName, process: stepName });
    setView('CaptureResults');
  };

  // The nav highlights the originating tab while a detail page is open so
  // the user keeps a sense of where they are in the IA.
  const navActive: Tab = (view === 'CaptureManage' || view === 'CaptureResults')
    ? detailOrigin
    : (view as Tab);

  return (
    <CredentialsProvider>
    <div className="min-h-screen bg-background text-foreground">
      <nav className="sticky top-0 z-30 border-b border-border-subtle bg-background/85 backdrop-blur-md backdrop-saturate-150 shadow-[0_1px_0_oklch(1_0_0/0.04)]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center gap-2 h-14">
            <ImageWithFallback
              src={image_dde8493fc3c936ece0e21797e939c09c940edb80}
              alt="MAMMA"
              className="h-7 w-auto mr-3 cursor-pointer"
              onClick={() => goToTab('Home')}
            />
            <NavButton active={navActive === 'Home'} onClick={() => goToTab('Home')}>Home</NavButton>
            <NavButton active={navActive === 'Captures'} onClick={() => goToTab('Captures')}>Captures</NavButton>
            <NavButton active={navActive === 'Tasks'} onClick={() => goToTab('Tasks')}>Tasks</NavButton>
            <NavButton active={navActive === 'Results'} onClick={() => goToTab('Results')}>Results</NavButton>
            <NavButton active={navActive === 'Docs'} onClick={() => goToTab('Docs')}>Docs</NavButton>
          </div>
        </div>
      </nav>

      <main>
        {view === 'Home' && (
          <Home
            onSubmitTask={() => goToTab('Tasks', { tasksSubView: 'submit' })}
            onTaskSubmitted={() => goToTab('Tasks')}
            onBrowseCaptures={() => goToTab('Captures')}
          />
        )}
        {view === 'Captures' && (
          <CapturesList onOpen={openCaptureForManage} />
        )}
        {view === 'Tasks' && (
          <Tasks
            onBrowseOutputs={browseOutputsFromCell}
            initialSubView={tasksInitialSubView}
          />
        )}
        {view === 'Docs' && <Docs />}
        {view === 'Results' && (
          <ResultsList onOpen={openCaptureForResults} />
        )}
        {view === 'CaptureManage' && selectedCapture && (
          <CaptureManage
            captureName={selectedCapture.name}
            jsonPath={selectedCapture.jsonPath}
            onBack={() => goToTab('Captures')}
            onDeleted={() => goToTab('Captures')}
            onViewTasks={() => goToTab('Tasks')}
            onViewResults={() => openCaptureForResults(selectedCapture.name, selectedCapture.jsonPath)}
          />
        )}
        {view === 'CaptureResults' && selectedCapture && (
          <CaptureDetail
            captureName={selectedCapture.name}
            onBack={() => goToTab(detailOrigin)}
            initial={outputsInitial ?? undefined}
          />
        )}
      </main>

      <Toaster />
    </div>
    </CredentialsProvider>
  );
}

function NavButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`relative px-3 py-1.5 rounded-md text-sm transition-colors ${
        active
          ? 'text-foreground bg-surface-2 ring-1 ring-inset ring-border'
          : 'text-foreground-muted hover:text-foreground hover:bg-surface-2/60'
      }`}
    >
      {children}
    </button>
  );
}
