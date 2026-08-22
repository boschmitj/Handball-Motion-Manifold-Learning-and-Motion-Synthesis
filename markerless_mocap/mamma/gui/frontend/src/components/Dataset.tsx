import { useState } from 'react';
import { ChevronDown, ChevronRight, Eye } from 'lucide-react';

interface Process {
  processName: string;
  processCode: string;
  status: 'Running' | 'Completed' | 'Failed' | 'Pending' | 'Cancelled';
}

interface Task {
  taskId: string;
  seqNames: string[];
  processes: Process[];
  createdAt: string;
}

interface Capture {
  captureName: string;
  tasks: Task[];
}

interface DatasetProps {
  onViewCapture: (captureName: string) => void;
}

export function Dataset({ onViewCapture }: DatasetProps) {
  const [expandedCaptures, setExpandedCaptures] = useState<Set<string>>(new Set());
  const [expandedTasks, setExpandedTasks] = useState<Set<string>>(new Set());

  // Mock data for captures
  const captures: Capture[] = [
    {
      captureName: 'capture_main_hall',
      tasks: [
        {
          taskId: 'task_00021',
          seqNames: ['seq_001', 'seq_002', 'seq_003'],
          createdAt: '2025-11-27 09:15:00',
          processes: [
            { processName: 'Create .npz', processCode: 'ma_cap', status: 'Completed' },
            { processName: 'Run Masks', processCode: 'ma_masks', status: 'Completed' },
            { processName: 'Run MammaNet', processCode: 'ma_2d', status: 'Completed' },
          ]
        }
      ]
    },
    {
      captureName: 'capture_dance_studio',
      tasks: [
        {
          taskId: 'task_00020',
          seqNames: ['seq_001'],
          createdAt: '2025-11-26 14:30:00',
          processes: [
            { processName: 'Create .npz', processCode: 'ma_cap', status: 'Completed' },
            { processName: 'Run Masks', processCode: 'ma_masks', status: 'Failed' },
          ]
        },
        {
          taskId: 'task_00022',
          seqNames: ['seq_001', 'seq_002'],
          createdAt: '2025-11-27 15:00:00',
          processes: [
            { processName: 'Create .npz', processCode: 'ma_cap', status: 'Completed' },
            { processName: 'Run Masks', processCode: 'ma_masks', status: 'Running' },
          ]
        }
      ]
    },
    {
      captureName: 'capture_outdoor',
      tasks: [
        {
          taskId: 'task_00019',
          seqNames: ['seq_001', 'seq_002'],
          createdAt: '2025-11-25 16:20:00',
          processes: [
            { processName: 'Create .npz', processCode: 'ma_cap', status: 'Completed' },
            { processName: 'Run Masks', processCode: 'ma_masks', status: 'Completed' },
            { processName: 'Run MammaNet', processCode: 'ma_2d', status: 'Completed' },
            { processName: 'Run Opt', processCode: 'ma_opt', status: 'Completed' },
          ]
        }
      ]
    },
    {
      captureName: 'capture_test',
      tasks: [
        {
          taskId: 'task_00018',
          seqNames: ['seq_001'],
          createdAt: '2025-11-24 11:00:00',
          processes: [
            { processName: 'Create .npz', processCode: 'ma_cap', status: 'Completed' },
            { processName: 'Rerun Vis', processCode: 'ma_vis', status: 'Completed' },
          ]
        }
      ]
    },
  ];

  const toggleCaptureExpansion = (captureName: string) => {
    const newExpanded = new Set(expandedCaptures);
    if (newExpanded.has(captureName)) {
      newExpanded.delete(captureName);
    } else {
      newExpanded.add(captureName);
    }
    setExpandedCaptures(newExpanded);
  };

  const toggleTaskExpansion = (taskId: string) => {
    const newExpanded = new Set(expandedTasks);
    if (newExpanded.has(taskId)) {
      newExpanded.delete(taskId);
    } else {
      newExpanded.add(taskId);
    }
    setExpandedTasks(newExpanded);
  };

  const getStatusBadgeColor = (status: string) => {
    switch (status) {
      case 'Running':
        return 'bg-blue-600/20 text-blue-400 border border-blue-500/30';
      case 'Completed':
        return 'bg-green-600/20 text-green-400 border border-green-500/30';
      case 'Failed':
      case 'Cancelled':
        return 'bg-red-600/20 text-red-400 border border-red-500/30';
      case 'Pending':
        return 'bg-yellow-600/20 text-yellow-400 border border-yellow-500/30';
      default:
        return 'bg-gray-600/20 text-gray-400 border border-gray-500/30';
    }
  };

  return (
    <div className="p-8">
      <div className="max-w-[1600px] mx-auto">
        <h2 className="text-3xl text-white mb-2">Dataset</h2>
        <p className="text-gray-400 mb-8">Browse captures and their processing tasks</p>

        {/* Captures List */}
        <div className="space-y-4">
          {captures.map((capture) => {
            const isCaptureExpanded = expandedCaptures.has(capture.captureName);
            return (
              <div key={capture.captureName} className="bg-[#0d1117] border border-gray-800 rounded-lg overflow-hidden">
                {/* Capture Header */}
                <div className="flex items-center justify-between px-6 py-4 hover:bg-[#1a2332]/50 transition-colors">
                  <button
                    onClick={() => toggleCaptureExpansion(capture.captureName)}
                    className="flex items-center gap-4 flex-1"
                  >
                    {isCaptureExpanded ? (
                      <ChevronDown className="w-5 h-5 text-gray-400" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-gray-400" />
                    )}
                    <div className="flex items-center gap-6">
                      <div className="text-left">
                        <div className="text-white">{capture.captureName}</div>
                        <div className="text-gray-400 text-sm mt-1">
                          {capture.tasks.length} {capture.tasks.length === 1 ? 'task' : 'tasks'}
                        </div>
                      </div>
                    </div>
                  </button>
                  <button
                    onClick={() => onViewCapture(capture.captureName)}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-500 transition-colors"
                  >
                    <Eye className="w-4 h-4" />
                    View Details
                  </button>
                </div>

                {/* Tasks List (Expanded) */}
                {isCaptureExpanded && (
                  <div className="border-t border-gray-800 bg-[#0a0e1a]">
                    <div className="p-4 space-y-3">
                      {capture.tasks.map((task) => {
                        const isTaskExpanded = expandedTasks.has(task.taskId);
                        return (
                          <div key={task.taskId} className="bg-[#0d1117] border border-gray-700 rounded-lg overflow-hidden">
                            {/* Task Header */}
                            <button
                              onClick={() => toggleTaskExpansion(task.taskId)}
                              className="w-full px-4 py-3 flex items-center justify-between hover:bg-[#1a2332]/50 transition-colors"
                            >
                              <div className="flex items-center gap-3">
                                {isTaskExpanded ? (
                                  <ChevronDown className="w-4 h-4 text-gray-400" />
                                ) : (
                                  <ChevronRight className="w-4 h-4 text-gray-400" />
                                )}
                                <div className="flex items-center gap-4">
                                  <span className="text-white font-mono text-sm">{task.taskId}</span>
                                  <div className="flex flex-wrap gap-1">
                                    {task.seqNames.map((seq) => (
                                      <span key={seq} className="px-2 py-0.5 bg-[#1a2332] text-gray-300 text-xs rounded border border-gray-700">
                                        {seq}
                                      </span>
                                    ))}
                                  </div>
                                  <span className="text-gray-400 text-sm">{task.createdAt}</span>
                                </div>
                              </div>
                              <div className="text-gray-400 text-sm">
                                {task.processes.length} {task.processes.length === 1 ? 'process' : 'processes'}
                              </div>
                            </button>

                            {/* Processes List (Expanded) */}
                            {isTaskExpanded && (
                              <div className="border-t border-gray-700 px-4 py-3 bg-[#0a0e1a]">
                                <div className="space-y-2">
                                  {task.processes.map((process, idx) => (
                                    <div key={idx} className="flex items-center justify-between py-2 px-3 bg-[#0d1117] rounded border border-gray-800">
                                      <div>
                                        <div className="text-gray-200 text-sm">{process.processName}</div>
                                        <div className="text-gray-500 text-xs font-mono mt-0.5">[{process.processCode}]</div>
                                      </div>
                                      <span className={`px-3 py-1 rounded-full text-xs ${getStatusBadgeColor(process.status)}`}>
                                        {process.status}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
