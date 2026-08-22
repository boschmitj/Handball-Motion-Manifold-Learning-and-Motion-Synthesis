import { useState } from 'react';
import { FileText, StopCircle, Eye } from 'lucide-react';

interface ProcessTask {
  id: string;
  taskId: string;
  captureName: string;
  seqNames: string[];
  processName: string;
  status: 'Running' | 'Completed' | 'Failed' | 'Pending' | 'Cancelled';
  jobId: string;
  createdAt: string;
  files: {
    log?: string;
    err?: string;
    out?: string;
    sub?: string;
    sh?: string;
  };
}

export function ProgressOfTasks() {
  const [selectedFile, setSelectedFile] = useState<{ name: string; content: string } | null>(null);
  const [confirmStop, setConfirmStop] = useState<string | null>(null);

  // Mock data for demonstration
  const tasks: ProcessTask[] = [
    {
      id: '1',
      taskId: 'task_001',
      captureName: 'capture_001',
      seqNames: ['seq_001'],
      processName: 'Create .npz',
      status: 'Completed',
      jobId: 'job_12345',
      createdAt: '2025-11-27 10:30:15',
      files: {
        log: 'Process started at 10:30:15\nLoading data...\nCreating .npz file...\nProcess completed successfully.',
        err: '',
        out: 'NPZ file created: /output/capture_001.npz\nTotal entries: 1024',
        sub: '#!/bin/bash\n#SBATCH --job-name=create_npz\n#SBATCH --output=output.log\n#SBATCH --time=01:00:00\npython create_npz.py',
        sh: '#!/bin/bash\nexport CUDA_VISIBLE_DEVICES=0\npython scripts/create_npz.py --input data/ --output output/'
      }
    },
    {
      id: '2',
      taskId: 'task_002',
      captureName: 'capture_001',
      seqNames: ['seq_001'],
      processName: 'Run Masks',
      status: 'Running',
      jobId: 'job_12346',
      createdAt: '2025-11-27 10:35:20',
      files: {
        log: 'Process started at 10:35:20\nLoading model...\nProcessing frame 1/500...\nProcessing frame 50/500...\nProcessing frame 100/500...',
        err: '',
        out: 'Current progress: 20%\nEstimated time remaining: 15 minutes',
        sub: '#!/bin/bash\n#SBATCH --job-name=run_masks\n#SBATCH --output=masks.log\n#SBATCH --time=02:00:00\npython run_masks.py',
        sh: '#!/bin/bash\nexport CUDA_VISIBLE_DEVICES=1\npython scripts/run_masks.py --input data/ --output masks/'
      }
    },
    {
      id: '3',
      taskId: 'task_003',
      captureName: 'capture_002',
      seqNames: ['seq_002'],
      processName: 'Run MammaNet',
      status: 'Failed',
      jobId: 'job_12347',
      createdAt: '2025-11-27 10:40:10',
      files: {
        log: 'Process started at 10:40:10\nLoading model...\nError: CUDA out of memory',
        err: 'RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 10.76 GiB total capacity)',
        out: '',
        sub: '#!/bin/bash\n#SBATCH --job-name=mammanet\n#SBATCH --output=mammanet.log\n#SBATCH --time=03:00:00\npython run_mammanet.py',
        sh: '#!/bin/bash\nexport CUDA_VISIBLE_DEVICES=0\npython scripts/run_mammanet.py --config configs/default.yaml'
      }
    },
    {
      id: '4',
      taskId: 'task_004',
      captureName: 'capture_002',
      seqNames: ['seq_002'],
      processName: 'Run Opt',
      status: 'Pending',
      jobId: 'job_12348',
      createdAt: '2025-11-27 10:45:30',
      files: {
        log: '',
        err: '',
        out: '',
        sub: '#!/bin/bash\n#SBATCH --job-name=run_opt\n#SBATCH --output=opt.log\n#SBATCH --time=04:00:00\npython run_opt.py',
        sh: '#!/bin/bash\nexport CUDA_VISIBLE_DEVICES=2\npython scripts/run_opt.py --iterations 1000'
      }
    },
    {
      id: '5',
      taskId: 'task_005',
      captureName: 'capture_003',
      seqNames: ['seq_003'],
      processName: 'Rerun Vis',
      status: 'Running',
      jobId: 'job_12349',
      createdAt: '2025-11-27 11:00:45',
      files: {
        log: 'Process started at 11:00:45\nGenerating visualizations...\nRendering frame 1/200...',
        err: '',
        out: 'Rendering progress: 10%',
        sub: '#!/bin/bash\n#SBATCH --job-name=rerun_vis\n#SBATCH --output=vis.log\n#SBATCH --time=01:30:00\npython rerun_vis.py',
        sh: '#!/bin/bash\nexport DISPLAY=:0\npython scripts/rerun_vis.py --input results/ --output visualizations/'
      }
    }
  ];

  const handleViewFile = (fileName: string, content: string) => {
    setSelectedFile({ name: fileName, content });
  };

  const handleStopProcess = (taskId: string) => {
    setConfirmStop(taskId);
  };

  const confirmStopProcess = (taskId: string) => {
    console.log('Stopping process:', taskId);
    // Handle stop process logic here
    setConfirmStop(null);
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
      <div className="max-w-[1400px] mx-auto">
        <h2 className="text-3xl text-white mb-2">Progress of Tasks</h2>
        <p className="text-gray-400 mb-8">Recent task submissions and currently running processes</p>
        
        {/* Tasks Table */}
        <div className="bg-[#0d1117] border border-gray-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-[#1a2332] border-b border-gray-800">
                <tr>
                  <th className="px-6 py-4 text-left text-gray-300">Task ID</th>
                  <th className="px-6 py-4 text-left text-gray-300">Capture Name</th>
                  <th className="px-6 py-4 text-left text-gray-300">Seq Names</th>
                  <th className="px-6 py-4 text-left text-gray-300">Process</th>
                  <th className="px-6 py-4 text-left text-gray-300">Status</th>
                  <th className="px-6 py-4 text-left text-gray-300">Job ID</th>
                  <th className="px-6 py-4 text-left text-gray-300">Created At</th>
                  <th className="px-6 py-4 text-left text-gray-300">Log Files</th>
                  <th className="px-6 py-4 text-left text-gray-300">Submission Files</th>
                  <th className="px-6 py-4 text-left text-gray-300">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {tasks.map((task) => (
                  <tr key={task.id} className="hover:bg-[#1a2332]/50 transition-colors">
                    <td className="px-6 py-4 text-gray-400 font-mono text-sm">{task.taskId}</td>
                    <td className="px-6 py-4 text-gray-200">{task.captureName}</td>
                    <td className="px-6 py-4">
                      <div className="flex flex-wrap gap-1">
                        {task.seqNames.map((seq) => (
                          <span key={seq} className="px-2 py-0.5 bg-[#1a2332] text-gray-300 text-xs rounded border border-gray-700">
                            {seq}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-gray-200">{task.processName}</td>
                    <td className="px-6 py-4">
                      <span className={`px-3 py-1 rounded-full text-xs ${getStatusBadgeColor(task.status)}`}>
                        {task.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-400 font-mono text-sm">{task.jobId}</td>
                    <td className="px-6 py-4 text-gray-400 text-sm">{task.createdAt}</td>
                    <td className="px-6 py-4">
                      <div className="flex gap-2">
                        {task.files.log && (
                          <button
                            onClick={() => handleViewFile(`${task.processName}.log`, task.files.log!)}
                            className="text-blue-400 hover:text-blue-300 transition-colors"
                            title="View .log file"
                          >
                            <FileText className="w-4 h-4" />
                          </button>
                        )}
                        {task.files.err && (
                          <button
                            onClick={() => handleViewFile(`${task.processName}.err`, task.files.err!)}
                            className="text-red-400 hover:text-red-300 transition-colors"
                            title="View .err file"
                          >
                            <FileText className="w-4 h-4" />
                          </button>
                        )}
                        {task.files.out && (
                          <button
                            onClick={() => handleViewFile(`${task.processName}.out`, task.files.out!)}
                            className="text-green-400 hover:text-green-300 transition-colors"
                            title="View .out file"
                          >
                            <FileText className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex gap-2">
                        {task.files.sub && (
                          <button
                            onClick={() => handleViewFile(`${task.processName}.sub`, task.files.sub!)}
                            className="text-purple-400 hover:text-purple-300 transition-colors"
                            title="View .sub file"
                          >
                            <FileText className="w-4 h-4" />
                          </button>
                        )}
                        {task.files.sh && (
                          <button
                            onClick={() => handleViewFile(`${task.processName}.sh`, task.files.sh!)}
                            className="text-orange-400 hover:text-orange-300 transition-colors"
                            title="View .sh file"
                          >
                            <FileText className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      {(task.status === 'Running' || task.status === 'Pending') && (
                        <button
                          onClick={() => handleStopProcess(task.id)}
                          className="text-red-400 hover:text-red-300 transition-colors flex items-center gap-1"
                        >
                          <StopCircle className="w-4 h-4" />
                          <span className="text-sm">Stop</span>
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* File Viewer Dialog */}
        {selectedFile && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-[#0d1117] border border-gray-800 rounded-lg max-w-4xl w-full max-h-[80vh] flex flex-col">
              <div className="flex items-center justify-between p-4 border-b border-gray-800">
                <h3 className="text-white flex items-center gap-2">
                  <Eye className="w-5 h-5" />
                  {selectedFile.name}
                </h3>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="text-gray-400 hover:text-gray-200 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="p-4 overflow-auto flex-1">
                <pre className="text-gray-300 text-sm font-mono whitespace-pre-wrap bg-[#1a2332] p-4 rounded">
                  {selectedFile.content || 'File is empty'}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* Stop Confirmation Dialog */}
        {confirmStop && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-[#0d1117] border border-gray-800 rounded-lg max-w-md w-full p-6">
              <h3 className="text-white mb-4">Confirm Stop Process</h3>
              <p className="text-gray-400 mb-6">
                Are you sure you want to stop this process? This action cannot be undone.
              </p>
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setConfirmStop(null)}
                  className="px-4 py-2 bg-[#1a2332] border border-gray-700 text-gray-300 rounded hover:bg-[#253044] hover:border-gray-600 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={() => confirmStopProcess(confirmStop)}
                  className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-500 transition-colors"
                >
                  OK
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}