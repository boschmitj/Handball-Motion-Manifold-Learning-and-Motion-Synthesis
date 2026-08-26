# Visualization examples

These three small scripts were used to inspect the data behind the penalty-processing pipeline. The TSV viewer checks the motion-capture skeleton and the fitted ball position frame by frame. The other scripts show a tracked penalty in 3D and compare the velocity and acceleration supplied by the fixture data with values recomputed from the positions.

The `data` folder contains one complete mocap throw (frames 2032–3019) and all 1,055 processed league penalties. The example folder can be copied and run on its own.

## Setup

Run these commands from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install matplotlib numpy pandas
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Run

```bash
python tsv_frame_viewer.py
python plot_penalty_3d.py --id 8008206
python plot_fixture_vs_recomputed_timeseries.py --id 8008206
```

The viewer opens an interactive window. Use the slider or the left and right arrow keys to change frames. The checkbox switches between the ball centre fitted from its markers and the ground-truth centre.

The plotting scripts write their PNG files to `plots_3d` and `timeseries`. Leave out `--id` to process every league penalty. This creates more than a thousand 3D plots and twice as many time-series plots, so it takes a while.

Add `--show` to `plot_penalty_3d.py` if you want to rotate a single 3D view interactively.
