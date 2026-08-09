# Simple Point-of-Release detector

Run the detector with:

```bash
cd PenaltyProcessing
python src/simple_release_detector.py --penalties penalties.csv --positions-dir games_position_files --output out/simple_penalty_trajectories.csv --penalty-id 8008206
```

Optional flags:
- `--include-unsuccessful` to process unsuccessful throws too.
- `--penalty-id <id>` to process a single penalty row.

The generated CSV contains:
- `trajectory_json`: compact JSON array of trajectory points.
- `release_point_json`: compact JSON object with the detected release point.
- `release_speed`, `release_accel`, `release_direction`, `release_idx`, `trajectory_point_count`.

The output is plot-friendly for the existing 3D plotter in `visualization/plot_penalty_3d.py`.
