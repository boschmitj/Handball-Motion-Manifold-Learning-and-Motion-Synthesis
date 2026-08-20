# End-to-end throw matching and trajectory reconstruction

Run all commands from the `PenaltyProcessing` directory:

```bash
cd /home/josh/BA/Handball-Motion-Manifold-Learning-and-Motion-Synthesis/PenaltyProcessing
```

If necessary, activate the project environment and install its dependencies:

```bash
conda activate BA
python -m pip install -r requirements.txt
```

## 1. Create the raw throw representations and matching features

The League input must be the previously generated
`simple_penalty_trajectories.csv`. The following example processes the
`throw_ul` Mocap recording:

```bash
python src/create_throw_representation.py \
  --throw-type throw_ul \
  --league-penalties penalties.csv \
  --league-csv out/run_20260816_144549/simple_penalty_trajectories.csv
```

The default output directory is `out/throw_features/`. Important outputs are:

```text
out/throw_features/raw_mocap.csv
out/throw_features/raw_league.csv
out/throw_features/features_mocap.csv
out/throw_features/features_league.csv
out/throw_features/throw_index.csv
```

`raw_league.csv` contains the League continuation from the synthetic PoR
through the first goal-line crossing. Bounce and rebound samples are retained.

Whenever raw representation generation changes, regenerate the feature files,
kNN results, and reconstructions rather than mixing old and new outputs.

## 2. Run weighted kNN once

Show the available presets and options:

```bash
python src/model/weighted_knn.py --help
```

Run with the default weight preset:

```bash
python src/model/weighted_knn.py \
  --mocap out/throw_features/features_mocap.csv \
  --league out/throw_features/features_league.csv \
  --output out/weighted_knn_matches.csv \
  -k 5
```

Select another predefined set with, for example:

```bash
--weight-preset height_focus
```

## 3. Run a randomized weight search

For example, evaluate 200 reproducible random weight configurations:

```bash
python src/model/weighted_knn.py \
  --mocap out/throw_features/features_mocap.csv \
  --league out/throw_features/features_league.csv \
  --random-weight-runs 200 \
  --random-seed 42 \
  -k 5
```

Random search creates a uniquely named directory such as:

```text
out/weighted_knn_random_20260818_175106_778543/
```

It contains every run and the overall reports:

```text
run_0001/weighted_knn_matches.csv
run_0001/metrics_by_throw_type.csv
run_0001/weights.json
...
random_search_summary.csv
best_weights.json
best_weighted_knn_matches.csv
```

In `random_search_summary.csv`:

- `balanced_mean_top1_distance` is lower-is-better.
- `balanced_score` is higher-is-better.
- `release_speed`, `release_angles`, and the other feature-group columns are
  weights. A larger number means that group influenced the run more strongly.

## 4. Reconstruct trajectories

### Use ordinary kNN output

```bash
python src/trajectory_reconstruction.py \
  --matches out/weighted_knn_matches.csv \
  --raw-league out/throw_features/raw_league.csv \
  --raw-mocap out/throw_features/raw_mocap.csv \
  --output out/reconstructed_matched_trajectories.csv \
  --target-hz 300 \
  --rank 1
```

### Use the overall best random-search result

Replace `<SEARCH_DIR>` with the directory created in step 3:

```bash
python src/trajectory_reconstruction.py \
  --matches <SEARCH_DIR>/best_weighted_knn_matches.csv \
  --raw-league out/throw_features/raw_league.csv \
  --raw-mocap out/throw_features/raw_mocap.csv \
  --output <SEARCH_DIR>/best_weighted_trajectory_reconstruction.csv \
  --target-hz 300 \
  --rank 1
```

### Use a specific random-search run

```bash
python src/trajectory_reconstruction.py \
  --random-search-dir <SEARCH_DIR> \
  --random-run-id 68 \
  --raw-league out/throw_features/raw_league.csv \
  --raw-mocap out/throw_features/raw_mocap.csv \
  --output <SEARCH_DIR>/trajectory_reconstruction.csv \
  --target-hz 300 \
  --rank 1
```

The output name automatically receives the run number:

```text
<SEARCH_DIR>/trajectory_reconstruction_run_0068.csv
```

### Select a run by important weight groups

The following command first keeps the 20 runs with the smallest balanced mean
top-1 distance. From that shortlist, it selects the run where all requested
weight groups have the highest joint importance:

```bash
python src/trajectory_reconstruction.py \
  --random-search-dir <SEARCH_DIR> \
  --select-weight-groups \
    release_speed \
    release_angles \
    relative_trajectory \
    release_height \
  --top-runs 20 \
  --raw-league out/throw_features/raw_league.csv \
  --raw-mocap out/throw_features/raw_mocap.csv
```

For a single preferred group:

```bash
python src/trajectory_reconstruction.py \
  --random-search-dir <SEARCH_DIR> \
  --select-weight-groups release_height \
  --top-runs 50 \
  --raw-league out/throw_features/raw_league.csv \
  --raw-mocap out/throw_features/raw_mocap.csv
```

The selected run ID is printed and added to the output filename. Valid weight
group names are:

```text
release_speed
release_angles
release_height
relative_trajectory
velocity_evolution
trajectory_angles
acceleration
absolute_por
```

## 5. Plot a reconstructed match

The Mocap and League IDs must identify a row that actually exists in the
selected reconstruction CSV:

```bash
python visualization/plot_reconstructed_match.py \
  --reconstructed out/reconstructed_matched_trajectories_run_0045.csv \
  --raw-league out/throw_features/raw_league.csv \
  --raw-mocap out/throw_features/raw_mocap.csv \
  --mocap-throw-id throw_ul_seg1 \
  --league-throw-id 12039519
```

Save the figure instead of opening a window:

```bash
python visualization/plot_reconstructed_match.py \
  --reconstructed out/reconstructed_matched_trajectories_run_0045.csv \
  --raw-league out/throw_features/raw_league.csv \
  --raw-mocap out/throw_features/raw_mocap.csv \
  --mocap-throw-id throw_ul_seg1 \
  --league-throw-id 12039519 \
  --output out/reconstructed_match_run_0045.png
```

## 6. Inspect an original League penalty trajectory

This is useful for checking the complete source trajectory and bounce before
raw representation generation:

```bash
python visualization/plot_penalty_3d.py \
  --input out/run_20260816_144549/simple_penalty_trajectories.csv \
  --shot-id 9400611 \
  --show
```

## Useful help commands

```bash
python src/create_throw_representation.py --help
python src/model/weighted_knn.py --help
python src/trajectory_reconstruction.py --help
python visualization/plot_reconstructed_match.py --help
python visualization/plot_penalty_3d.py --help
```

## Common mistakes

- Run commands from `PenaltyProcessing`, not the repository parent directory.
- Use `--input`, not `-input`, for `plot_penalty_3d.py`.
- The valid group name is `release_height`, not `release_hight`.
- Do not append `--help` to a command you intend to execute; argparse prints
  help and exits without performing the operation.
- `--output` for trajectory reconstruction must be a CSV file path, not only a
  directory.
- After regenerating `raw_league.csv`, also regenerate features, matches, and
  reconstructed trajectories.
