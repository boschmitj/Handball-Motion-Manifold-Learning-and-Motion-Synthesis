# Penalty Processing Tool

## Overview

This project matches penalty events from `penalties.csv` to the corresponding
tracking file in `games_position_files/`, reconstructs the ball trajectory, and
exports one enriched row per penalty.

The current pipeline uses:

- `penalties.csv` for penalty metadata
- `games_position_files/*_2_phases_positions.csv` for tracking data

The output includes the extracted trajectory, release point information, derived
metrics, and an issues file for unresolved rows.

## Project Layout

- `src/shot_matcher.py` - CLI entrypoint
- `src/penalty_processing.py` - main processing pipeline
- `src/ball_trajectory.py` - trajectory parsing and heuristics
- `src/fixture_resolution.py` - fixture lookup and run-folder creation
- `src/penalty_time_utils.py` - time parsing and name normalization helpers

## Inputs

### `penalties.csv`

Semicolon-separated penalty metadata. The pipeline expects at least these fields:

- `id`
- `home_team`
- `away_team`
- `timestamp_local_timezone`
- `distance`
- `success`
- `game_clock`

### `games_position_files/*_2_phases_positions.csv`

Tracking files with ball positions. The pipeline reads only rows where
`group name == Ball` and expects columns such as:

- `formatted local time`
- `x in m`
- `y in m`
- `z in m`
- `speed in m/s`
- `acceleration in m/s2`
- `direction of movement in deg`
- `ts in ms`

## Outputs

Each run creates a new numbered folder under `out/penalty_trajectories/`, for
example:

- `out/penalty_trajectories/run_1/penalty_trajectories.csv`
- `out/penalty_trajectories/run_1/penalty_trajectories_issues.csv`

The main CSV contains the matched trajectory, release point, and derived values.
The issues CSV records unresolved rows and fixture index warnings.

## How To Run

### 1. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

If you are using a virtual environment, activate it first.

### 2. Run the pipeline

The recommended command is from the repository root:

```bash
python3 src/shot_matcher.py
```

If you are already inside `src/`, you can run:

```bash
python3 shot_matcher.py
```

Both forms are equivalent. The repo-root form is safer because all default paths
are written relative to the project root.

### 3. Check available options

```bash
python3 src/shot_matcher.py --help
```

This prints all supported CLI flags and is the quickest way to confirm the
current defaults.

## Common Run Examples

### Process the full dataset

```bash
python3 src/shot_matcher.py
```

### Process only one penalty by id

```bash
python3 src/shot_matcher.py --penalty-id 12345
```

### Process only the first N matching penalties

```bash
python3 src/shot_matcher.py --limit 25
```

### Randomly sample penalties for a quick test run

```bash
python3 src/shot_matcher.py --random-test 10
```

### Include unsuccessful penalty throws

```bash
python3 src/shot_matcher.py --include-unsuccessful
```

### Extend the start window before the shot timestamp

```bash
python3 src/shot_matcher.py --extend-start-ms 500
```

You can combine these flags when needed, for example:

```bash
python3 src/shot_matcher.py --include-unsuccessful --limit 50 --extend-start-ms 250
```

## Command-Line Options

- `--penalties`: path to `penalties.csv`.
- `--positions-dir`: directory containing `*_2_phases_positions.csv` files.
- `--output-dir`: base output directory; the script creates `penalty_trajectories/run_N/` inside it.
- `--tol-y`, `--tol-z`: currently exposed as CLI settings and passed through the pipeline.
- `--limit`: process only the first N filtered penalties.
- `--random-test`: randomly sample N penalties after filtering.
- `--include-unsuccessful`: include unsuccessful throws instead of only successful ones.
- `--penalty-id`: process one specific penalty row.
- `--extend-start-ms`: extend the trajectory window backwards before the shot time.

## What The Pipeline Does

1. Load `penalties.csv`.
2. Filter rows according to the selected CLI flags.
3. Resolve each penalty to a matching tracking file in `games_position_files/`.
4. Load ball tracking rows from the matched fixture.
5. Align the shot timestamp to the tracking timeline.
6. Apply the existing distance-based plausibility corrections.
7. Build the final trajectory window.
8. Select and serialize the release point.
9. Write the output CSVs into a new run folder.

## Notes

- The pipeline creates a new `run_N` directory each time it runs.
- Some penalties may end up in the issues file if the matching fixture is missing,
  the timestamp cannot be parsed, or the trajectory cannot be reconstructed.
- The logic is the same as before the refactor; only the code organization changed.