# Weighted kNN throw retrieval

Run from `PenaltyProcessing`:

```bash
conda run -n BA python -m src.model.weighted_knn \
  --mocap out/throw_features/features_mocap.csv \
  --league out/throw_features/features_league.csv \
  --output out/throw_features/weighted_knn_matches.csv
```

The default is `k=5`. Override group-weight budgets with JSON, for example:

```bash
--weights '{"acceleration": 0, "relative_trajectory": 4}'
```

Select a named set from `PREDEFINED_WEIGHT_SETS`:

```bash
python -m src.model.weighted_knn --weight-preset height_focus
```

`--help` lists every available key and its complete weight dictionary. Add a
new entry to `PREDEFINED_WEIGHT_SETS` in `weighted_knn.py` to expose another
preset automatically.

Run a reproducible randomized weight search:

```bash
python -m src.model.weighted_knn \
  --mocap out/throw_features/features_mocap.csv \
  --league out/throw_features/features_league.csv \
  --random-weight-runs 100 \
  --random-seed 42
```

This creates a uniquely named `out/weighted_knn_random_*` directory containing
all match files, per-throw-type metrics, `random_search_summary.csv`, and
`best_weights.json`. Model selection uses rank-1 distance averaged within each
throw type and then averaged equally across types. The reported score is
`1 / (1 + balanced_distance)`, so lower distance and higher score are better.

In `random_search_summary.csv`, columns named after feature groups—such as
`release_speed`, `release_angles`, and `relative_trajectory`—are the weights
used by that run. They are neither scores nor distances; a larger value means
only that the feature group influenced that run more strongly.

Each `*_distance` is the unweighted mean normalized squared difference within
that available group. Each `*_contribution` is its additive contribution to
`total_distance`; contribution columns sum to the total. `compared_weight` and
`compared_feature_count` expose how much common data each pair had.

League population standard deviations provide the scales. Angular scales are
computed around the League circular mean, and every angular pair difference is
the shortest arc. Constant/all-missing League columns are ignored. Metadata
(`sampling_rate_hz`, point count, and duration) is never selected.
