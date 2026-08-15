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

Each `*_distance` is the unweighted mean normalized squared difference within
that available group. Each `*_contribution` is its additive contribution to
`total_distance`; contribution columns sum to the total. `compared_weight` and
`compared_feature_count` expose how much common data each pair had.

League population standard deviations provide the scales. Angular scales are
computed around the League circular mean, and every angular pair difference is
the shortest arc. Constant/all-missing League columns are ignored. Metadata
(`sampling_rate_hz`, point count, and duration) is never selected.
