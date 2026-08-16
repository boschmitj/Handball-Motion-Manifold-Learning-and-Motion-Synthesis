# Next Plan: Mocap-to-League Throw Matching

*Status: Analysis of current codebase vs. `mocap_league_throw_matching.md` goal — 2026-08-13*

## What's already implemented (per the goal doc)

| Goal-doc item | Status | Where |
|---|---|---|
| League trajectory extraction + PoR detection | ✅ Done | `src/penalty_processing.py`, `src/ball_trajectory.py`, `src/release_detector_trajectory_based.py` |
| Mocap PoR detection (3 methods) | ✅ Done | `src/mocap_por_detection_pipeline.py` |
| Mocap axis swap (Y→X, X→Y) | ✅ Done | `tools/swap_xy_tsv.py` |
| 180° rotation for opposite sides | ✅ Done | `_rotate_180_z` / `normalize_side` in `release_detector_trajectory_based.py` |
| Common kinematic recomputation (velocity/accel from positions) | ⚠️ Partially done | `compute_velocity_acceleration_from_points` exists, but Mocap uses a *different* scheme (`_compute_kinematics`) and League direction is not recomputed |
| Fixture-vs-recomputed validation plots | ✅ Done | `visualization/plot_fixture_vs_recomputed_timeseries.py` |

## The concrete gaps — what's needed next, in priority order

### Phase 0 — Data unification (prerequisite for everything else)

1. **Mocap coordinate translation to canonical court coords (NOT done).**
   The goal doc marks only the axis swap and 180° rotation as DONE. The translation
   `Mocap (0,0,0) ≈ (13 m − 0.40 m, 0, 0)` in League coordinates is still missing.
   Need a function that maps a Mocap throw's swapped coordinates into the canonical
   court frame (x toward goal, origin at court center).

2. **Unified throw representation module.**
   There is currently no single data structure holding both Mocap and League throws
   in a comparable canonical form. Create e.g. `src/throw_representation.py` that loads:
   - Mocap throws from `mocap_por/por_method_comparison_*_timeseries.csv`
     (has `trajectory_json`, `release_point_json`, `velocity_per_point`,
     `tangential_acceleration_per_point`), and
   - League throws from `out/.../simple_penalty_trajectories.csv`
     (has `trajectory_json`, `release_point_json`, `velocity_per_point`,
     `acceleration_per_point`),
   and normalizes both into one canonical `Throw` dataclass
   (canonical coords + PoR-relative coords + kinematics).

3. **PoR-relative representation.**
   Implement `p_rel(t) = p(t) − p_PoR` so every throw starts at `(0,0,0)`.
   This is the core reason matching can work across sources with different
   absolute release positions.

### Phase 1 — Feature extraction & normalization

4. **Unify kinematic recomputation.**
   The goal doc explicitly says "using the same kinematic recomputation methodology
   for both sources." Currently Mocap uses `_compute_kinematics` (central differences
   over frame indices) while League uses `compute_velocity_acceleration_from_points`
   (central differences over timestamps). Unify these, and add **direction
   recomputation** (`dir = atan2(v_y, v_x)`) for League data — the existing function
   only returns speed and accel, not direction.

5. **Feature extraction module.**
   Extract the primary matching features per throw:
   - release speed,
   - release direction,
   - release height,
   - velocity evolution over the first post-PoR interval,
   - direction evolution,
   - PoR-relative trajectory/displacement,
   - (secondary) acceleration characteristics + absolute release position.

6. **Feature normalization.**
   Normalize each feature group before weighting so acceleration (large units)
   doesn't dominate speed/direction.

### Phase 2 — Weighted kNN retrieval (the mandatory core)

7. **Implement weighted kNN retrieval** with
   `D = w_v·D_velocity + w_d·D_direction + w_t·D_trajectory + w_a·D_acceleration + w_s·D_spatial + w_c·D_type`,
   comparing each Mocap throw against all League throws and returning top-k.

8. **Temporal comparison at common elapsed times.**
   Compare at `0, 50, 100, 150, 200 ms, ...` after PoR, interpolating Mocap (300 Hz)
   at the League (20 Hz) timestamps — do NOT upsample League to 300 Hz.

9. **Partial trajectory comparison.**
   Use `T_common = min(T_Mocap, T_League)` and normalize the trajectory distance by
   the number of comparison points so short recordings aren't favored.

10. **Top-k output.**
    Serialize the top-k League throw IDs + distances to CSV/JSON for each Mocap throw.

-> Found throws are looking plausible
-> Need to be upsampled for animation and smoothed
-> For throws with z down then z up, need to find a plausible bounce location


### Phase 3 — Trajectory alignment for continuation

11. **Implement `L_aligned(t) = L(t) − L(0) + M(0)`** to translate the selected
    League trajectory so its PoR exactly equals the Mocap PoR, eliminating the
    visual discontinuity when switching from Mocap-driven to League-driven ball motion.

### Phase 4 — Evaluation & validation

12. **Qualitative validation** of top-k candidates (physical plausibility), e.g. a
    visualization script comparing Mocap throw vs. top-k League throws in
    PoR-relative coordinates.

### Phase 5 — Experimental extension (after the mandatory core works)

13. **Synthetic 20 Hz Mocap generation**:
    degrade 300 Hz Mocap → 20 Hz sampling → optional noise/filtering → recompute
    v, a, dir. This gives synthetic ground truth for evaluation.

14. **Learning-to-rank** on synthetic data:
    learn the feature weights of the same distance groups used by kNN (simple
    approach, no neural embedding), and compare Recall@1/3/5 + MRR against the
    manually-weighted kNN baseline.

## Key technical notes / gotchas

- **Direction convention**: Mocap `_compute_kinematics` already uses
  `atan2(vel[1], vel[0])` = `atan2(v_y, v_x)`, consistent with the goal doc's
  `0° = toward goal` — but only *after* coordinate canonicalization
  (swap + translation) is applied. Ensure direction is recomputed in canonical
  coords, not raw Mocap coords.
- **Mocap method choice**: The pipeline outputs 3 PoR methods per throw. Decide
  which method (or a consensus) feeds the matching — method 1 (hand-ball distance
  spike) is the most physically grounded for release detection.
- **League data availability**: `simple_penalty_trajectories.csv` already contains
  everything needed (trajectory, release point, recomputed velocity/accel per
  point), so the League pipeline does not need to be re-run — just load it.
- **Dependencies**: `requirements.txt` only has pandas/numpy/matplotlib. The kNN
  and learning-to-rank can be done with numpy/scipy (no sklearn strictly needed),
  but if `scipy.spatial.cKDTree` or `sklearn` is wanted, add them.

## Recommended next action

Start with **Phase 0 items 1–3** (Mocap translation + unified `Throw`
representation + PoR-relative coords). Everything downstream (features, kNN,
alignment) depends on having both sources in one canonical, PoR-relative form.
Once that module exists, the weighted kNN (Phase 2) is a relatively small,
self-contained addition.