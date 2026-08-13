# Mocap-to-League Throw Matching – Updated Concept

* **Goal:** For each recorded 7 m Mocap throw, retrieve the **top-k most similar throws from the handball league data**. The selected League throw will then be used to continue/drive the ball motion after the Mocap Point of Release (PoR).

* **Available data:**

  * **Mocap:** 300 Hz ball trajectory; PoR can be detected from player/ball tracking. After PoR, only ball positions are available and all other kinematic quantities must be derived.
  * **League:** 20 Hz ball trajectory. Position is recorded directly; velocity, acceleration, and direction can be recomputed from the recorded positional measurements.
  * For League data, the **recomputed velocity/acceleration based on positions should be preferred over the originally provided kinematic values**, because this already produced more physically plausible results and follows the same derivation approach as for Mocap.
  * Using the same kinematic recomputation methodology for both sources should reduce source-specific differences and make velocity, acceleration, and direction more comparable.
  * League trajectories usually provide only about 4–10 post-release points, sometimes more for slower/high-arc throws.
  * Mocap trajectories may end early because the ball hits the wall/ground or tracking is lost.
  * League Data is only 20Hz. Therefore, exact trajectory endpoints, exact bounce time, or exact trajectory peak cannot be assumed to be available.  
    Exact goal impact is given from penalties.csv row for throw id.
  * Mocap trajectories may end early. Therefore, exact trajectory endpoints, exact goal impact, or exact trajectory peak cannot be assumed to be available.

Prompt First Step:
Please do the mocap coordinate translation. So that 0,0,0 becomes 0, 12.6m, 0. Also convert mm to m. 
This should be incorporated into src/create_throw_representation.py
This file puts the puzzle up to this point together. It creates a simple_penalty_trajectories.csv with release_detector_trajectory_based.py with only the successful shots and those without deflection all mapped to same side.
It then uses mocap_por_detection pipeline to possibly parse multiple throws, with swap-xy enabled and the matching ball 3d and skeleton data for this throw (Throws will be organized into folders "throw_type" with subfolders for body, skeleton and ball data files). The throw_type directory to select will be passed to the create_throw_representation script. 
It should also translate the mocap points in xy plane (as discussed).
It should then write csv's containing the throw_id (for league data coming from the penalties.csv file, for the mocap throws just counting), The trajectory of the free flight, starting with the PoR, ending after crossing goal line (for league) or at the end found in @/PenaltyProcessing/src/mocap_por_detection_pipeline.py (for mocap data, so when ball either becomes unidentified or hits wall). It then writes csv's containing the release coords, velocity, acceleration, direction and the trajectory_json, containing the ball points (including coords, velocity vector, acceleration vector, magnitude of acceleration and velocity each, t_since_release, direction) and trajectory_point_count and trajectory_duration_ms as well as error/valid column and source.
IMPORTANT: Make sure that the velocity and acceleration are recomputed for data from the fixture files. Make sure that this computation is done in exactly the same way as for mocap throws.
From this a csv raw_mocap and a csv raw_league should be created.

Furthermore, a third csv should be created containing the throw id, trajectory_json_list of the whole mocap throw, so from throw segment start as found by mocap_por_detection_pipeline to throw end (as discussed earlier) and the index relative to that list, where the PoR is. And lastly also the global indices for segment start, PoR. Use also the recalculated ball center positions here.

* **Canonical coordinate system:**

  * League convention: `+x = towards goal`, `y = lateral`, `z = vertical`.
  * Mocap currently throws towards `+Y`, so Mocap axes must be transformed to the League convention. DONE
  * Mocap `Y → League X`, Mocap `X → League Y`, with the required sign corrections. DONE
  * Throws from opposite sides of the court should be rotated by **180° around the court center in the x–y plane**, so every throw is represented as travelling toward `+x`. DONE
  * Mocap coordinates must additionally be translated so that their physical location relative to the 7 m line matches the canonical court system. Conceptually, Mocap `0,0,0` is approximately `(13 m - 0.40 m, 0, 0)` in League coordinates: League `0,0,0` is the center of the court, the positive goal line is at `x = +20 m`, and the 7 m line is therefore at `x = +13 m`.

* **Preserve two coordinate representations:**

  * **Absolute/canonical court coordinates** for information such as release height, lateral position and location relative to the 7 m line.
  * **PoR-relative coordinates** for trajectory matching:

    $$
    p_{rel}(t)=p(t)-p_{PoR}
    $$

    so every throw starts at `(0,0,0)`.

* **Reason for PoR-relative matching:** No League throw will have exactly the same spatial PoR as a Mocap throw. Absolute positions should therefore not determine trajectory similarity. Instead, the retrieval should primarily compare the **motion of the ball after release**, independent of spatial translation.

* **Kinematic feature derivation:**

  * For both Mocap and League data, derive kinematic quantities from the recorded trajectory coordinates using the same methodology where possible.
  * Derived features include:
    * velocity vector and speed,
    * acceleration,
    * horizontal direction,
    * relative displacement after PoR.
  * Direction is defined as:

    $$
    dir=\operatorname{atan2}(v_y,v_x)
    $$

    in degrees, where `0°` means toward the goal after coordinate normalization.
  * Because Mocap is sampled at 300 Hz and League data at only 20 Hz, derivative computation and smoothing may need source-appropriate window sizes, while keeping the underlying definition of the quantities consistent.
  * Acceleration should still be treated more cautiously than velocity and direction because second-order differentiation is more noise-sensitive, especially at 20 Hz.

* **Primary matching features:**

  * release speed,
  * release direction,
  * release height,
  * velocity evolution during the first available post-PoR interval,
  * direction evolution,
  * PoR-relative trajectory/displacement.

* **Secondary matching features:**

  * acceleration or smoothed acceleration characteristics,
  * release position in absolute court coordinates,
  * broad throw type such as straight/hard, lob, or bounce.

* **Features that should not be relied on:**

  * exact maximum trajectory height,
  * exact time of maximum height,
  * exact bounce time,
  * exact wall/goal/ground impact time derived from the sparse trajectory,
  * trajectory duration until the last available point,
  * Mocap tracking endpoint,
  * precise impact-derived features from 20 Hz League data.

* **Temporal comparison:** Compare both sources at common elapsed times after PoR, for example:

  $$
  0,\ 50,\ 100,\ 150,\ 200\text{ ms}, ...
  $$

  Mocap can be sampled/interpolated at the League timestamps. League data should not be upsampled to 300 Hz and treated as if it contained additional information.

* **Partial trajectory comparison:** Only compare the interval that exists in both trajectories:

  $$
  T_{common}=\min(T_{Mocap},T_{League})
  $$

  and normalize the trajectory distance by the number of available comparison points so shorter recordings are not automatically favoured.

* **Primary retrieval method: weighted k-nearest neighbours / nearest-neighbour retrieval:**

  * No supervised ground truth exists that specifies which League throw is the correct match for a given Mocap throw.
  * Therefore, start with a **physics-informed weighted distance metric** rather than a model that requires labelled training data.
  * Each Mocap throw is compared against all League throws, and the `k` League throws with the smallest distance are returned.
  * Example distance:

    $$
    D =
    w_vD_{velocity}
    +w_dD_{direction}
    +w_tD_{trajectory}
    +w_aD_{acceleration}
    +w_sD_{spatial}
    +w_cD_{type}
    $$

  * Distances for different feature groups must be normalized before weighting so that features with numerically larger units, such as acceleration, do not dominate the result.
  * Initial weights should be manually chosen based on physical relevance and reliability, with higher weight on release velocity, release direction, and early trajectory evolution and lower weight on acceleration and absolute spatial position.

* **Experimental extension: learning-to-rank with synthetic training data:**

  * Use the weighted kNN retrieval as the **main baseline**.
  * As an additional experiment, generate synthetic League-like trajectories from Mocap data by degrading the 300 Hz Mocap trajectory to a League-like representation:

    $$
    300\text{ Hz Mocap}
    \rightarrow
    20\text{ Hz sampling}
    \rightarrow
    \text{optional noise/filtering}
    \rightarrow
    \text{recompute }v,a,dir
    $$

  * Because the synthetic 20 Hz trajectory originates from the same Mocap throw, its correct correspondence is known and can be used as synthetic ground truth.
  * Positive pair:

    $$
    M_i \leftrightarrow \tilde M_i
    $$

  * Negative pairs:

    $$
    M_i \leftrightarrow \tilde M_j,\quad i\neq j
    $$

  * Multiple degraded variants of the same Mocap throw can be generated to increase the amount of training data.
  * Training and test splits must be performed by **original Mocap throw**, not by synthetic variant, to avoid data leakage.
  * The learning-to-rank model should remain simple due to the limited implementation time. A suitable first approach is to learn the weights of the same feature distances already used by kNN rather than introducing a neural embedding model.
  * The learned ranking function can therefore use:

    $$
    score(M,L)=f(D_{velocity},D_{direction},D_{trajectory},D_{acceleration},...)
    $$

    and learn which differences should contribute most strongly to similarity.
  * This creates a direct comparison between:
    * **manually weighted similarity retrieval**, and
    * **learned similarity weighting from synthetic data**.
  * The synthetic training setup has an important limitation: synthetic 20 Hz Mocap data does not perfectly reproduce the tracking noise, filtering, and measurement characteristics of real League data. Therefore, results on synthetic data should not be treated as proof of performance on real League trajectories.

* **Evaluation:**

  * For synthetic data, exact correspondence is known, so quantitative retrieval metrics can be calculated:
    * Recall@1,
    * Recall@3,
    * Recall@5,
    * Mean Reciprocal Rank (MRR).
  * Compare weighted kNN and the learned ranker on the same held-out Mocap throws.
  * For real Mocap-to-League retrieval, where no exact ground truth exists, evaluate the top-k candidates manually/qualitatively based on physical plausibility and similarity.
  * Useful comparisons include ablations such as:
    * release speed + direction only,
    * + PoR-relative trajectory,
    * + velocity evolution,
    * + acceleration,
    * manually weighted vs. learned weights.

* **Separate physical similarity from spatial similarity:**

  * `D_kinematic / D_trajectory` determines whether two throws behave similarly.
  * `D_spatial` only expresses whether their original release positions were similar.
  * Spatial position should therefore have lower importance than release velocity, direction and trajectory behaviour.

* **Avoid the visual PoR discontinuity:** After selecting a League throw, translate its complete trajectory so that its PoR is exactly equal to the Mocap PoR:

  $$
  L_{aligned}(t)=L(t)-L(0)+M(0)
  $$

  This preserves the League trajectory's shape, velocity and direction while eliminating any positional jump when switching from Mocap-driven to League-driven ball motion.

* **Implementation scope / priority:**

  * **Mandatory:** coordinate normalization, common kinematic recomputation, PoR-relative representation, feature normalization, weighted kNN retrieval, top-k output, and qualitative validation.
  * **Experimental extension:** synthetic 20 Hz Mocap generation and a simple learning-to-rank model that learns feature weights.
  * More complex approaches such as neural embeddings, Siamese networks, or deep metric learning are out of scope and can be mentioned as future work.

* **Overall pipeline:**
  `Mocap trajectory → coordinate canonicalization → PoR detection → derive kinematics → PoR-relative representation → normalized feature comparison → weighted kNN top-k retrieval → optionally compare against learning-to-rank trained on synthetic 20 Hz Mocap → translate selected League trajectory to Mocap PoR → continue ball motion with League data.`
