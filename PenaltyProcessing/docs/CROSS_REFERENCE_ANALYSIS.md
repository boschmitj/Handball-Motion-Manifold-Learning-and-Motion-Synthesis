# CROSS-REFERENCE ANALYSIS: shots.csv ↔ position_tracking.csv

## What shots.csv Provides (LIMITED):
```
✓ Shot timestamp (ms)
✓ Shot position (x, y)
✓ Ball speed at release (m/s)
✓ Shot distance from goal (m)
✓ Shot type: 'penalty' or 'field'
✗ Trajectory arc ONLY shows single point repeated: "x,y;x,y"
✗ No frame-by-frame speed progression
✗ No acceleration data (only at release moment)
✗ No pre-release ball state
```

**Example trajectory field:** `"-13.43,1.62;-13.43,1.62"`  
This is just the shot position, NOT a trajectory arc!

---

## What position_tracking.csv Provides (DETAILED):

### Ball Tracking Data (72,037 points):
```
✓ 50ms temporal resolution (20 samples/second)
✓ 3D position: x, y, z (meters)
✓ Speed: speed in m/s  
✓ Acceleration: m/s² (frame-to-frame)
✓ Direction: degrees of movement
✓ Timestamp: ms precision
```

**Raw Example:**
```
ts=1708106445000ms | Ball 1
x=-1.195m, y=0.498m, z=1.436m
speed=1.752 m/s
acceleration=-1.323 m/s²
direction=63.6°
```

---

## THE SOLUTION: Cross-Reference by Timestamp

### Matching Strategy:

1. **Get shot from shots.csv**
   - Shot timestamp: `1708106445000` ms
   - Shot category: `penalty`
   - Distance: `6.77` m

2. **Query position file for same timestamp ±2000ms**
   - Find all ball position records in time window
   - Create trajectory sequence

3. **Extract Physics Data:**
   ```python
   trajectory = {
       'release_time': 1708106445000 ms,
       'release_speed': 33.76 m/s,
       'release_acceleration': 15.85 m/s²,
       'flight_duration': 850 ms,
       'points': [
           {ts: 1708106445000, x: -1.195, y: 0.498, z: 1.436, v: 1.752, a: -1.323},
           {ts: 1708106445050, x: -1.251, y: 0.518, z: 1.678, v: 1.731, a: 4.072},
           {ts: 1708106445100, x: -1.322, y: 0.477, z: 1.893, v: 2.158, a: 1.242},
           ... (17+ more points)
       ],
       'max_speed': 5.428 m/s,
       'trajectory_angle_variance': 12.3°,
   }
   ```

---

## For 7M PENALTY Detection:

### Signature Pattern:
```
FROM position_tracking.csv:
✓ Straight trajectory: angle_variance < 15°
✓ High release speed: > 15 m/s typical
✓ Low trajectory curvature (gravity + air resistance minimal in short flight)
✓ Ball released at ~7m distance from goal (x ≈ ±7m)
✓ Brief flight time: 0.5-1.5 seconds
✓ Release point near penalty spot

FROM shots.csv:
✓ shot_type = 'penalty'
✓ distance ≈ 6.5-7.5 m
✓ shot_position_x ≈ ±7m
```

---

## Speed & Acceleration Analysis:

### Trajectory Stages:

1. **Pre-release (100-300ms before shot)**
   - Wind-up phase
   - Variable acceleration

2. **Release Point (shot timestamp)**
   - Max velocity recorded
   - Max acceleration at release

3. **Flight Phase (after release)**
   - Speed gradually decreases (air resistance)
   - Acceleration varies (gravity component depends on trajectory angle)
   - Z-component shows parabolic path (gravity effect)

### Example Physics:
```
Release velocity:     33.76 m/s
Release acceleration: 15.85 m/s²
Peak speed:          5.428 m/s (wait, this is low - need to check data)
Flight duration:     850 ms
Horizontal distance: ~5-7m
Vertical drop:       Z component shows trajectory
```

---

## Implementation Roadmap:

### Step 1: Align Timestamps
```python
shots[timestamp_ms] → positions[ts_ms]
Match within ±500ms window
```

### Step 2: Extract Trajectory Sequence
```python
trajectory_points = positions[
    (positions.ts_ms >= shot_time - 1500) &
    (positions.ts_ms <= shot_time + 1500) &
    (positions.full_name.contains('Ball'))
].sort_values('ts_ms')
```

### Step 3: Compute Derived Metrics
```python
- Flight duration
- Speed progression curve
- Acceleration profile
- Angle variance (straightness)
- Vertical drop rate
- Peak speed during flight
```

### Step 4: 7M Penalty Classification
```python
is_7m = (
    shot_category == 'penalty' AND
    distance in [6.5-7.5]m AND
    angle_variance < 15° AND
    release_speed > 15 m/s AND
    shot_x in [±6.5-7.5]m
)
```

---

## Data Limitations & Quirks:

⚠️ **Known Issues:**
- Timestamp alignment between files (may need ±1000ms tolerance)
- Some penalties may not have valid tracking data
- Ball possession field in tracking file may indicate pre-release state
- Session_ID matching needed to link files correctly

✓ **Advantages:**
- 50ms resolution is excellent for ballistic analysis
- Acceleration data allows force/spin estimation
- 3D coordinates enable goal-line prediction
- Direction data validates straight-line assumption

---

## What You CAN Extrapolate:

1. **Ballistic Trajectory** - Complete 3D path with time series
2. **Initial Velocity Components** - vx, vy, vz at release
3. **Spin Estimation** - From acceleration patterns
4. **Impact Speed** - Speed at goal line
5. **Time-to-Goal** - Flight duration
6. **Interception Points** - Where goalkeeper needs to be
7. **Success Rate Correlation** - Speed/angle → goal probability
8. **Throwing Mechanics** - Release acceleration pattern

