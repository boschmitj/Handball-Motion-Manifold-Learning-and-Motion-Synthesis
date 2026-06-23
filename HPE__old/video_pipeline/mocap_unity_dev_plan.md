# Development Plan: MoCap Marker → Mixamo Avatar Animation in Unity (C#)

## TSV File Structure Summary

- **Header block** (rows 1–10): metadata fields (`FILE_VERSION`, `NO_OF_FRAMES=505`, `FREQUENCY=300 Hz`, `NO_OF_MARKERS=39`)
- **Data rows** start at row 11: `Frame | Time | [MarkerName X | Y | Z] × 39`
- **Coordinate units**: millimeters
- **Relevant markers** (with their TSV column names):

| Body part | Marker(s) in TSV |
|---|---|
| Left shoulder top (clavicular) | `Q_LShoulderTop` |
| Left arm/shoulder joint | `Q_LArm` |
| Left elbow | `Q_LElbowOut`, `Q_LElbowIn` |
| Left wrist | `Q_LWristIn`, `Q_LWristOut` |
| Left hand (MCP 2) | `Q_LHand2` |
| Right shoulder top | `Q_RShoulderTop` |
| Right arm/shoulder joint | `Q_RArm` |
| Right elbow | `Q_RElbowOut`, `Q_RElbowIn` |
| Right wrist | `Q_RWristIn`, `Q_RWristOut` |
| Right hand (MCP 2) | `Q_RHand2` |
| Head | `Q_HeadL`, `Q_HeadR`, `Q_HeadFront` |
| Spine/chest | `Q_Chest`, `Q_SpineThoracic2`, `Q_SpineThoracic12` |
| Waist/hips | `Q_WaistLFront`, `Q_WaistL`, `Q_WaistBack`, `Q_WaistR`, `Q_WaistRFront` |
| Left leg | `Q_LThighFrontLow`, `Q_LKneeOut`, `Q_LShinFrontHigh`, `Q_LAnkleOut`, `Q_LHeelBack`, `Q_LForefoot2`, `Q_LForefoot5` |
| Right leg | `Q_RThighFrontLow`, `Q_RKneeOut`, `Q_RShinFrontHigh`, `Q_RAnkleOut`, `Q_RHeelBack`, `Q_RForefoot2`, `Q_RForefoot5` |

---

## Mixamo Bone → Marker Mapping

This is the most critical conceptual layer of the project. Mixamo's rig does **not** correspond 1:1 to anatomical joints.

| Mixamo Bone | Driven by | Method |
|---|---|---|
| `LeftShoulder` | `Q_LShoulderTop` | Direct position |
| `LeftArm` (shoulder joint) | `Q_LArm` | Direct position |
| `LeftForeArm` (elbow) | avg(`Q_LElbowOut`, `Q_LElbowIn`) | Midpoint |
| `LeftHand` (wrist) | avg(`Q_LWristIn`, `Q_LWristOut`) | Midpoint |
| `LeftHandMiddle1` (MCP) | `Q_LHand2` | Direct position |
| *(mirror for Right side)* | | |
| `Head` | avg(`Q_HeadL`, `Q_HeadR`) | Midpoint, orient toward `Q_HeadFront` |
| `Spine` / `Spine1` | `Q_SpineThoracic12`, `Q_SpineThoracic2` | Direct |
| `Hips` | centroid of waist markers | Avg of 5 waist markers |
| `LeftUpLeg` | `Q_LThighFrontLow` | Direct |
| `LeftLeg` (knee) | `Q_LKneeOut` | Direct |
| `LeftFoot` | `Q_LAnkleOut` | Direct |
| `LeftToeBase` | avg(`Q_LForefoot2`, `Q_LForefoot5`) | Midpoint |
| *(mirror for Right side)* | | |

> **Note:** You only have `Q_LHand2` / `Q_RHand2` (MCP 2). No finger animation beyond MCP is possible with this data.

---

## Implementation Steps

### Step 1 – Parse the TSV at Runtime

- Skip the 9 metadata header rows; parse from the `Frame\tTime\t...` column header row onward.
- Build a `Dictionary<string, int>` mapping marker name (e.g. `"Q_LArm X"`) to its column index from the header row.
- Store each frame as `Dictionary<string, Vector3>` (keyed by marker name, value = position).
- Convert millimeters → meters on ingest (divide by 1000).

### Step 2 – Coordinate System Conversion

**This is the single most common source of broken animations.**

The MoCap system (likely Qualisys, given the `Q_` prefix and `.tsv` format) uses:
- **X** = right
- **Y** = up (sometimes: Y = forward, Z = up — verify from a known pose)
- **Z** = forward (or depth)

Unity uses:
- **X** = right
- **Y** = up
- **Z** = forward (left-handed)

Qualisys default is **right-handed, Y-up**. The conversion is:

```
Unity.x =  mocap.x  (mm → m)
Unity.y =  mocap.z  (mm → m)   ← swap Y and Z
Unity.z =  mocap.y  (mm → m)
```

> ⚠️ **Verify this from a T-pose frame**: confirm that the subject's head Y-position is highest, both arms extend laterally along X, and forward direction maps to +Z in Unity. If the system was calibrated differently, the axes may differ.

### Step 3 – Establish a Reference Pose (T-Pose Binding)

The marker positions are in **world/lab space**, but the avatar's bones are driven by **local rotations**. The approach is:

1. Pick a **reference frame** (ideally a T-pose or the first frame of calm standing).
2. Record the world-space marker positions at that frame as "bind pose" positions.
3. For each frame, compute **rotation deltas** relative to the bind pose to drive bone rotations.

This avoids needing an inverse kinematics solver for most bones.

### Step 4 – Compute Bone Rotations from Marker Positions

For each bone segment (e.g. upper arm):

1. Compute the **segment vector** from proximal to distal marker:  
   `segmentDir = distal_position - proximal_position`
2. Compute the **reference segment vector** from the bind pose frame.
3. Compute the rotation that takes the reference vector to the current vector:  
   `Quaternion rot = Quaternion.FromToRotation(refDir, currentDir)` 
4. Apply this as the bone's **local rotation**, composited with the bind pose local rotation.

For joints with two degrees of freedom (e.g. shoulder), you may need to also compute axial/twist rotation separately using a secondary marker as an up-vector reference.

### Step 5 – Drive the Animator via Script

Use Unity's `HumanPoseHandler` + `HumanPose` API (cleanest for Humanoid rigs):

```csharp
HumanPoseHandler poseHandler = new HumanPoseHandler(avatar, rootTransform);
HumanPose pose = new HumanPose();
poseHandler.GetHumanPose(ref pose);
// modify pose.muscles[] values
poseHandler.SetHumanPose(ref pose);
```

Alternatively, directly set `Transform.localRotation` on the bones retrieved via `Animator.GetBoneTransform(HumanBodyBones.xxx)`.

### Step 6 – Playback Controller

- Store all parsed frames in a `List<Frame>` on `Start()`.
- Use a coroutine or `Update()` with a frame timer driven by `Time.deltaTime` and the recorded frequency (300 Hz → `targetDeltaTime = 1f/300f`).
- Expose playback controls: play, pause, loop, scrub.

---

## Recommended C# File Structure

```
MoCapPlayer.cs
├── MoCapFrame (struct)         // per-frame data
├── MoCapParser (static class)  // reads .tsv, returns List<MoCapFrame>
├── MarkerMapper (class)        // converts markers → bone rotations
└── MoCapAnimator : MonoBehaviour
    ├── [SerializeField] TextAsset tsvFile
    ├── [SerializeField] Animator avatarAnimator
    ├── Start()   → parse + bind
    └── Update()  → advance frame + apply pose
```

---

## Pitfalls to Watch Out For

### Axis/Handedness
- Qualisys is right-handed; Unity is left-handed. Failing to swap axes = upside-down or mirrored animation.
- Don't assume — validate with a known-good static frame.

### Units
- TSV values are in **millimeters**. Unity works in **meters**. Forgetting this makes the avatar translate 1000× too far.

### Root Position
- The hip/root position in lab space needs to be **normalized** to the avatar's origin. Subtract the bind-pose hip world position so the avatar doesn't fly off-screen.

### Missing/Occluded Frames
- TSV marker data can have gaps (NaN or blank columns). Add null/NaN checks and either interpolate linearly or hold the previous valid frame.

### Mixamo "Shoulder" vs. Anatomical Shoulder
- The Mixamo `LeftShoulder` bone is the **clavicle**, not the glenohumeral joint. Do not drive it with `Q_LArm`. Drive it with `Q_LShoulderTop` (clavicular marker) and drive `LeftArm` with `Q_LArm`.

### Segment vs. Joint Rotation
- Markers are on the **skin**, not joints. At high speeds (throwing motion), skin sliding and soft tissue artefacts will introduce noise. Consider a simple moving-average filter (window of 3–5 frames) on marker positions before computing rotations.

### Rotation Singularities
- `Quaternion.FromToRotation` gives undefined results when vectors are anti-parallel (180° flip). Add a check and fall back to a perpendicular axis rotation.

### Bind Pose Frame Selection
- If frame 0 is mid-motion, the bind pose will be wrong. Either record a T-pose at the start of the capture, or expose a UI to select the bind frame index.

### Mixamo Rig Variability
- Not all Mixamo characters have identical bone hierarchies or rest poses. The plan above assumes a standard Humanoid-configured Mixamo character in Unity. Verify the rig is set to **Humanoid** (not Generic) in the Import Settings.

---

## Suggested Verification Checklist

- [ ] Parse first 5 frames and print `Q_LArm` position to Console — sanity check values.
- [ ] In a T-pose frame, confirm Unity Y-axis is the tallest coordinate.
- [ ] Animate only the right arm first before wiring all bones.
- [ ] Compare a frame mid-throw to video footage (if available) to confirm left/right is not mirrored.
- [ ] Check that hips don't drift upward or sideways over the clip.
