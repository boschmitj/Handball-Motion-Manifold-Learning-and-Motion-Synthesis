using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

/// <summary>
/// TsvRetargetPlayer: Implements the full mocap_unity_dev_plan workflow.
/// Parses Qualisys TSV marker data, establishes T-pose reference binding, computes bone rotations,
/// and drives a Mixamo Humanoid avatar via direct Transform manipulation.
/// 
/// Coordinate System: Qualisys is right-handed Y-up; Unity is left-handed Y-up.
/// Conversion: unity.x = mocap.x, unity.y = mocap.z, unity.z = mocap.y (with mm→m scaling).
/// </summary>
public class TsvRetargetPlayer : MonoBehaviour
{
    // ========== Serialized Fields ==========
    [Header("TSV Input")]
    public string tsvFilePath;
    public bool loadOnStart = true;

    [Header("Avatar")]
    public Transform hipsRoot; // Root transform of the Mixamo avatar
    public bool autoFindHips = true;

    [Header("Playback")]
    public bool playOnStart = true;
    public bool loop = true;
    public float playbackSpeed = 1f;

    [Header("Reference Binding")]
    [Tooltip("Frame index to use as T-pose reference. Use -1 to auto-detect calm frame (low motion).")]
    public int referenceFrameIndex = 0;
    public bool autoDetectReferenceFrame = false;

    [Header("Debug & Validation")]
    public bool verboseLogs = false;
    public bool drawMarkerGizmos = false;
    public float markerGizmoSize = 0.02f;

    // ========== Private Runtime State ==========
    private List<MoCapFrame> _frames = new List<MoCapFrame>();
    private int _currentFrameIndex = 0;
    private float _playbackTime = 0f;
    private bool _isPlaying = false;

    // Cached reference pose data
    private Dictionary<string, Vector3> _referenceBind; // marker name → world position at bind frame
    private Transform[] _boneTargets; // Mixamo bone hierarchy
    private Quaternion[] _bindLocalRotations; // Local rotations at T-pose
    private Quaternion[] _bindWorldRotations; // World rotations at bind per mapped bone
    private Vector3 _bindHipsWorldPos;
    private HumanPoseHandler _poseHandler;
    private HumanPose _humanPose;
    private float[] _bindSegmentLengths; // per BONE_MAPPINGS
    private Vector3[] _bindSegmentDirs;

    // ========== MoCapFrame Data Structure ==========
    [System.Serializable]
    private class MoCapFrame
    {
        public int frameIndex;
        public float timeSeconds;
        public Dictionary<string, Vector3> markerPositions; // marker name → world position (meters, Unity coords)
    }

    // ========== Marker → Bone Mapping ==========
    // Based on mocap_unity_dev_plan.md table
    [System.Serializable]
    private class BoneMapping
    {
        public HumanBodyBones bone;
        public List<string> markerNames;

        public BoneMapping(HumanBodyBones b, params string[] markers)
        {
            bone = b;
            markerNames = new List<string>(markers);
        }
    }

    private static readonly BoneMapping[] BONE_MAPPINGS = new[]
    {
        // Torso
        new BoneMapping(HumanBodyBones.Hips, "Q_WaistLFront", "Q_WaistL", "Q_WaistBack", "Q_WaistR", "Q_WaistRFront"),
        new BoneMapping(HumanBodyBones.Spine, "Q_SpineThoracic12", "Q_SpineThoracic2"),
        new BoneMapping(HumanBodyBones.Chest, "Q_Chest"),
        new BoneMapping(HumanBodyBones.UpperChest, "Q_Chest"),
        new BoneMapping(HumanBodyBones.Neck, "Q_Chest", "Q_HeadL", "Q_HeadR", "Q_HeadFront"),
        new BoneMapping(HumanBodyBones.Head, "Q_HeadL", "Q_HeadR", "Q_HeadFront"),

        // Left Arm
        new BoneMapping(HumanBodyBones.LeftShoulder, "Q_LShoulderTop"),
        new BoneMapping(HumanBodyBones.LeftUpperArm, "Q_LArm"),
        new BoneMapping(HumanBodyBones.LeftLowerArm, "Q_LElbowOut", "Q_LElbowIn"),
        new BoneMapping(HumanBodyBones.LeftHand, "Q_LWristIn", "Q_LWristOut"),

        // Right Arm
        new BoneMapping(HumanBodyBones.RightShoulder, "Q_RShoulderTop"),
        new BoneMapping(HumanBodyBones.RightUpperArm, "Q_RArm"),
        new BoneMapping(HumanBodyBones.RightLowerArm, "Q_RElbowOut", "Q_RElbowIn"),
        new BoneMapping(HumanBodyBones.RightHand, "Q_RWristIn", "Q_RWristOut"),

        // Left Leg
        new BoneMapping(HumanBodyBones.LeftUpperLeg, "Q_LThighFrontLow"),
        new BoneMapping(HumanBodyBones.LeftLowerLeg, "Q_LKneeOut"),
        new BoneMapping(HumanBodyBones.LeftFoot, "Q_LAnkleOut"),
        new BoneMapping(HumanBodyBones.LeftToes, "Q_LForefoot2", "Q_LForefoot5"),

        // Right Leg
        new BoneMapping(HumanBodyBones.RightUpperLeg, "Q_RThighFrontLow"),
        new BoneMapping(HumanBodyBones.RightLowerLeg, "Q_RKneeOut"),
        new BoneMapping(HumanBodyBones.RightFoot, "Q_RAnkleOut"),
        new BoneMapping(HumanBodyBones.RightToes, "Q_RForefoot2", "Q_RForefoot5"),
    };

    // ========== Lifecycle ==========
    private void Start()
    {
        if (!loadOnStart) return;

        if (!LoadAndBindTsv()) return;

        if (autoDetectReferenceFrame)
            referenceFrameIndex = DetectCalmFrame();

        if (!EstablishReferencePose(referenceFrameIndex)) return;

        // Initialize HumanPoseHandler
        var animator = GetComponent<Animator>();
        if (animator == null || animator.avatar == null || !animator.avatar.isValid || !animator.isHuman)
        {
            Debug.LogError("Animator/Humanoid avatar required for HumanPoseHandler approach");
            return;
        }
        _poseHandler = new HumanPoseHandler(animator.avatar, animator.transform);
        _humanPose = new HumanPose();
        _poseHandler.GetHumanPose(ref _humanPose);

        // cache bind segment info
        CacheBindSegments();

        // cache bone transforms and bind-world rotations for mapped bones
        _boneTargets = new Transform[BONE_MAPPINGS.Length];
        _bindWorldRotations = new Quaternion[BONE_MAPPINGS.Length];
        for (int i = 0; i < BONE_MAPPINGS.Length; i++)
        {
            var bt = animator.GetBoneTransform(BONE_MAPPINGS[i].bone);
            _boneTargets[i] = bt;
            _bindWorldRotations[i] = bt != null ? bt.rotation : Quaternion.identity;
        }

        BuildMuscleLookup();

        // disable the Animator so our manual transforms aren't overwritten
        animator.enabled = false;

        _isPlaying = playOnStart;
        if (_isPlaying) ApplyFrame(_currentFrameIndex);
    }

    private void Update()
    {
        if (!_isPlaying || _frames.Count == 0) return;

        _playbackTime += Time.deltaTime * Mathf.Max(0f, playbackSpeed);
        float frameDuration = 1f / 300f; // Assume 300 Hz; will override if TSV specifies
        if (_frames.Count > 0 && _frames[0].timeSeconds > 0)
            frameDuration = (_frames[_frames.Count - 1].timeSeconds - _frames[0].timeSeconds) / (_frames.Count - 1);
        else if (_frames.Count > 1)
            frameDuration = 1f / 300f;

        float totalDuration = _frames.Count * frameDuration;
        if (loop) _playbackTime %= totalDuration;
        else _playbackTime = Mathf.Min(_playbackTime, totalDuration - frameDuration);

        _currentFrameIndex = Mathf.Clamp(Mathf.RoundToInt(_playbackTime / frameDuration), 0, _frames.Count - 1);
        ApplyFrame(_currentFrameIndex);
    }

    private void OnDrawGizmos()
    {
        if (!drawMarkerGizmos || _frames == null || _frames.Count == 0 || _currentFrameIndex >= _frames.Count)
            return;

        MoCapFrame frame = _frames[_currentFrameIndex];
        Gizmos.color = Color.cyan;
        foreach (var kvp in frame.markerPositions)
        {
            Gizmos.DrawSphere(kvp.Value, markerGizmoSize);
        }
    }

    // ========== Loading & Parsing ==========
    private bool LoadAndBindTsv()
    {
        if (string.IsNullOrWhiteSpace(tsvFilePath))
        {
            Debug.LogError("TSV file path is empty");
            return false;
        }

        string path = tsvFilePath.Replace("\\", "/");
        if (!File.Exists(path))
        {
            Debug.LogError("TSV file not found: " + path);
            return false;
        }

        if (!TryParseTsv(path, out var frames))
            return false;

        _frames = frames;
        Log($"Loaded {_frames.Count} frames from TSV");

        if (!CacheAvatarBones())
            return false;

        return true;
    }

    private bool TryParseTsv(string path, out List<MoCapFrame> frames)
    {
        frames = new List<MoCapFrame>();
        string[] lines;

        try { lines = File.ReadAllLines(path); }
        catch (Exception ex)
        {
            Debug.LogError("Failed to read TSV: " + ex.Message);
            return false;
        }

        if (lines.Length < 11)
        {
            Debug.LogError("TSV too short (expected >= 11 rows)");
            return false;
        }

        // Extract metadata
        float fps = 300f;
        var metaDict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < 10; i++)
        {
            var parts = lines[i].Split('\t');
            if (parts.Length >= 2)
                metaDict[parts[0].Trim()] = parts[1].Trim();
        }
        if (metaDict.TryGetValue("FREQUENCY", out string freqStr) && float.TryParse(freqStr, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsedFps))
            fps = parsedFps;

        // Parse header row (row 10, index 9 in 0-based; but often row 11)
        int headerIdx = -1;
        for (int i = 9; i < Mathf.Min(20, lines.Length); i++)
        {
            if (lines[i].StartsWith("Frame\tTime", StringComparison.OrdinalIgnoreCase))
            {
                headerIdx = i;
                break;
            }
        }

        if (headerIdx < 0)
        {
            Debug.LogError("Could not find 'Frame\\tTime' header row");
            return false;
        }

        // Build column index map: marker name → (X col, Y col, Z col)
        var colMap = new Dictionary<string, (int x, int y, int z)>(StringComparer.OrdinalIgnoreCase);
        var headerParts = lines[headerIdx].Split('\t');
        for (int i = 2; i + 2 < headerParts.Length; i += 3)
        {
            string markerName = headerParts[i].TrimEnd(' ', '\t');
            if (markerName.EndsWith(" X", StringComparison.OrdinalIgnoreCase))
                markerName = markerName.Substring(0, markerName.Length - 2).Trim();
            else
                continue;

            if (!colMap.ContainsKey(markerName))
                colMap[markerName] = (i, i + 1, i + 2);
        }

        Log($"Parsed {colMap.Count} marker columns from TSV header");

        // Parse data rows
        for (int rowIdx = headerIdx + 1; rowIdx < lines.Length; rowIdx++)
        {
            string line = lines[rowIdx].Trim();
            if (string.IsNullOrEmpty(line)) continue;

            var cols = line.Split('\t');
            if (cols.Length < 2) continue;

            if (!int.TryParse(cols[0], out int frameNum)) continue;
            if (!float.TryParse(cols[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float timeVal)) continue;

            var frame = new MoCapFrame
            {
                frameIndex = frameNum,
                timeSeconds = timeVal,
                markerPositions = new Dictionary<string, Vector3>(StringComparer.OrdinalIgnoreCase)
            };

            // Parse marker positions
            foreach (var kvp in colMap)
            {
                string name = kvp.Key;
                (int xi, int yi, int zi) = kvp.Value;

                if (xi >= cols.Length || yi >= cols.Length || zi >= cols.Length) continue;

                if (!float.TryParse(cols[xi], NumberStyles.Float, CultureInfo.InvariantCulture, out float mx)) continue;
                if (!float.TryParse(cols[yi], NumberStyles.Float, CultureInfo.InvariantCulture, out float my)) continue;
                if (!float.TryParse(cols[zi], NumberStyles.Float, CultureInfo.InvariantCulture, out float mz)) continue;

                // Skip invalid (NaN) markers
                if (float.IsNaN(mx) || float.IsNaN(my) || float.IsNaN(mz)) continue;

                // Convert mm → m and apply coordinate system swap (Qualisys right-handed Y-up → Unity left-handed Y-up)
                // Unity.x = mocap.x, Unity.y = mocap.z, Unity.z = mocap.y
                Vector3 pos = new Vector3(mx, mz, my) * 0.001f;

                frame.markerPositions[name] = pos;
            }

            frames.Add(frame);
        }

        if (frames.Count == 0)
        {
            Debug.LogError("No valid frames parsed from TSV");
            return false;
        }

        Log($"Successfully parsed {frames.Count} frames");
        return true;
    }

    private bool CacheAvatarBones()
    {
        if (autoFindHips || hipsRoot == null)
        {
            hipsRoot = GetComponent<Transform>();
            // If no Animator on this object, try children
            var animator = GetComponent<Animator>();
            if (animator != null)
            {
                var hipsBone = animator.GetBoneTransform(HumanBodyBones.Hips);
                if (hipsBone != null)
                    hipsRoot = hipsBone;
            }
        }

        if (hipsRoot == null)
        {
            Debug.LogWarning("Hips root not found; will not apply root position");
        }

        Log("Avatar bones cached");
        return true;
    }

    // ========== Reference Pose (T-Pose) Binding ==========
    private bool EstablishReferencePose(int refFrameIdx)
    {
        if (refFrameIdx < 0 || refFrameIdx >= _frames.Count)
        {
            Debug.LogError($"Invalid reference frame index: {refFrameIdx} (range 0-{_frames.Count - 1})");
            return false;
        }

        MoCapFrame refFrame = _frames[refFrameIdx];
        _referenceBind = new Dictionary<string, Vector3>(refFrame.markerPositions);
        _bindHipsWorldPos = ComputeMarkerAverage(refFrame, "Q_WaistLFront", "Q_WaistL", "Q_WaistBack", "Q_WaistR", "Q_WaistRFront");

        Log($"Reference pose established from frame {refFrameIdx}");

        // Cache bind pose rotations from avatar (before any animation applied)
        var animator = GetComponent<Animator>();
        if (animator != null && animator.isHuman)
        {
            _bindLocalRotations = new Quaternion[(int)HumanBodyBones.LastBone];
            for (int i = 0; i < (int)HumanBodyBones.LastBone; i++)
            {
                var bone = animator.GetBoneTransform((HumanBodyBones)i);
                _bindLocalRotations[i] = bone != null ? bone.localRotation : Quaternion.identity;
            }
            Log("Bind pose rotations cached from Humanoid rig");
        }

        return true;
    }

    private void CacheBindSegments()
    {
        int N = BONE_MAPPINGS.Length;
        _bindSegmentLengths = new float[N];
        _bindSegmentDirs = new Vector3[N];

        // note: reference positions are computed per bone using centroid of mapped markers

        // Parent map for relevant bones
        var parent = new Dictionary<HumanBodyBones, HumanBodyBones>(){
            { HumanBodyBones.Spine, HumanBodyBones.Hips },
            { HumanBodyBones.Chest, HumanBodyBones.Spine },
            { HumanBodyBones.UpperChest, HumanBodyBones.Chest },
            { HumanBodyBones.Neck, HumanBodyBones.Chest },
            { HumanBodyBones.Head, HumanBodyBones.Neck },
            { HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftShoulder },
            { HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftUpperArm },
            { HumanBodyBones.LeftHand, HumanBodyBones.LeftLowerArm },
            { HumanBodyBones.RightUpperArm, HumanBodyBones.RightShoulder },
            { HumanBodyBones.RightLowerArm, HumanBodyBones.RightUpperArm },
            { HumanBodyBones.RightHand, HumanBodyBones.RightLowerArm },
            { HumanBodyBones.LeftUpperLeg, HumanBodyBones.Hips },
            { HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftUpperLeg },
            { HumanBodyBones.LeftFoot, HumanBodyBones.LeftLowerLeg },
            { HumanBodyBones.RightUpperLeg, HumanBodyBones.Hips },
            { HumanBodyBones.RightLowerLeg, HumanBodyBones.RightUpperLeg },
            { HumanBodyBones.RightFoot, HumanBodyBones.RightLowerLeg }
        };

        for (int i=0;i<BONE_MAPPINGS.Length;i++){
            var bm = BONE_MAPPINGS[i];
            Vector3 dir = Vector3.zero; float len=0f;
            if (TryGetReferenceBonePosition(bm.bone, out Vector3 childPos)){
                HumanBodyBones hb = bm.bone;
                if (parent.TryGetValue(hb, out HumanBodyBones pb) && TryGetReferenceBonePosition(pb, out Vector3 parentPos)){
                    dir = (childPos - parentPos);
                    len = dir.magnitude;
                    if (len>1e-6f) dir /= len; else dir = Vector3.zero;
                }
            }
            _bindSegmentDirs[i] = dir; _bindSegmentLengths[i]=len;
        }
    }

    private int DetectCalmFrame()
    {
        // Simple heuristic: find frame with lowest marker displacement from average position
        if (_frames.Count < 2) return 0;

        float minMotion = float.MaxValue;
        int calmFrameIdx = 0;

        for (int f = 0; f < _frames.Count; f++)
        {
            float motion = 0f;
            int count = 0;
            foreach (var pos in _frames[f].markerPositions.Values)
            {
                motion += pos.sqrMagnitude;
                count++;
            }
            if (count > 0) motion /= count;

            if (motion < minMotion)
            {
                minMotion = motion;
                calmFrameIdx = f;
            }
        }

        Log($"Auto-detected calm frame: {calmFrameIdx} (motion={minMotion})");
        return calmFrameIdx;
    }

    // ========== Frame Application & Bone Rotation Computation ==========
    private void ApplyFrame(int frameIdx)
    {
        if (frameIdx < 0 || frameIdx >= _frames.Count) return;

        MoCapFrame frame = _frames[frameIdx];

        // Compute root offset (hips) in world units relative to bind
        Vector3 hipsWorldPos = ComputeMarkerAverage(frame, "Q_WaistLFront", "Q_WaistL", "Q_WaistBack", "Q_WaistR", "Q_WaistRFront");
        Vector3 rootOffset = hipsWorldPos - _bindHipsWorldPos;

        // Apply root translation to the GameObject
        transform.position = hipsWorldPos - _bindHipsWorldPos;

        // For each mapped bone, compute current segment direction and apply world rotation
        for (int i = 0; i < BONE_MAPPINGS.Length; i++)
        {
            var bt = _boneTargets != null ? _boneTargets[i] : null;
            if (bt == null) continue;

            Vector3 bindDir = _bindSegmentDirs[i];
            if (bindDir == Vector3.zero) continue;

            Vector3 currDir = GetCurrentSegmentDir(frame, i);
            if (currDir == Vector3.zero) continue;

            Quaternion delta = Quaternion.FromToRotation(bindDir, currDir);
            bt.rotation = delta * _bindWorldRotations[i];
        }
    }

    private int FindMappingIndex(HumanBodyBones bone)
    {
        for (int i=0;i<BONE_MAPPINGS.Length;i++) if (BONE_MAPPINGS[i].bone==bone) return i; return -1;
    }

    private Vector3 GetCurrentSegmentDir(MoCapFrame frame, int mappingIndex)
    {
        var bm = BONE_MAPPINGS[mappingIndex];
        if (bm.markerNames.Count==0) return Vector3.zero;

        // child and parent positions are midpoints of their mapped markers
        Vector3 childPos = GetCurrentBonePosition(frame, bm.bone);
        if (childPos == Vector3.zero) return Vector3.zero;

        HumanBodyBones hb = bm.bone;
        HumanBodyBones parentBone = HumanBodyBones.Hips;
        // quick parent lookup
        switch(hb){
            case HumanBodyBones.Spine: parentBone = HumanBodyBones.Hips; break;
            case HumanBodyBones.Chest: parentBone = HumanBodyBones.Spine; break;
            case HumanBodyBones.UpperChest: parentBone = HumanBodyBones.Chest; break;
            case HumanBodyBones.Neck: parentBone = HumanBodyBones.Chest; break;
            case HumanBodyBones.Head: parentBone = HumanBodyBones.Neck; break;
            case HumanBodyBones.LeftUpperArm: parentBone = HumanBodyBones.LeftShoulder; break;
            case HumanBodyBones.LeftLowerArm: parentBone = HumanBodyBones.LeftUpperArm; break;
            case HumanBodyBones.LeftHand: parentBone = HumanBodyBones.LeftLowerArm; break;
            case HumanBodyBones.RightUpperArm: parentBone = HumanBodyBones.RightShoulder; break;
            case HumanBodyBones.RightLowerArm: parentBone = HumanBodyBones.RightUpperArm; break;
            case HumanBodyBones.RightHand: parentBone = HumanBodyBones.RightLowerArm; break;
            case HumanBodyBones.LeftUpperLeg: parentBone = HumanBodyBones.Hips; break;
            case HumanBodyBones.LeftLowerLeg: parentBone = HumanBodyBones.LeftUpperLeg; break;
            case HumanBodyBones.LeftFoot: parentBone = HumanBodyBones.LeftLowerLeg; break;
            case HumanBodyBones.RightUpperLeg: parentBone = HumanBodyBones.Hips; break;
            case HumanBodyBones.RightLowerLeg: parentBone = HumanBodyBones.RightUpperLeg; break;
            case HumanBodyBones.RightFoot: parentBone = HumanBodyBones.RightLowerLeg; break;
            default: parentBone = HumanBodyBones.Hips; break;
        }

        int pIdx = FindMappingIndex(parentBone);
        if (pIdx>=0){ Vector3 parentPos = GetCurrentBonePosition(frame, parentBone); if (parentPos!=Vector3.zero){ Vector3 dir = childPos - parentPos; if (dir.magnitude>1e-6f) return dir.normalized; else return Vector3.zero; }}
        return Vector3.zero;
    }

    private void ComputeAndWriteMuscles(MoCapFrame frame, ref HumanPose pose)
    {
        float[] muscles = pose.muscles;
        if (muscles==null || muscles.Length!=HumanTrait.MuscleCount) muscles = new float[HumanTrait.MuscleCount];

        // Optional: print muscle names once
        if (verboseLogs && Time.frameCount==0)
        {
            for (int i=0;i<HumanTrait.MuscleCount;i++) Debug.Log($"Muscle {i}: {HumanTrait.MuscleName[i]}");
        }

        // Spine & chest: use chest vs hips vector (centroid-based)
        int idxChest = FindMappingIndex(HumanBodyBones.Chest);
        Vector3 chestDir = Vector3.zero;
        if (idxChest >= 0)
        {
            Vector3 chestPos = GetCurrentBonePosition(frame, HumanBodyBones.Chest);
            Vector3 hipsPosCurr = GetCurrentBonePosition(frame, HumanBodyBones.Hips);
            if (chestPos != Vector3.zero && hipsPosCurr != Vector3.zero)
                chestDir = (chestPos - hipsPosCurr).normalized;
        }

        // Spine front-back -> use chest pitch relative to hips
        if (chestDir != Vector3.zero && idxChest >= 0)
        {
            float pitch = Vector3.SignedAngle(Vector3.ProjectOnPlane(_bindSegmentDirs[idxChest], Vector3.right), Vector3.ProjectOnPlane(chestDir, Vector3.right), Vector3.right);
            ApplyAngleMuscle(muscles, "Spine Front-Back", pitch);
        }

        // Spine left-right -> roll
        if (chestDir != Vector3.zero && idxChest >= 0)
        {
            float roll = Vector3.SignedAngle(Vector3.ProjectOnPlane(_bindSegmentDirs[idxChest], Vector3.forward), Vector3.ProjectOnPlane(chestDir, Vector3.forward), Vector3.forward);
            ApplyAngleMuscle(muscles, "Spine Left-Right", roll);
        }

        // Copy spine front-back into chest front-back (if available)
        CopyMuscle(muscles, "Chest Front-Back", "Spine Front-Back");

        // Upper arms
        int idxLUA = FindMappingIndex(HumanBodyBones.LeftUpperArm);
        if (idxLUA >= 0)
        {
            Vector3 bind = _bindSegmentDirs[idxLUA];
            Vector3 curr = GetCurrentSegmentDir(frame, idxLUA);
            if (bind != Vector3.zero && curr != Vector3.zero)
            {
                Vector3 axisUp = Vector3.Cross(bind, Vector3.up); if (axisUp.sqrMagnitude < 1e-6f) axisUp = Vector3.right; axisUp.Normalize();
                float upDown = Vector3.SignedAngle(bind, curr, axisUp);
                ApplyAngleMuscle(muscles, "Left Upper Arm Down-Up", upDown);
                Vector3 axisFwd = Vector3.Cross(bind, Vector3.forward); if (axisFwd.sqrMagnitude < 1e-6f) axisFwd = Vector3.up; axisFwd.Normalize();
                float fwdBack = Vector3.SignedAngle(bind, curr, axisFwd);
                ApplyAngleMuscle(muscles, "Left Upper Arm Front-Back", fwdBack);
            }
        }

        int idxRUA = FindMappingIndex(HumanBodyBones.RightUpperArm);
        if (idxRUA >= 0)
        {
            Vector3 bind = _bindSegmentDirs[idxRUA];
            Vector3 curr = GetCurrentSegmentDir(frame, idxRUA);
            if (bind != Vector3.zero && curr != Vector3.zero)
            {
                Vector3 axisUp = Vector3.Cross(bind, Vector3.up); if (axisUp.sqrMagnitude < 1e-6f) axisUp = Vector3.right; axisUp.Normalize();
                float upDown = Vector3.SignedAngle(bind, curr, axisUp);
                ApplyAngleMuscle(muscles, "Right Upper Arm Down-Up", upDown);
                Vector3 axisFwd = Vector3.Cross(bind, Vector3.forward); if (axisFwd.sqrMagnitude < 1e-6f) axisFwd = Vector3.up; axisFwd.Normalize();
                float fwdBack = Vector3.SignedAngle(bind, curr, axisFwd);
                ApplyAngleMuscle(muscles, "Right Upper Arm Front-Back", fwdBack);
            }
        }

        // Forearm stretch -> map based on length change
        int idxLFA = FindMappingIndex(HumanBodyBones.LeftLowerArm);
        if (idxLFA >= 0)
        {
            float bindLen = _bindSegmentLengths[idxLFA];
            Vector3 childPos = GetCurrentBonePosition(frame, HumanBodyBones.LeftLowerArm);
            Vector3 parentPos = GetCurrentBonePosition(frame, HumanBodyBones.LeftUpperArm);
            if (bindLen > 1e-6f && parentPos != Vector3.zero && childPos != Vector3.zero)
            {
                float currLen = (childPos - parentPos).magnitude;
                ApplyStretchMuscle(muscles, "Left Forearm Stretch", currLen, bindLen);
            }
        }
        int idxRFA = FindMappingIndex(HumanBodyBones.RightLowerArm);
        if (idxRFA >= 0)
        {
            float bindLen = _bindSegmentLengths[idxRFA];
            Vector3 childPos = GetCurrentBonePosition(frame, HumanBodyBones.RightLowerArm);
            Vector3 parentPos = GetCurrentBonePosition(frame, HumanBodyBones.RightUpperArm);
            if (bindLen > 1e-6f && parentPos != Vector3.zero && childPos != Vector3.zero)
            {
                float currLen = (childPos - parentPos).magnitude;
                ApplyStretchMuscle(muscles, "Right Forearm Stretch", currLen, bindLen);
            }
        }

        // Legs: upper leg front-back and knee stretch
        int idxLUL = FindMappingIndex(HumanBodyBones.LeftUpperLeg);
        if (idxLUL >= 0)
        {
            Vector3 bind = _bindSegmentDirs[idxLUL];
            Vector3 curr = GetCurrentSegmentDir(frame, idxLUL);
            if (bind != Vector3.zero && curr != Vector3.zero)
            {
                Vector3 axisFwd = Vector3.Cross(bind, Vector3.forward); if (axisFwd.sqrMagnitude < 1e-6f) axisFwd = Vector3.up;
                float ang = Vector3.SignedAngle(bind, curr, axisFwd);
                ApplyAngleMuscle(muscles, "Left Upper Leg Front-Back", ang);
            }
        }
        int idxLKL = FindMappingIndex(HumanBodyBones.LeftLowerLeg);
        if (idxLKL >= 0)
        {
            float bindLen = _bindSegmentLengths[idxLKL];
            Vector3 childPos = GetCurrentBonePosition(frame, HumanBodyBones.LeftLowerLeg);
            Vector3 parentPos = GetCurrentBonePosition(frame, HumanBodyBones.LeftUpperLeg);
            if (bindLen > 1e-6f && parentPos != Vector3.zero && childPos != Vector3.zero)
            {
                float currLen = (childPos - parentPos).magnitude;
                ApplyStretchMuscle(muscles, "Left Lower Leg Stretch", currLen, bindLen);
            }
        }
        int idxRUL = FindMappingIndex(HumanBodyBones.RightUpperLeg);
        if (idxRUL >= 0)
        {
            Vector3 bind = _bindSegmentDirs[idxRUL];
            Vector3 curr = GetCurrentSegmentDir(frame, idxRUL);
            if (bind != Vector3.zero && curr != Vector3.zero)
            {
                Vector3 axisFwd = Vector3.Cross(bind, Vector3.forward); if (axisFwd.sqrMagnitude < 1e-6f) axisFwd = Vector3.up;
                float ang = Vector3.SignedAngle(bind, curr, axisFwd);
                ApplyAngleMuscle(muscles, "Right Upper Leg Front-Back", ang);
            }
        }
        int idxRKL = FindMappingIndex(HumanBodyBones.RightLowerLeg);
        if (idxRKL >= 0)
        {
            float bindLen = _bindSegmentLengths[idxRKL];
            Vector3 childPos = GetCurrentBonePosition(frame, HumanBodyBones.RightLowerLeg);
            Vector3 parentPos = GetCurrentBonePosition(frame, HumanBodyBones.RightUpperLeg);
            if (bindLen > 1e-6f && parentPos != Vector3.zero && childPos != Vector3.zero)
            {
                float currLen = (childPos - parentPos).magnitude;
                ApplyStretchMuscle(muscles, "Right Lower Leg Stretch", currLen, bindLen);
            }
        }

        // Write back muscles
        pose.muscles = muscles;
    }

    private Quaternion ComputeBoneRotation(MoCapFrame frame, BoneMapping mapping)
    {
        // Get reference and current segment vectors from marker pairs
        List<Vector3> refSegments = new List<Vector3>();
        List<Vector3> currentSegments = new List<Vector3>();

        // Build segments: for each marker, find a paired child marker and compute direction vectors
        for (int i = 0; i < mapping.markerNames.Count; i++)
        {
            string markerName = mapping.markerNames[i];

            if (!_referenceBind.TryGetValue(markerName, out Vector3 refPos)) continue;
            if (!frame.markerPositions.TryGetValue(markerName, out Vector3 currentPos)) continue;

            // For single markers (e.g., Head), use direction from parent
            // For paired markers, compute the deviation
            if (i < mapping.markerNames.Count - 1)
            {
                string nextMarker = mapping.markerNames[i + 1];
                if (_referenceBind.TryGetValue(nextMarker, out Vector3 refNext) &&
                    frame.markerPositions.TryGetValue(nextMarker, out Vector3 currentNext))
                {
                    Vector3 refDir = (refNext - refPos).normalized;
                    Vector3 currDir = (currentNext - currentPos).normalized;
                    refSegments.Add(refDir);
                    currentSegments.Add(currDir);
                }
            }
        }

        if (refSegments.Count == 0)
            return Quaternion.identity;

        return BestFitRotation(refSegments, currentSegments);
    }

    private Quaternion BestFitRotation(List<Vector3> refDirs, List<Vector3> currentDirs)
    {
        if (refDirs.Count == 0) return Quaternion.identity;
        if (refDirs.Count == 1) return Quaternion.FromToRotation(refDirs[0], currentDirs[0]);

        // Build orthonormal bases from first two directions
        Vector3 r0 = refDirs[0].normalized;
        Vector3 r1 = refDirs[1].normalized;
        Vector3 c0 = currentDirs[0].normalized;
        Vector3 c1 = currentDirs[1].normalized;

        // Check for singularities
        if (Vector3.Dot(r0, r1) > 0.99f || Vector3.Dot(c0, c1) > 0.99f)
            return Quaternion.FromToRotation(r0, c0);

        Vector3 rz = Vector3.Cross(r0, r1).normalized;
        Vector3 ry = Vector3.Cross(rz, r0).normalized;

        Vector3 cz = Vector3.Cross(c0, c1).normalized;
        Vector3 cy = Vector3.Cross(cz, c0).normalized;

        // Construct rotation matrix
        Matrix4x4 from = Matrix4x4.identity;
        from.SetColumn(0, new Vector4(r0.x, r0.y, r0.z, 0));
        from.SetColumn(1, new Vector4(ry.x, ry.y, ry.z, 0));
        from.SetColumn(2, new Vector4(rz.x, rz.y, rz.z, 0));

        Matrix4x4 to = Matrix4x4.identity;
        to.SetColumn(0, new Vector4(c0.x, c0.y, c0.z, 0));
        to.SetColumn(1, new Vector4(cy.x, cy.y, cy.z, 0));
        to.SetColumn(2, new Vector4(cz.x, cz.y, cz.z, 0));

        Matrix4x4 rot = to * from.transpose;
        return rot.rotation;
    }

    // ========== Utility Methods ==========
    private Vector3 ComputeMarkerAverage(MoCapFrame frame, params string[] markerNames)
    {
        Vector3 sum = Vector3.zero;
        int count = 0;
        foreach (string name in markerNames)
        {
            if (frame.markerPositions.TryGetValue(name, out Vector3 pos))
            {
                sum += pos;
                count++;
            }
        }
        return count > 0 ? sum / count : Vector3.zero;
    }

    // ----------------- Helpers for muscle mapping and marker centroids -----------------
    private Dictionary<string, int> _muscleLookup;

    private void BuildMuscleLookup()
    {
        _muscleLookup = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < HumanTrait.MuscleCount; i++)
        {
            string name = HumanTrait.MuscleName[i];
            _muscleLookup[NormalizeKey(name)] = i;
        }
    }

    private static string NormalizeKey(string value)
    {
        if (string.IsNullOrEmpty(value)) return string.Empty;
        var sb = new System.Text.StringBuilder(value.Length);
        foreach (char c in value)
            if (char.IsLetterOrDigit(c)) sb.Append(char.ToLowerInvariant(c));
        return sb.ToString();
    }

    private bool TryGetMuscleIndex(string muscleName, out int index)
    {
        index = -1;
        if (_muscleLookup == null) return false;
        return _muscleLookup.TryGetValue(NormalizeKey(muscleName), out index);
    }

    private float NormalizeDegreesToMuscle(int muscleIndex, float degrees)
    {
        float min = HumanTrait.GetMuscleDefaultMin(muscleIndex);
        float max = HumanTrait.GetMuscleDefaultMax(muscleIndex);
        if (Mathf.Approximately(min, max)) return 0f;

        if (degrees >= 0f)
        {
            if (Mathf.Approximately(max, 0f)) return 0f;
            return Mathf.Clamp(degrees / max, -1f, 1f);
        }
        else
        {
            if (Mathf.Approximately(min, 0f)) return 0f;
            return Mathf.Clamp(degrees / -min, -1f, 1f);
        }
    }

    private void ApplyAngleMuscle(float[] muscles, string muscleName, float degrees)
    {
        if (!TryGetMuscleIndex(muscleName, out int idx)) return;
        if (idx < 0 || idx >= muscles.Length) return;
        muscles[idx] = NormalizeDegreesToMuscle(idx, degrees);
    }

    private void ApplyStretchMuscle(float[] muscles, string muscleName, float currentLength, float bindLength)
    {
        if (!TryGetMuscleIndex(muscleName, out int idx)) return;
        if (idx < 0 || idx >= muscles.Length || bindLength <= 1e-6f) return;
        muscles[idx] = Mathf.Clamp((currentLength - bindLength) / bindLength, -1f, 1f);
    }

    private void CopyMuscle(float[] muscles, string targetName, string sourceName)
    {
        if (!TryGetMuscleIndex(targetName, out int t)) return;
        if (!TryGetMuscleIndex(sourceName, out int s)) return;
        if (t < 0 || t >= muscles.Length || s < 0 || s >= muscles.Length) return;
        muscles[t] = muscles[s];
    }

    private bool TryGetReferenceBonePosition(HumanBodyBones bone, out Vector3 position)
    {
        position = Vector3.zero;
        int idx = FindMappingIndex(bone);
        if (idx < 0) return false;
        var bm = BONE_MAPPINGS[idx];
        if (bm.markerNames == null || bm.markerNames.Count == 0) return false;
        Vector3 sum = Vector3.zero; int count = 0;
        foreach (var m in bm.markerNames)
        {
            if (_referenceBind != null && _referenceBind.TryGetValue(m, out Vector3 p)) { sum += p; count++; }
        }
        if (count == 0) return false;
        position = sum / count; return true;
    }

    private bool TryGetCurrentBonePosition(MoCapFrame frame, HumanBodyBones bone, out Vector3 position)
    {
        position = Vector3.zero;
        int idx = FindMappingIndex(bone);
        if (idx < 0) return false;
        var bm = BONE_MAPPINGS[idx];
        if (bm.markerNames == null || bm.markerNames.Count == 0) return false;
        Vector3 sum = Vector3.zero; int count = 0;
        foreach (var m in bm.markerNames)
        {
            if (frame.markerPositions != null && frame.markerPositions.TryGetValue(m, out Vector3 p)) { sum += p; count++; }
        }
        if (count == 0) return false;
        position = sum / count; return true;
    }

    private Vector3 GetCurrentBonePosition(MoCapFrame frame, HumanBodyBones bone)
    {
        return TryGetCurrentBonePosition(frame, bone, out Vector3 pos) ? pos : Vector3.zero;
    }

    private void Log(string message)
    {
        if (verboseLogs)
            Debug.Log("[TsvRetargetPlayer] " + message);
    }

    // ========== Playback Control (Public API) ==========
    public void Play() { _isPlaying = true; }
    public void Pause() { _isPlaying = false; }
    public void Stop() { _isPlaying = false; _playbackTime = 0f; _currentFrameIndex = 0; }
    public void Seek(float timeSeconds) { _playbackTime = Mathf.Clamp(timeSeconds, 0f, _frames.Count / 300f); }
    public int GetFrameCount() => _frames.Count;
    public int GetCurrentFrameIndex() => _currentFrameIndex;
    public float GetDuration() => _frames.Count / 300f;
}
