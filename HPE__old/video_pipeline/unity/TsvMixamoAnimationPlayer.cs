using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

[Serializable]
public class TsvBoneBinding
{
    public string jointName;
    public Transform bone;
}

public class TsvMixamoAnimationPlayer : MonoBehaviour
{
    private readonly struct JointDef
    {
        public JointDef(string name, int parentIndex)
        {
            Name = name;
            ParentIndex = parentIndex;
        }

        public string Name { get; }
        public int ParentIndex { get; }
    }

    public enum SourceCoordinateSystem
    {
        UnityYUpLeftHanded,
        SourceYDownZForward,
        // Source where X axis points forward (direction of throw) and Z is up
        SourceXForwardZUp,
    }

    private static readonly JointDef[] TargetJoints =
    {
        new JointDef("Hips", -1),
        new JointDef("RightUpLeg", 0),
        new JointDef("RightLeg", 1),
        new JointDef("RightFoot", 2),
        new JointDef("RightToeBase", 3),
        new JointDef("LeftUpLeg", 0),
        new JointDef("LeftLeg", 4),
        new JointDef("LeftFoot", 5),
        new JointDef("LeftToeBase", 6),
        new JointDef("Spine", 0),
        new JointDef("Spine1", 7),
        new JointDef("Spine2", 8),
        new JointDef("Neck", 9),
        new JointDef("Head", 10),
        new JointDef("LeftShoulder", 9),
        new JointDef("LeftArm", 12),
        new JointDef("LeftForeArm", 13),
        new JointDef("LeftHand", 14),
        new JointDef("RightShoulder", 9),
        new JointDef("RightArm", 16),
        new JointDef("RightForeArm", 17),
        new JointDef("RightHand", 18),
    };

    private static readonly Dictionary<string, string[]> JointAliases = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
    {
        { "Hips", new[] { "hips", "pelvis", "mixamorighips" } },
        { "RightUpLeg", new[] { "rightupleg", "mixamorigrightupleg", "rightthigh" } },
        { "RightLeg", new[] { "rightleg", "mixamorigrightleg", "rightcalf" } },
        { "RightFoot", new[] { "rightfoot", "mixamorigrightfoot" } },
        { "RightToeBase", new[] { "righttoebase", "mixamorigrighttoebase", "righttoe", "righttoes" } },
        { "LeftUpLeg", new[] { "leftupleg", "mixamorigleftupleg", "leftthigh" } },
        { "LeftLeg", new[] { "leftleg", "mixamorigleftleg", "leftcalf" } },
        { "LeftFoot", new[] { "leftfoot", "mixamorigleftfoot" } },
        { "LeftToeBase", new[] { "lefttoebase", "mixamoriglefttoebase", "lefttoe", "lefttoes" } },
        { "Spine", new[] { "spine", "mixamorigspine" } },
        { "Spine1", new[] { "spine1", "mixamorigspine1", "spine01" } },
        { "Spine2", new[] { "spine2", "mixamorigspine2", "chest" } },
        { "Neck", new[] { "neck", "mixamorigneck" } },
        { "Head", new[] { "head", "mixamorighead" } },
        { "LeftShoulder", new[] { "leftshoulder", "mixamorigleftshoulder" } },
        { "LeftArm", new[] { "leftarm", "mixamorigleftarm", "leftupperarm", "leftshoulder" } },
        { "LeftForeArm", new[] { "leftforearm", "mixamorigleftforearm", "leftlowerarm", "leftelbow" } },
        { "LeftHand", new[] { "lefthand", "mixamoriglefthand", "leftwrist" } },
        { "RightShoulder", new[] { "rightshoulder", "mixamorigrightshoulder" } },
        { "RightArm", new[] { "rightarm", "mixamorigrightarm", "rightupperarm", "rightshoulder" } },
        { "RightForeArm", new[] { "rightforearm", "mixamorigrightforearm", "rightlowerarm", "rightelbow" } },
        { "RightHand", new[] { "righthand", "mixamorigrighthand", "rightwrist" } },
    };

    private static readonly string[] RequiredMarkers =
    {
        "Q_HeadL",
        "Q_HeadR",
        "Q_HeadFront",
        "Q_Chest",
        "Q_SpineThoracic2",
        "Q_SpineThoracic12",
        "Q_LShoulderTop",
        "Q_LArm",
        "Q_LElbowOut",
        "Q_LElbowIn",
        "Q_LWristIn",
        "Q_LWristOut",
        "Q_LHand2",
        "Q_RShoulderTop",
        "Q_RArm",
        "Q_RElbowOut",
        "Q_RElbowIn",
        "Q_RWristIn",
        "Q_RWristOut",
        "Q_RHand2",
        "Q_WaistLFront",
        "Q_WaistL",
        "Q_WaistBack",
        "Q_WaistR",
        "Q_WaistRFront",
        "Q_LThighFrontLow",
        "Q_LKneeOut",
        "Q_LShinFrontHigh",
        "Q_LAnkleOut",
        "Q_LForefoot2",
        "Q_RThighFrontLow",
        "Q_RKneeOut",
        "Q_RShinFrontHigh",
        "Q_RAnkleOut",
        "Q_RForefoot2",
    };

    private static readonly int[][] ChildrenPerJoint = BuildChildren();

    [Header("Input")]
    public bool loadOnStart = true;
    public string tsvFilePath;

    [Header("Bones")]
    public Transform hips;
    public bool autoMapByName = true;
    public List<TsvBoneBinding> manualBindings = new List<TsvBoneBinding>();

    [Header("Playback")]
    public bool playOnStart = true;
    public bool loop = true;
    public float playbackSpeed = 1f;
    public bool interpolate = true;
    public bool applyRootInWorldSpace;

    [Header("Source")]
    public SourceCoordinateSystem sourceCoordinateSystem = SourceCoordinateSystem.SourceYDownZForward;
    public float sourceUnitsToMeters = 0.001f;
    [Header("Calibration")]
    public bool enableCalibration = true;

    [Header("Solve")]
    public int calibrationFrames = 10;

    [Header("Diagnostics")]
    public bool verboseLogs;
    public bool drawDebugMarkers = false;
    public float debugMarkerScale = 0.02f;
    [SerializeField] private int loadedFrameCount;
    [SerializeField] private float detectedFps;
    [SerializeField] private int mappedBoneCount;
    [SerializeField] private string missingMarkersSummary;

    private Transform[] _jointTargets;
    private Quaternion[] _bindLocalRotations;
    private Quaternion[][] _solvedLocalRotations;
    private Vector3[] _rootPositions;
    // Per-frame joint positions computed from markers (after calibration offsets applied)
    private Vector3[,] _markerJointsPerFrame;
    private Vector3[] _calibrationOffsets;
    private float _secondsPerFrame;
    private float _time;
    private bool _isPlaying;
    private bool _isLoaded;
    private Vector3 _hipsStartLocalPosition;
    private Vector3 _hipsStartWorldPosition;

    private void Start()
    {
        if (!loadOnStart)
        {
            return;
        }

        if (!LoadFromTsv())
        {
            return;
        }

        _isPlaying = playOnStart;
    }

    private void Update()
    {
        if (!_isPlaying || !_isLoaded || _solvedLocalRotations == null || _solvedLocalRotations.Length == 0)
        {
            return;
        }

        _time += Time.deltaTime * Mathf.Max(0f, playbackSpeed);
        float clipDuration = _solvedLocalRotations.Length * _secondsPerFrame;
        if (clipDuration <= 1e-6f)
        {
            return;
        }

        if (loop)
        {
            _time %= clipDuration;
        }
        else if (_time >= clipDuration)
        {
            _time = clipDuration - _secondsPerFrame;
            _isPlaying = false;
        }

        float frameFloat = Mathf.Clamp(_time / _secondsPerFrame, 0f, _solvedLocalRotations.Length - 1);
        int i0 = Mathf.FloorToInt(frameFloat);
        int i1 = Mathf.Min(i0 + 1, _solvedLocalRotations.Length - 1);
        float alpha = interpolate ? (frameFloat - i0) : 0f;

        ApplyFrame(i0, i1, alpha);
    }

    public void Play()
    {
        _isPlaying = true;
    }

    public void Pause()
    {
        _isPlaying = false;
    }

    public void StopAndReset()
    {
        _isPlaying = false;
        _time = 0f;
        if (_isLoaded && _solvedLocalRotations != null && _solvedLocalRotations.Length > 0)
        {
            ApplyFrame(0, 0, 0f);
        }
    }

    [ContextMenu("Reload TSV")]
    public bool LoadFromTsv()
    {
        _isLoaded = false;
        loadedFrameCount = 0;
        detectedFps = 0f;
        mappedBoneCount = 0;
        missingMarkersSummary = string.Empty;

        if (string.IsNullOrWhiteSpace(tsvFilePath))
        {
            Debug.LogError("TsvMixamoAnimationPlayer: tsvFilePath is empty.");
            return false;
        }

        // Normalize path: convert backslashes to forward slashes and handle spaces/special chars
        string normalizedPath = tsvFilePath.Replace("\\", "/");
        
        if (!File.Exists(normalizedPath))
        {
            Debug.LogError($"TsvMixamoAnimationPlayer: TSV file not found at: {normalizedPath}");
            Debug.LogError($"  Original path was: {tsvFilePath}");
            Debug.LogError($"  Please verify the file exists and use forward slashes or full escaped path.");
            return false;
        }

        if (!TryParseTsv(normalizedPath, out float fps, out Vector3[,] markerFrames, out string[] markerNames))
        {
            return false;
        }

        // Build bone map and cache bind pose before solving
        BuildBoneMap();
        CacheBindPose();

        // NOTE: do not compute calibration offsets from the avatar bind pose here -
        // that biases the solver toward the scene T-pose. Calibration (if used)
        // must be derived from the marker data itself and/or applied after a
        // valid solve. For now, skip automatic bind-pose calibration.
        if (!TrySolveAnimation(markerFrames, markerNames, out _rootPositions, out _solvedLocalRotations))
        {
            return false;
        }

        detectedFps = fps;
        loadedFrameCount = _solvedLocalRotations.Length;
        _secondsPerFrame = 1f / Mathf.Max(1e-3f, fps);
        _time = 0f;

        // Do not re-cache the bind pose here; it must remain the original bind
        // rotations that we multiply by solved local quaternions.

        // Auto-enable debug markers when verbose logging is on to help debugging
        if (verboseLogs)
        {
            drawDebugMarkers = true;
        }

        if (_solvedLocalRotations.Length > 0)
        {
            Debug.Log($"TsvMixamoAnimationPlayer: solved {_solvedLocalRotations.Length} frames, mapped bones: {mappedBoneCount}/{TargetJoints.Length}");
            int identityCount = 0;
            for (int j = 0; j < TargetJoints.Length; j++)
            {
                if (_solvedLocalRotations[0][j] == Quaternion.identity) identityCount++;
            }
            Debug.Log($"TsvMixamoAnimationPlayer: frame0 identity local rotations: {identityCount}/{TargetJoints.Length}");
            ApplyFrame(0, 0, 0f);
        }

        if (hips != null)
        {
            _hipsStartLocalPosition = hips.localPosition;
            _hipsStartWorldPosition = hips.position;
        }

        if (_solvedLocalRotations.Length > 0)
        {
            ApplyFrame(0, 0, 0f);
        }

        _isLoaded = true;
        return true;
    }

    private bool TryParseTsv(string path, out float fps, out Vector3[,] markerFrames, out string[] markerNames)
    {
        fps = 0f;
        markerFrames = null;
        markerNames = null;

        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            Debug.LogError($"TsvMixamoAnimationPlayer: TSV file not found: {path}");
            return false;
        }

        string[] lines = File.ReadAllLines(path);
        if (lines.Length < 4)
        {
            Debug.LogError($"TsvMixamoAnimationPlayer: TSV is too short ({lines.Length} lines).");
            return false;
        }

        if (verboseLogs)
        {
            Debug.Log($"TsvMixamoAnimationPlayer: Loaded {lines.Length} lines from {path}");
        }

        var markerNameList = new List<string>();
        int headerRowIndex = -1;
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            if (line.StartsWith("FREQUENCY", StringComparison.OrdinalIgnoreCase))
            {
                string[] tokens = SplitTabs(line);
                if (tokens.Length > 1 && float.TryParse(tokens[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float parsedFps))
                {
                    fps = parsedFps;
                    if (verboseLogs)
                    {
                        Debug.Log($"TsvMixamoAnimationPlayer: Found frequency {fps} Hz");
                    }
                }
            }
            else if (line.StartsWith("MARKER_NAMES", StringComparison.OrdinalIgnoreCase))
            {
                string[] tokens = SplitTabs(line);
                if (verboseLogs)
                {
                    Debug.Log($"TsvMixamoAnimationPlayer: MARKER_NAMES line has {tokens.Length} tokens");
                }

                for (int tokenIndex = 1; tokenIndex < tokens.Length; tokenIndex++)
                {
                    if (!string.IsNullOrWhiteSpace(tokens[tokenIndex]))
                    {
                        markerNameList.Add(tokens[tokenIndex].Trim());
                    }
                }

                if (verboseLogs)
                {
                    Debug.Log($"TsvMixamoAnimationPlayer: Parsed {markerNameList.Count} marker names");
                }
            }
            else if (line.StartsWith("Frame\tTime", StringComparison.OrdinalIgnoreCase))
            {
                headerRowIndex = i;
                break;
            }
        }

        if (fps <= 1e-6f)
        {
            fps = 30f;
        }

        if (headerRowIndex < 0)
        {
            Debug.LogError("TsvMixamoAnimationPlayer: could not find Frame/Time header row.");
            return false;
        }

        if (markerNameList.Count == 0)
        {
            Debug.LogError($"TsvMixamoAnimationPlayer: MARKER_NAMES header is empty or not found. First few lines: {string.Join(" | ", System.Linq.Enumerable.Take(lines, 5))}");
            return false;
        }

        markerNames = markerNameList.ToArray();
        var markerToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < markerNames.Length; i++)
        {
            markerToIndex[markerNames[i]] = i;
        }

        var missing = new List<string>();
        for (int i = 0; i < RequiredMarkers.Length; i++)
        {
            if (!markerToIndex.ContainsKey(RequiredMarkers[i]))
            {
                missing.Add(RequiredMarkers[i]);
            }
        }

        if (missing.Count > 0)
        {
            missingMarkersSummary = string.Join(", ", missing);
            Debug.LogError($"TsvMixamoAnimationPlayer: missing required TSV markers: {missingMarkersSummary}");
            return false;
        }

        string[] headerColumns = SplitTabs(lines[headerRowIndex]);
        var markerColumnBase = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 2; i < headerColumns.Length; i++)
        {
            string col = headerColumns[i];
            if (string.IsNullOrWhiteSpace(col) || !col.EndsWith(" X", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string markerName = col.Substring(0, col.Length - 2).Trim();
            if (!markerColumnBase.ContainsKey(markerName))
            {
                markerColumnBase[markerName] = i;
            }
        }

        int dataStart = headerRowIndex + 1;
        var frameLines = new List<string>();
        for (int i = dataStart; i < lines.Length; i++)
        {
            if (!string.IsNullOrWhiteSpace(lines[i]))
            {
                frameLines.Add(lines[i]);
            }
        }

        if (frameLines.Count == 0)
        {
            Debug.LogError("TsvMixamoAnimationPlayer: no frame data rows found.");
            return false;
        }

        markerFrames = new Vector3[frameLines.Count, markerNames.Length];
        for (int frameIndex = 0; frameIndex < frameLines.Count; frameIndex++)
        {
            string[] columns = SplitTabs(frameLines[frameIndex]);
            for (int markerIndex = 0; markerIndex < markerNames.Length; markerIndex++)
            {
                string markerName = markerNames[markerIndex];
                if (!markerColumnBase.TryGetValue(markerName, out int baseCol))
                {
                    continue;
                }

                if (baseCol + 2 >= columns.Length)
                {
                    continue;
                }

                if (!TryParseFloat(columns[baseCol], out float xMm) ||
                    !TryParseFloat(columns[baseCol + 1], out float yMm) ||
                    !TryParseFloat(columns[baseCol + 2], out float zMm))
                {
                    continue;
                }

                Vector3 pos = new Vector3(xMm, yMm, zMm) * sourceUnitsToMeters;
                markerFrames[frameIndex, markerIndex] = ConvertPositionBasis(pos);
            }
        }

        // keep raw marker frames accessible for calibration/debug
        _markerJointsPerFrame = new Vector3[markerFrames.GetLength(0), TargetJoints.Length];

        return true;
    }

    private bool TrySolveAnimation(Vector3[,] markerFrames, string[] markerNames, out Vector3[] rootPositions, out Quaternion[][] localRotations)
    {
        rootPositions = null;
        localRotations = null;

        int frameCount = markerFrames.GetLength(0);
        int markerCount = markerFrames.GetLength(1);
        if (frameCount == 0 || markerCount == 0)
        {
            Debug.LogError("TsvMixamoAnimationPlayer: marker frame matrix is empty.");
            return false;
        }

        var markerToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < markerNames.Length; i++)
        {
            markerToIndex[markerNames[i]] = i;
        }
        Vector3[,] jointsPerFrame = new Vector3[frameCount, TargetJoints.Length];
        for (int frame = 0; frame < frameCount; frame++)
        {
            if (!TryBuildJointPositions(frame, markerFrames, markerToIndex, jointsPerFrame))
            {
                Debug.LogError($"TsvMixamoAnimationPlayer: failed to build joints for frame {frame}. Required markers may be invalid.");
                return false;
            }
        }

        // store marker-derived joints for debugging/inspection
        _markerJointsPerFrame = jointsPerFrame;

        Vector3[] restOffsets = ComputeRestOffsets(jointsPerFrame, Mathf.Clamp(calibrationFrames, 1, frameCount));
        rootPositions = new Vector3[frameCount];
        localRotations = new Quaternion[frameCount][];

        for (int frame = 0; frame < frameCount; frame++)
        {
            var worldRotations = new Quaternion[TargetJoints.Length];
            for (int j = 0; j < TargetJoints.Length; j++)
            {
                worldRotations[j] = Quaternion.identity;
            }

            for (int jointIndex = 0; jointIndex < TargetJoints.Length; jointIndex++)
            {
                int[] children = ChildrenPerJoint[jointIndex];
                var restDirs = new List<Vector3>(children.Length);
                var poseDirs = new List<Vector3>(children.Length);

                for (int k = 0; k < children.Length; k++)
                {
                    int childIndex = children[k];
                    Vector3 restDir = restOffsets[childIndex];
                    Vector3 poseDir = jointsPerFrame[frame, childIndex] - jointsPerFrame[frame, jointIndex];
                    if (restDir.sqrMagnitude < 1e-8f || poseDir.sqrMagnitude < 1e-8f)
                    {
                        continue;
                    }

                    restDirs.Add(restDir);
                    poseDirs.Add(poseDir);
                }

                if (restDirs.Count > 0)
                {
                    worldRotations[jointIndex] = BestFitRotation(restDirs, poseDirs);
                }
                else
                {
                    worldRotations[jointIndex] = Quaternion.identity;
                }
            }

            var localFrame = new Quaternion[TargetJoints.Length];
            for (int jointIndex = 0; jointIndex < TargetJoints.Length; jointIndex++)
            {
                int parent = TargetJoints[jointIndex].ParentIndex;
                if (parent < 0)
                {
                    localFrame[jointIndex] = worldRotations[jointIndex];
                }
                else
                {
                    localFrame[jointIndex] = Quaternion.Inverse(worldRotations[parent]) * worldRotations[jointIndex];
                }
            }

            localRotations[frame] = localFrame;
            rootPositions[frame] = jointsPerFrame[frame, 0];
        }

        if (rootPositions.Length > 0)
        {
            Vector3 baseRoot = rootPositions[0];
            for (int i = 0; i < rootPositions.Length; i++)
            {
                rootPositions[i] -= baseRoot;
            }
        }

        return true;
    }

    private bool TryBuildJointPositions(int frame, Vector3[,] markerFrames, Dictionary<string, int> markerToIndex, Vector3[,] outJoints)
    {
        Vector3 Head() => Avg(GetMarker(frame, markerFrames, markerToIndex, "Q_HeadL"), GetMarker(frame, markerFrames, markerToIndex, "Q_HeadR"), GetMarker(frame, markerFrames, markerToIndex, "Q_HeadFront"));
        Vector3 HipsCenter() => Avg(
            GetMarker(frame, markerFrames, markerToIndex, "Q_WaistLFront"),
            GetMarker(frame, markerFrames, markerToIndex, "Q_WaistL"),
            GetMarker(frame, markerFrames, markerToIndex, "Q_WaistBack"),
            GetMarker(frame, markerFrames, markerToIndex, "Q_WaistR"),
            GetMarker(frame, markerFrames, markerToIndex, "Q_WaistRFront")
        );

        Vector3 LeftWrist() => Avg(GetMarker(frame, markerFrames, markerToIndex, "Q_LWristIn"), GetMarker(frame, markerFrames, markerToIndex, "Q_LWristOut"));
        Vector3 RightWrist() => Avg(GetMarker(frame, markerFrames, markerToIndex, "Q_RWristIn"), GetMarker(frame, markerFrames, markerToIndex, "Q_RWristOut"));
        Vector3 LeftElbow() => Avg(GetMarker(frame, markerFrames, markerToIndex, "Q_LElbowOut"), GetMarker(frame, markerFrames, markerToIndex, "Q_LElbowIn"));
        Vector3 RightElbow() => Avg(GetMarker(frame, markerFrames, markerToIndex, "Q_RElbowOut"), GetMarker(frame, markerFrames, markerToIndex, "Q_RElbowIn"));
        Vector3 LeftFoot() => GetMarker(frame, markerFrames, markerToIndex, "Q_LAnkleOut");
        Vector3 RightFoot() => GetMarker(frame, markerFrames, markerToIndex, "Q_RAnkleOut");
        Vector3 LeftToe() => GetMarker(frame, markerFrames, markerToIndex, "Q_LForefoot2");
        Vector3 RightToe() => GetMarker(frame, markerFrames, markerToIndex, "Q_RForefoot2");

        try
        {
            outJoints[frame, 0] = HipsCenter();
            outJoints[frame, 1] = GetMarker(frame, markerFrames, markerToIndex, "Q_RThighFrontLow");
            outJoints[frame, 2] = GetMarker(frame, markerFrames, markerToIndex, "Q_RKneeOut");
            outJoints[frame, 3] = RightFoot();
            outJoints[frame, 4] = RightToe();
            outJoints[frame, 5] = GetMarker(frame, markerFrames, markerToIndex, "Q_LThighFrontLow");
            outJoints[frame, 6] = GetMarker(frame, markerFrames, markerToIndex, "Q_LKneeOut");
            outJoints[frame, 7] = LeftFoot();
            outJoints[frame, 8] = LeftToe();
            outJoints[frame, 9] = GetMarker(frame, markerFrames, markerToIndex, "Q_WaistBack");
            outJoints[frame, 10] = GetMarker(frame, markerFrames, markerToIndex, "Q_SpineThoracic12");
            outJoints[frame, 11] = GetMarker(frame, markerFrames, markerToIndex, "Q_Chest");
            outJoints[frame, 12] = Avg(outJoints[frame, 11], Head());
            outJoints[frame, 13] = Head();
            outJoints[frame, 14] = GetMarker(frame, markerFrames, markerToIndex, "Q_LShoulderTop");
            outJoints[frame, 15] = GetMarker(frame, markerFrames, markerToIndex, "Q_LArm");
            outJoints[frame, 16] = LeftElbow();
            outJoints[frame, 17] = LeftWrist();
            outJoints[frame, 18] = GetMarker(frame, markerFrames, markerToIndex, "Q_RShoulderTop");
            outJoints[frame, 19] = GetMarker(frame, markerFrames, markerToIndex, "Q_RArm");
            outJoints[frame, 20] = RightElbow();
            outJoints[frame, 21] = RightWrist();
            // apply calibration offsets if available
            if (_calibrationOffsets != null && _calibrationOffsets.Length == TargetJoints.Length)
            {
                for (int j = 0; j < TargetJoints.Length; j++)
                {
                    outJoints[frame, j] += _calibrationOffsets[j];
                }
            }

            return true;
        }
        catch (Exception ex)
        {
            if (verboseLogs)
            {
                Debug.LogError($"TsvMixamoAnimationPlayer: joint build failure at frame {frame}: {ex.Message}");
            }

            return false;
        }
    }

    private void BuildBoneMap()
    {
        var map = new Dictionary<string, Transform>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < manualBindings.Count; i++)
        {
            var binding = manualBindings[i];
            if (binding == null || binding.bone == null || string.IsNullOrWhiteSpace(binding.jointName))
            {
                continue;
            }

            map[NormalizeBoneName(binding.jointName)] = binding.bone;
        }

        if (autoMapByName)
        {
            foreach (Transform t in GetComponentsInChildren<Transform>(true))
            {
                string key = NormalizeBoneName(t.name);
                if (!map.ContainsKey(key))
                {
                    map[key] = t;
                }
            }
        }

        _jointTargets = new Transform[TargetJoints.Length];
        mappedBoneCount = 0;
        for (int i = 0; i < TargetJoints.Length; i++)
        {
            string jointName = TargetJoints[i].Name;
            if (TryFindBone(map, jointName, out Transform bone))
            {
                _jointTargets[i] = bone;
                mappedBoneCount++;
            }
            else if (verboseLogs)
            {
                Debug.LogWarning($"TsvMixamoAnimationPlayer: could not map bone for {jointName}. It will remain bind pose.");
            }
        }

        if (hips == null && _jointTargets.Length > 0)
        {
            hips = _jointTargets[0];
        }

        if (verboseLogs)
        {
            var mapped = new List<string>();
            for (int i = 0; i < _jointTargets.Length; i++)
            {
                if (_jointTargets[i] != null)
                    mapped.Add($"{TargetJoints[i].Name} -> {_jointTargets[i].name}");
                else
                    mapped.Add($"{TargetJoints[i].Name} -> <missing>");
            }
            Debug.Log($"TsvMixamoAnimationPlayer: mapped bones: {string.Join(", ", mapped)}");
        }
    }

    private void CacheBindPose()
    {
        if (_jointTargets == null)
        {
            return;
        }

        _bindLocalRotations = new Quaternion[_jointTargets.Length];
        for (int i = 0; i < _jointTargets.Length; i++)
        {
            _bindLocalRotations[i] = _jointTargets[i] != null ? _jointTargets[i].localRotation : Quaternion.identity;
        }
    }

    private void ApplyFrame(int frameA, int frameB, float t)
    {
        if (_solvedLocalRotations == null || _rootPositions == null || _jointTargets == null || _bindLocalRotations == null)
        {
            return;
        }

        if (frameA < 0 || frameA >= _solvedLocalRotations.Length || frameB < 0 || frameB >= _solvedLocalRotations.Length)
        {
            return;
        }

        ApplyRoot(frameA, frameB, t);

        for (int j = 0; j < _jointTargets.Length; j++)
        {
            Transform bone = _jointTargets[j];
            if (bone == null)
            {
                if (verboseLogs)
                {
                    Debug.LogWarning($"TsvMixamoAnimationPlayer: bone not mapped for joint {TargetJoints[j].Name} (index {j}). Leaving bind pose.");
                }
                continue;
            }

            Quaternion q0 = _solvedLocalRotations[frameA][j];
            Quaternion q1 = _solvedLocalRotations[frameB][j];
            Quaternion q = (frameA == frameB || !interpolate) ? q0 : Quaternion.Slerp(q0, q1, t);
            bone.localRotation = _bindLocalRotations[j] * q;
        }
    }

    private void ApplyRoot(int frameA, int frameB, float t)
    {
        if (hips == null || _rootPositions == null || _rootPositions.Length == 0)
        {
            return;
        }

        Vector3 p0 = _rootPositions[frameA];
        Vector3 p1 = _rootPositions[frameB];
        Vector3 p = (frameA == frameB || !interpolate) ? p0 : Vector3.Lerp(p0, p1, t);

        if (applyRootInWorldSpace)
        {
            hips.position = _hipsStartWorldPosition + p;
        }
        else
        {
            hips.localPosition = _hipsStartLocalPosition + p;
        }
    }

    private void OnDrawGizmos()
    {
        if (!drawDebugMarkers || _markerJointsPerFrame == null || _jointTargets == null)
        {
            return;
        }

        int frame = Mathf.Clamp((int)(_time / Mathf.Max(1e-6f, _secondsPerFrame)), 0, _markerJointsPerFrame.GetLength(0) - 1);
        for (int j = 0; j < TargetJoints.Length; j++)
        {
            Vector3 markerPos = _markerJointsPerFrame[frame, j];
            Gizmos.color = Color.yellow;
            Gizmos.DrawSphere(markerPos, debugMarkerScale);

            Transform tBone = _jointTargets != null && j < _jointTargets.Length ? _jointTargets[j] : null;
            if (tBone != null)
            {
                Gizmos.color = Color.cyan;
                Gizmos.DrawSphere(tBone.position, debugMarkerScale * 0.8f);
                Gizmos.color = Color.green;
                Gizmos.DrawLine(markerPos, tBone.position);
            }
        }
    }

    private Vector3 ConvertPositionBasis(Vector3 value)
    {
        if (sourceCoordinateSystem == SourceCoordinateSystem.SourceYDownZForward)
        {
            return new Vector3(value.x, -value.y, value.z);
        }

        if (sourceCoordinateSystem == SourceCoordinateSystem.SourceXForwardZUp)
        {
            // Source: X = forward, Z = up. Unity: Y = up, Z = forward.
            // Map (srcX, srcY, srcZ) -> (unityX, unityY, unityZ) = (srcY, srcZ, srcX)
            return new Vector3(value.y, value.z, value.x);
        }

        return value;
    }

    private static Vector3[] ComputeRestOffsets(Vector3[,] jointsPerFrame, int frameWindow)
    {
        Vector3[] rest = new Vector3[TargetJoints.Length];
        for (int jointIndex = 0; jointIndex < TargetJoints.Length; jointIndex++)
        {
            Vector3 sum = Vector3.zero;
            for (int frame = 0; frame < frameWindow; frame++)
            {
                sum += jointsPerFrame[frame, jointIndex];
            }

            rest[jointIndex] = sum / frameWindow;
        }

        Vector3[] offsets = new Vector3[TargetJoints.Length];
        for (int jointIndex = 0; jointIndex < TargetJoints.Length; jointIndex++)
        {
            int parent = TargetJoints[jointIndex].ParentIndex;
            offsets[jointIndex] = parent < 0 ? Vector3.zero : (rest[jointIndex] - rest[parent]);
        }

        return offsets;
    }

    private static Quaternion BestFitRotation(List<Vector3> fromDirs, List<Vector3> toDirs)
    {
        if (fromDirs.Count == 0)
        {
            return Quaternion.identity;
        }

        if (fromDirs.Count == 1)
        {
            return Quaternion.FromToRotation(fromDirs[0], toDirs[0]);
        }

        if (TryBuildBasis(fromDirs, out Matrix4x4 fromBasis) && TryBuildBasis(toDirs, out Matrix4x4 toBasis))
        {
            Matrix4x4 rot = toBasis * fromBasis.transpose;
            Quaternion basisQ = rot.rotation;

            for (int i = 2; i < fromDirs.Count; i++)
            {
                Vector3 rotated = basisQ * fromDirs[i];
                Quaternion correction = Quaternion.FromToRotation(rotated, toDirs[i]);
                basisQ = correction * basisQ;
            }

            return basisQ;
        }

        Quaternion q = Quaternion.FromToRotation(fromDirs[0], toDirs[0]);
        for (int i = 1; i < fromDirs.Count; i++)
        {
            Vector3 rotated = q * fromDirs[i];
            Quaternion correction = Quaternion.FromToRotation(rotated, toDirs[i]);
            q = correction * q;
        }

        return q;
    }

    private static bool TryBuildBasis(List<Vector3> vectors, out Matrix4x4 basis)
    {
        basis = Matrix4x4.identity;
        if (vectors.Count < 2)
        {
            return false;
        }

        Vector3 x = vectors[0].normalized;
        if (x.sqrMagnitude < 1e-8f)
        {
            return false;
        }

        for (int i = 1; i < vectors.Count; i++)
        {
            Vector3 z = Vector3.Cross(x, vectors[i]);
            if (z.sqrMagnitude < 1e-8f)
            {
                continue;
            }

            z.Normalize();
            Vector3 y = Vector3.Cross(z, x);
            y.Normalize();

            basis.SetColumn(0, new Vector4(x.x, x.y, x.z, 0f));
            basis.SetColumn(1, new Vector4(y.x, y.y, y.z, 0f));
            basis.SetColumn(2, new Vector4(z.x, z.y, z.z, 0f));
            basis.SetColumn(3, new Vector4(0f, 0f, 0f, 1f));
            return true;
        }

        return false;
    }

    private static int[][] BuildChildren()
    {
        var children = new List<int>[TargetJoints.Length];
        for (int i = 0; i < children.Length; i++)
        {
            children[i] = new List<int>();
        }

        for (int i = 0; i < TargetJoints.Length; i++)
        {
            int parent = TargetJoints[i].ParentIndex;
            if (parent >= 0)
            {
                children[parent].Add(i);
            }
        }

        var outArray = new int[TargetJoints.Length][];
        for (int i = 0; i < children.Length; i++)
        {
            outArray[i] = children[i].ToArray();
        }

        return outArray;
    }

    private static bool TryFindBone(Dictionary<string, Transform> map, string jointName, out Transform bone)
    {
        if (map.TryGetValue(NormalizeBoneName(jointName), out bone))
        {
            return true;
        }

        if (JointAliases.TryGetValue(jointName, out string[] aliases))
        {
            for (int i = 0; i < aliases.Length; i++)
            {
                if (map.TryGetValue(NormalizeBoneName(aliases[i]), out bone))
                {
                    return true;
                }
            }
        }

        bone = null;
        return false;
    }

    private static string NormalizeBoneName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return string.Empty;
        }

        string value = name.Trim().ToLowerInvariant();
        int colon = value.LastIndexOf(':');
        if (colon >= 0 && colon < value.Length - 1)
        {
            value = value.Substring(colon + 1);
        }

        var builder = new StringBuilder(value.Length);
        for (int i = 0; i < value.Length; i++)
        {
            char ch = value[i];
            if (char.IsLetterOrDigit(ch))
            {
                builder.Append(ch);
            }
        }

        return builder.ToString();
    }

    private static string[] SplitTabs(string line)
    {
        return line.Split('\t');
    }

    private void ComputeCalibrationOffsets(Vector3[,] markerFrames, string[] markerNames, int calibrationWindow)
    {
        try
        {
            int frameCount = Mathf.Clamp(calibrationWindow, 1, markerFrames.GetLength(0));
            var markerToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < markerNames.Length; i++) markerToIndex[markerNames[i]] = i;

            var jointsAvg = new Vector3[TargetJoints.Length];
            var temp = new Vector3[frameCount, TargetJoints.Length];
            for (int f = 0; f < frameCount; f++)
            {
                if (!TryBuildJointPositions(f, markerFrames, markerToIndex, temp))
                {
                    Debug.LogWarning($"TsvMixamoAnimationPlayer: calibration frame {f} failed to build joints.");
                }
                for (int j = 0; j < TargetJoints.Length; j++)
                {
                    jointsAvg[j] += temp[f, j];
                }
            }

            for (int j = 0; j < TargetJoints.Length; j++)
            {
                jointsAvg[j] /= frameCount;
            }

            // compute bind joint positions relative to bind root
            var bindRoot = (_jointTargets != null && _jointTargets.Length > 0 && _jointTargets[0] != null) ? _jointTargets[0].position : Vector3.zero;
            var bindRel = new Vector3[TargetJoints.Length];
            for (int j = 0; j < TargetJoints.Length; j++)
            {
                if (_jointTargets != null && j < _jointTargets.Length && _jointTargets[j] != null)
                {
                    bindRel[j] = _jointTargets[j].position - bindRoot;
                }
                else
                {
                    bindRel[j] = Vector3.zero;
                }
            }

            // compute marker root and relative marker positions
            var markerRoot = jointsAvg[0];
            _calibrationOffsets = new Vector3[TargetJoints.Length];
            for (int j = 0; j < TargetJoints.Length; j++)
            {
                Vector3 markerRel = jointsAvg[j] - markerRoot;
                _calibrationOffsets[j] = bindRel[j] - markerRel;
            }

            if (verboseLogs)
            {
                Debug.Log("TsvMixamoAnimationPlayer: computed calibration offsets for markers -> joints.");
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"TsvMixamoAnimationPlayer: calibration failed: {ex.Message}");
            _calibrationOffsets = null;
        }
    }

    private static bool TryParseFloat(string text, out float value)
    {
        return float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }

    private static Vector3 GetMarker(int frame, Vector3[,] markerFrames, Dictionary<string, int> markerToIndex, string markerName)
    {
        if (!markerToIndex.TryGetValue(markerName, out int markerIndex))
        {
            throw new InvalidOperationException($"Required marker not found: {markerName}");
        }

        return markerFrames[frame, markerIndex];
    }

    private static Vector3 Avg(Vector3 a, Vector3 b)
    {
        return (a + b) * 0.5f;
    }

    private static Vector3 Avg(Vector3 a, Vector3 b, Vector3 c)
    {
        return (a + b + c) / 3f;
    }

    private static Vector3 Avg(Vector3 a, Vector3 b, Vector3 c, Vector3 d, Vector3 e)
    {
        return (a + b + c + d + e) / 5f;
    }
}