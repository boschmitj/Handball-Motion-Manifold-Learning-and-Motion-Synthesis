using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[Serializable]
public class FrameData
{
    public float[] root_position;
    public float[] rotations_flat;
    public float[][] rotations;
}

[Serializable]
public class AnimationData
{
    public float frame_rate;
    public string coordinate_system;
    public string quaternion_format;
    public string root_joint;
    public string[] joint_names;
    public FrameData[] frames;
}

[Serializable]
public class BoneBinding
{
    public string jointName;
    public Transform bone;
}

public class JsonAnimationPlayer : MonoBehaviour
{
    public enum SourceCoordinateSystem
    {
        UnityYUpLeftHanded,
        SourceYDownZForward
    }

    [Header("Input")]
    public TextAsset jsonAnimation;
    public bool loadFromFilePath;
    public string jsonFilePath;

    [Header("Bones")]
    public Transform hips;
    public bool autoMapByName = true;
    public List<BoneBinding> manualBindings = new List<BoneBinding>();

    [Header("Playback")]
    public bool playOnStart = true;
    public bool loop = true;
    public float playbackSpeed = 1.0f;
    public bool interpolate = true;
    public bool applyRootInWorldSpace = false;

    [Header("Coordinate Conversion")]
    public SourceCoordinateSystem sourceCoordinateSystem = SourceCoordinateSystem.UnityYUpLeftHanded;

    private AnimationData _data;
    private Transform[] _jointTargets;
    private Quaternion[] _bindLocalRotations;
    private float _secondsPerFrame;
    private float _time;
    private bool _isPlaying;
    private Vector3 _hipsStartLocalPosition;
    private Vector3 _hipsStartWorldPosition;

    private static readonly Matrix4x4 BasisSourceToUnity = Matrix4x4.Scale(new Vector3(1f, -1f, 1f));

    private void Start()
    {
        LoadAnimation();
        BuildBoneMap();
        CacheBindPose();

        if (hips != null)
        {
            _hipsStartLocalPosition = hips.localPosition;
            _hipsStartWorldPosition = hips.position;
        }

        _isPlaying = playOnStart;
    }

    private void Update()
    {
        if (!_isPlaying || _data == null || _data.frames == null || _data.frames.Length == 0)
        {
            return;
        }

        _time += Time.deltaTime * Mathf.Max(0f, playbackSpeed);
        float clipDuration = _data.frames.Length * _secondsPerFrame;
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

        float frameFloat = Mathf.Clamp(_time / _secondsPerFrame, 0f, _data.frames.Length - 1);
        int i0 = Mathf.FloorToInt(frameFloat);
        int i1 = Mathf.Min(i0 + 1, _data.frames.Length - 1);
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
        if (_data != null && _data.frames != null && _data.frames.Length > 0)
        {
            ApplyFrame(0, 0, 0f);
        }
    }

    private void LoadAnimation()
    {
        string jsonText;
        if (loadFromFilePath)
        {
            if (string.IsNullOrWhiteSpace(jsonFilePath) || !File.Exists(jsonFilePath))
            {
                Debug.LogError("JsonAnimationPlayer: jsonFilePath is invalid.");
                return;
            }

            jsonText = File.ReadAllText(jsonFilePath);
        }
        else
        {
            if (jsonAnimation == null)
            {
                Debug.LogError("JsonAnimationPlayer: jsonAnimation TextAsset is not assigned.");
                return;
            }

            jsonText = jsonAnimation.text;
        }

        _data = JsonUtility.FromJson<AnimationData>(jsonText);
        if (_data == null || _data.frames == null || _data.frames.Length == 0 || _data.joint_names == null)
        {
            Debug.LogError("JsonAnimationPlayer: invalid JSON animation payload.");
            return;
        }

        for (int i = 0; i < _data.frames.Length; i++)
        {
            var frame = _data.frames[i];
            if (frame == null)
            {
                Debug.LogError($"JsonAnimationPlayer: frame {i} is null.");
                continue;
            }

            bool hasFlat = frame.rotations_flat != null && frame.rotations_flat.Length >= _data.joint_names.Length * 4;
            bool hasJagged = frame.rotations != null && frame.rotations.Length >= _data.joint_names.Length;
            if (!hasFlat && !hasJagged)
            {
                Debug.LogError(
                    $"JsonAnimationPlayer: frame {i} has no valid rotations payload. " +
                    "Expected rotations_flat (preferred) or rotations."
                );
            }
        }

        if (_data.frame_rate <= 1e-6f)
        {
            _data.frame_rate = 30f;
        }

        _secondsPerFrame = 1f / _data.frame_rate;

        if (!string.IsNullOrEmpty(_data.coordinate_system))
        {
            if (_data.coordinate_system.Equals("unity_y_up_left_handed", StringComparison.OrdinalIgnoreCase))
            {
                sourceCoordinateSystem = SourceCoordinateSystem.UnityYUpLeftHanded;
            }
            else
            {
                sourceCoordinateSystem = SourceCoordinateSystem.SourceYDownZForward;
            }
        }
    }

    private void BuildBoneMap()
    {
        if (_data == null)
        {
            return;
        }

        var map = new Dictionary<string, Transform>(StringComparer.OrdinalIgnoreCase);

        foreach (var binding in manualBindings)
        {
            if (binding != null && !string.IsNullOrWhiteSpace(binding.jointName) && binding.bone != null)
            {
                map[binding.jointName] = binding.bone;
            }
        }

        if (autoMapByName)
        {
            foreach (Transform t in GetComponentsInChildren<Transform>(true))
            {
                if (!map.ContainsKey(t.name))
                {
                    map[t.name] = t;
                }
            }
        }

        _jointTargets = new Transform[_data.joint_names.Length];
        for (int i = 0; i < _data.joint_names.Length; i++)
        {
            string jointName = _data.joint_names[i];
            if (map.TryGetValue(jointName, out var bone))
            {
                _jointTargets[i] = bone;
            }
        }

        if (hips == null)
        {
            if (!string.IsNullOrWhiteSpace(_data.root_joint) && map.TryGetValue(_data.root_joint, out var rootBone))
            {
                hips = rootBone;
            }
            else if (map.TryGetValue("Hips", out var fallbackHips))
            {
                hips = fallbackHips;
            }
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
        if (_jointTargets == null || _bindLocalRotations == null || _data == null || _data.frames == null)
        {
            return;
        }

        if (frameA < 0 || frameA >= _data.frames.Length || frameB < 0 || frameB >= _data.frames.Length)
        {
            return;
        }

        var f0 = _data.frames[frameA];
        var f1 = _data.frames[frameB];
        if (f0 == null || f1 == null)
        {
            return;
        }

        ApplyRoot(f0.root_position, f1.root_position, t);

        int jointCount = _jointTargets.Length;
        for (int j = 0; j < jointCount; j++)
        {
            Transform bone = _jointTargets[j];
            if (bone == null)
            {
                continue;
            }

            if (!TryGetJointQuaternion(f0, j, out var q0))
            {
                continue;
            }
            Quaternion q1 = q0;
            if (frameA != frameB)
            {
                TryGetJointQuaternion(f1, j, out q1);
            }
            Quaternion q = (frameA == frameB || !interpolate) ? q0 : Quaternion.Slerp(q0, q1, t);

            bone.localRotation = _bindLocalRotations[j] * q;
        }
    }

    private bool TryGetJointQuaternion(FrameData frame, int jointIndex, out Quaternion q)
    {
        q = Quaternion.identity;
        if (frame == null)
        {
            return false;
        }

        if (frame.rotations_flat != null)
        {
            int baseIndex = jointIndex * 4;
            if (baseIndex + 3 < frame.rotations_flat.Length)
            {
                q = ReadQuaternion(
                    new[]
                    {
                        frame.rotations_flat[baseIndex],
                        frame.rotations_flat[baseIndex + 1],
                        frame.rotations_flat[baseIndex + 2],
                        frame.rotations_flat[baseIndex + 3],
                    }
                );
                return true;
            }
        }

        if (frame.rotations != null && jointIndex < frame.rotations.Length)
        {
            var values = frame.rotations[jointIndex];
            if (values != null && values.Length >= 4)
            {
                q = ReadQuaternion(values);
                return true;
            }
        }

        return false;
    }

    private void ApplyRoot(float[] a, float[] b, float t)
    {
        if (hips == null || a == null || a.Length < 3)
        {
            return;
        }

        Vector3 p0 = ReadVector3(a);
        Vector3 p1 = (b != null && b.Length >= 3) ? ReadVector3(b) : p0;
        Vector3 root = interpolate ? Vector3.Lerp(p0, p1, t) : p0;

        if (applyRootInWorldSpace)
        {
            hips.position = _hipsStartWorldPosition + root;
        }
        else
        {
            hips.localPosition = _hipsStartLocalPosition + root;
        }
    }

    private Vector3 ReadVector3(float[] values)
    {
        Vector3 v = new Vector3(values[0], values[1], values[2]);
        if (sourceCoordinateSystem == SourceCoordinateSystem.SourceYDownZForward)
        {
            v = BasisSourceToUnity.MultiplyPoint3x4(v);
        }

        return v;
    }

    private Quaternion ReadQuaternion(float[] values)
    {
        Quaternion q = new Quaternion(values[0], values[1], values[2], values[3]);
        if (sourceCoordinateSystem == SourceCoordinateSystem.SourceYDownZForward)
        {
            q = ConvertQuaternionBasis(q, BasisSourceToUnity);
        }

        return q;
    }

    private static Quaternion ConvertQuaternionBasis(Quaternion q, Matrix4x4 basis)
    {
        Matrix4x4 r = Matrix4x4.Rotate(q);
        Matrix4x4 rOut = basis * r * basis.inverse;
        return rOut.rotation;
    }
}
