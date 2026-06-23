using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public class QuatTsvMixamoAnimator : MonoBehaviour
{
    [Header("Input")]
    public string tsvFilePath;
    public bool loadOnStart = true;

    [Header("Bones")]
    public Transform hips;
    public bool autoMapByName = true;

    [Header("Playback")]
    public bool playOnStart = true;
    public bool loop = true;
    public float playbackSpeed = 1f;

    [Header("Source")]
    public float sourceUnitsToMeters = 0.001f; // mm -> m
    public enum SourceCoordinate { XForward_ZUp, YDown_ZForward }
    public SourceCoordinate sourceCoordinate = SourceCoordinate.XForward_ZUp;

    [Header("Solve")]
    public int calibrationFrames = 10;
    public bool verboseLogs = true;

    // target joints (Mixamo-like order)
    private static readonly string[] JointNames = new[] {
        "Hips","RightUpLeg","RightLeg","RightFoot","LeftUpLeg","LeftLeg","LeftFoot",
        "Spine","Spine1","Spine2","Neck","Head","LeftShoulder","LeftArm","LeftForeArm","LeftHand",
        "RightShoulder","RightArm","RightForeArm","RightHand"
    };

    private static readonly int[] ParentIndices = new[] {
        -1,0,1,2,0,4,5,0,7,8,9,10,9,12,13,14,9,16,17,18
    };

    // runtime data
    private string[] _markerNames;
    private Vector3[,] _markerFrames; // [frame, marker]
    private int _frameCount;
    private float _secondsPerFrame = 1f / 300f;

    private Transform[] _targets;
    private Quaternion[] _bindLocalRot;
    private Quaternion[][] _solvedLocal;
    private Vector3[] _rootPositions;

    private float _time;
    private bool _isPlaying;

    private void Start()
    {
        if (!loadOnStart) return;
        if (!LoadAndSolve()) return;
        _isPlaying = playOnStart;
    }

    private void Update()
    {
        if (!_isPlaying || _solvedLocal == null || _solvedLocal.Length == 0) return;
        _time += Time.deltaTime * Mathf.Max(0f, playbackSpeed);
        float duration = _solvedLocal.Length * _secondsPerFrame;
        if (loop) _time %= duration; else _time = Mathf.Min(_time, duration - _secondsPerFrame);
        float frameF = Mathf.Clamp(_time / _secondsPerFrame, 0f, _solvedLocal.Length - 1);
        int i0 = Mathf.FloorToInt(frameF);
        int i1 = Mathf.Min(i0 + 1, _solvedLocal.Length - 1);
        float a = frameF - i0;
        ApplyFrame(i0, i1, a);
    }

    public bool LoadAndSolve()
    {
        if (string.IsNullOrWhiteSpace(tsvFilePath)) { Debug.LogError("tsvPath empty"); return false; }
        string path = tsvFilePath.Replace("\\","/");
        if (!File.Exists(path)) { Debug.LogError("TSV not found: " + path); return false; }

        if (!TryParseTsv(path, out _markerFrames, out _markerNames, out float fps)) return false;
        _frameCount = _markerFrames.GetLength(0);
        if (fps > 1e-6f) _secondsPerFrame = 1f / fps;

        BuildMap();
        CacheBindPose();

        if (!TrySolve(_markerFrames, _markerNames, out _rootPositions, out _solvedLocal)) return false;

        Debug.Log($"QuatTsvMixamoAnimator: solved {_solvedLocal.Length} frames");
        ApplyFrame(0,0,0f);
        return true;
    }

    private void BuildMap()
    {
        var map = new Dictionary<string, Transform>(StringComparer.OrdinalIgnoreCase);
        if (autoMapByName)
        {
            foreach (var t in GetComponentsInChildren<Transform>(true))
            {
                if (!map.ContainsKey(t.name)) map[t.name] = t;
            }
        }

        _targets = new Transform[JointNames.Length];
        for (int i = 0; i < JointNames.Length; i++)
        {
            string key = JointNames[i];
            if (map.TryGetValue(key, out var tr)) _targets[i] = tr; else _targets[i] = null;
        }
        if (hips == null && _targets.Length>0) hips = _targets[0];
    }

    private void CacheBindPose()
    {
        if (_targets == null) return;
        _bindLocalRot = new Quaternion[_targets.Length];
        for (int i=0;i<_targets.Length;i++) _bindLocalRot[i] = _targets[i] != null ? _targets[i].localRotation : Quaternion.identity;
    }

    private void ApplyFrame(int a,int b,float t)
    {
        if (_solvedLocal==null) return;
        Vector3 p = (_rootPositions==null||_rootPositions.Length==0)? Vector3.zero : ((_rootPositions[a]*(1f-t)+_rootPositions[b]*t));
        if (hips!=null) hips.localPosition = p;

        for (int j=0;j<_targets.Length;j++)
        {
            var bone = _targets[j];
            if (bone==null) continue;
            Quaternion q0 = _solvedLocal[a][j];
            Quaternion q1 = _solvedLocal[b][j];
            Quaternion q = (a==b)? q0 : Quaternion.Slerp(q0,q1,t);
            bone.localRotation = _bindLocalRot[j] * q;
        }
    }

    // ---------- Parsing ----------
    private bool TryParseTsv(string path, out Vector3[,] frames, out string[] markers, out float fps)
    {
        frames = null; markers = null; fps = 300f;
        string[] lines = File.ReadAllLines(path);
        var markerList = new List<string>();
        int headerIdx = -1;
        for (int i=0;i<lines.Length;i++)
        {
            string L = lines[i];
            if (L.StartsWith("MARKER_NAMES", StringComparison.OrdinalIgnoreCase))
            {
                var tok = L.Split('\t');
                for (int j=1;j<tok.Length;j++) if(!string.IsNullOrWhiteSpace(tok[j])) markerList.Add(tok[j].Trim());
            }
            else if (L.StartsWith("FREQUENCY", StringComparison.OrdinalIgnoreCase))
            {
                var tok=L.Split('\t'); if (tok.Length>1 && float.TryParse(tok[1],NumberStyles.Float,CultureInfo.InvariantCulture,out float pf)) fps=pf;
            }
            else if (L.StartsWith("Frame\tTime", StringComparison.OrdinalIgnoreCase)) { headerIdx=i; break; }
        }
        if (markerList.Count==0 || headerIdx<0) { Debug.LogError("TSV parse failed: no markers or header"); return false; }
        markers = markerList.ToArray();
        // build column base
        var headerCols = lines[headerIdx].Split('\t');
        var colBase = new Dictionary<string,int>(StringComparer.OrdinalIgnoreCase);
        for (int i=2;i<headerCols.Length;i++) { string c=headerCols[i]; if(c.EndsWith(" X",StringComparison.OrdinalIgnoreCase)) { string name=c.Substring(0,c.Length-2).Trim(); if(!colBase.ContainsKey(name)) colBase[name]=i; }}
        var data = new List<string>();
        for (int i=headerIdx+1;i<lines.Length;i++) if(!string.IsNullOrWhiteSpace(lines[i])) data.Add(lines[i]);
        frames = new Vector3[data.Count, markers.Length];
        for (int f=0; f<data.Count; f++)
        {
            var cols = data[f].Split('\t');
            for (int m=0;m<markers.Length;m++)
            {
                if (!colBase.TryGetValue(markers[m], out int baseCol)) continue;
                if (baseCol+2>=cols.Length) continue;
                if (!float.TryParse(cols[baseCol],NumberStyles.Float,CultureInfo.InvariantCulture,out float x)) continue;
                float y = float.Parse(cols[baseCol+1],CultureInfo.InvariantCulture);
                float z = float.Parse(cols[baseCol+2],CultureInfo.InvariantCulture);
                Vector3 p = new Vector3(x,y,z)*sourceUnitsToMeters;
                frames[f,m] = ConvertBasis(p);
            }
        }
        return true;
    }

    private Vector3 ConvertBasis(Vector3 v)
    {
        if (sourceCoordinate==SourceCoordinate.XForward_ZUp) return new Vector3(v.y, v.z, v.x);
        return new Vector3(v.x, -v.y, v.z);
    }

    // ---------- Solve ----------
    private bool TrySolve(Vector3[,] markerFrames, string[] markerNames, out Vector3[] roots, out Quaternion[][] localQuats)
    {
        roots = null; localQuats = null;
        int F = markerFrames.GetLength(0);
        int M = markerFrames.GetLength(1);
        if (F==0 || M==0) return false;

        var mToIdx = new Dictionary<string,int>(StringComparer.OrdinalIgnoreCase);
        for (int i=0;i<markerNames.Length;i++) mToIdx[markerNames[i]] = i;

        var joints = new Vector3[F, JointNames.Length];
        for (int f=0; f<F; f++)
        {
            if (!BuildJointsFromMarkers(f, markerFrames, mToIdx, joints)) { Debug.LogError("Build joints failed at frame "+f); return false; }
        }

        // rest offsets
        var rest = new Vector3[JointNames.Length];
        int cal = Math.Min(calibrationFrames, F);
        for (int j=0;j<JointNames.Length;j++) { Vector3 s=Vector3.zero; for (int f=0;f<cal;f++) s+=joints[f,j]; rest[j]=s/cal; }
        var offsets = new Vector3[JointNames.Length];
        for (int j=0;j<JointNames.Length;j++) { int p=ParentIndices[j]; offsets[j] = (p<0)? Vector3.zero : (rest[j]-rest[p]); }

        roots = new Vector3[F]; localQuats = new Quaternion[F][];
        for (int f=0; f<F; f++)
        {
            var worldR = new Quaternion[JointNames.Length];
            for (int j=0;j<JointNames.Length;j++) worldR[j] = Quaternion.identity;
            for (int j=0;j<JointNames.Length;j++)
            {
                var children = GetChildren(j);
                var restDirs = new List<Vector3>();
                var poseDirs = new List<Vector3>();
                foreach (var c in children)
                {
                    Vector3 rdir = offsets[c];
                    Vector3 pdir = joints[f,c] - joints[f,j];
                    if (rdir.sqrMagnitude<1e-8 || pdir.sqrMagnitude<1e-8) continue;
                    restDirs.Add(rdir);
                    poseDirs.Add(pdir);
                }
                if (restDirs.Count>0) worldR[j] = BestFit(restDirs, poseDirs);
                else worldR[j] = Quaternion.identity;
            }
            var local = new Quaternion[JointNames.Length];
            for (int j=0;j<JointNames.Length;j++)
            {
                int p = ParentIndices[j];
                local[j] = (p<0)? worldR[j] : Quaternion.Inverse(worldR[p]) * worldR[j];
            }
            localQuats[f] = local;
            roots[f] = joints[f,0];
        }

        // normalize root to first frame
        Vector3 baseRoot = roots[0];
        for (int i=0;i<roots.Length;i++) roots[i] -= baseRoot;
        return true;
    }

    private int[] GetChildren(int j)
    {
        var list = new List<int>();
        for (int i=0;i<ParentIndices.Length;i++) if (ParentIndices[i]==j) list.Add(i);
        return list.ToArray();
    }

    private Quaternion BestFit(List<Vector3> restDirs, List<Vector3> poseDirs)
    {
        if (restDirs.Count==0) return Quaternion.identity;
        if (restDirs.Count==1) return Quaternion.FromToRotation(restDirs[0].normalized, poseDirs[0].normalized);
        // build bases from first two
        Vector3 r0 = restDirs[0].normalized; Vector3 r1 = restDirs[1].normalized;
        Vector3 p0 = poseDirs[0].normalized; Vector3 p1 = poseDirs[1].normalized;
        Vector3 rz = Vector3.Cross(r0,r1); if (rz.sqrMagnitude<1e-8) return Quaternion.FromToRotation(r0,p0);
        rz.Normalize(); Vector3 ry = Vector3.Cross(rz,r0); ry.Normalize();
        Matrix4x4 from = Matrix4x4.identity; from.SetColumn(0,new Vector4(r0.x,r0.y,r0.z,0)); from.SetColumn(1,new Vector4(ry.x,ry.y,ry.z,0)); from.SetColumn(2,new Vector4(rz.x,rz.y,rz.z,0));
        Vector3 pz = Vector3.Cross(p0,p1); pz.Normalize(); Vector3 py = Vector3.Cross(pz,p0); py.Normalize();
        Matrix4x4 to = Matrix4x4.identity; to.SetColumn(0,new Vector4(p0.x,p0.y,p0.z,0)); to.SetColumn(1,new Vector4(py.x,py.y,py.z,0)); to.SetColumn(2,new Vector4(pz.x,pz.y,pz.z,0));
        Matrix4x4 rot = to * from.transpose;
        Quaternion q = rot.rotation;
        // refine with remaining vectors
        for (int i=2;i<restDirs.Count;i++) { Vector3 r = q * restDirs[i]; Quaternion c = Quaternion.FromToRotation(r, poseDirs[i]); q = c * q; }
        return q;
    }

    // Build joints mapping per user's marker semantics
    private bool BuildJointsFromMarkers(int frame, Vector3[,] markerFrames, Dictionary<string,int> mToIdx, Vector3[,] outJoints)
    {
        try
        {
            Vector3 Head = Avg(GetM(frame,mToIdx,"Q_HeadL"), GetM(frame,mToIdx,"Q_HeadR"), GetM(frame,mToIdx,"Q_HeadFront"));
            Vector3 Chest = GetM(frame,mToIdx,"Q_Chest");
            Vector3 Spine12 = GetM(frame,mToIdx,"Q_SpineThoracic12");
            Vector3 Spine2 = GetM(frame,mToIdx,"Q_SpineThoracic2");
            Vector3 WaistLFront = GetM(frame,mToIdx,"Q_WaistLFront");
            Vector3 WaistL = GetM(frame,mToIdx,"Q_WaistL");
            Vector3 WaistBack = GetM(frame,mToIdx,"Q_WaistBack");
            Vector3 WaistR = GetM(frame,mToIdx,"Q_WaistR");
            Vector3 WaistRFront = GetM(frame,mToIdx,"Q_WaistRFront");

            Vector3 LShoulderTop = GetM(frame,mToIdx,"Q_LShoulderTop");
            Vector3 LArm = GetM(frame,mToIdx,"Q_LArm");
            Vector3 LElbow = Avg(GetM(frame,mToIdx,"Q_LElbowOut"), GetM(frame,mToIdx,"Q_LElbowIn"));
            Vector3 LWrist = Avg(GetM(frame,mToIdx,"Q_LWristIn"), GetM(frame,mToIdx,"Q_LWristOut"));
            Vector3 LHand = GetM(frame,mToIdx,"Q_LHand2");

            Vector3 RShoulderTop = GetM(frame,mToIdx,"Q_RShoulderTop");
            Vector3 RArm = GetM(frame,mToIdx,"Q_RArm");
            Vector3 RElbow = Avg(GetM(frame,mToIdx,"Q_RElbowOut"), GetM(frame,mToIdx,"Q_RElbowIn"));
            Vector3 RWrist = Avg(GetM(frame,mToIdx,"Q_RWristIn"), GetM(frame,mToIdx,"Q_RWristOut"));
            Vector3 RHand = GetM(frame,mToIdx,"Q_RHand2");

            Vector3 LThigh = GetM(frame,mToIdx,"Q_LThighFrontLow");
            Vector3 LKnee = GetM(frame,mToIdx,"Q_LKneeOut");
            Vector3 LAnkle = GetM(frame,mToIdx,"Q_LAnkleOut");
            Vector3 LForefoot2 = GetM(frame,mToIdx,"Q_LForefoot2");

            Vector3 RThigh = GetM(frame,mToIdx,"Q_RThighFrontLow");
            Vector3 RKnee = GetM(frame,mToIdx,"Q_RKneeOut");
            Vector3 RAnkle = GetM(frame,mToIdx,"Q_RAnkleOut");
            Vector3 RForefoot2 = GetM(frame,mToIdx,"Q_RForefoot2");

            // Hips center
            outJoints[frame,0] = Avg(WaistLFront, WaistL, WaistBack, WaistR, WaistRFront);
            // Right leg
            outJoints[frame,1] = WaistRFront; // RightUpLeg approx waistRFront per user
            outJoints[frame,2] = RKnee;
            outJoints[frame,3] = RAnkle;
            // Left leg
            outJoints[frame,4] = WaistLFront; // LeftUpLeg approx waistLFront
            outJoints[frame,5] = LKnee;
            outJoints[frame,6] = LAnkle;
            // Spine & torso
            outJoints[frame,7] = WaistBack; // Spine
            outJoints[frame,8] = Spine12;   // Spine1
            outJoints[frame,9] = Chest;     // Spine2
            outJoints[frame,10] = Avg(Chest, Head); // Neck approx
            outJoints[frame,11] = Head;
            // Left arm chain: leave clavicle static at chest, use shoulder top as upper arm
            outJoints[frame,12] = Chest; // LeftShoulder (leave in chest/bind region — no clavicle animation)
            outJoints[frame,13] = LShoulderTop; // LeftArm: use Q_LShoulderTop as the upper-arm proxy
            outJoints[frame,14] = LElbow; // LeftForeArm
            outJoints[frame,15] = LWrist; // LeftHand approx wrist
            // Right arm
            outJoints[frame,16] = RShoulderTop;
            outJoints[frame,17] = RArm;
            outJoints[frame,18] = RElbow;
            outJoints[frame,19] = RWrist;

            return true;
        }
        catch (Exception ex)
        {
            if (verboseLogs) Debug.LogError("BuildJoints error: " + ex.Message);
            return false;
        }
    }

    private Vector3 GetM(int f, Dictionary<string,int> mToIdx, string name)
    {
        if (!mToIdx.TryGetValue(name, out int idx)) throw new InvalidOperationException("Marker missing: " + name);
        return _markerFrames[f, idx];
    }

    private static Vector3 Avg(params Vector3[] vs)
    {
        Vector3 s=Vector3.zero; foreach(var v in vs) s+=v; return s/vs.Length;
    }
}
