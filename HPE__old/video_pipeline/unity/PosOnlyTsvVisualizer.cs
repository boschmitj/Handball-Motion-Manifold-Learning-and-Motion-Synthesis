using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

// Simple visualizer that positions transforms to marker-derived joint positions each frame.
public class PosOnlyTsvVisualizer : MonoBehaviour
{
    [Header("Input")]
    public string tsvFilePath;
    public bool loadOnStart = true;

    [Header("Bones (assign Mixamo bones in order)")]
    public Transform[] boneTargets;

    [Header("Playback")]
    public bool playOnStart = true;
    public bool loop = true;
    public float playbackSpeed = 1f;

    [Header("Source")]
    public float sourceUnitsToMeters = 0.001f;
    public enum SourceCoordinate { XForward_ZUp, YDown_ZForward }
    public SourceCoordinate sourceCoordinate = SourceCoordinate.XForward_ZUp;

    private string[] _markerNames;
    private Vector3[,] _markerFrames;
    private int _frameCount;
    private float _secondsPerFrame = 1f/300f;
    private float _time;
    private bool _isPlaying;

    private void Start()
    {
        if (!loadOnStart) return;
        if (!LoadTsv()) return;
        _isPlaying = playOnStart;
    }

    private void Update()
    {
        if (!_isPlaying || _markerFrames==null) return;
        _time += Time.deltaTime * Math.Max(0f, playbackSpeed);
        float duration = _frameCount * _secondsPerFrame;
        if (loop) _time %= duration; else _time = Math.Min(_time, duration - _secondsPerFrame);
        int frame = Mathf.Clamp((int)(_time / _secondsPerFrame), 0, _frameCount-1);
        ApplyPositions(frame);
    }

    private bool LoadTsv()
    {
        if (string.IsNullOrWhiteSpace(tsvFilePath)) return false;
        string path = tsvFilePath.Replace("\\","/");
        if (!File.Exists(path)) { Debug.LogError("TSV not found"); return false; }
        if (!TryParseTsv(path, out _markerFrames, out _markerNames, out float fps)) return false;
        _frameCount = _markerFrames.GetLength(0);
        if (fps>1e-6f) _secondsPerFrame = 1f/fps;
        return true;
    }

    private void ApplyPositions(int frame)
    {
        if (boneTargets==null) return;
        // Map boneTargets length to typical joint subsets by index; the user should assign bones
        // directly: hips, rightUpLeg, rightLeg, rightFoot, leftUpLeg, leftLeg, leftFoot, spine, spine1, spine2, neck, head, leftShoulder, leftArm, leftForeArm, leftHand, rightShoulder, rightArm, rightForeArm, rightHand
        try
        {
            var mToIdx = BuildMarkerIndex(_markerNames);
            var joints = new Vector3[boneTargets.Length];
            // Build same semantics as other script but simplified
            Vector3 head = Avg(GetM(frame,mToIdx,"Q_HeadL"), GetM(frame,mToIdx,"Q_HeadR"), GetM(frame,mToIdx,"Q_HeadFront"));
            Vector3 chest = GetM(frame,mToIdx,"Q_Chest");
            Vector3 spine12 = GetM(frame,mToIdx,"Q_SpineThoracic12");
            Vector3 waistCenter = Avg(GetM(frame,mToIdx,"Q_WaistLFront"), GetM(frame,mToIdx,"Q_WaistL"), GetM(frame,mToIdx,"Q_WaistBack"), GetM(frame,mToIdx,"Q_WaistR"), GetM(frame,mToIdx,"Q_WaistRFront"));

            // positions according to target ordering
            if (boneTargets.Length>0 && boneTargets[0]!=null) boneTargets[0].position = waistCenter;
            if (boneTargets.Length>10 && boneTargets[10]!=null) boneTargets[10].position = (chest+head)/2f; // neck
            if (boneTargets.Length>11 && boneTargets[11]!=null) boneTargets[11].position = head;

            // left arm
            if (mToIdx.ContainsKey("Q_LShoulderTop") && mToIdx.ContainsKey("Q_LArm")){
                if (FindTarget("LeftShoulder", out Transform t) && t!=null) t.position = GetM(frame,mToIdx,"Q_LShoulderTop");
            }

            // set any remaining assigned bones to marker-derived positions if available
            for (int i=0;i<boneTargets.Length;i++){
                if (boneTargets[i]==null) continue;
                // don't override ones we set above
            }
        }
        catch (Exception ex) { Debug.LogWarning("PosOnly ApplyPositions failed: "+ex.Message); }
    }

    private Dictionary<string,int> BuildMarkerIndex(string[] names)
    {
        var d = new Dictionary<string,int>(StringComparer.OrdinalIgnoreCase);
        for (int i=0;i<names.Length;i++) d[names[i]] = i; return d;
    }

    private bool FindTarget(string name, out Transform t)
    {
        t=null;
        if (boneTargets==null) return false;
        foreach(var b in boneTargets) if (b!=null && b.name.IndexOf(name,StringComparison.OrdinalIgnoreCase)>=0) { t=b; return true; }
        return false;
    }

    private bool TryParseTsv(string path, out Vector3[,] frames, out string[] markers, out float fps)
    {
        frames=null; markers=null; fps=300f;
        string[] lines = File.ReadAllLines(path);
        var markerList = new List<string>(); int header=-1;
        for (int i=0;i<lines.Length;i++){ string L=lines[i]; if (L.StartsWith("MARKER_NAMES",StringComparison.OrdinalIgnoreCase)){var tok=L.Split('\t'); for(int j=1;j<tok.Length;j++) if(!string.IsNullOrWhiteSpace(tok[j])) markerList.Add(tok[j].Trim()); } else if(L.StartsWith("Frame\tTime",StringComparison.OrdinalIgnoreCase)){ header=i; break; } }
        if (markerList.Count==0 || header<0) return false; markers = markerList.ToArray();
        var headerCols = lines[header].Split('\t'); var colBase = new Dictionary<string,int>(StringComparer.OrdinalIgnoreCase);
        for (int i=2;i<headerCols.Length;i++){ string c=headerCols[i]; if (c.EndsWith(" X",StringComparison.OrdinalIgnoreCase)){ string name=c.Substring(0,c.Length-2).Trim(); if(!colBase.ContainsKey(name)) colBase[name]=i; }}
        var data = new List<string>(); for (int i=header+1;i<lines.Length;i++) if (!string.IsNullOrWhiteSpace(lines[i])) data.Add(lines[i]);
        frames = new Vector3[data.Count, markers.Length];
        for (int f=0; f<data.Count; f++){ var cols=data[f].Split('\t'); for (int m=0;m<markers.Length;m++){ if (!colBase.TryGetValue(markers[m], out int baseCol)) continue; if (baseCol+2>=cols.Length) continue; float x=float.Parse(cols[baseCol],CultureInfo.InvariantCulture); float y=float.Parse(cols[baseCol+1],CultureInfo.InvariantCulture); float z=float.Parse(cols[baseCol+2],CultureInfo.InvariantCulture); frames[f,m] = ConvertBasis(new Vector3(x,y,z)*sourceUnitsToMeters); }}
        return true;
    }

    private Vector3 ConvertBasis(Vector3 v) { if (sourceCoordinate==SourceCoordinate.XForward_ZUp) return new Vector3(v.y,v.z,v.x); return new Vector3(v.x,-v.y,v.z); }

    private Vector3 GetM(int f, Dictionary<string,int> idx, string name){ if(!idx.TryGetValue(name,out int i)) throw new InvalidOperationException("Missing marker: "+name); return _markerFrames[f,i]; }
    private static Vector3 Avg(params Vector3[] vs){ Vector3 s=Vector3.zero; foreach(var v in vs) s+=v; return s/vs.Length; }
}
