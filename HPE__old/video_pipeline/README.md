# Handball Throw -> Unity Humanoid 3D Pipeline

End-to-end monocular video pipeline:

MP4 -> frames -> 2D keypoints (ViTPose/MMPose) -> COCO->H36M + normalization -> 3D lifting (MotionBERT) -> smoothing -> retargeting -> BVH export -> Streamlit visualization.

## 1) Reproducible Environment (Python 3.10, conda)

```bash
cd /home/josh/Bachelorarbeit/video_pipeline
conda env create -f environment.yml
conda activate handball3d
```

If you need CPU-only PyTorch, remove `pytorch-cuda=12.1` from [environment.yml](environment.yml).

### MotionBERT install

```bash
bash scripts/setup_motionbert.sh /home/josh/Bachelorarbeit/video_pipeline/third_party
```

This creates MotionBERT at:

- `/home/josh/Bachelorarbeit/video_pipeline/third_party/MotionBERT`

## 2) Project Architecture

- [video_processing](video_processing): MP4 loading and frame extraction
- [pose_estimation_vitpose](pose_estimation_vitpose): ViTPose inference via MMPose
- [pose_lifting_motionbert](pose_lifting_motionbert): MotionBERT lifting wrapper
- [preprocessing](preprocessing): normalization + COCO->H36M mapping
- [postprocessing](postprocessing): smoothing, grounding, visualization
- [skeleton](skeleton): humanoid skeleton/joint naming
- [retargeting](retargeting): positions -> joint rotations
- [export](export): BVH writing
- [frontend](frontend): Streamlit UI
- [scripts](scripts): CLI entrypoint
- [tests](tests): unit tests for independent modules

## 3) Required Model Files

### ViTPose

Provide MMPose config + checkpoint paths, for example:

- `configs/body_2d_keypoint/topdown_heatmap/coco/vitpose_small_coco_256x192.py`
- `vitpose_small.pth`

### MotionBERT

Provide MotionBERT config + checkpoint paths from your clone.

Default wrapper calls:

- `third_party/MotionBERT/infer_wild.py`

If your MotionBERT revision uses different argument names, update [motionbert_lifter.py](pose_lifting_motionbert/motionbert_lifter.py#L64).

## 4) Run Pipeline (Strict Order Implemented)

```bash
python -m scripts.run_pipeline \
  --video data/samples/input_throw.mp4 \
  --output_dir outputs/run01 \
  --vitpose_config /abs/path/to/vitpose_config.py \
  --vitpose_checkpoint /abs/path/to/vitpose_checkpoint.pth \
  --motionbert_repo /home/josh/Bachelorarbeit/video_pipeline/third_party/MotionBERT \
  --motionbert_config /abs/path/to/motionbert_config.yaml \
  --motionbert_checkpoint /abs/path/to/motionbert_checkpoint.bin \
  --vitpose_device cuda:0 \
  --motionbert_device cuda:0
```

Generated outputs include:

- `pose2d_coco.npy` and `pose2d_coco.json`
- `pose2d_overlay.mp4`
- `pose2d_h36m_norm.npy`
- `pose3d_motionbert.npy`
- `pose3d_smoothed.npy`
- `animation_mixamo.json` (direct Unity runtime animation input)
- `animation_mixamo.bvh`
- `animation_mixamo.fbx` (auto-converted from BVH via Blender CLI)

## 4.1) Direct JSON Animation Path (No Mecanim, No BVH/FBX Required)

The pipeline now exports `animation_mixamo.json` with:

- `frame_rate`
- `joint_names`
- per frame:
  - `root_position` (for hips)
  - `rotations` (local quaternions in `xyzw`, same order as `joint_names`)

Unity runtime script is provided at [unity/JsonAnimationPlayer.cs](unity/JsonAnimationPlayer.cs).

Typical Unity setup:

1. Add `JsonAnimationPlayer` to your avatar root object.
2. Assign JSON file as a `TextAsset` or file path.
3. Assign `hips` (or let auto-map find `Hips`).
4. Enable `autoMapByName` if your Mixamo bone names match.

Example schema is provided at [data/samples/animation_json_example.json](data/samples/animation_json_example.json).

### Blender CLI requirement for FBX export

BVH->FBX conversion is done automatically after BVH export. Ensure Blender is
available on PATH as `blender`.

If Blender is in a custom location, pass:

```bash
python -m scripts.run_pipeline ... --blender_executable /abs/path/to/blender
```

If you only want BVH output, disable FBX conversion:

```bash
python -m scripts.run_pipeline ... --disable_fbx_export
```

## 5) Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

UI features:

- Upload MP4 video
- View original + 2D overlay video
- View 3D skeleton preview
- View joint trajectory plot
- Download BVH

## 6) Joint Mapping (COCO -> Human3.6M)

Implemented in [joint_mapping.py](preprocessing/joint_mapping.py).

- Direct mapped joints use explicit index table
- Pelvis = midpoint(left_hip, right_hip)
- Thorax = midpoint(left_shoulder, right_shoulder)
- Spine = midpoint(pelvis, thorax)
- Neck/head fallback from nose (confidence decay)

This guarantees required H36M-17 output shape for MotionBERT input.

## 7) Retargeting Math

Implemented in [mixamo_retargeter.py](retargeting/mixamo_retargeter.py).

1. Bone direction per frame:
   - `v_pose = child_joint - parent_joint`
2. Rest direction from sequence-average rest offsets
3. Quaternion alignment from rest direction to pose direction
4. Local joint rotation per bone from aligned quaternion
5. Bone-length normalization to target rig proportion

Quaternion alignment uses a robust axis-angle construction with degeneracy handling.

## 8) Coordinate System Handling

Implemented in [bvh_exporter.py](export/bvh_exporter.py#L16).

- Model-side uses right-handed convention
- Unity expects left-handed coordinates
- Conversion used for BVH export: flip Z axis
  - `(x, y, z) -> (x, y, -z)`

Euler channels are exported as `Zrotation Xrotation Yrotation` to match BVH channel declarations.

## 9) Example Sample Output

A sample BVH file is included:

- [data/samples/sample_output.bvh](data/samples/sample_output.bvh)

## 10) Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Current tests cover:

- mapping shapes
- normalization + denormalization consistency
