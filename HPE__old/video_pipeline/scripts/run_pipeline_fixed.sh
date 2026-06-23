#!/usr/bin/env bash
set -euo pipefail

# Fixed-parameter runner for reproducible execution.
# Usage:
#   bash scripts/run_pipeline_fixed.sh --output_dir outputs/runX [--video data/samples/other.mp4] [--blender_executable /path/to/blender]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTPUT_DIR=""
VIDEO_PATH="$ROOT_DIR/data/samples/sample_video2.mp4"
BLENDER_EXECUTABLE_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --video)
      VIDEO_PATH="$2"
      shift 2
      ;;
    --blender_executable)
      BLENDER_EXECUTABLE_ARG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash scripts/run_pipeline_fixed.sh --output_dir outputs/runX [--video data/samples/other.mp4] [--blender_executable /path/to/blender]"
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Missing required argument: --output_dir"
  exit 1
fi

if [[ "$VIDEO_PATH" != /* ]]; then
  VIDEO_PATH="$ROOT_DIR/$VIDEO_PATH"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Input video not found: $VIDEO_PATH"
  exit 1
fi

VITPOSE_CONFIG="$ROOT_DIR/third_party/ViTPose/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py"
VITPOSE_CHECKPOINT="$ROOT_DIR/third_party/ViTPose/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth"
MOTIONBERT_REPO="$ROOT_DIR/third_party/MotionBERT"
MOTIONBERT_CONFIG="$ROOT_DIR/configs/motionbert_h36m_global_infer.yaml"
MOTIONBERT_CHECKPOINT="$ROOT_DIR/third_party/MotionBERT/checkpoint/pose3d/MB_train_h36m/best_epoch.bin"

BLENDER_EXECUTABLE="${BLENDER_EXECUTABLE_ARG:-${BLENDER_EXECUTABLE:-}}"
FBX_ARGS=()
DISABLE_FBX_EXPORT=0

if [[ -z "$BLENDER_EXECUTABLE" ]]; then
  if command -v blender >/dev/null 2>&1; then
    BLENDER_EXECUTABLE="$(command -v blender)"
  elif [[ -x "/mnt/c/Program Files (x86)/Steam/steamapps/common/Blender/blender.exe" ]]; then
    BLENDER_EXECUTABLE="/mnt/c/Program Files (x86)/Steam/steamapps/common/Blender/blender.exe"
  else
    echo "Blender executable not found. Running pipeline without FBX export."
    DISABLE_FBX_EXPORT=1
  fi
fi

if [[ -n "$BLENDER_EXECUTABLE" ]] && [[ ! -x "$BLENDER_EXECUTABLE" ]]; then
  echo "Provided BLENDER_EXECUTABLE is not executable: $BLENDER_EXECUTABLE"
  echo "Running pipeline without FBX export."
  DISABLE_FBX_EXPORT=1
fi

if [[ $DISABLE_FBX_EXPORT -eq 1 ]]; then
  FBX_ARGS+=(--disable_fbx_export)
elif [[ -n "$BLENDER_EXECUTABLE" ]]; then
  FBX_ARGS+=(--blender_executable "$BLENDER_EXECUTABLE")
fi

echo "Running pipeline"
echo "  video: $VIDEO_PATH"
echo "  output: $OUTPUT_DIR"
if [[ $DISABLE_FBX_EXPORT -eq 1 ]]; then
  echo "  fbx export: disabled"
else
  echo "  blender: $BLENDER_EXECUTABLE"
fi

cd "$ROOT_DIR"
conda run -n handball3d python -m scripts.run_pipeline \
  --video "$VIDEO_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --vitpose_config "$VITPOSE_CONFIG" \
  --vitpose_checkpoint "$VITPOSE_CHECKPOINT" \
  --motionbert_repo "$MOTIONBERT_REPO" \
  --motionbert_config "$MOTIONBERT_CONFIG" \
  --motionbert_checkpoint "$MOTIONBERT_CHECKPOINT" \
  --vitpose_device cuda:0 \
  --motionbert_device cuda:0 \
  "${FBX_ARGS[@]}"

echo "Done."
