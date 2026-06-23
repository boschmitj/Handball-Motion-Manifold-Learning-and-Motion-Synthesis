#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_motionbert.sh /absolute/path/to/third_party

TARGET_DIR=${1:-"$PWD/third_party"}
mkdir -p "$TARGET_DIR"
pushd "$TARGET_DIR" >/dev/null

if [[ ! -d MotionBERT ]]; then
  git clone https://github.com/Walter0807/MotionBERT.git
fi

pushd MotionBERT >/dev/null
python -m pip install -r requirements.txt
# pip install -e .
popd >/dev/null
popd >/dev/null

echo "MotionBERT setup completed in $TARGET_DIR/MotionBERT"
