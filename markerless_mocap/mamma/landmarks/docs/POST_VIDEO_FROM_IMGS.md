# post_video_from_imgs.py

Create MP4 videos by vertically stacking camera views for each body in every sequence.

## Dataset layout
```
<dataset_dir>/<sequence>/<camera>/<body>/
    frame_00001.jpg|png
    frame_00002.jpg|png
    ...
```
All cameras must contain the same bodies; frames are aligned by sorted filename and truncated to the shortest camera.

## Usage
```
python utils/post_video_from_imgs.py --dataset_dir <dataset_dir> --output_dir <out_dir> --fps 30 --num_workers 4 --overwrite
```
- One process per sequence (up to `num_workers`).
- Outputs MP4s at `<out_dir>/<sequence>/<body>.mp4`.
- Supports `.jpg`, `.jpeg`, `.png` frames.

