# transform video to images
# Usage: python video2image.py --video_path /path/to/video --output_path /path/to/output --fps 1
import cv2
import os
import argparse

def video2image(video_path, output_path, fps):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % fps == 0:
            cv2.imwrite(os.path.join(output_path, f'img_{str(count).zfill(4)}.jpg'), frame)
        count += 1
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help='path to video')
    parser.add_argument('--output_path', type=str, required=True, help='path to output')
    parser.add_argument('--fps', type=int, default=1, help='frame per second')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    video2image(args.video_path, args.output_path, args.fps)
