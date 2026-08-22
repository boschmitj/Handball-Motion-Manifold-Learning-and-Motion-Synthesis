#!/usr/bin/env python
"""SAM3 Video Text Prompt Demo — based on the official SAM3 notebook.

Tests SAM3's text prompt "person" on a local video to understand:
1. How many people are detected on the prompt frame
2. How many IDs exist after propagation (new discoveries?)
3. What the masks look like at different frames

Usage:
    python playground/sam3_video_prompt_demo.py --video data_local/IOI_09.mp4 --start 10 --end 70
    python playground/sam3_video_prompt_demo.py --video data_local/IOI_20.mp4 --start 70 --end 250
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


TAB10 = matplotlib.colormaps["tab10"]


def propagate_in_video(predictor, session_id):
    """Propagate from frame 0 to end — exactly like the official demo."""
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def extract_frames(video_path, start=None, end=None, out_dir=None):
    """Extract frames from MP4 to a temp directory for SAM3."""
    if out_dir is None:
        out_dir = "playground/extracted_frames"
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps is None or source_fps <= 0:
        source_fps = 24.0
    s = start or 0
    e = end or total

    frames_for_vis = []
    for i in range(s, e):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(out_dir, f"{i - s:06d}.jpg")
        cv2.imwrite(fname, frame)
        frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Extracted {len(frames_for_vis)} frames to {out_dir}")
    return out_dir, frames_for_vis, float(source_fps)


def render_overlay_frame(frame_rgb, frame_out):
    """Render SAM3 masks and object IDs on a single RGB frame."""
    vis = frame_rgb.copy()
    if frame_out is None:
        return vis

    fids = frame_out.get("out_obj_ids", frame_out.get("obj_ids", []))
    if hasattr(fids, "tolist"):
        fids = fids.tolist()

    masks = frame_out.get("out_binary_masks", frame_out.get("binary_masks", None))
    if masks is None:
        return vis

    for i, oid in enumerate(fids):
        if i >= masks.shape[0]:
            continue
        m = masks[i].cpu().numpy() if hasattr(masks[i], "cpu") else masks[i]
        if m.ndim == 3:
            m = m[0]
        mask = m > 0
        if not np.any(mask):
            continue

        color = np.array(TAB10(int(oid) % 10)[:3]) * 255
        vis[mask] = (0.6 * vis[mask] + 0.4 * color).astype(np.uint8)

        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(
                vis,
                str(oid),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return vis


def main():
    parser = argparse.ArgumentParser(description="SAM3 Video Text Prompt Demo")
    parser.add_argument("--video", required=True, help="Path to MP4 video")
    parser.add_argument("--start", type=int, default=None, help="Start frame (global)")
    parser.add_argument("--end", type=int, default=None, help="End frame (global)")
    parser.add_argument("--prompt", default="person", help="Text prompt (default: person)")
    parser.add_argument("--prompt_frame", type=int, default=0, help="Frame index for text prompt (relative to range)")
    parser.add_argument("--out", default="playground/output", help="Output directory")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS for MP4 (default: source FPS)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Step 1: Extract frames
    print(f"\n=== Extracting frames from {args.video} [{args.start}:{args.end}] ===")
    frames_dir, video_frames, source_fps = extract_frames(args.video, args.start, args.end)

    # Step 2: Build predictor (exactly like the demo)
    print("\n=== Building SAM3 video predictor ===")
    from sam3.model_builder import build_sam3_video_predictor
    gpus_to_use = list(range(torch.cuda.device_count()))
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # Step 3: Start session
    print(f"\n=== Starting session on {frames_dir} ===")
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=frames_dir)
    )
    session_id = response["session_id"]
    print(f"Session ID: {session_id}")

    # Step 4: Reset + text prompt (exactly like the demo)
    predictor.handle_request(
        request=dict(type="reset_session", session_id=session_id)
    )

    print(f"\n=== Adding text prompt '{args.prompt}' on frame {args.prompt_frame} ===")
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=args.prompt_frame,
            text=args.prompt,
        )
    )
    out = response["outputs"]

    # Parse initial detection
    obj_ids = out.get("out_obj_ids", out.get("obj_ids", []))
    if hasattr(obj_ids, 'tolist'):
        obj_ids = obj_ids.tolist()
    print(f"Detected {len(obj_ids)} objects on frame {args.prompt_frame}: {obj_ids}")

    # Save prompt frame visualization
    if len(video_frames) > args.prompt_frame:
        masks = out.get("out_binary_masks", out.get("binary_masks", None))
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(video_frames[args.prompt_frame])
        if masks is not None:
            for i, oid in enumerate(obj_ids):
                if i < masks.shape[0]:
                    m = masks[i].cpu().numpy() if hasattr(masks[i], 'cpu') else masks[i]
                    if m.ndim == 3:
                        m = m[0]
                    color = TAB10(i % 10)[:3]
                    overlay = np.zeros((*m.shape, 4))
                    overlay[m > 0] = (*color, 0.4)
                    ax.imshow(overlay)
                    ys, xs = np.where(m > 0)
                    if len(xs) > 0:
                        ax.text(xs.mean(), ys.mean(), str(oid), color='white',
                                fontsize=14, fontweight='bold', ha='center', va='center',
                                bbox=dict(facecolor=color, alpha=0.7, pad=2))
        ax.set_title(f"Prompt frame {args.prompt_frame}: {len(obj_ids)} detections")
        ax.axis('off')
        plt.savefig(os.path.join(args.out, "prompt_frame.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {args.out}/prompt_frame.png")

    # Step 5: Propagate (exactly like the demo)
    print(f"\n=== Propagating through video ===")
    outputs_per_frame = propagate_in_video(predictor, session_id)
    print(f"Propagation complete: {len(outputs_per_frame)} frames")

    # Step 6: Analyze results
    all_ids = set()
    for fidx, frame_out in outputs_per_frame.items():
        if frame_out is None:
            continue
        fids = frame_out.get("out_obj_ids", frame_out.get("obj_ids", []))
        if hasattr(fids, 'tolist'):
            fids = fids.tolist()
        all_ids.update(fids)

    new_ids = sorted(all_ids - set(obj_ids))
    print(f"\nInitial IDs: {obj_ids}")
    print(f"All IDs after propagation: {sorted(all_ids)}")
    print(f"New IDs discovered during propagation: {new_ids}")

    # Per-ID analysis
    print(f"\n=== Per-ID Analysis ===")
    for oid in sorted(all_ids):
        frames_present = []
        for fidx in sorted(outputs_per_frame.keys()):
            frame_out = outputs_per_frame[fidx]
            if frame_out is None:
                continue
            fids = frame_out.get("out_obj_ids", frame_out.get("obj_ids", []))
            if hasattr(fids, 'tolist'):
                fids = fids.tolist()
            masks = frame_out.get("out_binary_masks", frame_out.get("binary_masks", None))
            if oid in fids and masks is not None:
                idx = fids.index(oid)
                if idx < masks.shape[0]:
                    m = masks[idx].cpu().numpy() if hasattr(masks[idx], 'cpu') else masks[idx]
                    area = m.sum()
                    if area > 0:
                        frames_present.append((fidx, int(area)))

        if frames_present:
            avg_area = np.mean([a for _, a in frames_present])
            is_new = "NEW" if oid in new_ids else ""
            print(f"  ID {oid:2d}: {len(frames_present):3d}/{len(outputs_per_frame)} frames, "
                  f"avg area={avg_area:>8.0f}px, "
                  f"range=[{frames_present[0][0]}, {frames_present[-1][0]}] {is_new}")

    # Step 7: Render all frames and export MP4
    print(f"\n=== Rendering full propagation video ===")
    if video_frames:
        out_fps = args.fps if args.fps is not None else source_fps
        out_fps = float(out_fps) if out_fps and out_fps > 0 else 24.0

        h, w = video_frames[0].shape[:2]
        out_mp4_path = os.path.join(args.out, "propagation_all_frames.mp4")
        writer = cv2.VideoWriter(
            out_mp4_path,
            cv2.VideoWriter.fourcc(*"mp4v"),
            out_fps,
            (w, h),
        )

        written = 0
        for fidx, frame_rgb in enumerate(video_frames):
            frame_out = outputs_per_frame.get(fidx)
            vis_rgb = render_overlay_frame(frame_rgb, frame_out)
            writer.write(cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
            written += 1

        writer.release()
        print(f"Saved: {out_mp4_path} ({written} frames @ {out_fps:.2f} FPS)")
    else:
        print("No frames available to render video.")

    # Cleanup
    predictor.handle_request(dict(type="close_session", session_id=session_id))
    predictor.shutdown()
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
