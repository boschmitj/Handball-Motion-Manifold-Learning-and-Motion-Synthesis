import os
import numpy as np
from PIL import Image
from moviepy.editor import ImageClip, concatenate_videoclips


# ----- MASK -> RED RGBA -----
def mask_to_red_rgba(img):
    """Convert a white/black mask to red with alpha transparency."""
    img = img.convert("L")  # grayscale mask
    np_mask = np.array(img)

    rgba = np.zeros((np_mask.shape[0], np_mask.shape[1], 4), dtype=np.uint8)

    # white foreground
    fg = np_mask > 128

    rgba[..., 0][fg] = 255   # R
    rgba[..., 1][fg] = 0     # G
    rgba[..., 2][fg] = 0     # B
    rgba[..., 3][fg] = 255   # A

    return Image.fromarray(rgba)


def main():
    bodies = [
              "body_00",
              "body_01",
              ]
    data = [
        "inputs",
            # "masks",
            "preds"
            ]
    cam_ids = [
               "IOI_01",
               "IOI_02",
               "IOI_03",
               "IOI_19",
               "IOI_20",
           ]

    for body in bodies:
        for folder in data:
            for cam_id in cam_ids:
                input_folder = f"out/test_03_grappling2_025_grappling2/{cam_id}/{body}/{folder}"
                print("Processing folder:", input_folder)
                output_video = "output.mp4"
                fps = 20

                seconds_per_step = .1  # how long each slide takes

                # ----- LOAD IMAGES -----
                files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")])
                images = [mask_to_red_rgba(Image.open(os.path.join(input_folder, f))) for f in files]

                W, H = images[0].size

                # shift so next image overlaps 75% → move by 25%
                shift = int(W * 0.25)
                shift = 0

                # final width: base W + shift * (N - 1)
                final_width = W + shift * (len(images) - 1)

                frames = []

                # ============================================================
                # BUILD CUMULATIVE FRAMES
                # ============================================================
                for i in range(len(images)):

                    # white background canvas (no alpha)
                    canvas = Image.new("RGB", (final_width, H), (255, 255, 255))

                    # paste all previous masks including this one
                    for j in range(i + 1):
                        x = j * shift
                        # pase just the current image without alpha
                        canvas.paste(images[j].convert("RGB"), (x, 0))

                    frames.append(np.array(canvas))  # moviepy needs numpy arrays

                # ============================================================
                # CREATE VIDEO CLIPS
                # ============================================================
                clips = [
                    ImageClip(f).set_duration(seconds_per_step)
                    for f in frames
                ]

                final_video = concatenate_videoclips(clips, method="compose")

                # ============================================================
                # SAVE AS MP4 (no alpha)
                # ============================================================
                output_path = os.path.join(input_folder, output_video)

                final_video.write_videofile(
                    output_path,
                    codec="libx264",
                    fps=fps,
                    ffmpeg_params=["-pix_fmt", "yuv420p"],
                )

                print("Saved:", output_path)
                final_video.close()

if __name__ == "__main__":
    main()