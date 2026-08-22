"""Interactive GUI for clicking on people to track.

Opens a tkinter window showing video frames. The user clicks on each person
to create positive point prompts for SAM.

Controls:
    Keys 0-9:     Switch active person ID
    Left-click:   Add a point for the active person
    Right-click:  Remove the nearest point for the active person
    +/-:          Zoom in/out
    Slider:       Navigate frames
    Close window: Finish and return clicks

Returns:
    Dict mapping person_id (int) -> {frame_idx: [(x, y), ...]}
"""
import torch
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np


# Distinct colors for up to 10 people
PERSON_COLORS = [
    "#FF0000",  # 0: red
    "#0066FF",  # 1: blue
    "#00CC00",  # 2: green
    "#FF9900",  # 3: orange
    "#CC00CC",  # 4: magenta
    "#00CCCC",  # 5: cyan
    "#FFCC00",  # 6: yellow
    "#FF66CC",  # 7: pink
    "#6633FF",  # 8: purple
    "#99CC00",  # 9: lime
]


def show_images_gui(inference_state, sam_version="sam2"):
    """Open an interactive GUI for clicking on people in video frames.

    Args:
        inference_state: SAM inference state containing loaded video frames.
        sam_version: "sam2", "sam3", or "sam3_prompt" — determines image normalization.

    Returns:
        Dict mapping person_id (int) -> {frame_idx: [(x, y), ...]}
        Coordinates are in the network resolution (will be converted by caller).
    """
    images = inference_state['images'].detach().cpu().float()
    if sam_version in ("sam3", "sam3_prompt"):
        img_mean = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
        img_std = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
    else:
        img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    images = images * img_std + img_mean
    images = torch.clamp(images, 0, 1)
    images = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    # clicks[person_id] = {frame_idx: [(x, y), ...]}
    clicks = {}
    active_person = [0]
    zoom_factor = [1.0]
    current_index = [0]

    def update_image():
        idx = current_index[0]
        img = Image.fromarray(images[idx])
        zf = zoom_factor[0]
        img = img.resize((int(img.width * zf), int(img.height * zf)), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(img)

        # Draw all person clicks on this frame
        for pid, frame_clicks in clicks.items():
            if idx in frame_clicks:
                color = PERSON_COLORS[pid % len(PERSON_COLORS)]
                for x, y in frame_clicks[idx]:
                    r = 6
                    draw.ellipse(
                        (x * zf - r, y * zf - r, x * zf + r, y * zf + r),
                        fill=color, outline="white", width=1,
                    )
                    # Draw person ID label next to the dot
                    draw.text((x * zf + r + 2, y * zf - r), str(pid),
                              fill=color)

        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

        # Status bar
        total_points = sum(
            len(pts) for fc in clicks.values() for pts in fc.values()
        )
        n_people = len([p for p, fc in clicks.items() if any(fc.values())])
        status_label.config(
            text=f"Frame {idx + 1}/{len(images)}  |  "
                 f"Active: Person {active_person[0]}  |  "
                 f"{n_people} people, {total_points} points total"
        )

        # Highlight active person button
        for pid, btn in person_buttons.items():
            if pid == active_person[0]:
                btn.config(relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED)

    def on_slider_change(value):
        current_index[0] = int(value)
        update_image()

    def on_image_click(event):
        zf = zoom_factor[0]
        x, y = int(event.x / zf), int(event.y / zf)
        pid = active_person[0]
        if pid not in clicks:
            clicks[pid] = {}
        if current_index[0] not in clicks[pid]:
            clicks[pid][current_index[0]] = []
        clicks[pid][current_index[0]].append((x, y))
        update_image()

    def on_image_right_click(event):
        zf = zoom_factor[0]
        x, y = int(event.x / zf), int(event.y / zf)
        pid = active_person[0]
        if pid in clicks and current_index[0] in clicks[pid]:
            pts = clicks[pid][current_index[0]]
            if pts:
                closest = min(pts, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
                pts.remove(closest)
                if not pts:
                    del clicks[pid][current_index[0]]
                update_image()

    def set_active_person(pid):
        active_person[0] = pid
        update_image()

    def on_key_press(event):
        if event.char.isdigit():
            set_active_person(int(event.char))
        elif event.keysym == 'plus' or event.keysym == 'equal':
            zoom_factor[0] *= 1.1
            update_image()
        elif event.keysym == 'minus':
            zoom_factor[0] /= 1.1
            update_image()

    def on_close():
        root.destroy()

    # --- Build UI ---
    root = tk.Tk()
    root.title("MAMMA Masks — Interactive Person Selection")
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Instructions
    instr = tk.Label(
        root,
        text="Keys 0-9: switch person  |  Left-click: add point  |  "
             "Right-click: remove  |  +/-: zoom  |  Close when done",
        font=("Helvetica", 10), fg="gray",
    )
    instr.pack(pady=2)

    # Image
    img_label = tk.Label(root)
    img_label.pack()
    img_label.bind("<Button-1>", on_image_click)
    img_label.bind("<Button-3>", on_image_right_click)

    # Status bar
    status_label = tk.Label(root, text="", font=("Helvetica", 11, "bold"))
    status_label.pack(pady=2)

    # Frame slider
    slider = tk.Scale(
        root, from_=0, to=len(images) - 1, orient=tk.HORIZONTAL,
        command=on_slider_change, length=400,
    )
    slider.pack(fill=tk.X, padx=10)

    # Person buttons (0-9)
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)
    person_buttons = {}
    for pid in range(10):
        color = PERSON_COLORS[pid]
        btn = tk.Button(
            btn_frame, text=f" {pid} ", width=3,
            bg=color, fg="white", font=("Helvetica", 10, "bold"),
            command=lambda p=pid: set_active_person(p),
        )
        btn.pack(side=tk.LEFT, padx=2)
        person_buttons[pid] = btn

    root.bind("<Key>", on_key_press)

    update_image()
    root.mainloop()

    # Convert to the format expected by process_first_video_interactive:
    # {person_id: {frame_idx: [(x, y), ...]}} — only include non-empty
    result = {}
    for pid, frame_clicks in clicks.items():
        non_empty = {f: pts for f, pts in frame_clicks.items() if pts}
        if non_empty:
            result[pid] = non_empty
    return result
