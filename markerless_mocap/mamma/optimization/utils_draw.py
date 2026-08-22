import os
import cv2
import numpy as np
import tqdm
from utils.utils_camera import get_projected_points, RenderCamera, project_points, project_points_np_tensor
from utils.video_tools import create_video_from_images


def draw_2d_pts(img_size, pts2d, background_img=None, img_fn=None, color=None, variance_scale=0.001):
    if background_img is not None:
        image = background_img
    else:
        h, w = img_size
        image = np.zeros((h, w, 3), dtype=np.uint8)
    # Draw points on the image
    if color is None:
        color = (0, 255, 0)
        color = np.random.randint(0, 256, 3).tolist()
    for point in pts2d:
        x, y = point[:2]
        if point.shape[0] == 3:
            color = int(255*(point[-1]/variance_scale))
            color = (0,255-color,color)
        cv2.circle(image, (int(x), int(y)), radius=0, color=color, thickness=4)  # Draw a filled circle for each point

    # Show the image with points
    if img_fn is not None:
        cv2.imwrite(img_fn, image)


def save_images(body_id, cameras_metadata_fns, smplx_out_gt, smplx_out_pred, faces, batch_size, imgs_pth, pred_fns, out_folder, out_fn, save_one_cam=False):
    if save_one_cam:
        valid_idx = [0]
    else:
        valid_idx = list(range(len(cameras_metadata_fns)))
    for i, camera_metadata_fn in enumerate(cameras_metadata_fns):
        if i not in valid_idx:
            continue
        cam_metadata = np.load(camera_metadata_fn, allow_pickle=True)
        cam_id = os.path.splitext(os.path.basename(camera_metadata_fn))[0]
        render_cam = RenderCamera(faces)
        render_cam.add_camera(cam_metadata["cam_int"])

        frame_n = smplx_out_gt.vertices.shape[0]
        save_pth = os.path.join(out_folder, "sampled_imgs", out_fn)
        save_folder_name = f"{save_pth}/{cam_id}"
        os.makedirs(save_folder_name, exist_ok=True)
        pts2d = project_points_np_tensor(smplx_out_gt.vertices.cpu().detach().numpy(), cam_metadata["cam_int"], cam_metadata["cam_ext"])
        img_fn =os.path.join(imgs_pth, cam_metadata["img_abs_path"][0])
        img_bgr = cv2.imread(img_fn)
        img_h, img_w = img_bgr.shape[:2]
        padding = 0.2
        pts2d = pts2d[:, :, :2]
        x_min, x_max = pts2d[:, :, 0].min(axis=1), pts2d[:, :, 0].max(axis=1)
        y_min, y_max = pts2d[:, :, 1].min(axis=1), pts2d[:, :, 1].max(axis=1)
        x_min = np.clip((x_min - padding * (x_max - x_min)).astype(int), a_min=0, a_max=None)
        x_max = np.clip((x_max + padding * (x_max - x_min)).astype(int), a_min=None, a_max=img_w)
        y_min = np.clip((y_min - padding * (y_max - y_min)).astype(int), a_min=0, a_max=None)
        y_max = np.clip((y_max + padding * (y_max - y_min)).astype(int), a_min=None, a_max=img_h)

        max_size = max(max((x_max-x_min)), max(y_max-y_min))
        target_size = 768

        scale = (max_size)/target_size  # if cropping around body
        print("scale", scale)
        every_n_frames = 1
        save_ext = "png"
        for frame_n_i in tqdm.tqdm(range(0, frame_n, every_n_frames)):
            img_fn =os.path.join(imgs_pth, cam_metadata["img_abs_path"][frame_n_i])
            render_cam.render(smplx_out_gt, frame_n_i, cam_metadata["cam_ext"],
                              cam_metadata["cam_int"], img_fn, smplx_out_pred,
                              save_name=os.path.join(save_folder_name, f"img_{frame_n_i:04d}.{save_ext}"), scale_img=scale, padding=padding,
                              output_size=target_size, max_size=max_size)
        try:
            create_video_from_images(save_folder_name, os.path.join(save_folder_name, "video.mp4"), extension=save_ext)
        except Exception as e:
            print("Error creating video", e)
            pass
