import os
import cv2
import numpy as np


def to_homogeneous(points):
    return np.hstack((points, np.ones((points.shape[0], 1))))


def project_points(points3d, intrinsics, extrinsics):
    points3d_h = to_homogeneous(points3d)
    points2d_h = intrinsics@((extrinsics @ points3d_h.T)[:3,])
    points2d = (points2d_h[:2,] / points2d_h[2:,]).T
    return points2d


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


def plot_3d_points(out_path, cam_imgs_dir, cam_metadata, smplx_out, cam_id, body_id, batch_size):
    for frame_n in range(batch_size):
        img_fn = os.path.join(cam_imgs_dir, f"{frame_n:04d}.png")
        img = cv2.imread(img_fn)
        # rotate 90 degrees
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if bool(cam_metadata["cam_portrait"]) else img

        joint_3d = smplx_out.joints[frame_n,].cpu().detach().numpy()

        extrinsics = cam_metadata['cam_ext']
        intrinsics = cam_metadata['cam_int']

        points2d_proj = project_points(joint_3d, intrinsics, extrinsics)

        img_out_fn = os.path.join(out_path, f"{cam_id}_{frame_n:04d}.png")
        draw_2d_pts(None, points2d_proj, background_img=img, img_fn=None, color=(0, 255, 0))
        draw_2d_pts(None, cam_metadata["joints_2d"][frame_n][body_id], background_img=img, img_fn=img_out_fn, color=(255, 0, 0))
