import os
import numpy as np
import cv2
import trimesh
from scipy.ndimage import distance_transform_edt


class DrawUV:
    def __init__(self, mesh_obj_pth="assets/smplx_uv.obj", uv_img_pth="assets/smplx_uv.png", downsampled_verts_mat_path="assets/verts_512.pkl"):
        # load obj file
        mesh = trimesh.load_mesh(mesh_obj_pth)
        mesh.merge_vertices(merge_tex=True)
        uv = mesh.visual.uv
        self.uv_img_size = 1024//2
        uv = uv * self.uv_img_size
        downsampled_verts_mat = np.load(downsampled_verts_mat_path, allow_pickle=True).detach().cpu().numpy()
        self.donwsampled_uv = np.einsum("ij,jb->ib", downsampled_verts_mat, uv)

        # read with alpha channel
        uv_img = cv2.imread(uv_img_pth, cv2.IMREAD_UNCHANGED)
        uv_img = cv2.resize(uv_img, (self.uv_img_size, self.uv_img_size))

        trans_mask = uv_img[:,:,3] == 0
        uv_img[trans_mask] = [255, 255, 255, 255]
        self.uv_img = uv_img

    def new_uv_img(self):
        return self.uv_img.copy()

    def expand_colors(self, image):
        """
        Expand colors over an image without blending.

        Args:
            image (numpy.ndarray): An NxMx3 image with some pixels having color and the rest being black (0, 0, 0).

        Returns:
            numpy.ndarray: The modified image with expanded colors.
        """
        mask = np.any(image > 0, axis=-1)  # Find pixels with color
        distances, indices = distance_transform_edt(~mask, return_indices=True)  # Get nearest non-zero pixel

        expanded_image = image[indices[0], indices[1]]  # Expand nearest color to blank pixels

        return np.clip(expanded_image, 0, 255).astype(np.uint8)

    def draw_visibility_img(self, img, i, color=(0, 0, 255, 255)):
        # for i in range(self.donwsampled_uv.shape[0]):
        cv2.circle(img, (int(self.donwsampled_uv[i, 0]),
                                    self.uv_img_size-int(self.donwsampled_uv[i, 1])),
                                    2, color, -1)