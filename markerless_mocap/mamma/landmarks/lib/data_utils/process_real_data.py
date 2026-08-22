'''
python -m lib.data_utils.process_real_data --seqs_dir <path/to/extra_data>/real/WhiteRabbit --downsampled_verts_mat assets/verts_512.pkl
'''
import os, glob
import argparse
import numpy as np
import torch
from lib.data_utils.utils_smplx import get_smplx_forward, get_smplx_models
from lib.data_utils.utils_draw import project_points, draw_2d_pts
import torch.multiprocessing as mp
import tqdm
import cv2

from configs.constants import PATHS


NUM_BETAS = 11  # 11
MODEL_TYPE = 'smplx'
ROTATED_CAMS = [10, 11, 12, 13, 14, 15, 16]
SMPLX_MODELS = PATHS.SMPLX_LOCKHEAD_MODELS
SMPLX_BUN = PATHS.BODY_MODELS_PTH


def get_iou(bbox1, bbox2):
    intersection = np.maximum(np.minimum(bbox1[:, 1], bbox2[:, 1]) - np.maximum(bbox1[:, 0], bbox2[:, 0]), np.zeros_like(bbox1[:, 0]))
    union = np.maximum(bbox1[:, 1], bbox2[:, 1]) - np.minimum(bbox1[:, 0], bbox2[:, 0])
    iou = np.prod(intersection, 1) / np.prod(union, 1)
    return iou


def correct_markers_distortion(markers2d, dist_coeffs,):
    a = 1
    sign = -1
    w0 = sign*dist_coeffs[2]
    w1 = sign*dist_coeffs[3]
    w2 = 0 #-dist_coeffs[4]
    xdc = dist_coeffs[0]
    ydc = dist_coeffs[1]
    xpp = xdc
    ypp = ydc
    dx = markers2d[..., 0] - xdc
    dy = a*markers2d[..., 1] - ydc
    r = np.sqrt(dx**2 + dy**2)
    s = 1 + w0*r**2 + w1*r**4 + w2*r**6
    xc = s*dx + xpp
    yc = (s*dy + ypp)/a
    return np.stack([xc, yc], axis=-1)


def create_xycoordinates_from_img(img):
    h, w = img.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def undistord_image(img, distortion_coef):
    xx, yy = create_xycoordinates_from_img(img)
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    points = correct_markers_distortion(points, distortion_coef)
    points = points.reshape(xx.shape[0], xx.shape[1], 2)

    # having the distortion map, cv2.remap does dst(x,y) =  src(map_x(x,y),map_y(x,y))
    dst = cv2.remap(img, points[:,:,0].astype(np.float32), points[:,:,1].astype(np.float32), cv2.INTER_LINEAR)
    return dst


def pad_to_square(image):
    """
    Pad an image with zeros to make it square.

    Parameters:
    image (numpy.ndarray): The input image of shape (n, m, 3)

    Returns:
    numpy.ndarray: The square image with zero padding
    """
    # Get the original dimensions
    n, m, _ = image.shape

    # Determine the size of the square image
    size = max(n, m)

    # Calculate padding amounts
    pad_top = (size - n) // 2
    pad_bottom = size - n - pad_top
    pad_left = (size - m) // 2
    pad_right = size - m - pad_left

    # Apply padding
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    return padded_image


def plot_2d_ldmks(args):
    pred_ldmks2d, gt_ldmks, img_fn, rad_dist, save_pth= args
    img = cv2.imread(img_fn)
    if rad_dist is not None:
        img = undistord_image(img, rad_dist)
    img_orig = img.copy()
    img_h, img_w = img.shape[:2]
    scale = 1
    if pred_ldmks2d.shape[-1] == 3:
        pred_ldmks2d[:, -1] = pred_ldmks2d[:, -1] * scale**2

    # if cam_id >= 10:
    draw_2d_pts(img.shape[:2], pred_ldmks2d, background_img=img, color=(0, 255, 0), variance_scale=0.001*500)
    if gt_ldmks is not None:
        draw_2d_pts(img.shape[:2], gt_ldmks, background_img=img, color=(0, 0, 255), variance_scale=0.001*500)
    img = np.concatenate([img_orig, img], axis=1)
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    if save_pth is not None:
        cv2.imwrite(save_pth, img)
    return pad_to_square(img)


def get_projected_points(smplx_out, cam_metadata, batch_size):
    pts2d_gt_np = []
    for frame_n in range(batch_size):
        extrinsics = cam_metadata['cam_ext']
        intrinsics = cam_metadata['cam_int']
        pts3d = smplx_out.vertices[frame_n,].cpu().detach().numpy()
        points2d_proj = project_points(pts3d, intrinsics, extrinsics)
        pts2d_gt_np.append(points2d_proj)
    pts2d_gt_np = np.stack(pts2d_gt_np, axis=0)
    return pts2d_gt_np


def read_gt(metadata_world: dict, smplx_model, smplx_model_pth, num_betas, batch_size: int, body_id: int = 0, device: str = "cuda", v_template=None):
    smplx_pose = torch.tensor(metadata_world["pose_world"][:, body_id]).float().to(device)
    smplx_trans = torch.tensor(metadata_world["pose_trans_world"][:, body_id]).float().to(device)
    smplx_betas = torch.tensor(metadata_world["shape"][body_id])[None,].repeat((batch_size, 1)).float().to(device)

    gender = "neutral"
    if v_template is not None:
        import smplx
        gender = metadata_world["gender"][0] if v_template is not None else "neutral"
        print("Using custom v_template")
        smplx_model_gt = {gender: smplx.create(smplx_model_pth, model_type=MODEL_TYPE,
                                        gender=gender,
                                        ext='npz',
                                        flat_hand_mean=True,
                                        num_pca_comps=45,
                                        num_betas=num_betas, v_template=v_template[body_id],
                                        use_pca=False).to(device)}
    else:
        smplx_model_gt = smplx_model

    smplx_out = get_smplx_forward(smplx_pose,
                                    smplx_betas[:, :num_betas],
                                    smplx_trans,
                                    gender=gender,
                                    smplx_models=smplx_model_gt)

    return smplx_out


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--seqs_dir', type=str, default='/path/to/sequences', help='Path to sequences directory')
    args.add_argument('--gt_folder', type=str, default='gt', help='GT folder name')
    args.add_argument('--device', type=str, default='cuda', help='Device to run on')
    args.add_argument('--downsampled_verts_mat', type=str, default=None, help='Path to downsampled verts matrix')
    return args.parse_args()


def process_seqs(seqs, gt_folder, device="cuda", downsampled_verts_mat=None,
                 dataset_name='vicon', iter_mode=False, cam_id: str ="", flat_hand=True,
                 use_predicted_mask=False):
    if dataset_name in ['vicon', "whiterabbit"]:
        smplx_model_pth = SMPLX_MODELS
        num_betas = 11  # doesn't matter, its v-poser
        num_pca_comps = 45
    elif dataset_name == 'harmony4d':
        smplx_model_pth = SMPLX_MODELS
        num_betas = 16  # doesn't matter, its v-poser
        num_pca_comps = 45
    elif dataset_name == 'hi4d':
        gt_folder = ''
        smplx_model_pth = SMPLX_MODELS
        num_betas = 16  # doesn't matter, its v-poser
        num_pca_comps = 45
    elif dataset_name == 'rich':
        gt_folder = ''
        num_betas = 10
        flat_hand= False
        smplx_model_pth = SMPLX_BUN
        num_pca_comps=12
    elif dataset_name == 'chi3d':
        gt_folder = ''
        num_betas = 10
        smplx_model_pth = SMPLX_MODELS
        num_pca_comps=12
    elif dataset_name == 'moyo':
        smplx_model_pth = SMPLX_MODELS
        num_betas = 100
        num_pca_comps = 45
        flat_hand = True
    elif dataset_name == 'weastcoast':
        smplx_model_pth = SMPLX_MODELS
        num_betas = 100
        num_pca_comps = 45
        flat_hand = True
    elif dataset_name == 'h36m':
        smplx_model_pth = SMPLX_MODELS
        num_betas = 100
        num_pca_comps = 45
        flat_hand = True
        gt_folder = ''
    elif dataset_name == "mosh" or dataset_name == "data_release":
        smplx_model_pth = SMPLX_MODELS
        num_betas = 100
        num_pca_comps = 45
        flat_hand = True

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    smplx_model = get_smplx_models(smplx_model_pth, num_betas, model_type=MODEL_TYPE, device=device, flat_hand=flat_hand, num_pca_comps=num_pca_comps)  # for bedlam is true
    device = "cuda"
    for seq in seqs:
        seq_name = os.path.basename(seq)
        if downsampled_verts_mat is not None:
            downsampled_verts_mat = np.load(downsampled_verts_mat, allow_pickle=True)
        gt_path = os.path.join(seq, gt_folder)
        metadata_world = np.load(os.path.join(gt_path, 'global.npz'))
        if cam_id == "":
            if dataset_name in ["vicon", "whiterabbit", "harmony4d", "rich", "chi3d", "weastcoast", "hi4d", "h36m", "mosh", "data_release"]:
                gt_files = sorted(glob.glob(os.path.join(gt_path, 'IOI_*.npz')))
            elif dataset_name == "moyo":
                gt_files = sorted(glob.glob(os.path.join(gt_path, 'YOGI_Cam_*.npz')))
        else:
            if dataset_name in ["vicon", "whiterabbit", "harmony4d", "rich", "chi3d", "weastcoast", "hi4d", "h36m", "mosh", "data_release"]:
                gt_files = sorted(glob.glob(os.path.join(gt_path, f'IOI_{cam_id}*.npz')))
            elif dataset_name == "moyo":
                gt_files = sorted(glob.glob(os.path.join(gt_path, f'YOGI_Cam_{cam_id}*.npz')))

        # if len(gt_files) == 0:

        assert len(gt_files) > 0, f"No gt files found in {gt_path} either with IOI_ or YOGI_Cam_ \n for {gt_files} \n with cam_id {cam_id}"

        v_template = metadata_world["v_template"] if "v_template" in metadata_world else None
        batch_size = metadata_world["frames_len"].item()
        bodies = int(metadata_world["people_len"]) if "people_len" in metadata_world else 1

        for body_id in range(bodies):
            if "vertices_3d_world" not in metadata_world and "3d_keypoints_17" not in metadata_world:
                smplx_out_gt = read_gt(metadata_world, smplx_model, smplx_model_pth, num_betas, batch_size, body_id, device, v_template=v_template)
                # NOTE: we assume there's just two people in the scene, we need to extend this or improve this part of the code
                if bodies > 1:
                    body_id_other = 1 - body_id
                    smplx_out_gt_other = read_gt(metadata_world, smplx_model, smplx_model_pth, num_betas, batch_size, body_id_other, device, v_template=v_template)

            for gt_file in gt_files:
                camera_metadata = np.load(gt_file)
                if "vertices_3d_world" in metadata_world:
                    metadata_world["vertices_3d_world"][:, body_id]
                    camera_metadata['cam_ext']
                    camera_metadata['cam_int']
                    vertices3d_cam = np.einsum("mn,fvn->fvm", camera_metadata['cam_ext'][:3, :3], metadata_world["vertices_3d_world"][:, body_id]) + camera_metadata['cam_ext'][:3, 3][None, None, :]
                    pts2d_gt_np = np.einsum("mn,fvn->fvm", camera_metadata['cam_int'], vertices3d_cam)
                    pts2d_gt_np = pts2d_gt_np[:, :, :2] / pts2d_gt_np[:, :, 2:]

                elif "3d_keypoints_17" in metadata_world:
                    vertices3d_cam = np.einsum("mn,fvn->fvm", camera_metadata['cam_ext'][:3, :3], metadata_world["3d_keypoints_17"]) + camera_metadata['cam_ext'][:3, 3][None, None, :]
                    pts2d_gt_np = np.einsum("mn,fvn->fvm", camera_metadata['cam_int'], vertices3d_cam)
                    pts2d_gt_np = pts2d_gt_np[:, :, :2] / pts2d_gt_np[:, :, 2:]
                else:
                    pts2d_gt_np = get_projected_points(smplx_out_gt, camera_metadata, batch_size)
                if downsampled_verts_mat is not None:
                    pts2d_gt_np = torch.einsum("ij,bjk->bik", downsampled_verts_mat, torch.from_numpy(pts2d_gt_np).float()).numpy()
                print(f"Processing {seq_name} {body_id} {gt_file}", "with size: ", pts2d_gt_np.shape, flush=True)

                if bodies > 1:
                    body_id_other = 1 - body_id
                    if "vertices_3d_world" in metadata_world:
                        metadata_world["vertices_3d_world"][:, body_id_other]
                        camera_metadata['cam_ext']
                        camera_metadata['cam_int']
                        vertices3d_cam_other = np.einsum("mn,fvn->fvm", camera_metadata['cam_ext'][:3, :3], metadata_world["vertices_3d_world"][:, body_id_other]) + camera_metadata['cam_ext'][:3, 3][None, None, :]
                        pts2d_gt_np_other = np.einsum("mn,fvn->fvm", camera_metadata['cam_int'], vertices3d_cam_other)
                        pts2d_gt_np_other = pts2d_gt_np_other[:, :, :2] / pts2d_gt_np_other[:, :, 2:]

                    elif "3d_keypoints_17" in metadata_world:
                        vertices3d_cam_other = np.einsum("mn,fvn->fvm", camera_metadata['cam_ext'][:3, :3], metadata_world["3d_keypoints_17"]) + camera_metadata['cam_ext'][:3, 3][None, None, :]
                        pts2d_gt_np_other = np.einsum("mn,fvn->fvm", camera_metadata['cam_int'], vertices3d_cam_other)
                        pts2d_gt_np_other = pts2d_gt_np_other[:, :, :2] / pts2d_gt_np_other[:, :, 2:]
                    else:
                        pts2d_gt_np_other = get_projected_points(smplx_out_gt_other, camera_metadata, batch_size)
                    if downsampled_verts_mat is not None:
                        pts2d_gt_np_other = torch.einsum("ij,bjk->bik", downsampled_verts_mat, torch.from_numpy(pts2d_gt_np_other).float()).numpy()
                    print(f"Processing {seq_name} {body_id_other} {gt_file}", "with size: ", pts2d_gt_np_other.shape, flush=True)

                if "gt_smplx_contact" in metadata_world:
                    contact_gt = metadata_world["gt_smplx_contact"][:, body_id]
                    contact_gt = torch.einsum("ij,bj->bi", downsampled_verts_mat, torch.from_numpy(contact_gt).float()).numpy()
                else:
                    contact_gt = None

                if "floor_contact_mask" in metadata_world:
                    floor_contact_gt = metadata_world["floor_contact_mask"][:, body_id]
                    floor_contact_gt = torch.einsum("ij,bj->bi", downsampled_verts_mat, torch.from_numpy(floor_contact_gt).float()).numpy()
                else:
                    floor_contact_gt = None

                if iter_mode:
                    for j in range(0, pts2d_gt_np.shape[0]):
                        img_path_mask = ''
                        if dataset_name in ['vicon', "whiterabbit"]:
                            img_path_mask = os.path.join(seq, "masks", f"IOI_{cam_id}")
                            img_path_mask = os.path.join(img_path_mask, f"{j:06d}_{body_id:02d}.png")

                            if use_predicted_mask:
                                cam_name = str(camera_metadata["cam_name"])
                                img_fn = camera_metadata["img_abs_path"][j]
                                pred_mask_dir = os.path.join(f"{PATHS.sam2_masks()}/sam2_segment_westcoast/output_masks/whiterabbit", seq_name, cam_name, "masks")
                                img_fn = os.path.basename(img_fn)
                                img_fn = "mask_" + f"{j:04d}_{body_id+1:02d}.png"
                                img_path_mask = os.path.join(pred_mask_dir, img_fn)

                        elif dataset_name == "harmony4d":
                            path_dir = seq.split("/harmony4d")[0]
                            img_fn = camera_metadata["img_abs_path"][j].split("harmony4d/")[-1]
                            img_fn = os.path.join(path_dir, "harmony4d", img_fn)
                            img_path_mask = img_fn.replace(".jpg", "").replace("rectified_images", "masks")
                            if use_predicted_mask:
                                base_name = os.path.basename(img_path_mask)
                                base_name = f'{int(base_name)-1:05d}'
                                dir_name = os.path.dirname(img_path_mask)
                                img_path_mask = os.path.join(dir_name, base_name)
                                img_path_mask = img_path_mask.replace("masks", "sam2_masks").replace("/rgb/0", "_rgb/masks/mask_")+ f"_{body_id+1:02d}.png"
                            else:
                                img_path_mask = img_path_mask+ f"_{body_id:02d}.png"

                        elif dataset_name == "rich":
                            img_fn = camera_metadata["img_abs_path"][j]
                            img_path_mask = camera_metadata["mask_abs_path"][j]

                            if use_predicted_mask:
                                cam_name = str(camera_metadata["cam_name"])
                                pred_mask_dir = os.path.join(f"{PATHS.sam2_masks()}/sam2_segment_westcoast/output_masks/rich", seq_name, cam_name, "masks")
                                img_fn = os.path.basename(img_fn)
                                img_fn = "mask_" + f"{j:04d}_{body_id+1:02d}.png"
                                img_path_mask = os.path.join(pred_mask_dir, img_fn)

                        elif dataset_name == "moyo":
                            img_fn = camera_metadata["img_abs_path"][j]
                            img_path_mask = camera_metadata["mask_abs_path"][j]

                            if use_predicted_mask:
                                cam_name = str(camera_metadata["cam_name"])
                                pred_mask_dir = os.path.join(f"{PATHS.sam2_masks()}/sam2_segment_westcoast/output_masks/moyo", seq_name, cam_name, "masks")
                                img_fn = os.path.basename(img_fn)
                                img_fn = "mask_" + f"{j:04d}_{body_id+1:02d}.png"
                                img_path_mask = os.path.join(pred_mask_dir, img_fn)

                        elif dataset_name == "hi4d":
                            img_fn = camera_metadata["img_abs_path"][j]
                            img_path_mask = camera_metadata["mask_abs_path"][j]

                            if use_predicted_mask:
                                cam_name = str(camera_metadata["cam_name"])
                                pred_mask_dir = os.path.join(f"{PATHS.sam2_masks()}/sam2_segment_hi4d/output_masks/hi4d", seq_name, cam_name, "masks")
                                img_fn = os.path.basename(img_fn)
                                img_fn = "mask_" + f"{j:04d}_{body_id+1:02d}.png"
                                img_path_mask = os.path.join(pred_mask_dir, img_fn)

                            else:
                                img_path_mask = img_path_mask+ f"_{body_id:02d}.png"

                        elif dataset_name == "mosh":
                            img_fn = camera_metadata["img_abs_path"][j]
                            img_path_mask = camera_metadata["mask_abs_path"][j]
                            if use_predicted_mask:
                                cam_name = str(camera_metadata["cam_name"])
                                mosh_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(gt_file))))
                                pred_mask_dir = os.path.join(f"{PATHS.extra_data()}/real/MAMMAEval_ViconMosh/sam2/{mosh_dir}", seq_name, cam_name, "masks")
                                img_fn = os.path.basename(img_fn)
                                img_fn = "mask_" + f"{j:04d}_{body_id+1:02d}.png"
                                img_path_mask = os.path.join(pred_mask_dir, img_fn)

                        elif dataset_name == "data_release":
                            img_fn = camera_metadata["img_abs_path"][j]
                            img_path_mask = camera_metadata["mask_abs_path"][j]
                            if use_predicted_mask:
                                cam_name = str(camera_metadata["cam_name"])
                                mosh_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(gt_file))))
                                pred_mask_dir = os.path.join(f"{PATHS.extra_data()}/real/MPI_Dance/sam2/{mosh_dir}", seq_name, cam_name, "masks")
                                img_fn = os.path.basename(img_fn)
                                img_fn = "mask_" + f"{j:04d}_{body_id+1:02d}.png"
                                img_path_mask = os.path.join(pred_mask_dir, img_fn)

                        elif dataset_name == "h36m":
                            img_fn = camera_metadata["img_abs_path"][j]
                            img_path_mask = None
                        elif dataset_name == "chi3d":
                            img_fn = camera_metadata["img_abs_path"][j]
                            img_path_mask = camera_metadata["mask_abs_path"][j]

                            if use_predicted_mask:
                                cam_name = str(camera_metadata["cam_name"])
                                pred_mask_dir = os.path.join(f"{PATHS.sam2_masks()}/sam2_segment_chi3d/output_masks/chi3d", seq_name, cam_name, "masks")
                                img_fn = os.path.basename(img_fn)
                                img_fn = "mask_" + img_fn.replace(".png", f"_{body_id+1:02d}.png")
                                img_path_mask = os.path.join(pred_mask_dir, img_fn)

                        elif dataset_name == "weastcoast":
                            img_fn = camera_metadata["img_abs_path"][j]
                            img_path_mask = camera_metadata["mask_abs_path"][j]

                            if use_predicted_mask:
                                cam_name = str(camera_metadata["cam_name"])
                                pred_mask_dir = os.path.join(f"{PATHS.sam2_masks()}/sam2_segment_westcoast/output_masks/westcoast", seq_name, cam_name, "masks")
                                img_fn = os.path.basename(img_fn)
                                img_fn = "mask_" + f"{j:04d}_{body_id+1:02d}.png"
                                img_path_mask = os.path.join(pred_mask_dir, img_fn)

                            else:
                                img_path_mask = img_path_mask+ f"_{body_id:02d}.png"

                        if dataset_name == "h36m":
                            x1, y1, x2, y2 = camera_metadata['bboxes'][j, 0], camera_metadata['bboxes'][j, 1], camera_metadata['bboxes'][j, 2], camera_metadata['bboxes'][j, 3]
                        else:
                            x1, y1, x2, y2 = np.min(pts2d_gt_np[j,:, 0]), np.min(pts2d_gt_np[j,:,1]), np.max(pts2d_gt_np[j,:,0]), np.max(pts2d_gt_np[j,:,1])
                        bbox = np.array([x1, y1, x2, y2])[None,]

                        person1_bbx = np.array([x1, y1, x2, y2]).reshape(1, 2, 2)
                        if bodies > 1:
                            x1, y1, x2, y2 = np.min(pts2d_gt_np_other[j,:, 0]), np.min(pts2d_gt_np_other[j,:,1]), np.max(pts2d_gt_np_other[j,:,0]), np.max(pts2d_gt_np_other[j,:,1])
                            person2_bbx = np.array([x1, y1, x2, y2]).reshape(1, 2, 2)
                            iou = get_iou(person1_bbx, person2_bbx).item()
                        else:
                            iou = 0

                        contact_person_frame = contact_gt[j] if contact_gt is not None else None
                        floor_contact_gt_person_frame = floor_contact_gt[j] if floor_contact_gt is not None else None

                        yield pts2d_gt_np[j], bbox, camera_metadata["img_abs_path"][j], img_path_mask, body_id, j, iou, contact_person_frame, floor_contact_gt_person_frame

                else:
                    cam_frame_fns = camera_metadata["img_abs_path"]
                    args_3d_gt = [(pts2d_gt_np[j, :, :2], None, os.path.join("", cam_frame_fns[j]), camera_metadata["vicon_radial_2"], f"img_{j:02d}.png") for j in range(0, pts2d_gt_np.shape[0], 15)]

                    pool = mp.Pool(2)
                    imgs_3d_gt = []
                    for result in tqdm.tqdm(pool.imap(plot_2d_ldmks, args_3d_gt), total=len(args_3d_gt)):
                        imgs_3d_gt.append(result)
                        pass
                    pool.close()
                    pool.join()


def main():
    args = parser()
    seqs_dir = args.seqs_dir
    gt_folder = args.gt_folder

    # find only folders
    seqs = [os.path.join(seqs_dir, f) for f in os.listdir(seqs_dir) if os.path.isdir(os.path.join(seqs_dir, f))]
    process_seqs(seqs, gt_folder, args.device, args.downsampled_verts_mat, args.dataset_name)


if __name__ == '__main__':
    main()