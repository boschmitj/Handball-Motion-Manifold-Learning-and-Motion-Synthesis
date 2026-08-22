import torch
import sys, os

from omegaconf import OmegaConf
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import cv2
from .HumanPoseEstimation import HumanPoseEstimationDataset as Dataset
from ..models.models_2d.utils.transform import fliplr_joints, get_affine_transform
from lib.datasets.augmentations import extreme_cropping, augment_image, augment_mask, xyxy2cs
import webdataset as wds
import random
import joblib
import json
from configs.constants import PATHS


def convert_cvimg_to_tensor(cvimg: np.array):
    """
    Convert image from HWC to CHW format.
    Args:
        cvimg (np.array): Image of shape (H, W, 3) as loaded by OpenCV.
    Returns:
        np.array: Output image of shape (3, H, W).
    """
    # from h,w,c(OpenCV) to c,h,w
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    # from int to float
    img = img.astype(np.float32)
    return img


class BEDLAM_WD(Dataset):
    def __init__(self,
                 label_path=None,
                 subsample_pts_fn=None,
                 task='landmark_2d',
                 org_w=1920.0,
                 org_h=1080.0,
                 normalize_plus_min_one=False,
                 hand_weight=1.0,
                 **kwargs):
        super(BEDLAM_WD, self).__init__(**kwargs)
        if subsample_pts_fn is None:
            subsample_pts_fn = PATHS.SUBSAMPLE_PTS_DIR

        # 512-vertex flip pairs: left half (0..255) <-> right half (256..511).
        self.flip_pairs = [[i, i + 256] for i in range(256)]
        joints_weight = [1]*self.max_num_joints
        self.joints_weight = np.array(joints_weight).reshape(self.max_num_joints, 1)
        # 512-vertex SMPL-X body partition: indices grouped by part, then
        # concatenated and argsorted to produce the permutation that
        # mvhead.py uses to reorder per-part predictions back to natural order.
        parts_path = os.path.join(os.path.dirname(__file__), "_smplx_512_body_parts.json")
        with open(parts_path, "r") as f:
            self.body_parts_dict = json.load(f)
        body_idx = (
            self.body_parts_dict["body"]
            + self.body_parts_dict["left_hand"] + self.body_parts_dict["right_hand"]
            + self.body_parts_dict["left_feet"] + self.body_parts_dict["right_feet"]
            + self.body_parts_dict["left_head"] + self.body_parts_dict["right_head"]
        )
        self.parts2body_idx = np.argsort(np.array(body_idx))
        self.hand_weight = hand_weight

        get_matrix_fn = os.path.join(subsample_pts_fn, "verts_512.pkl")
        self.smplx2landmarks = joblib.load(get_matrix_fn).numpy()

        half_desired_n_verts = self.num_joints//2
        self.smplx2landmarks = np.concatenate([self.smplx2landmarks[:half_desired_n_verts],
                                self.smplx2landmarks[self.smplx2landmarks.shape[0]//2:self.smplx2landmarks.shape[0]//2+half_desired_n_verts]], axis=0)
        self.ldmks2d_idx = self.smplx2landmarks.argmax(axis=-1)

        self.mean=255*np.array([0.485, 0.456, 0.406])
        self.std=255*np.array([0.229, 0.224, 0.225])

        self.scale_crop = 1.0
        if "BEDLAM_MASKS_WD" in label_path and self.aspect_ratio == 1.0:
            # BEDLAM BBX is from the eyes, so we need to increase the crop
            self.scale_crop = 1.3

        self.org_w = org_w
        self.org_h = org_h
        self.normalize_plus_min_one = normalize_plus_min_one

        if self.is_train:
            dataset_tar_list_fn = os.path.join(label_path, "tar_train_list.txt")
        else:
            dataset_tar_list_fn = os.path.join(label_path, "tar_eval_list.txt")

        unique_names = set()
        web_dataset_tars = []
        total_data = 0
        with open(dataset_tar_list_fn, 'r') as f:
            # read list and remove \n
            for line in f:
                tar_fn = line.strip()
                web_dataset_tars.append(os.path.join(label_path, tar_fn))
                scene_name = os.path.dirname(tar_fn)
                if scene_name not in unique_names:
                    # open json
                    with open(os.path.join(label_path, scene_name, "metadata.json"), 'r') as json_file:
                        scene_data = json.load(json_file)
                        total_data += scene_data["count_crop"]
                    unique_names.add(scene_name)

                web_dataset_tars = sorted(web_dataset_tars)
        self.web_dataset_tars = web_dataset_tars

        self.length = total_data
        self.total_valid_ldmks_idxs = self.generate_new_idxs_for_less_ldmks(self.num_joints, self.max_num_joints)

    @staticmethod
    def get_hand_feet_head_idxs(vertices, side_tol=1e-4, is_left=True):
        ''' It is expected to have the vertices in the SMPLX model in T-pose
        '''

        if is_left:
            body_side = vertices[:, 0] > side_tol
            hand = vertices[:, 0] > (vertices[body_side, 0].max() - 0.15)
        else:
            body_side = vertices[:, 0] < side_tol
            hand = vertices[:, 0] < (vertices[body_side, 0].min() + 0.15)

        feet = (vertices[:, 1] < (vertices[body_side, 1].min() + 0.10)) & body_side
        head = (vertices[:, 1] > (vertices[body_side, 1].max() - 0.20)) & body_side

        return hand, feet, head


    @staticmethod
    def generate_new_idxs_for_less_ldmks(new_n_ldmks, total_n_ldmks):
        '''
        When we generated the dataset, we used 512 landmarks, 256 for one half and 256 for the other half.
        Thus, we need to generate the new indexes for the landmarks in case we want to use less landmarks.
        '''

        assert new_n_ldmks % 2 == 0, "new_n_ldmks should be even"
        assert total_n_ldmks % 2 == 0, "total_n_ldmks should be even"
        assert new_n_ldmks <= total_n_ldmks, "new_n_ldmks should be less than total_n_ldmks"
        assert new_n_ldmks > 0, "new_n_ldmks should be greater than 0"
        assert total_n_ldmks > 0, "total_n_ldmks should be greater than 0"

        valid_joints_idx = np.arange(total_n_ldmks)
        if new_n_ldmks == total_n_ldmks:
            return valid_joints_idx

        half_total_n_ldmks = total_n_ldmks//2
        half_new_n_ldmks = new_n_ldmks//2
        valid_joint_idxs_right = valid_joints_idx[:half_new_n_ldmks]
        valid_joint_idxs_left = valid_joints_idx[half_total_n_ldmks: half_total_n_ldmks + half_new_n_ldmks]
        total_valid_joints_idx = np.concatenate([valid_joint_idxs_right, valid_joint_idxs_left])
        return total_valid_joints_idx

    def load_tars_as_webdataset(self, train: bool,
            resampled=False,
            epoch_size=None,
            cache_dir=None,
            **kwargs) -> Dataset:
        """
        Loads the dataset from a webdataset tar file.
        """

        def split_data(source):
            for item in source:
                bodies = item["data.pyd"]
                for body, bodies_data in bodies.items():
                    if bodies_data["person_visitiblity_rate"] < 0.15:
                        continue
                    yield {
                        "__key__": item["__key__"] + f"_{body}",
                        "jpg": item["jpg"],
                        "data.pyd": bodies_data,
                        "mask.jpg": item["mask.jpg"],
                    }

        # Load the dataset
        if epoch_size is not None:
            resampled = True

        urls= self.web_dataset_tars
        dataset = wds.WebDataset(urls,
                                nodesplitter=wds.split_by_node,
                                shardshuffle=True,
                                resampled=resampled,
                                cache_dir=cache_dir,)
        if train:
            dataset = dataset.shuffle(100)
        dataset = dataset.decode('rgb8').rename(jpg='jpg;jpeg;png')

        # Process the dataset
        dataset = dataset.compose(split_data)

        # Process the dataset further
        dataset = dataset.map(lambda x: self.process_webdataset_tar_item(x, train,))
        if epoch_size is not None:
            dataset = dataset.with_epoch(epoch_size)

        return dataset

    def process_webdataset_tar_item(self, data, train):
        joints_data = {}
        image = data['jpg'].copy()
        mask = data['mask.jpg'].copy()
        person_idx = int(data["data.pyd"]['person_idx'])

        img_name = data['__key__']
        if "closeup" in img_name:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        joints = data["data.pyd"]["vertices2d"].copy()
        joints_vis = data["data.pyd"]["vertex_visibility"].copy()

        try:
            joints_floor_contact = data["data.pyd"]["floor_contact_mask"].copy()
        except:
            print(data["data.pyd"]["imgname"], data["data.pyd"].keys())

        joints = joints[self.ldmks2d_idx]
        joints_vis = joints_vis[self.ldmks2d_idx]
        joints_floor_contact = joints_floor_contact[self.ldmks2d_idx][:, None]

        contact_thresh = 0.01

        if 'sdf_vertices' in data["data.pyd"] and type(data["data.pyd"]['sdf_vertices']) is not list:
            joint_contact = (data["data.pyd"]['sdf_vertices'][self.ldmks2d_idx] < contact_thresh).astype(np.float32)
        else:
            joint_contact = np.zeros_like(joints_vis)
        joint_contact = joint_contact * (1-joints_vis)  # if visible, not in contact

        is_random_crop = torch.rand(1).item() if self.is_train else 1.0
        if self.is_train and (is_random_crop < self.extreme_cropping_prob):
            try:
                bbox, rescale = extreme_cropping(joints, self.body_parts_dict, image.shape[1], image.shape[0])
                c, s = xyxy2cs(bbox, self.aspect_ratio, self.pixel_std)
                s = s * rescale
            except:
                c = np.array(data["data.pyd"]["center"]).copy()
                s = data["data.pyd"]["scale"].copy()
                s = s / 1.2 * self.scale_crop # remove CLIFF_SCALE_FACTOR
                s = np.array([s,s])

        else:
            c = np.array(data["data.pyd"]["center"]).copy()
            s = data["data.pyd"]["scale"].copy()
            s = s / 1.2 * self.scale_crop  # remove CLIFF_SCALE_FACTOR
            s = np.array([s,s])

        score = 1
        r = 0

        # Apply data augmentation
        is_flipped = False
        if self.is_train:

            sf = self.scale_factor
            rf = self.rotation_factor

            if self.scale:
                s = s * np.clip(random.random() * sf + 1, 1 - sf, 1)  # A random scale factor in [1 - sf, 1 + sf]

            if self.trans_factor > 0 and random.random() < self.trans_prob:
                # multiplying by self.pixel_std removes the 200 scale
                trans_x = np.random.uniform(-self.trans_factor, self.trans_factor) * self.pixel_std * s[0]
                trans_y = np.random.uniform(-self.trans_factor, self.trans_factor) * self.pixel_std * s[1]
                c[0] = c[0] + trans_x
                c[1] = c[1] + trans_y

            if self.rotate_prob and random.random() < self.rotate_prob:
                r = np.clip(random.random() * rf, -rf * 2, rf * 2)  # A random rotation factor in [-2 * rf, 2 * rf]
            else:
                r = 0

            if self.flip_prob and random.random() < self.flip_prob:
                image = image[:, ::-1, :]
                joints, joints_vis = fliplr_joints(joints, joints_vis, image.shape[1], self.flip_pairs)
                c[0] = image.shape[1] - c[0] - 1
                is_flipped = True

            if torch.rand(1).item() < self.img_aug_prob:
                # Numbers taken from bedlam/core/datasets/utils.py get_example
                image = augment_image(image)

        # Apply affine transform on joints and image
        trans = get_affine_transform(c, s, self.pixel_std, r, self.image_size)
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )
        mask = cv2.warpAffine(
            mask,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_NEAREST
        )

        mask = ((mask[...,0] == (person_idx+1))).astype(np.float32)
        if self.is_train:
            mask = augment_mask(mask)
        mask = mask[..., None]

        joints = np.concatenate([joints, np.ones((joints.shape[0], 1))], axis=1)
        joints = joints@trans.T

        # Convert image to tensor and normalize
        if self.transform is not None:  # I could remove this check
            image  = convert_cvimg_to_tensor(image)  # convert from HWC to CHW
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]
            mask = convert_cvimg_to_tensor(mask)  # convert from HWC to CHW

        valid_joints_x = (np.zeros(joints.shape[0]) <= joints[:, 0]) & (joints[:, 0] < self.image_size[0])
        valid_joints_y = (np.zeros(joints.shape[0]) <= joints[:, 1]) & (joints[:, 1] < self.image_size[1])
        valid_joints = (valid_joints_x & valid_joints_y)*1
        valid_joints = valid_joints[:, None]
        joints_vis = joints_vis * valid_joints # NOTE: DOUBLE CHECK THIS!!!!

        # Calculate the weight based on the points inside the image

        # get joints outside the image
        target_weight = valid_joints.copy().astype(np.float32)
        if joints[valid_joints[:,0]==0].shape[0] > 0:
            # normalize joints outside the image to [-1, 1]
            outside_joints = 2*(joints[valid_joints[:,0]==0]/self.image_size - 0.5)
            # calculate the distance of the joints to the center
            outside_joints = np.linalg.norm(outside_joints, axis=1)
            beta = 2. #0.25 #4
            outside_joints= np.exp(-beta*np.abs(outside_joints-1))
            target_weight[valid_joints[:,0]==0, 0] = outside_joints

        if True: #"vertex_idx" in self.labels:
            for key, val in self.body_parts_dict.items():
                # add 1 to the joints like hands, feet and head
                if valid_joints[val].sum() > 0 and key in ["left_hand", "right_hand", "left_feet", "right_feet"]:
                    target_weight[val] = target_weight[val] + target_weight[val] #valid_joints[val] #(valid_joints[val, 0] >= 1) * 1
                    if key in ["left_hand", "right_hand"]:
                        target_weight[val] = target_weight[val] * self.hand_weight

        target = np.array([0])

        # scale joints to [0, 1]
        joints[:, 0] /= self.image_size[0]
        joints[:, 1] /= self.image_size[1]

        if self.normalize_plus_min_one:
            # scale joints to [-1, 1]
            joints = joints * 2 - 1

        # Update metadata
        joints_data['image_name'] = img_name
        joints_data['joints'] = joints[self.total_valid_ldmks_idxs].astype(np.float32)
        joints_data['joints_visibility'] = joints_vis[self.total_valid_ldmks_idxs].astype(np.float32)
        joints_data['joints_contact'] = joint_contact[self.total_valid_ldmks_idxs].astype(np.float32)
        joints_data['center'] = c
        joints_data['scale'] = s
        joints_data['rotation'] = r
        joints_data['flip'] = is_flipped
        joints_data['score'] = score
        joints_data['image'] = image.astype(np.float32)
        joints_data['mask'] = mask[:1, ].astype(np.float32)
        joints_data['target'] = target.astype(np.float32)
        joints_data['target_weight'] = target_weight.astype(np.float32)[self.total_valid_ldmks_idxs]
        joints_data['person_visitiblity_rate'] = data["data.pyd"]['person_visitiblity_rate']
        joints_data['joints_floor_contact'] = joints_floor_contact.astype(np.float32)
        return joints_data

class MixedWebDataset(wds.WebDataset):
    def __init__(self, train_data_cfg, is_train: bool = False) -> None:
        super(wds.WebDataset, self).__init__()
        metadata = dict(train_data_cfg['datasets'])
        dataset_config = dict(OmegaConf.to_container(train_data_cfg, resolve=True))
        datasets = []
        weights = []
        print(f"is train: {is_train}")
        for dataset_name, metadata_cfg in metadata.items():
            try:
                dataset = BEDLAM_WD(**dataset_config, label_path=metadata_cfg['label_path'],
                                    hand_weight=metadata_cfg['hand_weight'],)
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                continue
            datasets.append(dataset.load_tars_as_webdataset(train=is_train, epoch_size=dataset.length))
            weights.append(metadata_cfg['weights'])
            print(f"  {dataset_name}: {dataset.length:,}")
        weights = np.array(weights) / np.sum(weights)
        self.append(wds.RandomMix(datasets, weights))


def main():
    """Smoke-test BEDLAM_WD: load one batch and save side-by-side viz
    (image + mask with joint overlay coloured by target_weight) per
    sample to ./test/sample_*.png.

    Run from repo root: python -m lib.datasets.BEDLAM_WD
    """
    import matplotlib.pyplot as plt
    from hydra import compose, initialize_config_dir

    CFG_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs/train/models_2d",
    )
    OmegaConf.register_new_resolver("mult", lambda x, y: x * y)
    OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("concat", lambda x: np.concatenate(x))
    OmegaConf.register_new_resolver("sorted", lambda x: np.argsort(x))

    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name="config")

    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 8
    train_data_cfg = cfg.data['train']
    normalize_plus_min_one = train_data_cfg.get('normalize_plus_min_one', False)

    dataset = MixedWebDataset(train_data_cfg, is_train=False).with_epoch(
        1000 // batch_size
    ).shuffle(100)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=False,
    )

    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]

    batch = next(iter(dataloader))
    print("Batch keys:", list(batch.keys()))
    print("Image shape:", batch['image'].shape)

    for i, (img, mask, joints, part_weight) in enumerate(zip(
        batch['image'], batch['mask'],
        batch['joints'].numpy(), batch['target_weight'],
    )):
        if normalize_plus_min_one:
            joints = (joints + 1) / 2
        fig, ax = plt.subplots(1, 2)
        img_norm = (img.numpy() * std + mean).transpose(1, 2, 0)
        ax[0].imshow(img_norm)
        ax[1].imshow(mask.numpy().transpose(1, 2, 0), interpolation='none')
        scatter = ax[1].scatter(
            joints[:, 0] * img.shape[2], joints[:, 1] * img.shape[1],
            c=part_weight[:, 0], cmap='plasma', s=1, vmin=0., vmax=2,
        )
        plt.colorbar(scatter, ax=ax)
        plt.savefig(f'{output_dir}/sample_{i:04d}.png')
        plt.close()


if __name__ == '__main__':
    main()
