"""
Parts of the code are taken or adapted from
https://github.com/mkocabas/EpipolarPose/blob/master/lib/utils/img_utils.py
"""
import torch
import numpy as np
import random
import cv2
import torch.utils
import albumentations as A
from albumentations import ImageOnlyTransform


def xyxy2cs(bbox, aspect_ratio, pixel_std, scale_factor=0.75):
    x1, y1, x2, y2 = bbox
    center = np.array([x1 + x2, y1 + y2]) * 0.5
    w = x2 - x1
    h = y2 - y1

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_factor

    return center, scale


def extreme_cropping(joints, body_parts_dict, img_w, img_h):
    valid_joints_x = (np.zeros(joints.shape[0]) <= joints[:, 0]) & (joints[:, 0] < img_w)
    valid_joints_y = (np.zeros(joints.shape[0]) <= joints[:, 1]) & (joints[:, 1] < img_h)
    valid_joints = (valid_joints_x & valid_joints_y)*1
    valid_joints = valid_joints[:, None]

    valid_body_names = []
    valid_weights = []
    for body_part, idx in body_parts_dict.items():
        if np.sum(valid_joints[idx]) > 20:
            valid_body_names.append(body_part)
            if "body" in body_part:
                valid_weights.append(4)
            else:
                valid_weights.append(1)
    valid_weights = torch.tensor(valid_weights)
    valid_weights = valid_weights/torch.sum(valid_weights)

    random_index = torch.multinomial(valid_weights, 1).item()
    random_body_part = valid_body_names[random_index]
    ramdom_body_idx = body_parts_dict[random_body_part]
    keypoints = np.concatenate([joints[ramdom_body_idx], valid_joints[ramdom_body_idx]], axis=-1)
    # x1, y1, x2, y2 from keypoints
    bbox = [np.min(keypoints[:, 0]), np.min(keypoints[:, 1]), np.max(keypoints[:, 0]), np.max(keypoints[:, 1])]
    rescale = 1.2 if "body" in random_body_part else 4
    return bbox, rescale


def augment_image(image):
    aug_comp = [A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                            A.ImageCompression(20, 100, p=0.1),
                            A.RandomRain(blur_value=4, p=0.1),
                            A.MotionBlur(blur_limit=(3, 15),  p=0.2),
                            A.Blur(blur_limit=(3, 9), p=0.1),
                            A.RandomSnow(brightness_coeff=1.5,
                            snow_point_lower=0.2, snow_point_upper=0.4)]
    aug_mod = [A.CLAHE((1, 11), (10, 10), p=0.2), A.ToGray(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5],
            elementwise=True, per_channel=True, p=0.2),
            A.HueSaturationValue(hue_shift_limit=20,
            sat_shift_limit=30, val_shift_limit=20,
            always_apply=False, p=0.2),
            A.Posterize(p=0.1),
            A.RandomGamma(gamma_limit=(80, 200), p=0.1),
            A.Equalize(mode='cv', p=0.1)]
    albumentation_aug = A.Compose([A.OneOf(aug_comp,
                                p=0.3),
                                A.OneOf(aug_mod,
                                p=0.3)])
    return albumentation_aug(image=image)['image']


class random_mask_noise(ImageOnlyTransform):
    def __init__(self, drop_prob=0.01, add_prob=0.01, p=0.5):
        super().__init__(p=p)
        self.drop_prob = drop_prob
        self.add_prob = add_prob

    def get_params_dependent_on_data(self, params, data):
        return {"drop_prob": self.drop_prob, "add_prob": self.add_prob}

    def apply(self, img, drop_prob, add_prob, **params):
        noise = np.random.rand(*img.shape)
        img = img.copy()
        img[noise < drop_prob] = 0
        img[noise > 1 - add_prob] = 1
        return img


class remove_pixels(ImageOnlyTransform):
    def __init__(self, extreme_removal=False, p=0.5):
        super().__init__(p=p)
        self.extreme_removal = extreme_removal

    def get_params_dependent_on_data(self, params, data):
        return {"extreme_removal": self.extreme_removal,}

    def apply(self, img, extreme_removal, **params):
        noise = np.random.rand(*img.shape)
        img = img.copy()
        if extreme_removal:
            img[noise < np.random.uniform(0.8, 1.)] = 0
        else:
            img[noise < np.random.uniform(0.1, 0.6)] = 0
        return img


class no_mask(ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def get_params_dependent_on_data(self, params, data):
        return {}

    def apply(self, img, **params):
        return img*0


class random_oval_mask(ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def get_params_dependent_on_data(self, params, data):
        return {}

    def apply(self, img, **params):
        img_idx = np.argwhere(img == 1)
        if len(img_idx) == 0:
            return img

        contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        x = np.hstack((contours[0][:, 0, 0], contours[0][:, 0, 0][0]))
        y = np.hstack((contours[0][:, 0, 1], contours[0][:, 0, 1][0]))
        random_contour_idx = np.random.choice(len(x))
        x, y = x[random_contour_idx], y[random_contour_idx]
        center = (x, y)
        h, w = img.shape

        axes = (np.random.randint(0, w//4), np.random.randint(0, h//4))
        angle = np.random.randint(0, 360)
        cv2.ellipse(img, center, axes, angle, 0, 360, (1,), -1)
        # if np.random.rand() > 0.5:
        return img


class remove_mask_areas(ImageOnlyTransform):
    def __init__(self, p=0.5, upper_part=True, left_part=True, percentage_up=0.3, percentage_left=0.3):
        super().__init__(p=p)
        self.upper_part = upper_part
        self.left_part = left_part
        self.percentage_up = percentage_up
        self.percentage_left = percentage_left

    def get_params_dependent_on_data(self, params, data):
        return {"upper_part": self.upper_part, "left_part": self.left_part,
                "percentage_up": self.percentage_up, "percentage_left": self.percentage_left}

    def apply(self, img, upper_part, left_part, percentage_up, percentage_left, **params):
        h, w = img.shape
        if upper_part:
            img[:int(h*percentage_up), :] = 0
        else:
            img[int(h*(1-percentage_up)):, :] = 0
        if left_part:
            img[:, :int(w*percentage_left)] = 0
        else:
            img[:, int(w*(1-percentage_left)):] = 0
        return img


def augment_mask(mask):
    aug_comp = A.Compose([
                A.Morphological(scale=mask.shape[0]//25, operation='dilation', p=0.5),
                A.Morphological(scale=mask.shape[0]//25, operation='erosion', p=0.5),
                remove_mask_areas(p=.5, upper_part=random.random()>0.5, left_part=random.random()>0.5,
                                  percentage_up=random.uniform(0.1, 0.6), percentage_left=random.uniform(0.1, 0.6)),
                remove_pixels(extreme_removal=1, p=0.3),
                random_oval_mask(p=.5),
                random_mask_noise(drop_prob=0.01, add_prob=0.01, p=.1),
                no_mask(p=0.07),
                ])

    return aug_comp(image=mask)['image']