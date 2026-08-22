import numpy as np
from torchvision import transforms

class DatasetRegistration(type):
    """
    Metaclass for registering different datasets
    """
    def __init__(cls, name, bases, nmspc):
        super().__init__(name, bases, nmspc)
        if not hasattr(cls, 'registry'):
            cls.registry = dict()
        cls.registry[name] = cls

    # Metamethods, called on class objects:
    def __iter__(cls):
        return iter(cls.registry)

    def __str__(cls):
        return str(cls.registry)

class Dataset(metaclass=DatasetRegistration):
    """
    Base Dataset class
    """
    def __init__(self, *args, **kwargs):
        pass

class HumanPoseEstimationDataset(Dataset):
    """
    HumanPoseEstimationDataset class.

    Generic class for HPE datasets.
    """
    def __init__(self, is_train=True, image_width=288, image_height=384,
                 scale=True, scale_factor=0.35, flip_prob=0.5, rotate_prob=0.5, trans_prob=0.5, rotation_factor=45., half_body_prob=0.3, trans_factor=0.0,
                 use_different_joints_weight=False, extreme_cropping_prob=0.1, img_aug_prob=0.9, heatmap_sigma=3, max_res=10, num_joints=17, total_num_joints=512,
                 flip_pairs=None, joints_weight=None,
                 **kwargs):

        self.max_res = max_res

        self.is_train = is_train
        self.scale = scale  # ToDo Check
        self.scale_factor = scale_factor
        self.trans_factor = trans_factor
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.trans_prob = trans_prob
        self.rotation_factor = rotation_factor
        self.half_body_prob = half_body_prob
        self.use_different_joints_weight = use_different_joints_weight  # ToDo Check
        self.extreme_cropping_prob = extreme_cropping_prob
        self.img_aug_prob = img_aug_prob
        self.heatmap_sigma = heatmap_sigma

        self.image_size = (image_width, image_height)
        self.aspect_ratio = image_width * 1.0 / image_height

        self.heatmap_size = (int(image_width / 4), int(image_height / 4))
        self.heatmap_type = 'gaussian'
        self.pixel_std = 200  # I don't understand the meaning of pixel_std (=200) in the original implementation

        self.num_joints = num_joints
        self.max_num_joints = total_num_joints
        self.num_joints_half_body = 8

        if flip_pairs:
            self.flip_pairs = flip_pairs
        if joints_weight:
            self.joints_weight = np.array(joints_weight).reshape(self.max_num_joints, 1)
        else:
            self.use_different_joints_weight = False

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Default values
        self.bbox_thre = 1.0
        self.image_thre = 0.0
        self.in_vis_thre = 0.2
        self.nms_thre = 1.0
        self.oks_thre = 0.9

    #         for joint_id in range(self.num_joints):
    #             # Check that any part of the gaussian is in-bounds
    #                     or br[0] < 0 or br[1] < 0:
    #                 # If not, just return the image as is
    #                 continue

    #             if v > 0.5:
    #                     g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    #         raise NotImplementedError

    #     if self.use_different_joints_weight:


