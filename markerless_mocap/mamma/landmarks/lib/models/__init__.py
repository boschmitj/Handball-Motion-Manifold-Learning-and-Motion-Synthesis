from .models_2d.denseldmks2d_vit import DenseLdmks2DViT
from .models_2d.denseldmks2d_hrnet import DenseLdmks2DHRNet

__all__ = [
    "DenseLdmks2DViT",
    "DenseLdmks2DHRNet",
    "build_model",
]


def build_model(cfg, **kwargs):
    model = eval(cfg.model_name)(cfg, **kwargs)
    return model