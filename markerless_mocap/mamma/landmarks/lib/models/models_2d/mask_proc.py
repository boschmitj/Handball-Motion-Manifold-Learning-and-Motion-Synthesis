import torch.nn as nn
from ..backbone.vit import to_2tuple
from ..models_2d.layer_norm import LayerNorm2d


class MaskEmbedding(nn.Module):
    '''
    Code taken from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py
    '''
    def __init__(self, embed_dim, patch_size, mask_in_chans = 16, activation = nn.GELU):
        super(MaskEmbedding, self).__init__()
        patch_size = to_2tuple(patch_size)
        ratio = 1

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=patch_size, stride=(patch_size[0] * ratio)),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=patch_size, stride=(patch_size[0] * ratio)),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def forward(self, x):
        x = self.mask_downscaling(x)
        return x
