
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from .transformer_detr import TransformerDecoder, TransformerDecoderLayer
import torch.nn as nn
import torch.nn.functional as F

class MvCameraHMR(nn.Module):
    def __init__(self,
                 n_views=8,
                 d_feat=768,
                 n_heads=8,
                 n_layers=6,
                 n_landmarks=43,
                 d_transl=3,
                 dropout=0.1,
                 uncertainty=False,
                 visibility=False,
                 contact=False,
                 aggregate_type='avg'):
        """
        Setup Multi-view Feature Transformer
        """
        super().__init__()

        self.aggregate_type = aggregate_type
        self.output_dim = 3 if uncertainty else 2
        self.visibility = visibility
        self.contact = contact
        if visibility:
            self.output_dim += 1
        if contact:
            self.output_dim += 1

        self.dropout = nn.Dropout(dropout)
        self.to_token_embedding = nn.Linear(1, d_feat)
        self.intraview_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_feat, nhead=n_heads, dim_feedforward=1024, activation='gelu'),
            num_layers=n_layers
        )

        self.declandmark = nn.Linear(d_feat, n_landmarks * self.output_dim)

    def forward(self, feats, masks):
        """
        Forward pass of the Multi-view Feature Transformer
        Args:
            feats (torch.Tensor): Features from multiple views
            masks (torch.Tensor): Masks indicating valid views
        Returns:
            torch.Tensor: Multi-view feature representation
        """

        B = feats.shape[0]

        # Decode ViT image path
        p_token = torch.zeros(B, 1, 1).to(feats.device)
        p_token = self.to_token_embedding(p_token)
        p_token = self.dropout(p_token)
        p_token = rearrange(p_token, 'B L D -> L B D')
        feats = rearrange(feats, 'B D H W -> (H W) B D')
        feats = self.intraview_decoder(p_token, feats).squeeze(0)

        # Decode 2D landmarks + (uncertainty)
        landmark = self.declandmark(feats).view(B, -1, self.output_dim)

        # Construct predictions
        if self.contact and self.visibility:
            pred = dict(
                joints2d=landmark[...,:-2] if self.visibility else landmark[...,:-1],
                visibility=landmark[...,-2:-1],
                contact=landmark[...,-1:],
            )
        elif self.contact and not self.visibility:
            pred = dict(
                joints2d=landmark[...,:-1],
                visibility=None,
                contact=landmark[...,-1:],
            )
        elif not self.contact and self.visibility:
            pred = dict(
                joints2d=landmark[...,:-1],
                visibility=landmark[...,-1:],
                contact=None,
            )
        else:
            pred = dict(
                joints2d=landmark,
                visibility=None,
                contact=None,
            )
        return pred


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def initialize_last_layer(self):
        # Extract the last layer
        last_layer = self.layers[-1]

        # Apply custom initialization to the first output
        nn.init.constant_(last_layer.weight, 0.0)  # Optional: Initialize the first weight to 0
        nn.init.uniform_(last_layer.bias[:2], a=-2, b=2)  # Optional: Initialize the first bias to 0
        last_layer.bias.data[:2].clamp_(-1, 1)  # Optional: Clamp the first bias to -1 and 1
        # # Optionally initialize the remaining weights/biases as needed
        # for i in range(1, last_layer.weight.size(0)):  # For other outputs

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # print last layer weights
        return x


class MammaNetDecoder(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_heads=8,
                 n_layers=6,
                 n_landmarks=512,
                 transformer_dim_feedforward=2048,
                 ldmks_dim=2,
                 dropout=0.1,
                 uncertainty=True,
                 visibility=True,
                 contact=True,
                 floor_contact=False,

                 ):

        super(MammaNetDecoder, self).__init__()

        self.output_dim = ldmks_dim+1 if uncertainty else ldmks_dim
        self.visibility = visibility
        self.contact = contact
        self.floor_contact = floor_contact

        activation = 'gelu'
        self.return_intermediate_dec = True
        normalize_before = False
        self.query_embed = nn.Embedding(n_landmarks, d_model)

        decoder_layer = TransformerDecoderLayer(d_model, n_heads, transformer_dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder_detr = TransformerDecoder(decoder_layer, n_layers, decoder_norm,
                                          return_intermediate=self.return_intermediate_dec)
        self.landmarks = MLP(d_model, d_model, self.output_dim, 3)
        if self.visibility:
            self.vis_prob = nn.Linear(d_model, 1)
        if self.contact:
            self.contact_prob = nn.Linear(d_model, 1)
        if self.floor_contact:
            self.floor_contact_prob = nn.Linear(d_model, 1)

    def forward(self, src, pos_embed, mask=None):
        patch_pos_embed = pos_embed[:, 1:]
        bs = src.shape[0]
        patch_pos_embed = patch_pos_embed.permute(1, 0, 2)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        src = rearrange(src, 'B D H W -> (H W) B D')

        hs = self.decoder_detr(tgt, src, memory_key_padding_mask=None, pos=patch_pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2) #, memory.permute(1, 2, 0).view(bs, c, h, w)

        # Construct predictions
        pred = dict(
            joints2d=self.landmarks(hs)[-1],
            visibility=self.vis_prob(hs)[-1] if self.visibility else None,
            contact=self.contact_prob(hs)[-1] if self.contact else None,
            floor_contact=self.floor_contact_prob(hs)[-1] if self.floor_contact else None,
        )

        return pred

