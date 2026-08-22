#  Copyright (c) 2023 Max Planck Society
#  License: https://bedlam.is.tuebingen.mpg.de/license.html

import numpy as np


class GeometryModifierCutHands:
    """
    Cut the hands from the SMPL-X mesh. This is done by removing the vertices of the hands and changing the faces
    accordingly. The hands get often stuck in the clothing during simulation and are the cause of many simulation
    failures.
    """

    def __init__(self, hand_removal_data_fname='./support_data/hand_removal.npz'):
        self.hand_removal_data_fname = hand_removal_data_fname
        self.hand_vids_to_remove, self.wrist_right_vids, self.wrist_left_vids, self.updated_faces = \
            self.load_hand_removal_data()
        self.vertices_to_keep = np.setdiff1d(np.arange(10475), self.hand_vids_to_remove)

    def load_hand_removal_data(self):
        data = np.load(self.hand_removal_data_fname)
        return [data[k] for k in ['hand_vids_to_remove', 'wrist_right_vids', 'wrist_left_vids', 'faces_after_removal']]

    def cut_hands(self, smplx_vertices):
        v = smplx_vertices[self.vertices_to_keep, :]
        v = np.vstack((v, v[self.wrist_left_vids, :].mean(axis=0), v[self.wrist_right_vids, :].mean(axis=0)))
        return v


if __name__ == '__main__':
    # Demo: cut hands from a batch of SMPL-X meshes.
    N, V = 3, 10475  # frames, SMPL-X vertex count
    v_seq = np.random.randn(N, V, 3)
    geo_mod = GeometryModifierCutHands('./utils/hand_removal.npz')
    v_cut = np.stack([geo_mod.cut_hands(v) for v in v_seq])
    print(f"Input shape:  {v_seq.shape}")
    print(f"Output shape: {v_cut.shape}  (hands removed, wrist centroids added)")