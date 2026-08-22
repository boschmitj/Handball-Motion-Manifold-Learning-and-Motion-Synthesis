import numpy as np
import os
import glob
import pandas as pd
import tqdm
import sys
try:
    from utils.mesh_post_proc import GeometryModifierCutHands
except ImportError:
    from mesh_post_proc import GeometryModifierCutHands
import argparse

"""
Parts of the code are adapted from https://github.com/akanazawa/hmr
"""
import numpy as np
geo_mod = GeometryModifierCutHands('./utils/hand_removal.npz')
body_verts = geo_mod.vertices_to_keep
hand_verts = geo_mod.hand_vids_to_remove


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat, scale, R, t


def procrustes_analysis_batch(S1, S2, valid_indices=None, S1_extra=None):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    if S1_extra is not None:
        S1_hat_extra = np.zeros_like(S1_extra)
    for i in range(S1.shape[0]):
        if valid_indices is not None:
            S1_hat_, scale, R, t = compute_similarity_transform(S1[i, valid_indices], S2[i, valid_indices])
            S1_hat[i] = (scale*R.dot(S1[i].T) + t).T
        else:
            S1_hat[i], scale, R, t = compute_similarity_transform(S1[i], S2[i])
        if S1_extra is not None:
            S1_hat_extra[i] = (scale*R.dot(S1_extra[i].T) + t).T
    if S1_extra is not None:
        return S1_hat, S1_hat_extra, scale, R, t
    else:
        return S1_hat, scale, R, t


def scale_and_translation_transform_batch(P, T):
    """
    First Normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of N 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of N reference 3D meshes.
    :return: P transformed
    """
    P_mean = np.mean(P, axis=1, keepdims=True)
    P_trans = P - P_mean
    P_scale = np.sqrt(np.sum(P_trans ** 2, axis=(1, 2), keepdims=True) / P.shape[1])
    P_normalised = P_trans / P_scale

    T_mean = np.mean(T, axis=1, keepdims=True)
    T_scale = np.sqrt(np.sum((T - T_mean) ** 2, axis=(1, 2), keepdims=True) / T.shape[1])

    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed


def compute_jitter(joints, fps=30):
    jitter = np.linalg.norm((joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),axis=2,).mean(axis=-1)
    return jitter / 10.0

def body25_to_skel19(joints):
    map_idx = np.array([8, 1, 9, 12, 0, 2, 5, 10, 13, 17, 18, 3, 6, 11, 14, 4, 7, 19, 22], dtype=np.int32)
    return joints[..., map_idx, :]


def smplx2skel19(vertices, regressor, smplidx):
    smpl = vertices[:, smplidx]
    body25 = np.einsum('ik,bkd->bid', regressor, smpl)
    return body25_to_skel19(body25)

def smplx2skel17(vertices, regressor, smplidx):
    smpl = vertices[:, smplidx]
    body17 = np.einsum('ik,bkd->bid', regressor, smpl)
    return body17

