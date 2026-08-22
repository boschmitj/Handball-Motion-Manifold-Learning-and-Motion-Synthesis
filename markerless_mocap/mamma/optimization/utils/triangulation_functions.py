import torch


def triangulate_batch(pts2d_list, Ks_list, extrs_list,
                      vis_list=None, img_size_px=512,
                      use_uncertainties=True, channel_is_logvar=False,
                      vis_thresh=0.0, clamp_sigma=(1.0, 24.0)):
    """
    pts2d_list: list of [T, N, 2 or 3] (pixel coords)
    Ks_list: list of [T,3,3]
    extrs_list: list of [T,4,4]
    vis_list: list of [T,N] or None
    Returns:
        pts3d_world [T,N,3], valid_mask [T,N]
    """
    print("Triangulating batch with {} views ...".format(len(pts2d_list)))
    device = Ks_list[0].device
    dtype  = Ks_list[0].dtype
    n_cams = len(pts2d_list)
    T, N = pts2d_list[0].shape[:2]

    Ps = []
    for c in range(n_cams):
        K, E = Ks_list[c], extrs_list[c]
        Ps.append(K @ E[:, :3, :])                          # [T,3,4]
    Ps = torch.stack(Ps, dim=1)                             # [T,n_cams,3,4]

    pts2d = torch.stack(pts2d_list, dim=1)                  # [T,n_cams,N,2 or 3]
    vis   = torch.stack(vis_list, dim=1) if vis_list is not None else \
            torch.ones(T, n_cams, N, device=device, dtype=dtype)

    valid = (vis > vis_thresh) & (pts2d[..., :2].abs().sum(-1) > 0)
    valid_f = valid.float()

    if use_uncertainties and pts2d.shape[-1] == 3:
        third = pts2d[..., 2]
        sigma = torch.exp(0.5 * third) * (img_size_px / 2.0) if channel_is_logvar else \
                third * (img_size_px / 2.0)
        sigma = sigma.clamp(*clamp_sigma)
        w = 1.0 / (sigma * sigma)
    else:
        w = torch.ones_like(valid_f)
    w = w * valid_f
    w_sqrt = torch.sqrt(w + 1e-12)

    u, v = pts2d[..., 0], pts2d[..., 1]
    P0, P1, P2 = Ps[..., 0, :], Ps[..., 1, :], Ps[..., 2, :]

    r1 = (u.unsqueeze(-1) * P2.unsqueeze(2) - P0.unsqueeze(2)) * w_sqrt.unsqueeze(-1)
    r2 = (v.unsqueeze(-1) * P2.unsqueeze(2) - P1.unsqueeze(2)) * w_sqrt.unsqueeze(-1)
    A = torch.cat([r1, r2], dim=1).permute(0, 2, 1, 3).contiguous()  # [T,N,2n,4]

    # Flatten batch for a single SVD call
    TN, M, D = T*N, A.shape[2], 4
    A_flat = A.view(TN, M, D)
    # SVD(A): A = U S Vh, right singular vector = last row of Vh
    # do svd in cpu because it's faster
    A_flat = A_flat.cpu()
    U, S, Vh = torch.linalg.svd(A_flat, full_matrices=False)
    Xh = Vh[:, -1, :]                                       # [TN, 4]
    Xh = Xh / (Xh[:, 3:4] + 1e-8)
    Xw = Xh[:, :3].view(T, N, 3)
    Xw = Xw.to(device=device, dtype=dtype)

    cam_count = valid.sum(dim=1)
    ok = cam_count >= 2
    Xw = Xw.masked_fill(~ok.unsqueeze(-1), 0.0)
    print("Triangulation done.")
    return Xw, ok
