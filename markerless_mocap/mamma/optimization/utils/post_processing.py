import torch
import math

def unwrap_axis_angle_sequence(poses_aa: torch.Tensor) -> torch.Tensor:
    """
    poses_aa: [F, J*3] axis-angle sequence.
    Returns a new tensor with 2π jumps removed per joint.
    """
    poses = poses_aa.clone()
    F, D = poses.shape
    J = D // 3
    poses = poses.view(F, J, 3)

    for j in range(J):
        prev = poses[0, j]  # [3]
        for t in range(1, F):
            cur = poses[t, j]

            # Convert both to angle + axis
            angle_prev = prev.norm()
            angle_cur  = cur.norm()

            if angle_prev < 1e-8 or angle_cur < 1e-8:
                # very small rotation, nothing to unwrap
                poses[t, j] = cur
                prev = poses[t, j]
                continue

            axis_prev = prev / angle_prev
            axis_cur  = cur  / angle_cur

            # Make axes as consistent as possible (avoid opposite axes)
            if torch.dot(axis_prev, axis_cur) < 0:
                axis_cur  = -axis_cur
                angle_cur = -angle_cur

            # Find integer k that minimizes |(angle_cur + 2πk) - angle_prev|
            k = torch.round((angle_prev - angle_cur) / (2 * math.pi))
            angle_cur_unwrapped = angle_cur + k * 2 * math.pi

            poses[t, j] = axis_cur * angle_cur_unwrapped
            prev = poses[t, j]

    return poses.view(F, D)

