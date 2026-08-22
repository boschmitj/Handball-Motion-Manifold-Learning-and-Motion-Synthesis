
import torch
import smplx

FLAT_HAND = True  # for bedlam is true
def get_smplx_models(model_folder, num_betas=11, model_type='smplx', device="cuda", flat_hand=FLAT_HAND, num_pca_comps=45):
    smplx_model_male = smplx.create(model_folder, model_type=model_type,
                                gender='male',
                                ext='npz',
                                num_betas=num_betas,
                                flat_hand_mean=flat_hand,  # True
                                num_pca_comps=num_pca_comps,
                                use_pca=False).to(device)

    smplx_model_female = smplx.create(model_folder, model_type=model_type,
                                    gender='female',
                                    ext='npz',
                                    num_betas=num_betas,
                                    flat_hand_mean=flat_hand,  # True
                                    num_pca_comps=num_pca_comps,
                                    use_pca=False).to(device)

    smplx_model_neutral = smplx.create(model_folder, model_type=model_type,
                                    gender='neutral',
                                    ext='npz',
                                    flat_hand_mean=flat_hand,  # False
                                    num_betas=num_betas,
                                    num_pca_comps=num_pca_comps,
                                    use_pca=False).to(device)

    return {"male": smplx_model_male, "female": smplx_model_female, "neutral": smplx_model_neutral}


def get_smplx_forward(poses, betas, trans, gender, smplx_models, only_body_pose=None, expression=None):
    assert str(gender) in ["male", "female", "neutral"], f"gener is {smplx_models[str(gender)]}, which is not male, female or neutral"

    if expression is None:
        n_expression = smplx_models[gender].num_expression_coeffs
        expression = torch.zeros([poses.shape[0], n_expression]).to(poses.device)

    model_out = smplx_models[str(gender)](betas=betas,
                        global_orient=poses[:, :3],
                        body_pose=poses[:, 3:66] if only_body_pose is None else only_body_pose,
                        left_hand_pose=poses[:, 75:120],
                        right_hand_pose=poses[:, 120:165],
                        jaw_pose=poses[:, 66:69],
                        leye_pose=poses[:, 69:72],
                        reye_pose=poses[:, 72:75],
                        transl=trans,
                        expression=expression)

    return model_out

def get_smplx_forward_per_parts(poses_global, pose_body, pose_left_hand, pose_right_hand, pose_jaw, pose_leye, pose_reye,
                                betas, trans, gender, smplx_models, only_body_pose=None, expression=None):
    assert str(gender) in ["male", "female", "neutral"], f"gener is {smplx_models[str(gender)]}, which is not male, female or neutral"

    if expression is None:
        n_expression = smplx_models[gender].num_expression_coeffs
        expression = torch.zeros([pose_body.shape[0], n_expression]).to(pose_body.device)

    model_out = smplx_models[str(gender)](betas=betas,
                        global_orient=poses_global,
                        body_pose=pose_body if only_body_pose is None else only_body_pose,
                        left_hand_pose=pose_left_hand,
                        right_hand_pose=pose_right_hand,
                        jaw_pose=pose_jaw,
                        leye_pose=pose_leye,
                        reye_pose=pose_reye,
                        transl=trans,
                        expression=expression)

    return model_out
