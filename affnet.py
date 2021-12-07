from dataclasses import dataclass
import math
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

import pickle
import os
import cv2 as cv
import kornia as KR
import kornia.feature as KF
import kornia.geometry as KG
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from kornia_moons.feature import *
from config import CartesianConfig
from hard_net_descriptor import HardNetDescriptor
from utils import Timer
from rectification import get_rotation_matrix
from kornia_moons.feature import visualize_LAF
from kornia_utils import *

@dataclass
class PointsStyle:
    ts: torch.tensor
    phis: torch.tensor
    color: str
    size: float


def round_and_clamp_coords_torch(coords, max_0_excl, max_1_excl):

    # round and write elsewhere, then work in-place
    coords = torch.round(coords)
    torch.clamp(coords[:, 0], 0, max_0_excl - 1, out=coords[:, 0])
    torch.clamp(coords[:, 1], 0, max_1_excl - 1, out=coords[:, 1])
    coords = coords.to(torch.long)
    return coords


# NOTE : also basically version of utils.get_kpts_normals, but for torch!!!
# TODO there is some duplicate code with get_kpts_normals_indices
def get_kpts_components_indices(components_indices_np, valid_components_dict, laffs_no_scale):

    Timer.start_check_point("get_kpts_components_indices")

    coords = round_and_clamp_coords_torch(laffs_no_scale[0, :, :, 2], components_indices_np.shape[1], components_indices_np.shape[0])

    components_indices_linear = components_indices_np[coords[:, 1], coords[:, 0]]
    components_indices_linear_and_invalid = torch.ones_like(coords[:, 0]) * -1
    for valid_component in valid_components_dict:
        components_indices_linear_and_invalid[components_indices_linear == valid_component] = valid_component

    Timer.end_check_point("get_kpts_components_indices")

    return components_indices_linear_and_invalid[None]


def get_rotation_matrices(angles):
    R = torch.cos(angles)
    R = R.repeat(1, 1, 2, 2)
    R[:, :, 1, 0] = torch.sin(angles)
    R[:, :, 0, 1] = -R[:, :, 1, 0]
    return R


def compose_lin_maps(ts, phis, lambdas, psis):

    if lambdas is None:
        lambdas = 1.0 / torch.sqrt(ts)
        lambdas = lambdas.repeat(1, 1, 1, 1)
    if psis is None:
        psis = torch.zeros_like(phis)

    # def get_rotation_matrices(angles):
    #     R = torch.cos(angles)
    #     R = R.repeat(1, 1, 2, 2)
    #     R[:, :, 1, 0] = torch.sin(angles)
    #     R[:, :, 0, 1] = -R[:, :, 1, 0]
    #     return R

    R_psis = get_rotation_matrices(psis)
    R_phis = get_rotation_matrices(phis)

    T_ts = torch.zeros_like(ts)
    T_ts = T_ts.repeat(1, 1, 2, 2)
    T_ts[:, :, 0, 0] = ts
    T_ts[:, :, 1, 1] = 1

    lin_maps = lambdas * R_psis @ T_ts @ R_phis
    return lin_maps, lambdas, R_psis, T_ts, R_phis


# def compose_lin_maps(ts, phis):
#
#     R_phis = get_rotation_matrices(phis)
#
#     T_ts = torch.zeros_like(ts)
#     T_ts = T_ts.repeat(1, 1, 2, 2)
#     T_ts[:, :, 0, 0] = ts
#     T_ts[:, :, 1, 1] = 1
#
#     lin_maps = T_ts @ R_phis
#     return lin_maps, T_ts, R_phis


# TODO handle CUDA
def decompose_lin_maps_lambda_psi_t_phi(l_maps, asserts=True):

    assert len(l_maps.shape) == 4
    Timer.start_check_point("decompose_lin_maps")

    # NOTE for now just disallow CUDA
    assert l_maps.device == torch.device('cpu')

    U, s, V = torch.svd(l_maps)
    V = torch.transpose(V, dim0=2, dim1=3)

    lambdas = torch.ones(l_maps.shape[:2])

    def assert_decomposition():
        d = torch.diag_embed(s)
        product = lambdas.view(1, -1, 1, 1) * U @ d @ V
        close_cond = torch.allclose(product, l_maps, atol=1e-05)
        assert close_cond

    assert_decomposition()

    if asserts:
        assert torch.all(torch.sgn(s[:, :, 0]) == torch.sgn(s[:, :, 1]))
        assert torch.all(s[:, :, 0] != 0)

    # TODO this is probably useless as factor will be 1
    factor = torch.sgn(s[:, :, :1])
    U = factor[:, :, :, None] * U
    s = factor * s

    # lambda <- s[1]
    # s <- [[t, 0], [0, 1]], t >= 1
    lambdas = s[:, :, 1].clone()
    s = s / s[:, :, 1:]

    assert_decomposition()

    if asserts:
        assert torch.all(s[:, :, 0] >= 1)
        assert torch.all(s[:, :, 1] == 1)

    dets_u = torch.det(U)
    dets_v = torch.det(V)
    if asserts:
        assert torch.allclose(dets_v, dets_u, atol=1e-07)
        assert torch.allclose(torch.abs(dets_v), torch.tensor(1.0), atol=1e-07)

    # it could be that det U[:, :, i] == det V[:, :, i] == -1, therefore I need to negate row(U, 0) and column(V, 0) -> two reflections
    factor_rows_columns = torch.where(dets_v > 0.0, 1.0, -1.0).view(1, -1, 1)
    U[:, :, :, 0] = factor_rows_columns * U[:, :, :, 0]
    V[:, :, 0, :] = factor_rows_columns * V[:, :, 0, :]

    assert_decomposition()

    dets_u = torch.det(U)
    dets_v = torch.det(V)
    if asserts:
        assert torch.allclose(dets_v, dets_u, atol=1e-07)
        assert torch.allclose(dets_v, torch.tensor(1.0), atol=1e-07)

    # phi in (0, pi), if not, V <- -V and U <- -U
    phi_norm_factor = torch.where(V[:, :, :1, 1:] > 0, -1.0, 1.0)
    V = V * phi_norm_factor
    U = U * phi_norm_factor

    def assert_rotation(A, angle):
        sins_ang = torch.sin(angle)
        if asserts:
            assert torch.allclose(sins_ang, -A[:, :, 0, 1], atol=1e-03)
            assert torch.allclose(sins_ang, A[:, :, 1, 0], atol=1e-03)
            assert torch.allclose(A[:, :, 1, 1], A[:, :, 0, 0], atol=1e-05)

    phis = torch.arccos(torch.clamp(V[:, :, 0, 0], -1.0, 1.0))
    assert_rotation(V, phis)

    psis = torch.arcsin(-torch.clamp(U[:, :, 0, 1], -1.0, 1.0))
    assert_rotation(U, psis)

    ts = s[:, :, 0]

    assert_decomposition()
    Timer.end_check_point("decompose_lin_maps")

    return lambdas, psis, ts, phis


def draw(radius, ts, phis, color, size, ax):

    ts_logs = torch.log(ts)
    xs = torch.cos(phis) * ts_logs
    ys = torch.sin(phis) * ts_logs

    ax.plot(xs, ys, 'o', color=color, markersize=size)


def prepare_plot(radius: float, ax):

    log_radius = math.log(radius)

    ax.set_aspect(1.0)

    factor = 1.2
    ax.set_xlim((-factor*log_radius, factor*log_radius))
    ax.set_ylim((-factor*log_radius, factor*log_radius))

    circle = Circle((0, 0), log_radius, color='r', fill=False)
    ax.add_artist(circle)


# TODO export somehow?
# def get_normals(normals, K):
#
#     Hs = get_rectification_homographies(normals, K)
#     Hs_as_affine = Hs[:, :, :2, :2]
#     det_Hs = torch.det(Hs_as_affine).sqrt().unsqueeze(2).unsqueeze(3)
#     Hs_as_affine = Hs_as_affine / det_Hs
#
#     # TODO CONTINUE asserts=False did not work on the whole set of normals (index 3081)
#     # TODO invert=?
#     _, _, ts_h, phis_h = decompose_lin_maps(Hs_as_affine, asserts=True)
#
#     return ts_h, phis_h
#

def get_aff_map(t, phi, component_mask, invert_first):

    lin_map, T_t, R_phi = compose_lin_maps(t, phi)[0]

    assert invert_first, "current impl needs invert_first to be set to True"

    aff_map = torch.zeros((1, 2, 3))
    aff_map[:, :2, :2] = lin_map

    coords = torch.where(component_mask)
    min_x = coords[1].min()
    max_x = coords[1].max()
    min_y = coords[0].min()
    max_y = coords[0].max()

    corner_pts = torch.tensor([[min_x, min_y],
                               [min_x, max_y],
                               [max_x, max_y],
                               [max_x, min_y]], dtype=torch.float)[None]

    H = KR.geometry.convert_affinematrix_to_homography(aff_map)
    corner_pts_new = KR.geometry.transform_points(H, corner_pts)

    aff_map[:, :, 2] = -torch.tensor([corner_pts_new[0, :, 0].min(), corner_pts_new[0, :, 1].min()])

    new_w = int((corner_pts_new[0, :, 0].max() - corner_pts_new[0, :, 0].min()).item())
    new_h = int((corner_pts_new[0, :, 1].max() - corner_pts_new[0, :, 1].min()).item())
    return aff_map, new_h, new_w, lin_map, T_t, R_phi


def plot_space_of_tilts(label, img_name, valid_component, normal_index, exp_r, point_styles: list):

    fig, ax = plt.subplots()
    plt.title("{}: {} - component {} / normal {}".format(label, img_name, valid_component, normal_index))
    prepare_plot(exp_r, ax)
    for point_style in point_styles:
        draw(exp_r, point_style.ts, point_style.phis, point_style.color, point_style.size, ax)
    plt.show(block=False)


def visualize_LAF_custom(img, LAF, img_idx=0, color='r', title="", **kwargs):
    x, y = KR.feature.laf.get_laf_pts_to_draw(KR.feature.laf.scale_laf(LAF, 0.5), img_idx)
    plt.figure(**kwargs)
    plt.title(title)
    plt.imshow(KR.utils.tensor_to_image(img[img_idx]))
    plt.plot(x, y, color)
    plt.show(block=False)
    return


def get_rectified_keypoints(normals,
                            components_indices,
                            valid_components_dict,
                            img,
                            K,
                            descriptor,
                            img_name,
                            fixed_rotation_vector=None,
                            clip_angle=None,
                            show=False,
                            save=False,
                            out_prefix=None,
                            rotation_factor=1.0,
                            all_unrectified=False):

    print()
    # for component_index in valid_components_dict:
    #     R = get_rotation
    #     T, bounding_box = get_perspective_transform(R, K)
    #     rectified_img = warpPerspective(img, T, bounding_box)
    #     kps, descs = descriptor.detectAndCompute(rectified, None)
    #     * warp back
    #     * filter the kpts
    #     * update kpts location
    #     * add the keypoints
    # add keypoint with not belonging to any plane
    # return kpts, descs, unrectified_indices !!

# TODO document
def get_corners_of_mask(mask):
    coords = torch.where(mask)
    min_x = coords[1].min()
    max_x = coords[1].max()
    min_y = coords[0].min()
    max_y = coords[0].max()

    corner_pts = torch.tensor([[min_x, min_y],
                               [min_x, max_y],
                               [max_x, max_y],
                               [max_x, min_y]], dtype=torch.float).T[None]
    return corner_pts


def warp_affine(img_t, mask, affine_map, mode='biliner'):
    """
    Helper method to warp a torch image img_t with a centralized affine_map (t=0) so that mask
    will just fit in the minimal target img possible
    :param img_t: 
    :param mask: 
    :param affine_map: 
    :param mode: 
    :return: warped_img, new_h, new_w
    """
    corner_pts = get_corners_of_mask(mask)
    #H = KR.geometry.convert_affinematrix_to_homography(affine_map)
    #corner_pts_new = KR.geometry.transform_points(H, corner_pts)
    corner_pts_new = affine_map[:, :, :2] @ corner_pts

    affine_map[:, :, 2] = -torch.tensor([corner_pts_new[0, 0, :].min(), corner_pts_new[0, 1, :].min()])

    new_w = int((corner_pts_new[0, 0, :].max() - corner_pts_new[0, 0, :].min()).item())
    new_h = int((corner_pts_new[0, 1, :].max() - corner_pts_new[0, 1, :].min()).item())

    warped_img = KR.geometry.warp_affine(img_t, affine_map, dsize=(new_h, new_w), mode=mode)
    return warped_img, new_h, new_w


def warp_image(img, tilt, phi, img_mask, blur_param=0.8, invert_first=True, show_all=True):

    assert invert_first, "current impl needs invert_first to be set to True"
    show_all = False
    if show_all:
        show_torch_img(img, title="before warping")

    def tilt_img(img_fc, til):

        # TODO 2 interpolation = 'nearest' ??!!
        img_tilt = KR.geometry.transform.rescale(img_fc, (1.0, 1.0 / tilt), interpolation='nearest')

        if show_all:
            show_torch_img(img_tilt, title="blurred img - rescale")

        return img_tilt

    def blur(img_fc, tilt, blur_param=0.8):
        blur_amplification = 1.0
        sigma_x = blur_amplification * blur_param * math.sqrt(tilt * tilt - 1)
        kernel_size = 2 * math.ceil(sigma_x * 3.0) + 1
        kernel = KR.filters.get_gaussian_kernel1d(kernel_size, sigma_x)[None].unsqueeze(1)
        print("kernal shape: {}".format(kernel.shape))
        # will kernel be in a good shape? (#dimensions, but also the shape?)
        img_blurred = KR.filters.filter2d(img_fc, kernel)

        if show_all:
            show_torch_img(img_blurred, title="rotated then blurred")
            img_blurred_y = KR.filters.filter2d(img_fc, kernel.view(1, kernel.shape[2], 1))
            show_torch_img(img_blurred_y, title="rotated then blurred in the other direction")

        return img_blurred

    def warp_rotate(angle, img_fc, img_mask):
        s = math.sin(angle)
        c = math.cos(angle)
        affine_rotation = torch.tensor([[c, -s, 0.0], [s, c, 0.0]]).unsqueeze(0)
        rotated_img, new_h, new_w = warp_affine(img_fc, img_mask, affine_rotation, mode='bilinear')
        if show_all:
            show_torch_img(rotated_img, title="rotated img")
        return affine_rotation, rotated_img, new_h, new_w

    affine_transform, img_rotated, new_h, new_w = warp_rotate(phi, img, img_mask)
    img_blurred = blur(img_rotated, tilt, blur_param=blur_param)

    affine_transform[0, 0, :] = affine_transform[0, 0, :] * 1.0 / tilt
    img_tilt = tilt_img(img_blurred, tilt)

    show_all = True
    if show_all:
        show_torch_img(img_tilt, title="final rectification")


        img_warped_test = KR.geometry.warp_affine(img, affine_transform, dsize=(new_h, new_w))
        show_torch_img(img_warped_test, title="warped by one sigle affine transform")

        aff_map_inv = KR.geometry.transform.invert_affine_transform(affine_transform)
        img_warped_back = KR.geometry.warp_affine(img_warped_test, aff_map_inv, dsize=(img.shape[2], img.shape[3]))
        show_torch_img(img_warped_back, title="img warped back")

    return img_tilt, affine_transform


def affnet_rectify(img_name, hardnet_descriptor, img_data, conf_map):

    Timer.start_check_point("affnet_rectify")

    tilt_r = conf_map.get("affnet_tilt_r", 5.8)
    invert_first = conf_map.get("invert_first", True)
    # TODO - test the None functionality
    affnet_hard_net_filter = conf_map.get("affnet_hard_net_filter", 1)
    show_affnet = conf_map.get("show_affnet", True)

    #laffs_no_scale = get_lafs(img_data.img, hardnet_descriptor, img_name, conf_map)

    # TODO - check and remove - is COLOR_BGR2RGB necessary? If so, then previous experiments (with HardNet) were invalid?
    img = cv.cvtColor(img_data.img, cv.COLOR_BGR2RGB)
    # plt.figure(figsize=(9, 9))
    # plt.title(img_name)
    # plt.imshow(img)
    # plt.show()
    # plt.close()

    identity_kps, identity_descs, identity_laffs = hardnet_descriptor.detectAndCompute(img, give_laffs=True, filter=affnet_hard_net_filter)
    # torch
    kpts_component_indices = get_kpts_components_indices(img_data.components_indices, img_data.valid_components_dict, identity_laffs)
    mask_no_valid_component = (kpts_component_indices == -1)[0]

    all_kps = [kps for i, kps in enumerate(identity_kps) if mask_no_valid_component[i]] # []
    all_descs = identity_descs[mask_no_valid_component] # np.zeros((0, 128), dtype=np.float32)
    all_laffs = identity_laffs[:, mask_no_valid_component] # orch.zeros(1, 0, 2, 3)

    laffs_scale = KF.get_laf_scale(identity_laffs)
    laffs_no_scale = KF.scale_laf(identity_laffs, 1. / laffs_scale)

    if show_affnet:
        timg = KR.image_to_tensor(img, False).float() / 255.
        title = "{} - all unrectified affnet features".format(img_name)
        visualize_LAF_custom(timg, identity_laffs, title=title, figsize=(8, 12))
        title = "{} - all unrectified affnet features - no valid component".format(img_name)
        visualize_LAF_custom(timg, identity_laffs[:, mask_no_valid_component], title=title, figsize=(8, 12))

    affnet_lin_maps = laffs_no_scale[:, :, :, :2]

    if invert_first:
        affnet_lin_maps = torch.inverse(affnet_lin_maps)

    _, _, ts, phis = decompose_lin_maps_lambda_psi_t_phi(affnet_lin_maps)

    # plot_space_of_tilts -> all initial
    label = "inverted all" if invert_first else "not inverted all"
    print("{}: count: {}".format(label, ts.shape))
    plot_space_of_tilts(label, img_name, 0, 0, tilt_r, [
        PointsStyle(ts=ts, phis=phis, color="b", size=0.5),
    ])

    for current_component in img_data.valid_components_dict:

        normal_index = img_data.valid_components_dict[current_component]
        print("processing component->normal: {} -> {}".format(current_component, normal_index))

        mask = kpts_component_indices == current_component
        ts_affnet, phis_affnet = ts[mask], phis[mask]

        t_mean_affnet = torch.mean(ts_affnet)
        phi_mean_affnet = torch.mean(phis_affnet)

        label = "inverted unrectified" if invert_first else "not inverted unrectified"
        print("{}: count: {}".format(label, ts_affnet.shape))
        plot_space_of_tilts(label, img_name, current_component, normal_index, tilt_r, [
            #PointsStyle(ts=ts_normal, phis=phis_normal, color="black", size=5),
            PointsStyle(ts=ts_affnet, phis=phis_affnet, color="b", size=0.5),
            PointsStyle(ts=t_mean_affnet, phis=phi_mean_affnet, color="r", size=3)
        ])

        mask_img_component = torch.from_numpy(img_data.components_indices == current_component)

        t_img = KR.image_to_tensor(img_data.img, False).float() / 255.
        img_warped_t, aff_map = warp_image(t_img, t_mean_affnet.item(), phi_mean_affnet.item(), mask_img_component, invert_first=invert_first)
        img_warped = k_to_img_np(img_warped_t)

        # testing...
        # aff_maps, new_h, new_w, t_img = get_aff_map(img, torch.tensor([2.0]), torch.tensor([0.0]))

        # aff_maps, new_h, new_w, t_img = get_aff_map(img_data.img, t_mean_affnet, phi_mean_affnet, mask_img_component, invert_first)
        # img_warped_t = KR.geometry.warp_affine(t_img, aff_maps, dsize=(new_h, new_w))
        # img_warped = KR.tensor_to_image(img_warped_t * 255.0).astype(dtype=np.uint8)

        if show_affnet:
            img_normal_component_title = "{} - warped component {}, normal {}".format(img_name, current_component, normal_index)
            plt.figure(figsize=(6, 8))
            plt.title(img_normal_component_title)
            plt.imshow(img_warped)
            plt.show()

        kps_warped, descs_warped, laffs_final = hardnet_descriptor.detectAndCompute(img_warped, give_laffs=True, filter=affnet_hard_net_filter)

        aff_maps_inv = KR.geometry.transform.invert_affine_transform(aff_map)
        img_warped_back_t = KR.geometry.warp_affine(img_warped_t, aff_maps_inv, dsize=(t_img.shape[2], t_img.shape[3]))
        show_torch_img(img_warped_back_t, "warped back from caller")

        kps_t = torch.tensor([kp.pt + (1,) for kp in kps_warped])
        kpt_s_back = aff_maps_inv.repeat(kps_t.shape[0], 1, 1) @ kps_t.unsqueeze(2)
        kpt_s_back = kpt_s_back.squeeze(2)

        laffs_final[0, :, :, 2] = kpt_s_back

        kpt_s_back_int = torch.round(kpt_s_back).to(torch.long)
        mask = (kpt_s_back_int[:, 1] < img_data.img.shape[0]) & (kpt_s_back_int[:, 1] >= 0) & (kpt_s_back_int[:, 0] < img_data.img.shape[1]) & (kpt_s_back_int[:, 0] >= 0)
        print("invalid back transformed pixels: {}/{}".format(mask.shape[0] - mask.sum(), mask.shape[0]))

        kpt_s_back_int[~mask, 0] = 0
        kpt_s_back_int[~mask, 1] = 0
        mask = (mask) & (img_data.components_indices[kpt_s_back_int[:, 1], kpt_s_back_int[:, 0]] == current_component)
        mask = mask.to(torch.bool)

        kps = []
        for i, kp in enumerate(kps_warped):
            if mask[i]:
                kp.pt = (kpt_s_back[i][0].item(), kpt_s_back[i][1].item())
                kps.append(kp)
        descs = descs_warped[mask]
        laffs_final = laffs_final[:, mask]

        img_normal_component_title = "{} - rectified features for component {}, normal {}".format(img_name, current_component, normal_index)
        visualize_LAF_custom(t_img, laffs_final, title=img_normal_component_title,  figsize=(8, 12))

        all_kps.extend(kps)
        all_descs = np.vstack((all_descs, descs))
        all_laffs = torch.cat((all_laffs, laffs_final), 1)

        scale_l_final = KF.get_laf_scale(laffs_final)
        laffs_final_no_scale = KF.scale_laf(laffs_final, 1. / scale_l_final)
        _, _, ts_affnet_final, phis_affnet_final = decompose_lin_maps_lambda_psi_t_phi(laffs_final_no_scale[:, :, :, :2])

        label = "inverted rectified" if invert_first else "not inverted rectified"
        print("{}: count: {}".format(label, ts_affnet_final.shape))
        plot_space_of_tilts(label, img_name, current_component, normal_index, tilt_r, [
            #PointsStyle(ts=ts_normal, phis=phis_normal, color="black", size=5),
            PointsStyle(ts=ts_affnet_final, phis=phis_affnet_final, color="b", size=0.5),
        ])

    title = "{} - all rectified features".format(img_name)
    visualize_LAF_custom(t_img, all_laffs, title=title, figsize=(8, 12))

    all_scales = KF.get_laf_scale(all_laffs)
    all_laffs_no_scale = KF.scale_laf(all_laffs, 1. / all_scales)
    _, _, ts_affnet_final, phis_affnet_final = decompose_lin_maps_lambda_psi_t_phi(all_laffs_no_scale[:, :, :, :2])

    label = "all inverted rectified" if invert_first else "all not inverted rectified"
    print("{}: count: {}".format(label, ts_affnet_final.shape))
    plot_space_of_tilts(label, img_name, "-", "-", tilt_r, [
        PointsStyle(ts=ts_affnet_final, phis=phis_affnet_final, color="b", size=0.5),
    ])

    Timer.end_check_point("affnet_rectify")

    # TODO
    unrectified_indices = None
    return all_kps, all_descs, unrectified_indices


def draw_test():

    radius = 1.7
    ts = torch.tensor([1.1319, 1.1319, 1.2136, 1.0592, 1.1286, 1.1663, 1.2231, 1.2231, 1.1127,
                 1.0755, 1.0778, 1.2580, 1.0351, 1.2732, 1.2732, 1.2206, 1.1509, 1.0904,
                 1.0982, 1.0432, 1.3149, 1.0875, 1.0457, 1.0357, 1.1554, 1.2691, 1.2691,
                 1.0834, 1.0124, 1.0124, 1.1793, 1.2422, 1.0970, 1.2479, 1.2478, 1.0951,
                 1.1189, 1.0922, 1.0922, 1.0861, 1.2390, 1.0681, 1.0788, 1.1042, 1.2036,
                 1.1800, 1.1800, 1.1368, 1.1368, 1.2107, 1.1130, 1.0565, 1.1601, 1.1735,
                 1.1735, 1.0962, 1.1921, 1.0994, 1.0994, 1.2375, 1.1585, 1.0956, 1.0956,
                 1.0915, 1.2143, 1.1329, 1.0327, 1.2184, 1.2300, 1.3192, 1.0538, 1.0538,
                 1.1630, 1.1138, 1.1916, 1.0691, 1.0681, 1.0289, 1.0809, 1.1317, 1.0821,
                 1.0266, 1.0524, 1.1183, 1.2366, 1.1436, 1.2093, 1.2093, 1.1545, 1.0598,
                 1.0598, 1.1063, 1.1583, 1.1583, 1.0944, 1.1603, 1.2142, 1.2142, 1.0572,
                 1.1491])
    phis = torch.tensor([0.1547, 0.1547, 1.4570, 1.7965, 0.2988, 0.0480, 1.3214, 1.3214, 0.7572,
                   0.2797, 1.0581, 1.5154, 2.9855, 0.1490, 0.1490, 0.5565, 0.1747, 0.4184,
                   0.8013, 2.0616, 0.2092, 1.0492, 0.5837, 2.8053, 0.6010, 3.1406, 3.1407,
                   1.0849, 1.4009, 1.4009, 1.5825, 0.3531, 0.3736, 0.2089, 0.2089, 1.0472,
                   0.8741, 1.0096, 1.0096, 3.0395, 0.3065, 1.6334, 1.3562, 1.2099, 0.3060,
                   0.6847, 0.6847, 0.8206, 0.8206, 0.1486, 1.1090, 3.0709, 1.2815, 0.2579,
                   0.2579, 0.8093, 0.1903, 0.5502, 0.5502, 0.4062, 1.2590, 1.3575, 1.3575,
                   1.3447, 0.5134, 0.2113, 0.3479, 1.1235, 3.0801, 0.2207, 0.8382, 0.8382,
                   0.2779, 0.2045, 1.3080, 2.6112, 0.3788, 0.7550, 1.2412, 0.6423, 3.0673,
                   2.3306, 1.7669, 1.3823, 0.8896, 0.2179, 0.4288, 0.4288, 0.9286, 0.5084,
                   0.5084, 0.7006, 0.4556, 0.4556, 1.4091, 0.2774, 1.0528, 1.0528, 1.0057,
                   0.5571])

    # plt.figure()
    fig, ax = plt.subplots()
    plt.title("img foo - 1st normal")
    prepare_plot(radius, ax)
    draw(radius, ts, phis, "b", 1, ax)
    plt.show()


def decomposition_test():

    t = torch.tensor(1.5220)
    phi = torch.tensor(2.2653)
    print("orig t: {}, phi: {}".format(t, phi))

    def dec_and_print(lin_map):
        lambda_, psi, t, phi = decompose_lin_maps_lambda_psi_t_phi(lin_map, asserts=True)
        print("l: {}, psi: {}, t: {}, phi: {}".format(lambda_, psi, t, phi))
        return lambda_, psi, t, phi


    projected_lin_map, lambdas, R_psis, T_ts, R_phis = compose_lin_maps(t, phi)
    assert torch.all(projected_lin_map == lambdas * R_psis @ T_ts @ R_phis)

    print("decomposition scalar and matrices: lambda, R_1, T, R_2: \n {} \n {} \n {} \n {}".format(lambdas, R_psis, T_ts, R_phis))
    print(" = lin map: {}:".format(projected_lin_map))

    lambdas_back, psis_back, ts_back, phis_back = dec_and_print(projected_lin_map)

    projected_lin_map_back, lambdas, R_psis, T_ts, R_phis = compose_lin_maps(ts_back, phis_back, lambdas_back, psis_back)
    assert torch.all(projected_lin_map_back == lambdas * R_psis @ T_ts @ R_phis)

    assert torch.allclose(projected_lin_map_back, projected_lin_map)

    print("decomposition scalar and matrices: lambda, R_1, T, R_2: \n {} \n {} \n {} \n {}".format(lambdas, R_psis, T_ts, R_phis))
    print(" = lin map: {}:".format(projected_lin_map_back))

    projected_lin_map_inv = torch.inverse(projected_lin_map)
    print(" inverse: {}:".format(projected_lin_map_inv))

    lambdas_inv, psis_inv, ts_inv, phis_inv = dec_and_print(projected_lin_map_inv)


if __name__ == "__main__":
    #main_demo()
    decomposition_test()
    #draw_test()

# CONTINUE:
#   IMPORTANT: a) get_kpts_normals(normals, laffs_no_scale).unsqueeze(dim=0) -> get_kpts_normals_representatives!!!!
#   IMPORTANT: b) compare sets of affines (original or inverses) within one plane
#   IMPORTANT: c) compare sets of affines' decomposition (original or inverses) within one plane
#             -  basically the idea in Rodriquez is to ignore the tail of the distribution (cluster)

# (background - get_normal_vec_from_decomposition(ts, phis) is most likely wrong - at least because of the missing calibration
# get_normal_vec_from_decomposition - probably doesn't work as expected - implement 5,6 from affine_decomposition.pdf

# def get_rotation_matrices(unit_rotation_vectors, thetas):
#
#     # Rodrigues formula
#     # R = I + sin(theta) . K + (1 - cos(theta)).K**2
#
#     K = torch.zeros([*unit_rotation_vectors.shape[:2], *[3, 3]])
#
#     # K[:, :, 0, 0] = 0.0
#     K[:, :, 0, 1] = -unit_rotation_vectors[:, :, 2]
#     K[:, :, 0, 2] = unit_rotation_vectors[:, :, 1]
#     K[:, :, 1, 0] = unit_rotation_vectors[:, :, 2]
#     # K[:, :, 1, 1] = 0.0
#     K[:, :, 1, 2] = -unit_rotation_vectors[:, :, 0]
#
#     K[:, :, 2, 0] = -unit_rotation_vectors[:, :, 1]
#     K[:, :, 2, 1] = unit_rotation_vectors[:, :, 0]
#     # K[:, :, 2, 2] = 0.0
#
#     a = torch.eye(3)[None, None]
#
#     b = torch.sin(thetas).unsqueeze(3) * K
#
#     c = (1.0 - torch.cos(thetas)).unsqueeze(3) * K @ K
#
#     R = a + b + c
#
#     a0 = torch.eye(3)
#     b0 = torch.sin(thetas[0, 0]) * K[0, 0]
#     c0 = (1.0 - torch.cos(thetas[0, 0])) * K[0, 0] @ K[0, 0]
#     R0 = a0 + b0 + c0
#
#     R1 = get_rotation_matrix(unit_rotation_vectors[0, 0], thetas[0, 0])
#
#     assert torch.all(R[0,0] == R0)
#     assert torch.allclose(R[0,0], torch.from_numpy(R1).to(torch.float))
#
#     return R


# def get_rectification_rotations(normals):
#
#     # now the normals will be "from" me, "inside" the surfaces
#     normals = -torch.from_numpy(normals)[None]
#
#     z = torch.tensor([[[0.0, 0.0, 1.0]]]).expand(-1, normals.shape[1], -1)
#     print()
#
#     # this handles the case when there is only one dominating plane
#
#     assert torch.all(normals[..., 2] > 0)
#
#     rotation_vectors = torch.cross(normals, z)
#     rotation_vector_norms = torch.norm(rotation_vectors, dim=2).unsqueeze(2)
#     unit_rotation_vectors = rotation_vectors / rotation_vector_norms
#
#     Rs = get_rotation_matrices(unit_rotation_vectors, rotation_vector_norms)
#     assert torch.allclose(torch.det(Rs), torch.tensor(1.0))
#     return Rs
#
#

# def get_rectification_homographies(normals, K):
#     Rs = get_rectification_rotations(normals)
#     K = torch.from_numpy(K).to(torch.float)[None, None]
#     K_inv = torch.inverse(K).to(torch.float)
#     Hs = K @ Rs @ K_inv
#     return Hs
#
#

# def get_lafs(img, hardnet_desctiptor, img_name, conf_map={}):
#
#     affnet_hard_net_filter = conf_map.get("affnet_hard_net_filter", None)
#
#     # TODO - check and remove - is COLOR_BGR2RGB necessary? If so, then previous experiments with HardNet were invalid?
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     plt.figure(figsize=(9, 9))
#     plt.title(img_name)
#     plt.imshow(img)
#     plt.show()
#     plt.close()
#
#     kps, descs, laffs = hardnet_desctiptor.detectAndCompute(img, give_laffs=True, filter=affnet_hard_net_filter)
#     scale1 = KF.get_laf_scale(laffs)
#     lafs_no_scale = KF.scale_laf(laffs, 1. / scale1)
#
#     if conf_map.get("show_affnet", True):
#         timg = KR.image_to_tensor(img, False).float() / 255.
#         title = "{} - all unrectified affnet features".format(img_name)
#         visualize_LAF_custom(timg, laffs, title=title, figsize=(8, 12))
#
#     return lafs_no_scale

# NOTE : basically version of utils.get_kpts_normals, but for torch!!!
# def get_kpts_normals_indices(components_indices, valid_components_dict, laffs_no_scale):
#
#     Timer.start_check_point("get_kpts_normals_indices")
#
#     coords = laffs_no_scale[0, :, :, 2]
#     coords = torch.round(coords)
#     torch.clamp(coords[:, 0], 0, components_indices.shape[1] - 1, out=coords[:, 0])
#     torch.clamp(coords[:, 1], 0, components_indices.shape[0] - 1, out=coords[:, 1])
#     coords = coords.to(torch.long)
#
#     component_indices = components_indices[coords[:, 1], coords[:, 0]]
#     normals_indices = torch.ones_like(coords[:, 0]) * -1
#     for valid_component in valid_components_dict:
#         normals_indices[component_indices == valid_component] = valid_components_dict[valid_component]
#
#     Timer.end_check_point("get_kpts_normals_indices")
#
#     return normals_indices[None]

# def main_demo():
#
#     Timer.start()
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     Timer.start_check_point("HardNetDescriptor")
#     hardnet = HardNetDescriptor(sift_descriptor=cv.SIFT_create(None), device=device)
#     Timer.end_check_point("HardNetDescriptor")
#
#     pipeline = prepare_pipeline()
#
#     conf_map = {"affnet_tilt_r": 5.8}
#
#     #l = ["frame_0000000070_2", "frame_0000001525_1", "frame_0000001865_1"]
#     for img_name in ["frame_0000000070_2"]:
#
#         img_data = pipeline.process_image(img_name, order=0)[0]
#         assert len(img_data.normals.shape) == 2
#
#         conf_map["invert_first"] = False
#         affnet_rectify(img_name, hardnet, img_data, conf_map)
#         conf_map["invert_first"] = True
#         affnet_rectify(img_name, hardnet, img_data, conf_map)
#
#     Timer.log_stats()

# # TODO provide img as a parameter to save some computation
# def get_laffs_no_scale_p_cached(img_file_path, img_name, descriptor, cache_laffs):
#     cache_laffs_fn = "work/laffs_no_scale.pt"
#     if cache_laffs and os.path.exists(cache_laffs_fn):
#         Timer.start_check_point("laffs_no_scale cache read")
#         laffs_no_scale = torch.load(cache_laffs_fn)
#         Timer.end_check_point("laffs_no_scale cache read")
#     else:
#         Timer.start_check_point("laffs_no_scale computation")
#         _, _, laffs_no_scale = get_lafs(img_file_path, descriptor, img_name)
#         Timer.end_check_point("laffs_no_scale computation")
#         Timer.start_check_point("laffs_no_scale saving")
#         torch.save(laffs_no_scale, cache_laffs_fn)
#         Timer.end_check_point("laffs_no_scale saving")
#     return laffs_no_scale

# TODO circular dependence
# def prepare_pipeline():
#
#     Timer.start_check_point("prepare_pipeline")
#
#     pipeline, config_map = Pipeline.configure("config.txt", None)
#     pipeline.output_dir = "affnet_demo"
#     all_configs = CartesianConfig.get_configs(config_map)
#     config, cache_map = all_configs[0]
#     pipeline.config = config
#     pipeline.cache_map = cache_map
#
#     pipeline.start()
#     Timer.end_check_point("prepare_pipeline")
#
#     return pipeline




