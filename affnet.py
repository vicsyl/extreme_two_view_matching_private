import math
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

import pickle
import os
import cv2 as cv
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from pipeline import Pipeline
from kornia_moons.feature import *
from config import CartesianConfig
from hard_net_descriptor import HardNetDescriptor
from utils import Timer
from rectification import get_rotation_matrix

# def get_local_descriptors(img, cv2_sift_kpts, kornia_descriptor):
#     # We will not train anything, so let's save time and memory by no_grad()
#     with torch.no_grad():
#         kornia_descriptor.eval()
#         timg = K.color.rgb_to_grayscale(K.image_to_tensor(img, False).float()) / 255.
#         lafs = laf_from_opencv_SIFT_kpts(cv2_sift_kpts)
#         # We will estimate affine shape of the feature and re-orient the keypoints with the OriNet
#         affine = KF.LAFAffNetShapeEstimator(True)
#         orienter = KF.LAFOrienter(32, angle_detector=KF.OriNet(True))
#         orienter.eval()
#         affine.eval()
#         lafs2 = affine(lafs, timg)
#         lafs_new = orienter(lafs2, timg)
#         patches = KF.extract_patches_from_pyramid(timg, lafs_new, 32)
#         B, N, CH, H, W = patches.size()
#         # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
#         # So we need to reshape a bit :)
#         descs = kornia_descriptor(patches.view(B * N, CH, H, W)).view(B * N, -1)
#     return descs.detach().cpu().numpy(), lafs2
#
#
# def sift_korniadesc_matching(fname1, fname2, descriptor):
#     img1 = cv.cvtColor(cv.imread(fname1), cv.COLOR_BGR2RGB)
#     img2 = cv.cvtColor(cv.imread(fname2), cv.COLOR_BGR2RGB)
#
#     sift = cv.SIFT_create(8000)
#     kps1 = sift.detect(img1, None)
#     kps2 = sift.detect(img2, None)
#     # That is the only change in the pipeline -- descriptors
#     descs1, lafs_new1 = get_local_descriptors(img1, kps1, descriptor)
#     descs2, lafs_new2 = get_local_descriptors(img2, kps2, descriptor)
#     # The rest is the same, as for SIFT
#
#     dists, idxs = KF.match_smnn(torch.from_numpy(descs1), torch.from_numpy(descs2), 0.95)
#     tentatives = cv2_matches_from_kornia(dists, idxs)
#     src_pts = np.float32([kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 2)
#     dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 2)
#     F, inliers_mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, 0.75, 0.99, 100000)
#     inliers_mask = 1 * inliers_mask
#     draw_params = dict(matchColor=(255, 255, 0),  # draw matches in yellow color
#                        singlePointColor=None,
#                        matchesMask=inliers_mask.ravel().tolist(),  # draw only inliers
#                        flags=2)
#     img_out = cv.drawMatches(img1, kps1, img2, kps2, tentatives, None, **draw_params)
#     plt.figure(figsize=(15, 15))
#     #fig, ax = plt.subplots(figsize=(15, 15))
#     plt.imshow(img_out, interpolation='nearest')
#     plt.show()
#     print(f'{inliers_mask.sum()} inliers found')
#     return lafs_new1, lafs_new2
#
#
# def orig_main():
#
#     fname1 = 'kn_church-2.jpg'
#     fname2 = 'kn_church-8.jpg'
#
#     hardnet = KF.HardNet(True)
#     lafs1, lafs2 = sift_korniadesc_matching(fname1, fname2, hardnet)
#
#     img1 = cv.cvtColor(cv.imread(fname1), cv.COLOR_BGR2RGB)
#     timg1 = K.image_to_tensor(img1, False).float() / 255.
#
#     # Let's visualize some of the local features
#     visualize_LAF(timg1, lafs1[:, 1200:1500], figsize=(8, 12))
#
#     scale1 = KF.get_laf_scale(lafs1)
#     lafs_no_scale = KF.scale_laf(lafs1, 1. / scale1)
#
#     u, s, v = torch.linalg.svd(lafs_no_scale[0:, :, :2, :2].reshape(-1, 4).t())
#     aff1 = u[0].reshape(2, 2)
#     aff_full = torch.cat([aff1, torch.tensor([800, 400]).view(2, 1)], dim=1)[None]
#
#     aff_full = torch.cat([aff1, torch.tensor([100, 100]).view(2, 1)], dim=1)[None]
#     img1_warped = K.geometry.rotate(K.geometry.warp_affine(timg1, aff_full, dsize=(1000, 800)),
#                                     torch.tensor(-90.))
#     plt.figure(figsize=(6, 8))
#     plt.imshow(K.tensor_to_image(img1_warped[0, :, 100:600, 350:])[:, ::-1])
#     plt.show()


# def normals_caching_scheme():
#     cache_normals_fn = "work/normals.pt"
#     if os.path.exists(cache_normals_fn):
#         Timer.start_check_point("normals cache read")
#         with open(cache_normals_fn, "rb") as f:
#             normals_at_kps = pickle.load(f)
#         Timer.end_check_point("normals cache read")
#     else:
#         Timer.start_check_point("normals computation")
#         normals = pipeline.process_image(img_name, idx % 2)[1]
#         normals_at_kps = get_kpts_normals(normals, laffs_no_scale).unsqueeze(dim=0)
#         Timer.end_check_point("normals computation")
#         Timer.start_check_point("normals cache save")
#         with open(cache_normals_fn, "wb") as f:
#             pickle.dump(normals_at_kps, f)
#         Timer.end_check_point("normals cache save")


def get_rotation_matrices(unit_rotation_vectors, thetas):

    # Rodrigues formula
    # R = I + sin(theta) . K + (1 - cos(theta)).K**2

    K = torch.zeros([*unit_rotation_vectors.shape[:2], *[3, 3]])

    # K[:, :, 0, 0] = 0.0
    K[:, :, 0, 1] = -unit_rotation_vectors[:, :, 2]
    K[:, :, 0, 2] = unit_rotation_vectors[:, :, 1]
    K[:, :, 1, 0] = unit_rotation_vectors[:, :, 2]
    # K[:, :, 1, 1] = 0.0
    K[:, :, 1, 2] = -unit_rotation_vectors[:, :, 0]

    K[:, :, 2, 0] = -unit_rotation_vectors[:, :, 1]
    K[:, :, 2, 1] = unit_rotation_vectors[:, :, 0]
    # K[:, :, 2, 2] = 0.0

    a = torch.eye(3)[None, None]

    b = torch.sin(thetas).unsqueeze(3) * K

    c = (1.0 - torch.cos(thetas)).unsqueeze(3) * K @ K

    R = a + b + c

    a0 = torch.eye(3)
    b0 = torch.sin(thetas[0, 0]) * K[0, 0]
    c0 = (1.0 - torch.cos(thetas[0, 0])) * K[0, 0] @ K[0, 0]
    R0 = a0 + b0 + c0

    R1 = get_rotation_matrix(unit_rotation_vectors[0, 0], thetas[0, 0])

    assert torch.all(R[0,0] == R0)
    assert torch.allclose(R[0,0], torch.from_numpy(R1).to(torch.float))

    return R


def get_rectification_rotations(normals):

    # now the normals will be "from" me, "inside" the surfaces
    normals = -torch.from_numpy(normals)[None]

    z = torch.tensor([[[0.0, 0.0, 1.0]]]).expand(-1, normals.shape[1], -1)
    print()

    # this handles the case when there is only one dominating plane

    assert torch.all(normals[..., 2] > 0)

    rotation_vectors = torch.cross(normals, z)
    rotation_vector_norms = torch.norm(rotation_vectors, dim=2).unsqueeze(2)
    unit_rotation_vectors = rotation_vectors / rotation_vector_norms

    Rs = get_rotation_matrices(unit_rotation_vectors, rotation_vector_norms)
    assert torch.allclose(torch.det(Rs), torch.tensor(1.0))
    return Rs


def get_rectification_homographies(normals, K):
    Rs = get_rectification_rotations(normals)
    K = torch.from_numpy(K).to(torch.float)[None, None]
    K_inv = torch.inverse(K).to(torch.float)
    Hs = K @ Rs @ K_inv
    return Hs


def get_lafs(file_path, descriptor, img_name):

    img = cv.cvtColor(cv.imread(file_path), cv.COLOR_BGR2RGB)
    plt.figure(figsize=(9, 9))
    plt.title(img_name)
    plt.imshow(img)
    plt.close()

    kps, descs, laffs = descriptor.detectAndCompute(img, mask=None, give_laffs=True)

    timg = K.image_to_tensor(img, False).float() / 255.

    # Let's visualize some of the local features
    visualize_LAF(timg, laffs[:, ::10], figsize=(8, 12))

    scale1 = KF.get_laf_scale(laffs)
    lafs_no_scale = KF.scale_laf(laffs, 1. / scale1)

    # u, s, v = torch.linalg.svd(lafs_no_scale[0:, :, :2, :2].reshape(-1, 4).t())
    # aff1 = u[0].reshape(2, 2)
    # aff_full = torch.cat([aff1, torch.tensor([800, 400]).view(2, 1)], dim=1)[None]
    #
    # aff_full = torch.cat([aff1, torch.tensor([100, 100]).view(2, 1)], dim=1)[None]
    # img1_warped = K.geometry.rotate(K.geometry.warp_affine(timg1, aff_full, dsize=(1000, 800)),
    #                                 torch.tensor(-90.))
    # plt.figure(figsize=(6, 8))
    # plt.imshow(K.tensor_to_image(img1_warped[0, :, 100:600, 350:])[:, ::-1])
    # plt.show()

    return kps, descs, lafs_no_scale


def prepare_pipeline():

    Timer.start_check_point("prepare_pipeline")
    parser = argparse.ArgumentParser(prog='pipeline')
    parser.add_argument('--output_dir', help='output dir')
    args = parser.parse_args()

    pipeline, config_map = Pipeline.configure("config.txt", args)
    all_configs = CartesianConfig.get_configs(config_map)
    config, cache_map = all_configs[0]
    pipeline.config = config
    pipeline.cache_map = cache_map

    pipeline.start()
    Timer.end_check_point("prepare_pipeline")

    return pipeline


# def get_kpts_normals(normals, laffs_no_scale):
#
#     Timer.start_check_point("get_kpts_normals")
#
#     coords = laffs_no_scale[0, :, :, 2]
#     coords = torch.round(coords)
#     torch.clamp(coords[:, 0], 0, normals.shape[1] - 1, out=coords[:, 0])
#     torch.clamp(coords[:, 1], 0, normals.shape[0] - 1, out=coords[:, 1])
#     coords = coords.to(torch.long)
#
#     normals_kpts = normals[coords[:, 1], coords[:, 0]]
#
#     Timer.end_check_point("get_kpts_normals")
#
#     return normals_kpts
#
#


# NOTE : basically version of utils.get_kpts_normals, but for torch!!!
def get_kpts_normals_indices(components_indices, valid_components_dict, laffs_no_scale):

    Timer.start_check_point("get_kpts_normals_indices")

    coords = laffs_no_scale[0, :, :, 2]
    coords = torch.round(coords)
    torch.clamp(coords[:, 0], 0, components_indices.shape[1] - 1, out=coords[:, 0])
    torch.clamp(coords[:, 1], 0, components_indices.shape[0] - 1, out=coords[:, 1])
    coords = coords.to(torch.long)

    component_indices = components_indices[coords[:, 1], coords[:, 0]]
    normals_indices = torch.ones_like(coords[:, 0]) * -1
    for valid_component in valid_components_dict:
        normals_indices[component_indices == valid_component] = valid_components_dict[valid_component]

    Timer.end_check_point("get_kpts_normals_indices")

    return normals_indices[None]


def decompose_lin_maps(l_maps, asserts=True):

    Timer.start_check_point("decompose_lin_maps")

    l_maps = torch.inverse(l_maps)

    # TODO watch out for CUDA efficiency
    U, s, V = torch.svd(l_maps)
    V = torch.transpose(V, dim0=2, dim1=3)

    if asserts:
        assert torch.all(torch.sgn(s[:, :, 0]) == torch.sgn(s[:, :, 1]))
        assert torch.all(s[:, :, 0] != 0)

    factor = torch.sgn(s[:, :, :1])
    U = factor[:, :, :, None] * U
    s = factor * s
    lambdas = s[:, :, 1].clone()
    s = s / s[:, :, 1:]

    if asserts:
        assert torch.all(s[:, :, 0] >= 1)
        assert torch.all(s[:, :, 1] == 1)

    dets_u = torch.det(U)
    dets_v = torch.det(V)
    if asserts:
        assert torch.allclose(dets_v, dets_u, atol=1e-07)
        assert torch.allclose(torch.abs(dets_v), torch.tensor(1.0), atol=1e-07)

    def swap_rows(A):
        for col in [0, 1]:
            tmp_el = A[:, :, 0, col].clone()
            A[:, :, 0, col] = torch.where(dets_v < 0.0, A[:, :, 1, col], A[:, :, 0, col])
            A[:, :, 1, col] = torch.where(dets_v < 0.0, tmp_el, A[:, :, 1, col])

    swap_rows(U)
    swap_rows(V)

    dets_u = torch.det(U)
    dets_v = torch.det(V)
    if asserts:
        assert torch.allclose(dets_v, dets_u, atol=1e-07)
        assert torch.allclose(dets_v, torch.tensor(1.0), atol=1e-07)

    # phi in (0, pi)
    phi_norm_factor = torch.where(V[:, :, :1, 1:] > 0, -1.0, 1.0)
    V = V * phi_norm_factor
    U = U * phi_norm_factor

    def assert_rotation(A, angle):
        sins_ang = torch.sin(angle)
        if asserts:
            assert torch.allclose(sins_ang, -A[:, :, 0, 1], atol=1e-03)
            assert torch.allclose(sins_ang, A[:, :, 1, 0], atol=1e-03)
            assert torch.allclose(A[:, :, 1, 1], A[:, :, 0, 0], atol=1e-05)

    phis = torch.arccos(V[:, :, 0, 0])
    assert_rotation(V, phis)

    psis = torch.arcsin(-torch.clamp(U[:, :, 0, 1], -1.0, 1.0))
    #psis_norm_factor = torch.where(U[:, :, :1, 1:] > 0, -1.0, 1.0)
    #psis = psis * psis_norm_factor[:, :, 0, 0]

    assert_rotation(U, psis)

    ts = s[:, :, 0]

    Timer.end_check_point("decompose_lin_maps")

    return lambdas, psis, ts, phis


# def get_rectification_rotation_vectors(normals):
#
#     # now the normals will be "from" me, "inside" the surfaces
#     normals = -normals
#
#     z = torch.tensor([0.0, 0.0, 1.0])
#
#     # this handles the case when there is only one dominating plane
#     assert normals[:, 2] > 0
#
#     rotation_vectors = torch.cross(normals, z)
#     rotation_vector_norms = np.linalg.norm(rotation_vectors, axis=1) # sins_theta
#     unit_rotation_vector = rotation_vectors / rotation_vector_norms
#
#
#     # theta = math.asin(abs_sin_theta) * rotation_factor
#     # # TODO !!!
#     # theta = min(theta, math.pi * 4.0/9.0)
#     #
#     # R = get_rotation_matrix(unit_rotation_vector, theta)
#     # det = np.linalg.det(R)
#     # assert math.fabs(det - 1.0) < 0.0001
#     # return R

# def get_normal_vec_from_decomposition(ts, phis):
#     Timer.start_check_point("get_normal_vec_from_decomposition")
#     sin_theta = torch.sqrt(ts ** 2 - 1) / ts
#     xs = torch.cos(phis) * sin_theta
#     ys = torch.sin(phis) * sin_theta
#     zs = torch.sqrt(1 - xs ** 2 - ys ** 2)
#     norms_from_lafs = torch.cat((xs.unsqueeze(2), ys.unsqueeze(2), zs.unsqueeze(2)), 2)
#
#     Timer.end_check_point("get_normal_vec_from_decomposition")
#     return norms_from_lafs


def get_laffs_no_scale_p_cached(img_file_path, img_name, descriptor):
    cache_laffs_fn = "work/laffs_no_scale.pt"
    if os.path.exists(cache_laffs_fn):
        Timer.start_check_point("laffs_no_scale cache read")
        laffs_no_scale = torch.load(cache_laffs_fn)
        Timer.end_check_point("laffs_no_scale cache read")
    else:
        Timer.start_check_point("laffs_no_scale computation")
        _, _, laffs_no_scale = get_lafs(img_file_path, descriptor, img_name)
        Timer.end_check_point("laffs_no_scale computation")
        Timer.start_check_point("laffs_no_scale saving")
        torch.save(laffs_no_scale, cache_laffs_fn)
        Timer.end_check_point("laffs_no_scale saving")
    return laffs_no_scale


def draw(radius, ts, phis, color, size, ax):

    # upper bound on radii
    log_radius = math.log(radius)
    print("foo draw: r = {}\n ts = {} \n phis = {} \n style = {}".format(log_radius, ts[:100], phis[:100], color))

    ts_logs = torch.log(ts)
    xs = torch.cos(phis) * ts_logs
    ys = torch.sin(phis) * ts_logs

    ax.plot(xs, ys, 'o', color=color, markersize=size)


def prepare_plot(radius: float, ax):

    log_radius = math.log(radius)

    ax.set_aspect(1.0)

    ax.set_xlim((-2*log_radius, 2*log_radius))
    ax.set_ylim((-2*log_radius, 2*log_radius))

    circle = Circle((0, 0), log_radius, color='r', fill=False)
    ax.add_artist(circle)


# TODO export somehow?
def get_normals(normals, K):

    Hs = get_rectification_homographies(normals, K)
    Hs_as_affine = Hs[:, :, :2, :2]
    det_Hs = torch.det(Hs_as_affine).sqrt().unsqueeze(2).unsqueeze(3)
    Hs_as_affine = Hs_as_affine / det_Hs

    # TODO CONTINUE asserts=False did not work on the whole set of normals (index 3081)
    _, _, ts_h, phis_h = decompose_lin_maps(Hs_as_affine, asserts=True)

    return ts_h, phis_h


def main():

    Timer.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Timer.start_check_point("HardNetDescriptor")
    hardnet = HardNetDescriptor(sift_descriptor=cv.SIFT_create(None), device=device)
    Timer.end_check_point("HardNetDescriptor")

    pipeline = prepare_pipeline()

    # dfl = ["frame_0000000070_2", "frame_0000001525_1", "frame_0000001865_1"]
    dfl = ["frame_0000000070_2"]
    for idx, img_name in enumerate(dfl):
        img_file_path = pipeline.scene_info.get_img_file_path(img_name)

        laffs_no_scale = get_laffs_no_scale_p_cached(img_file_path, img_name, hardnet)
        _, _, ts, phis = decompose_lin_maps(laffs_no_scale[:, :, :, :2])

        img_data = pipeline.process_image(img_name, idx % 2)[0]
        kpts_normals_indices = get_kpts_normals_indices(img_data.components_indices, img_data.valid_components_dict, laffs_no_scale)

        K = pipeline.scene_info.get_img_K(img_name)
        ts_h, phis_h = get_normals(img_data.normals, K)

        exp_r = 1.7
         # -1 for invalid/no normal #AGAIN the ambiguity of whether 1 normal is (3,) it (1,3)
        assert len(img_data.normals.shape) == 2
        for i in range(img_data.normals.shape[0]):

            fig, ax = plt.subplots()
            plt.title("{} - {}th normal".format(img_name, i))
            prepare_plot(exp_r, ax)

            ts_normal, phis_normal = ts_h[:, i], phis_h[:, i]
            draw(exp_r, ts_normal, phis_normal, "black", 5, ax)

            mask = kpts_normals_indices == i
            ts_affnet, phis_affnet = ts[mask], phis[mask]
            draw(exp_r, ts_affnet, phis_affnet, "b", 0.5, ax)

            plt.show()

    Timer.log_stats()


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


if __name__ == "__main__":
    main()
    #draw_test()

# CONTINUE:
#   IMPORTANT: a) get_kpts_normals(normals, laffs_no_scale).unsqueeze(dim=0) -> get_kpts_normals_representatives!!!!
#   IMPORTANT: b) compare sets of affines (original or inverses) within one plane
#   IMPORTANT: c) compare sets of affines' decomposition (original or inverses) within one plane
#             -  basically the idea in Rodriquez is to ignore the tail of the distribution (cluster)

# (background - get_normal_vec_from_decomposition(ts, phis) is most likely wrong - at least because of the missing calibration
# get_normal_vec_from_decomposition - probably doesn't work as expected - implement 5,6 from affine_decomposition.pdf

