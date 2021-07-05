import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from resize import resample_nearest_numpy
from utils import Timer, identity_map_from_range_of_iter, get_rotation_matrix
from scene_info import SceneInfo, read_cameras
from connected_components import get_and_show_components, read_img_normals_info, get_connected_components
from img_utils import show_or_close
from depth_to_normals import compute_normals
from config import Config


def get_rectification_rotation(normal, rotation_factor=1.0):

    # now the normals will be "from" me, "inside" the surfaces
    normal = -normal

    z = np.array([0.0, 0.0, 1.0])

    # this handles the case when there is only one dominating plane

    assert normal[2] > 0
    rotation_vector = np.cross(normal, z)
    rotation_vector_norm = abs_sin_theta = np.linalg.norm(rotation_vector)
    unit_rotation_vector = rotation_vector / rotation_vector_norm
    theta = math.asin(abs_sin_theta) * rotation_factor
    theta = min(theta, math.pi * 4.0/9.0)

    R = get_rotation_matrix(unit_rotation_vector, theta)
    det = np.linalg.det(R)
    assert math.fabs(det - 1.0) < 0.0001
    return R


def add_third_row(column_vecs):
    return np.vstack((column_vecs, np.ones(column_vecs.shape[1])))


def get_perspective_transform(R, K, K_inv, component_indices, index, scale=1.0):

    unscaled = True
    while unscaled:

        coords = np.where(component_indices == index)
        coords = np.array([coords[1], coords[0]])
        coords = add_third_row(coords)

        # TODO rethink if I really need K
        # (I can guess)
        P = K @ R @ K_inv
        if scale != 1.0:
            unscaled = False
            print("old scale: {}".format(scale))
            P[:2, :] *= scale

        new_coords = P @ coords
        new_coords = new_coords / new_coords[2, :]
        #new_coords[2] = 1.0

        min_row = min(new_coords[1])
        max_row = max(new_coords[1])
        min_col = min(new_coords[0])
        max_col = max(new_coords[0])

        dst = np.float32([[min_col, min_row], [min_col, max_row - 1], [max_col - 1, max_row - 1], [max_col - 1, min_row]])
        dst = np.transpose(dst)
        dst = add_third_row(dst)

        if unscaled:
            new_bb_size = (max_row - min_row) * (max_col - min_col)
            # len(coords) * factor(=1.3) = new_bb
            scale = np.sqrt((coords.shape[1] * 2.0) / new_bb_size)
            if scale == 1.0:
                unscaled = False
                break



    translate_vec_new = (-np.min(dst[0]), -np.min(dst[1]))
    translate_matrix_new = np.array([
        [1, 0, translate_vec_new[0]],
        [0, 1, translate_vec_new[1]],
        [0, 0, 1],
    ])

    dst = translate_matrix_new @ dst
    P = translate_matrix_new @ P
    bounding_box_new = (math.ceil(np.max(dst[0])), math.ceil(np.max(dst[1])))

    return P, bounding_box_new


def split_keypoints_into_raw(kps):
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    ors = np.array([k.angle for k in kps], dtype=np.float32)
    scs = np.array([k.size for k in kps], dtype=np.float32)
    return pts, ors, scs


def get_rectified_keypoints(normals,
                            components_indices,
                            valid_components_dict,
                            img,
                            K,
                            descriptor,
                            img_name,
                            show=False,
                            save=False,
                            out_prefix=None,
                            rotation_factor=1.0):

    Timer.start_check_point("rectification")

    K_inv = np.linalg.inv(K)

    all_orig_pts = np.zeros((0, 2), dtype=np.float32)
    all_new_pts = np.zeros((0, 2), dtype=np.float32)
    all_scs = np.zeros(0, dtype=np.float32)
    all_ors = np.zeros(0, dtype=np.float32)
    all_descs = np.zeros((0, 128), dtype=np.float32)
    all_kps = []

    # components_in_colors will be used for other visualizations
    # components_in_colors = get_and_show_components(components_indices, valid_components_dict, show=False)

    for component_index in valid_components_dict:

        normal_index = valid_components_dict[component_index]
        normal = normals[normal_index]

        R = get_rectification_rotation(normal, rotation_factor)

        T, bounding_box = get_perspective_transform(R, K, K_inv, components_indices, component_index)
        assert bounding_box[0] * bounding_box[1] < 10**8, "warping to an img that is too big, skipping"

        T_inv = np.linalg.inv(T)

        rectified = cv.warpPerspective(img, T, bounding_box)

        kps, descs = descriptor.detectAndCompute(rectified, None)
        orig_pts, ors, scs = split_keypoints_into_raw(kps)
        new_pts = cv.perspectiveTransform(orig_pts.reshape(-1, 1, 2), T_inv)
        new_pts = new_pts.reshape(-1, 2)

        if new_pts is not None:
            kps_int_coords = np.int32(new_pts)

            h, w, _ = img.shape
            first = kps_int_coords[:, 0]
            first = np.where(0 <= first, first, 0)
            first = np.where(first < w, first, 0)
            seconds = kps_int_coords[:, 1]
            seconds = np.where(0 <= seconds, seconds, 0)
            # TODO: :, 0) => this I think is bad!!!
            seconds = np.where(seconds < h, seconds, 0)
            kps_int_coords[:, 0] = first
            kps_int_coords[:, 1] = seconds

            cluster_mask_bool = np.array([components_indices[kps_int_coord[1], [kps_int_coord[0]]] == component_index for kps_int_coord in kps_int_coords])
            cluster_mask_bool = cluster_mask_bool.reshape(-1)

            # TODO a) add assert
            # TODO b) on a clean WC, debug and rename the local vars (also, possibly speed up)
            # TODO c) new_kps[:, 0, 0/1] still out of bounds (i.e. negative)
            descs = descs[cluster_mask_bool]
            new_pts = new_pts[cluster_mask_bool]
            orig_pts = orig_pts[cluster_mask_bool]
            ors = ors[cluster_mask_bool]
            scs = scs[cluster_mask_bool]

            kps = [kp for i, kp in enumerate(kps) if cluster_mask_bool[i]]

            cv.drawKeypoints(rectified, kps, rectified, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            for kpi, kp in enumerate(kps):
                kp.pt = tuple(new_pts[kpi].tolist())

            all_kps.extend(kps)
            all_descs = np.vstack((all_descs, descs))
            all_new_pts = np.vstack((all_new_pts, new_pts))
            all_orig_pts = np.vstack((all_orig_pts, orig_pts))
            all_scs = np.hstack((all_scs, scs))
            all_ors = np.hstack((all_ors, ors))

        if show or save:
            plt.figure()
            plt.title("{} - component: {},\n {} kpts \n normal: {}".format(img_name, component_index, descs.shape[0], normals[normal_index]))
            plt.imshow(rectified)
            if save:
                plt.savefig("{}_rectified_component_{}.jpg.".format(out_prefix, component_index))
            show_or_close(show)

        # if show_components:
        #     rectified_components = components_in_colors.astype(np.float32) / 255
        #     rectified_components = cv.warpPerspective(rectified_components, T, bounding_box)
        #     plt.imshow(rectified_components)
        #     show_or_close(show)


    # TODO corner case - None, [], ...
    kps, descs = descriptor.detectAndCompute(img, None)
    pts, ors, scs = split_keypoints_into_raw(kps)

    kps_floats = np.float32([kp.pt for kp in kps])
    # TODO is this the way to round it?
    kps_ints = np.int32(kps_floats)
    in_img_mask = kps_ints[:, 0] >= 0
    in_img_mask = np.logical_and(in_img_mask, kps_ints[:, 0] < img.shape[1])
    in_img_mask = np.logical_and(in_img_mask, kps_ints[:, 1] >= 0)
    in_img_mask = np.logical_and(in_img_mask, kps_ints[:, 1] < img.shape[0])
    kps_ints = kps_ints[in_img_mask]
    kps = [kp for i, kp in enumerate(kps) if in_img_mask[i]]

    descs = descs[in_img_mask]
    pts = pts[in_img_mask]
    ors = ors[in_img_mask]
    scs = scs[in_img_mask]

    valid_keys_set = set(valid_components_dict.keys())
    all_indices_set = set(range(np.max(components_indices) + 1))
    non_valid_indices = list(all_indices_set - valid_keys_set)

    filter_non_valid = np.zeros(kps_ints.shape[0])
    for non_valid_index in non_valid_indices:
        filter_non_valid = np.logical_or(filter_non_valid, components_indices[kps_ints[:, 1], kps_ints[:, 0]] == non_valid_index)

    kps = [kp for i, kp in enumerate(kps) if filter_non_valid[i]]

    descs = descs[filter_non_valid]
    pts = pts[filter_non_valid]
    ors = ors[filter_non_valid]
    scs = scs[filter_non_valid]

    all_kps.extend(kps)
    all_descs = np.vstack((all_descs, descs))
    all_new_pts = np.vstack((all_new_pts, pts))
    all_orig_pts = np.vstack((all_orig_pts, pts))
    all_scs = np.hstack((all_scs, scs))
    all_ors = np.hstack((all_ors, ors))

    if show or save:
        no_component_img = img.copy()
        cv.drawKeypoints(no_component_img, kps, no_component_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(10, 10))
        plt.title("{} - no valid component: {} kpts".format(img_name, descs.shape[0]))
        plt.imshow(no_component_img)
        if save:
            plt.savefig("{}_rectified_no_valid_component.jpg.".format(out_prefix))
        show_or_close(show)

        all_img = img.copy()
        cv.drawKeypoints(all_img, all_kps, all_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(10, 10))
        plt.title("All keypoints: {} kpts".format(all_descs.shape[0]))
        plt.imshow(all_img)
        if save:
            plt.savefig("{}_rectified_all.jpg.".format(out_prefix))
        show_or_close(show)
        print("{} keypoints found".format(len(all_kps)))

    Timer.end_check_point("rectification")

    return all_kps, all_descs, all_orig_pts, all_new_pts, all_ors, all_scs


def possibly_upsample_normals(img, normal_indices):

    if img.shape[0] != normal_indices.shape[0]:
        # needs upsampling
        #epsilon = 0.0001
        epsilon = 0.003
        hard_epsilon = 0.1
        aspect_ratio_diff = abs(img.shape[0] / normal_indices.shape[0] - img.shape[1] / normal_indices.shape[1])
        if aspect_ratio_diff >= hard_epsilon:
            raise Exception("{} and {} not of the same aspect ratio".format(normal_indices.shape, img.shape))
        else:
            if img.shape[0] < normal_indices.shape[0]:
                print("WARNING: img.shape[0] < normal_indices.shape[0]")
            if aspect_ratio_diff >= epsilon:
                print("WARNING: {} and {} not of the same aspect ratio".format(normal_indices.shape, img.shape))
            print("Will upsample the normals")
            normal_indices = resample_nearest_numpy(normal_indices, img.shape[0], img.shape[1])

    return normal_indices


def show_rectifications(scene_info: SceneInfo, normals_parent_dir, original_input_dir, limit, interesting_dirs=None):

    if interesting_dirs is not None:
        dirs = interesting_dirs
    else:
        dirs = [dirname for dirname in sorted(os.listdir(normals_parent_dir)) if os.path.isdir("{}/{}".format(normals_parent_dir, dirname))]
        dirs = sorted(dirs)
        if limit is not None:
            dirs = dirs[0:limit]

    if len(dirs) == 0:
        print("WARNING: no normals data!")

    for img_name in dirs:
        print("Processing: {}".format(img_name))

        img_file_path = '{}/{}.jpg'.format(original_input_dir, img_name)
        img = cv.imread(img_file_path, None)

        normals, normal_indices = read_img_normals_info(normals_parent_dir, img_name)
        if normals is None:
            print("depth data for img_name is probably missing, skipping")
            continue

        show = True
        if show:
            get_and_show_components(normal_indices, identity_map_from_range_of_iter(normals), normals)

        # manual "extension" point
        # normals = np.array(
        #     [[ 0.33717412, -0.30356583, -0.89115733],
        #      [-0.68118596, -0.23305716, -0.6940245 ]]
        # )
        # normals = np.array(
        #     [
        #      [ 0.33717412, -0.30356583, -0.89115733],
        #      [-0.91, -0.25, -0.31]],
        # )
        # for i in range(len(normals)):
        #     norm = np.linalg.norm(normals[i])
        #     normals[i] /= norm
        #     print("normalized: {}".format(normals[i]))

        normal_indices = possibly_upsample_normals(img, normal_indices)
        K = scene_info.get_img_K(img_name)
        components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals)), True)

        get_rectified_keypoints(normals,
                                components_indices,
                                valid_components_dict,
                                img,
                                K,
                                descriptor=cv.SIFT_create(),
                                img_name=img_name,
                                show=True)


def play_with_rectification(scene_info: SceneInfo, normals_parent_dir, original_input_dir, limit, interesting_dirs=None):

    if interesting_dirs is not None:
        dirs = interesting_dirs
    else:
        dirs = [dirname for dirname in sorted(os.listdir(normals_parent_dir)) if os.path.isdir("{}/{}".format(normals_parent_dir, dirname))]
        dirs = sorted(dirs)
        if limit is not None:
            dirs = dirs[0:limit]

    if len(dirs) == 0:
        print("WARNING: no normals data!")

    for img_name in dirs:
        print("Processing: {}".format(img_name))

        img_file_path = '{}/{}.jpg'.format(original_input_dir, img_name)
        img = cv.imread(img_file_path, None)

        normals, normal_indices = read_img_normals_info(normals_parent_dir, img_name)
        if normals is None:
            print("depth data for img_name is probably missing, skipping")
            continue

        show_domponents = False
        if show_domponents:
            get_and_show_components(normal_indices, identity_map_from_range_of_iter(normals), normals)

        # manual "extension" point
        # normals = np.array(
        #     [[ 0.33717412, -0.30356583, -0.89115733],
        #      [-0.68118596, -0.23305716, -0.6940245 ]]
        # )
        # normals = np.array(
        #     [
        #      [ 0.33717412, -0.30356583, -0.89115733],
        #      [-0.91, -0.25, -0.31]],
        # )
        # for i in range(len(normals)):
        #     norm = np.linalg.norm(normals[i])
        #     normals[i] /= norm
        #     print("normalized: {}".format(normals[i]))

        normal_indices = possibly_upsample_normals(img, normal_indices)
        K = scene_info.get_img_K(img_name)
        components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals)), True)

        all_kps, all_descs = get_rectified_keypoints(normals,
                                components_indices,
                                valid_components_dict,
                                img,
                                K,
                                descriptor=cv.SIFT_create(),
                                img_name=img_name,
                                show=True)


def play_main():

    Timer.start()

    interesting_files = []

    scene_info = SceneInfo.read_scene("scene1", lazy=True)

    #play_iterate(scene_info, "original_dataset/scene1/images", limit=20, interesting_files=interesting_files)

    Timer.end()


interests = [
    # "frame_0000001670_1.jpg",
    # "frame_0000000705_3.jpg",
    "frame_0000000535_3.jpg",
    "frame_0000000450_3.jpg",
    # "frame_0000000910_3.jpg",
    # "frame_0000000870_4.jpg",
    # "frame_0000002185_1.jpg",
    # "frame_0000000460_4.jpg",
    # "frame_0000000165_1.jpg",
    # "frame_0000000335_1.jpg",
    # "frame_0000000355_1.jpg",
    # "frame_0000000250_1.jpg",

]

pairs_int_files = interesting_dirs = [
    # "frame_0000001535_4.jpg",
    # "frame_0000000305_1.jpg",
    # "frame_0000001135_4.jpg",
    # "frame_0000001150_4.jpg",
    # "frame_0000000785_2.jpg",
    # "frame_0000000710_2.jpg",
    # "frame_0000000155_4.jpg",
    # "frame_0000002375_1.jpg",
    # "frame_0000000535_3.jpg",
    # "frame_0000000450_3.jpg",
    # "frame_0000000895_4.jpg",
    # "frame_0000000610_2.jpg",
    # "frame_0000000225_3.jpg",
    # "frame_0000000265_4.jpg",
    # "frame_0000000105_2.jpg",
    # "frame_0000000365_3.jpg",
    # "frame_0000001785_3.jpg",
    # "frame_0000000125_1.jpg",
    # "frame_0000000910_3.jpg",
    # "frame_0000000870_4.jpg",
    # "frame_0000002230_1.jpg",
    # "frame_0000000320_3.jpg",
    # "frame_0000000315_3.jpg",
    # "frame_0000000085_4.jpg",
    # "frame_0000002070_1.jpg",
    # "frame_0000000055_2.jpg",
    # "frame_0000001670_1.jpg",
    # "frame_0000000705_3.jpg",
    "frame_0000000345_1.jpg",
    "frame_0000001430_4.jpg",
    "frame_0000002185_1.jpg",
    "frame_0000000460_4.jpg",
    "frame_0000001175_3.jpg",
    "frame_0000001040_4.jpg",
    "frame_0000000165_1.jpg",
    "frame_0000000335_1.jpg",
    "frame_0000001585_4.jpg",
    "frame_0000001435_4.jpg",
    "frame_0000000110_4.jpg",
    "frame_0000000130_3.jpg",
    "frame_0000000445_2.jpg",
    "frame_0000000755_3.jpg",
    "frame_0000000355_1.jpg",
    "frame_0000000250_1.jpg",
    ]

if __name__ == "__main__":

    play_main()

    Timer.start()

    #interesting_dirs = ["frame_0000000145_2"]
    #interesting_dirs = ["frame_0000000015_4"]

    scene_info = SceneInfo.read_scene("scene1", lazy=True)

    show_rectifications(scene_info, "work/scene1/normals/svd", "original_dataset/scene1/images", limit=1, interesting_dirs=interesting_dirs)

    Timer.end()


def get_rectified_keypoints_new(normals,
                            components_indices,
                            valid_components_dict,
                            img,
                            K,
                            descriptor,
                            img_name,
                            show=False,
                            save=False,
                            out_prefix=None,
                            rotation_factor=1.0):

    Timer.start_check_point("new rectification")

    K_inv = np.linalg.inv(K)

    all_orig_pts = np.zeros((0, 2), dtype=np.float32)
    all_new_pts = np.zeros((0, 2), dtype=np.float32)
    all_scs = np.zeros(0, dtype=np.float32)
    all_ors = np.zeros(0, dtype=np.float32)
    all_descs = np.zeros((0, 128), dtype=np.float32)
    all_kps = []

    for component_index in valid_components_dict:

        normal_index = valid_components_dict[component_index]
        normal = normals[normal_index]

        R = get_rectification_rotation(normal, rotation_factor)

        #T, bounding_box = get_perspective_transform(R, K, K_inv, components_indices, component_index)
        T, bounding_box = get_perspective_transform_new(R, K, K_inv, components_indices, component_index, normal)
        assert bounding_box[0] * bounding_box[1] < 10**8, "warping to an img that is too big, skipping"

        rectified = cv.warpPerspective(img, T, bounding_box)

        kps, descs = descriptor.detectAndCompute(rectified, None)

        if len(kps) > 0:

            orig_pts, ors, scs = split_keypoints_into_raw(kps)
            T_inv = np.linalg.inv(T)
            new_pts = cv.perspectiveTransform(orig_pts.reshape(-1, 1, 2), T_inv)
            if new_pts is None:
                new_pts = np.zeros((0, 1, 2))
            new_pts = new_pts.reshape(-1, 2)

            #kps_int_coords = np.int32(new_pts).reshape(-1, 2)
            # TODO improvement DONE: round
            kps_int_coords = np.round(new_pts).astype(dtype=np.int32)
            #kps_int_coords = np.int32(new_pts)

            h, w, _ = img.shape

            kps_int_coords_mask = kps_int_coords[:, 0] >= 0
            kps_int_coords_mask = np.logical_and(kps_int_coords_mask, kps_int_coords[:, 1] >= 0)
            kps_int_coords_mask = np.logical_and(kps_int_coords_mask, kps_int_coords[:, 0] < w)
            kps_int_coords_mask = np.logical_and(kps_int_coords_mask, kps_int_coords[:, 1] < h)
            kps_int_coords_mask_x_y = np.repeat(kps_int_coords_mask.reshape(-1, 1), 2, axis=1)
            kps_int_coords = np.where(kps_int_coords_mask_x_y, kps_int_coords, np.zeros(2, np.int32))

            # first = kps_int_coords[:, 0]
            # first = np.where(0 <= first, first, 0)
            # first = np.where(first < w, first, 0)
            # seconds = kps_int_coords[:, 1]
            # seconds = np.where(0 <= seconds, seconds, 0)
            # # TODO: :, 0) => this I think is bad!!!
            # seconds = np.where(seconds < h, seconds, 0)
            # kps_int_coords[:, 0] = first
            # kps_int_coords[:, 1] = seconds

            cluster_mask_bool = (components_indices[kps_int_coords[:, 1], [kps_int_coords[:, 0]]] == component_index)[0]
            cluster_mask_bool = np.logical_and(cluster_mask_bool, kps_int_coords_mask)
            # does it brake?
            #assert np.sum(kps_int_coords_mask) == kps_int_coords_mask.shape

            # cluster_mask_bool = np.array([components_indices[kps_int_coord[1], [kps_int_coord[0]]] == component_index for kps_int_coord in kps_int_coords])
            # cluster_mask_bool = cluster_mask_bool.reshape(-1)

            descs = descs[cluster_mask_bool]
            new_pts = new_pts[cluster_mask_bool]
            orig_pts = orig_pts[cluster_mask_bool]
            ors = ors[cluster_mask_bool]
            scs = scs[cluster_mask_bool]

            kps = [kp for i, kp in enumerate(kps) if cluster_mask_bool[i]]
            if show or save:
                cv.drawKeypoints(rectified, kps, rectified, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            for kpi, kp in enumerate(kps):
                kp.pt = tuple(new_pts[kpi].tolist())


            all_kps.extend(kps)
            all_descs = np.vstack((all_descs, descs))
            all_new_pts = np.vstack((all_new_pts, new_pts))
            all_orig_pts = np.vstack((all_orig_pts, orig_pts))
            all_scs = np.hstack((all_scs, scs))
            all_ors = np.hstack((all_ors, ors))

        # end if

        if show or save:
            plt.figure()
            plt.title("{} - component: {},\n kpts: {} \n normal: {}".format(img_name, component_index, descs.shape[0], normals[normal_index]))
            plt.imshow(rectified)
            if save:
                plt.savefig("{}_rectified_component_{}.jpg.".format(out_prefix, component_index))
            show_or_close(show)

        # if show_components:
        #     rectified_components = components_in_colors.astype(np.float32) / 255
        #     rectified_components = cv.warpPerspective(rectified_components, T, bounding_box)
        #     plt.imshow(rectified_components)
        #     show_or_close(show)


    # TODO corner case - None, [], ...

    kps, descs = descriptor.detectAndCompute(img, None)
    pts, ors, scs = split_keypoints_into_raw(kps)

    # TODO is this the way to round it?
    kps_int_coords = np.round(pts).astype(dtype=np.int32)
    #kps_int_coords = np.int32(pts)

    h, w, _ = img.shape

    kps_int_coords_mask = kps_int_coords[:, 0] >= 0
    kps_int_coords_mask = np.logical_and(kps_int_coords_mask, kps_int_coords[:, 1] >= 0)
    kps_int_coords_mask = np.logical_and(kps_int_coords_mask, kps_int_coords[:, 0] < w)
    kps_int_coords_mask = np.logical_and(kps_int_coords_mask, kps_int_coords[:, 1] < h)
    kps_int_coords_mask_x_y = np.repeat(kps_int_coords_mask.reshape(-1, 1), 2, axis=1)
    kps_int_coords = np.where(kps_int_coords_mask_x_y, kps_int_coords, np.zeros(2, np.int32))

    # in_img_mask = kps_int_coords[:, 0] >= 0
    # in_img_mask = np.logical_and(in_img_mask, kps_int_coords[:, 0] < img.shape[1])
    # in_img_mask = np.logical_and(in_img_mask, kps_int_coords[:, 1] >= 0)
    # in_img_mask = np.logical_and(in_img_mask, kps_int_coords[:, 1] < img.shape[0])
    #
    # kps_ints = kps_ints[in_img_mask]
    # kps = [kp for i, kp in enumerate(kps) if in_img_mask[i]]
    # descs = descs[in_img_mask]

    valid_keys_set = set(valid_components_dict.keys())
    all_indices_set = set(range(np.max(components_indices) + 1))
    non_valid_indices = list(all_indices_set - valid_keys_set)

    non_valid_mask = np.zeros(kps_int_coords.shape[0])
    for non_valid_index in non_valid_indices:
        non_valid_mask = np.logical_or(non_valid_mask, components_indices[kps_int_coords[:, 1], kps_int_coords[:, 0]] == non_valid_index)

    non_valid_mask = np.logical_and(non_valid_mask, kps_int_coords_mask)

    kps = [kp for i, kp in enumerate(kps) if non_valid_mask[i]]

    descs = descs[non_valid_mask]
    pts = pts[non_valid_mask]
    ors = ors[non_valid_mask]
    scs = scs[non_valid_mask]

    all_kps.extend(kps)
    all_descs = np.vstack((all_descs, descs))
    all_new_pts = np.vstack((all_new_pts, pts))
    all_orig_pts = np.vstack((all_orig_pts, pts))
    all_scs = np.hstack((all_scs, scs))
    all_ors = np.hstack((all_ors, ors))

    if show or save:
        no_component_img = img.copy()
        cv.drawKeypoints(no_component_img, kps, no_component_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(10, 10))
        plt.title("{} - no valid component: {} kpts".format(img_name, descs.shape[0]))
        plt.imshow(no_component_img)
        if save:
            plt.savefig("{}_rectified_no_valid_component.jpg.".format(out_prefix))
        show_or_close(show)

        all_img = img.copy()
        cv.drawKeypoints(all_img, all_kps, all_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(10, 10))
        plt.title("All keypoints: {} kpts".format(all_descs.shape[0]))
        plt.imshow(all_img)
        if save:
            plt.savefig("{}_rectified_all.jpg.".format(out_prefix))
        show_or_close(show)
        print("{} keypoints found".format(len(all_kps)))

    Timer.end_check_point("new rectification")
    return all_kps, all_descs, all_orig_pts, all_new_pts, all_ors, all_scs


def get_perspective_transform_new(R, K, K_inv, component_indices, index, normal, scale=1.0):

    coords = np.where(component_indices == index)
    coords = np.array([coords[1], coords[0]])
    coords = add_third_row(coords)

    # I need K here
    P = K @ R @ K_inv
    if scale == 1.0:
        scale = np.sqrt((1.0 - normal[0] ** 2) * (1.0 - normal[1] ** 2))
        scale *= 0.71
    print("new scale: {}".format(scale))
    P[:2, :] *= scale

    new_coords = P @ coords
    new_coords = new_coords / new_coords[2, :]

    min_row = min(new_coords[1])
    max_row = max(new_coords[1])
    min_col = min(new_coords[0])
    max_col = max(new_coords[0])

    dst = np.float32([[min_col, min_row], [min_col, max_row - 1], [max_col - 1, max_row - 1], [max_col - 1, min_row]])
    dst = np.transpose(dst)
    dst = add_third_row(dst)

    translate_vec_new = (-np.min(dst[0]), -np.min(dst[1]))
    translate_matrix_new = np.array([
        [1, 0, translate_vec_new[0]],
        [0, 1, translate_vec_new[1]],
        [0, 0, 1],
    ])

    dst = translate_matrix_new @ dst
    P = translate_matrix_new @ P
    bounding_box_new = (math.ceil(np.max(dst[0])), math.ceil(np.max(dst[1])))

    return P, bounding_box_new


def get_mask_for_components(component_indices, w, h, components, pts):

    int_coords = np.round(pts).astype(dtype=np.int32)

    mask1 = int_coords[:, 0] >= 0
    mask1 = np.logical_and(mask1, int_coords[:, 1] >= 0)
    mask1 = np.logical_and(mask1, int_coords[:, 0] < w)
    mask1 = np.logical_and(mask1, int_coords[:, 1] < h)
    int_coords_mask_x_y = np.repeat(mask1.reshape(-1, 1), 2, axis=1)
    int_coords = np.where(int_coords_mask_x_y, int_coords, np.zeros(2, np.int32))

    mask2 = np.zeros(mask1.size)
    for component in components:
        mask2 = np.logical_or(mask2, component_indices[int_coords[:, 1], int_coords[:, 0]] == component)

    return np.logical_and(mask1, mask2)

