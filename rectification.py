import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from resize import upsample_nearest_numpy
from utils import Timer, identity_map_from_range_of_iter, get_rotation_matrix
from scene_info import SceneInfo, read_cameras
from connected_components import show_components, read_img_normals_info, get_connected_components
from img_utils import show_or_close
from depth_to_normals import compute_normals
from config import Config
#from matching import rich_split_points, find_correspondences, draw_matches, find_and_draw_homography, apply_inliers_on_list


def get_rectification_rotation(normal):

    # now the normals will be "from" me, "inside" the surfaces
    normal = -normal

    z = np.array([0.0, 0.0, 1.0])

    # this handles the case when there is only one dominating plane

    assert normal[2] > 0
    rotation_vector = np.cross(normal, z)
    rotation_vector_norm = sin_theta = np.linalg.norm(rotation_vector)
    unit_rotation_vector = rotation_vector / rotation_vector_norm
    theta = math.asin(sin_theta)

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

        P = K @ R @ K_inv
        if scale != 1.0:
            unscaled = False
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

# FIXME: keypoints not belonging to any component are simply disregarded
def get_rectified_keypoints(normals, components_indices, valid_components_dict, img, K, descriptor, img_name, out_dir=None, show=False):

    K_inv = np.linalg.inv(K)

    all_descs = None
    all_kps = []

    # components_in_colors will be used for other visualizations
    components_in_colors = show_components(components_indices, valid_components_dict, show=False)

    for component_index in valid_components_dict:

        normal_index = valid_components_dict[component_index]
        normal = normals[normal_index]
        threshold_degrees = 80 # [degrees]
        angle_rad = math.acos(np.dot(normal, np.array([0, 0, -1])))
        angle_degrees = angle_rad * 180 / math.pi
        print("angle: {} vs. angle threshold: {}".format(angle_degrees, threshold_degrees))
        if angle_degrees >= threshold_degrees:
            print("WARNING: two sharp of an angle with the -z axis, skipping the rectification")
            continue
        else:
            print("angle ok")

        R = get_rectification_rotation(normals[valid_components_dict[component_index]])

        T, bounding_box = get_perspective_transform(R, K, K_inv, components_indices, component_index)
        #TODO this is too defensive (and wrong) I think, I can warp only the plane
        if bounding_box[0] * bounding_box[1] > 10**8:
            print("warping to an img that is too big, skipping")
            continue

        T_inv = np.linalg.inv(T)

        rectified = cv.warpPerspective(img, T, bounding_box)

        kps, descs = descriptor.detectAndCompute(rectified, None)

        kps_raw = np.float32([kp.pt for kp in kps]).reshape(-1, 1, 2)

        new_kps = cv.perspectiveTransform(kps_raw, T_inv)

        if new_kps is not None:
            kps_int_coords = np.int32(new_kps).reshape(-1, 2)

            h, w, _ = img.shape
            first = kps_int_coords[:, 0]
            first = np.where(0 <= first, first, 0)
            first = np.where(first < w, first, 0)
            seconds = kps_int_coords[:, 1]
            seconds = np.where(0 <= seconds, seconds, 0)
            seconds = np.where(seconds < h, seconds, 0)
            kps_int_coords[:, 0] = first
            kps_int_coords[:, 1] = seconds

            cluster_mask_bool = np.array([components_indices[kps_int_coord[1], [kps_int_coord[0]]] == component_index for kps_int_coord in kps_int_coords])
            cluster_mask_bool = cluster_mask_bool.reshape(-1)

            descs = descs[cluster_mask_bool]

            # TODO new_kps[:, 0, 0/1] still out of bounds (i.e. negative)
            new_kps = new_kps[cluster_mask_bool]

            kps = [kp for i, kp in enumerate(kps) if cluster_mask_bool[i]]

            cv.drawKeypoints(rectified, kps, rectified, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            for kpi, kp in enumerate(kps):
                kp.pt = tuple(new_kps[kpi, 0].tolist())

            all_kps.extend(kps)

            if all_descs is None:
                all_descs = descs
            else:
                all_descs = np.vstack((all_descs, descs))

        plt.title("normal {}".format(normals[normal_index]))
        plt.imshow(rectified)
        show_or_close(show)

        rectified_components = components_in_colors.astype(np.float32) / 255
        rectified_components = cv.warpPerspective(rectified_components, T, bounding_box)
        plt.imshow(rectified_components)
        show_or_close(show)

        if out_dir is not None:
            path = "{}/rectified_{}_{}.jpg".format(out_dir, img_name, component_index)
            plt.savefig(path)

        # img_rectified = cv.polylines(decolorize(img), [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
        # plt.imshow(img_rectified)
        # plt.show(block=False)
    return all_kps, all_descs


def possibly_upsample_normals(img, normal_indices):

    if img.shape[0] != normal_indices.shape[0]:
        # needs upsampling
        epsilon = 0.0001
        if img.shape[0] < normal_indices.shape[0]:
            raise Exception("img.shape[0] < normal_indices.shape[0] not expected")
        elif abs(img.shape[0] / normal_indices.shape[0] - img.shape[1] / normal_indices.shape[1]) >= epsilon:
            raise Exception("{} and {} not of the same aspect ratio".format(normal_indices.shape, img.shape))
        else:
            print("Will upsample the normals")
            normal_indices = upsample_nearest_numpy(normal_indices, img.shape[0], img.shape[1])

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
            show_components(normal_indices, identity_map_from_range_of_iter(normals), normals)

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
            show_components(normal_indices, identity_map_from_range_of_iter(normals), normals)

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

    # interesting_files = interests
    #
    # scene_info = SceneInfo.read_scene("scene1", lazy=True)
    #
    # play_iterate(scene_info, "original_dataset/scene1/images", limit=20, interesting_files=interesting_files)

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

    # Timer.start()
    #
    # #interesting_dirs = ["frame_0000000145_2"]
    # interesting_dirs = ["frame_0000000015_4"]
    #
    # scene_info = SceneInfo.read_scene("scene1", lazy=True)
    #
    # show_rectifications(scene_info, "work/scene1/normals/svd", "original_dataset/scene1/images", limit=1, interesting_dirs=None)
    #
    # Timer.end()

