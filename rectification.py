import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from resize import upsample_nearest_numpy
from utils import Timer
from scene_info import SceneInfo, read_cameras
from connected_components import show_components, read_img_normals_info, get_connected_components


def get_rotation_matrix(unit_rotation_vector, theta):

    # Rodrigues formula
    # R = I + sin(theta) . K + (1 - cos(theta)).K**2

    K = np.array([
        [0.0, -unit_rotation_vector[2], unit_rotation_vector[1]],
        [unit_rotation_vector[2], 0.0, -unit_rotation_vector[0]],
        [-unit_rotation_vector[1], unit_rotation_vector[0], 0.0],
    ])
    a = np.eye(3)
    b = math.sin(theta) * K
    c = (1.0 - math.cos(theta)) * K @ K
    return a + b + c


# refactor: just one
def get_rectification_rotations(normals):

    # now the normals will be "from" me, "inside" the surfaces
    normals = -normals

    z = np.array([0.0, 0.0, 1.0])
    Rs = []

    # this handles the case when there is only one dominating plane
    if len(normals.shape) == 1:
        normals = normals.reshape(1, -1)
    for _, normal in enumerate(normals):
        assert normal[2] > 0
        rotation_vector = np.cross(normal, z)
        rotation_vector_norm = sin_theta = np.linalg.norm(rotation_vector)
        unit_rotation_vector = rotation_vector / rotation_vector_norm
        theta = math.asin(sin_theta)

        R = get_rotation_matrix(unit_rotation_vector, theta)
        det = np.linalg.det(R)
        assert math.fabs(det - 1.0) < 0.0001
        Rs.append(R)

    return Rs


def add_third_row(column_vecs):
    return np.vstack((column_vecs, np.ones(column_vecs.shape[1])))


def get_perspective_transform(R, K, K_inv, component_indices, index):

    coords = np.where(component_indices == index)
    coords = np.array([coords[1], coords[0]])
    coords = add_third_row(coords)

    P = K @ R @ K_inv

    new_coords = P @ coords
    new_coords /= new_coords[2, :]

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


def get_rectified_keypoints(normals, components_indices, valid_components_dict, img, K, descriptor, img_name, out_dir=None):

    K_inv = np.linalg.inv(K)
    Rs = get_rectification_rotations(normals)

    all_descs = None
    all_kps = []

    # TODO show
    components_in_colors = show_components(components_indices, valid_components_dict.keys())

    for component_index in valid_components_dict:

        normal_index = valid_components_dict[component_index]
        R = Rs[normal_index]

        T, bounding_box = get_perspective_transform(R, K, K_inv, components_indices, component_index)
        #TODO this is too defensive (and wrong) I think, I can warp only the plane
        if bounding_box[0] * bounding_box[1] > 10**8:
            print("warping to an img that is too big, skipping")
            continue

        T_inv = np.linalg.inv(T)

        rectified = cv.warpPerspective(img, T, bounding_box)

        rectified_components = components_in_colors.astype(np.float32) / 255
        rectified_components = cv.warpPerspective(rectified_components, T, bounding_box)

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

        plt.figure()
        #plt.figure(dpi=600)
        plt.title("normal {}".format(normals[normal_index]))
        plt.imshow(rectified)
        plt.show()
        plt.imshow(rectified_components)
        plt.show()
        if out_dir is not None:
            plt.savefig("{}/rectified_{}_{}.jpg".format(out_dir, img_name, component_index))

        # img_rectified = cv.polylines(decolorize(img), [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
        # plt.imshow(img_rectified)
        # plt.show()
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
            show_components(normal_indices, range(len(normals)))

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

        get_rectified_keypoints(normals, components_indices, valid_components_dict, img, K, descriptor= cv.SIFT_create(), img_name=img_name)


if __name__ == "__main__":

    Timer.start()

    interesting_dirs = ["frame_0000000145_2"]

    scene_info = SceneInfo.read_scene("scene1")

    show_rectifications(scene_info, "work/scene1/normals/simple_diff_mask", "original_dataset/scene1/images", limit=1, interesting_dirs=None)

    Timer.end()
