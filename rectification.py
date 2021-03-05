import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import time
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
    #coords = np.vstack((coords, np.ones(coords.shape[1])))
    coords = add_third_row(coords)

    min_row_or_new = min(coords[1])
    max_row_or_new = max(coords[1])
    min_col_or_new = min(coords[0])
    max_col_or_new = max(coords[0])
    src_new_bb = np.float32([[min_col_or_new, min_row_or_new], [min_col_or_new, max_row_or_new - 1], [max_col_or_new - 1, max_row_or_new - 1], [max_col_or_new - 1, min_row_or_new]])

    scale = 0.1
    scale_matrix = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ])

    T_new = K @ R @ K_inv @ scale_matrix
    T_old = T_new
    print("second T: {}".format(T_new))

    new_coords = T_new @ coords
    new_coords /= new_coords[2, :]

    min_row_new = min(new_coords[1])
    max_row_new = max(new_coords[1])
    min_col_new = min(new_coords[0])
    max_col_new = max(new_coords[0])

    dst_new = np.float32([[min_col_new, min_row_new], [min_col_new, max_row_new - 1], [max_col_new - 1, max_row_new - 1], [max_col_new - 1, min_row_new]])
    dst_new = np.transpose(dst_new)
    dst_new = add_third_row(dst_new)

    src_old = get_bounding_box(component_indices, index)
    # src_try = np.transpose(np.squeeze(src_old, axis=1))
    # src_try = add_third_row(src_try)

    dst_old = cv.perspectiveTransform(src_old, T_old)
    #dst_try = T_old @ src_try

    mins_old = (np.min(dst_old[:, 0, 0]), np.min(dst_old[:, 0, 1]))
    t0_old = -mins_old[0]
    t1_old = -mins_old[1]
    translate_matrix_old = np.array([
        [1, 0, t0_old],
        [0, 1, t1_old],
        [0, 0, 1],
    ])
    print("Translating old by:\n{}".format(translate_matrix_old))
    T_old = translate_matrix_old @ T_old
    dst_old = cv.perspectiveTransform(src_old, T_old)
    bounding_box_old = (np.max(dst_old[:, 0, 0]), np.max(dst_old[:, 0, 1]))
    print("bounding_box_old: {}".format(bounding_box_old))


    translate_vec_new = (-np.min(dst_new[0]), -np.min(dst_new[1]))
    translate_matrix_new = np.array([
        [1, 0, translate_vec_new[0]],
        [0, 1, translate_vec_new[1]],
        [0, 0, 1],
    ])
    #translate_matrix_new = (translate_matrix_new + translate_matrix_old) / 2

    print("Translating new by:\n{}".format(translate_matrix_new))

    dst_new = translate_matrix_new @ dst_new
    T_new = translate_matrix_new @ T_new
    bounding_box_new = (math.ceil(np.max(dst_new[0])), math.ceil(np.max(dst_new[1])))
    print("bounding_box_new: {}".format(bounding_box_new))

    return T_new, bounding_box_new


def get_bounding_box(normal_indices, index):

    rows, col = np.where(normal_indices == index)
    min_row = min(rows)
    max_row = max(rows)
    min_col = min(col)
    max_col = max(col)
    src = np.float32([[min_col, min_row], [min_col, max_row - 1], [max_col - 1, max_row - 1], [max_col - 1, min_row]]).reshape(-1, 1, 2)
    return src


def get_rectified_keypoints_all(normals, normal_indices, img, K, descriptor, img_file, out_dir=None):
    # NOTE this should handle the invalid normal index (=3) well
    get_rectified_keypoints(normals, normal_indices, list(range(len(normals))), img, K, descriptor, img_file, out_dir)


def get_rectified_keypoints(normals, normal_indices, valid_indices, img, K, descriptor, img_file, out_dir=None):

    #TODO remove valid_indices

    K_inv = np.linalg.inv(K)

    Rs = get_rectification_rotations(normals)

    components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals)), True)
    components_in_colors = show_components(components_indices, valid_components_dict.keys())

    all_descs = None
    all_kps = []

    for component_index in valid_components_dict:

        normal_index = valid_components_dict[component_index]
        R = Rs[normal_index]

        T, bounding_box = get_perspective_transform(R, K, K_inv, components_indices, component_index)

        # src = get_bounding_box(components_indices, component_index)
        #
        # scale = 1.0
        # scale_matrix = np.array([
        #     [scale, 0, 0],
        #     [0, scale, 0],
        #     [0, 0, 1],
        # ])
        #
        # T = K @ R @ K_inv @ scale_matrix
        # print("first T: {}".format(T))
        # dst = cv.perspectiveTransform(src, T)
        # mins = (np.min(dst[:, 0, 0]), np.min(dst[:, 0, 1]))
        # t0 = -mins[0]
        # t1 = -mins[1]
        # translate = np.array([
        #     [1, 0, t0],
        #     [0, 1, t1],
        #     [0, 0, 1],
        # ])
        # print("Translating by:\n{}".format(translate))
        # dst2 = cv.perspectiveTransform(src, T)
        # dst3 = cv.perspectiveTransform(dst2, translate)
        # T = translate @ T
        # dst = cv.perspectiveTransform(src, T)
        #
        # bounding_box = (np.max(dst[:, 0, 0]), np.max(dst[:, 0, 1]))
        #TODO this is too defensive (and wrong) I think, I can warp only the plane
        if bounding_box[0] * bounding_box[1] > 10**8:
            print("warping to an img that is too big, skipping")
            continue

        T_inv = np.linalg.inv(T)

        print("rotation: \n {}".format(R))
        print("transformation: \n {}".format(T))
        #print("transformation2: \n {}".format(T2))

        # print("src: \n {}".format(src))
        # print("dst: \n {}".format(dst))

        print("bounding box: {}".format(bounding_box))
        #print("bounding box2: {}".format(bounding_box2))
        rectified = cv.warpPerspective(img, T, bounding_box)

        comp_to_rect = components_in_colors.astype(np.float32) / 255
        rectified_comp = cv.warpPerspective(comp_to_rect, T, bounding_box)

        #rectified_indices = cv.warpPerspective(normal_indices, T, bounding_box)

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

            new_kps = new_kps[cluster_mask_bool]

            kps = [kp for i, kp in enumerate(kps) if cluster_mask_bool[i]]

            cv.drawKeypoints(rectified, kps, rectified, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            for kpi, kp in enumerate(kps):
                kp.pt = tuple(new_kps[kpi, 0].tolist())

            print("adding {} keypoints".format(len(kps)))

            all_kps.extend(kps)

            if all_descs is None:
                all_descs = descs
            else:
                all_descs = np.vstack((all_descs, descs))

        # plt.figure()
        # plt.title("normal {}".format(normals[normal_index]))
        # plt.imshow(rectified_indices * 50)
        # plt.show()

        plt.figure()
        #plt.figure(dpi=600)
        plt.title("normal {}".format(normals[normal_index]))
        plt.imshow(rectified)
        plt.show()
        plt.imshow(rectified_comp)
        plt.show()
        if out_dir is not None:
            plt.savefig("{}/rectified_{}_{}.jpg".format(out_dir, img_file, component_index))

        # img_rectified = cv.polylines(decolorize(img), [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
        # plt.imshow(img_rectified)
        # plt.show()
    return all_kps, all_descs


def show_rectifications(scene_info: SceneInfo, parent_dir, original_input_dir, limit, interesting_dirs=None):

    if interesting_dirs is not None:
        dirs = interesting_dirs
    else:
        dirs = [dirname for dirname in sorted(os.listdir(parent_dir)) if os.path.isdir("{}/{}".format(parent_dir, dirname))]
        dirs = sorted(dirs)
        if limit is not None:
            dirs = dirs[0:limit]

    for img_name in dirs:

        K = scene_info.get_img_K(img_name)

        img_file_path = '{}/{}.jpg'.format(original_input_dir, img_name)
        img = cv.imread(img_file_path, None)
        normals, normal_indices = read_img_normals_info(parent_dir, img_name)

        show = True
        if show:
            show_components(normal_indices, range(len(normals)))

        # normals = np.array(
        #     [[ 0.33717412, -0.30356583, -0.89115733],
        #      [-0.68118596, -0.23305716, -0.6940245 ]]
        # )
        normals = np.array(
            [
             [ 0.33717412, -0.30356583, -0.89115733],
             [-0.80, -0.1, -0.60]],
        )

        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            normals[i] /= norm
            print("normalized: {}".format(normals[i]))

        get_rectified_keypoints_all(normals, normal_indices, img, K, cv.SIFT_create(), img_name)


if __name__ == "__main__":

    Timer.start()

    interesting_dirs = ["frame_0000000145_2"]

    scene_info = SceneInfo.read_scene("scene1")
    Timer.check_point("scene info read")

    show_rectifications(scene_info, "work/scene1/normals/simple_diff_mask_sigma_5", "original_dataset/scene1/images", limit=1, interesting_dirs=interesting_dirs)

    Timer.check_point("All done")
