import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import time
from scene_info import read_cameras

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


# TODO the idea of this method was to define a bounding box and only this box could be transformed, however, warpPerspectife (AFAIK)
# always transforms the whole image... so there is not easy way to do that
def get_bounding_box(normals, normal_indices, index, img_remove):

    # rows, col = np.where(normal_indices == index)
    # min_row = min(rows)
    # max_row = max(rows)
    # min_col = min(col)
    # max_col = max(col)
    # src = np.float32([[min_row, min_col], [min_row, max_col - 1], [max_row - 1, max_col - 1], [max_row - 1, min_col]]).reshape(-1, 1, 2)

    h, w, _ = img_remove.shape
    src = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    return src


def get_rectified_keypoints(normals, normal_indices, img, K, K_inv, descriptor, show=False, out_dir=None):

    Rs = get_rectification_rotations(normals)

    all_descs = None
    all_kps = []

    for normal_index, R in enumerate(Rs):

        src = get_bounding_box(normals, normal_indices, normal_index, img)


        T = K @ R @ K_inv
        dst = cv.perspectiveTransform(src, T)
        mins = (np.min(dst[:, 0, 0]), np.min(dst[:, 0, 1]))
        if mins[0] < 0 or mins[1] < 0:
            translate = np.array([
                [1, 0, -mins[0]],
                [0, 1, -mins[1]],
                [0, 0, 1],
            ])
            T = translate @ T
            dst = cv.perspectiveTransform(src, T)

        bounding_box = (np.max(dst[:, 0, 0]), np.max(dst[:, 0, 1]))
        # TODO this is too defensive (and wrong) I think, I can warp only the plane
        if bounding_box[0] * bounding_box[1] > 10**8:
            print("warping to an img that is too big, skipping")
            continue

        T_inv = np.linalg.inv(T)

        print("rotation: \n {}".format(R))
        print("transformation: \n {}".format(T))
        print("src: \n {}".format(src))
        print("dst: \n {}".format(dst))

        recified = cv.warpPerspective(img, T, bounding_box)

        kps, descs = descriptor.detectAndCompute(recified, None)

        kps_raw = np.float32([kp.pt for kp in kps]).reshape(-1, 1, 2)

        new_kps = cv.perspectiveTransform(kps_raw, T_inv)

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

        cluster_mask_bool = np.array([normal_indices[kps_int_coord[1], [kps_int_coord[0]], 0] == normal_index for kps_int_coord in kps_int_coords]).reshape(-1)

        descs = descs[cluster_mask_bool]
        new_kps = new_kps[cluster_mask_bool]
        kps = [kp for i, kp in enumerate(kps) if cluster_mask_bool[i]]

        cv.drawKeypoints(recified, kps, recified, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        for kpi, kp in enumerate(kps):
            kp.pt = tuple(new_kps[kpi, 0].tolist())

        print("adding {} keypoints".format(len(kps)))

        all_kps.extend(kps)

        if all_descs is None:
            all_descs = descs
        else:
            all_descs = np.vstack((all_descs, descs))

        plt.figure()
        #plt.figure(dpi=300)
        plt.title("normal {}".format(normals[normal_index]))
        plt.imshow(recified)
        #plt.show()
        if out_dir is not None:
            plt.savefig("{}/rectified_{}.jpg".format(out_dir, normal_index))

        # img_rectified = cv.polylines(decolorize(img), [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
        # plt.imshow(img_rectified)
        # plt.show()
    return all_kps, all_descs


# original_input_dir - to scene info
def read_img_normals_info(parent_dir, img_name_dir):

    if not os.path.isdir("{}/{}".format(parent_dir, img_name_dir)):
        return None, None

    paths_png = glob.glob("{}/{}/*.png".format(parent_dir, img_name_dir))
    paths_txt = glob.glob("{}/{}/*.txt".format(parent_dir, img_name_dir))

    if paths_png is None or paths_txt is None:
        print(".txt or .png file doesn't exist in {}!".format(img_name_dir))
        raise

    normals = np.loadtxt(paths_txt[0], delimiter=',')
    normal_indices = cv.imread(paths_png[0], None)
    return normals, normal_indices


def show_rectifications(parent_dir, original_input_dir, limit):

    # TODO
    cameras = read_cameras("scene1")
    K = cameras[1801].get_K()
    K_inv = np.linalg.inv(K)

    # /Users/vaclav/ownCloud/SVP/project/work/scene1/normals/simple_diff_mask_sigma_5
    dirs = [dirname for dirname in sorted(os.listdir(parent_dir)) if os.path.isdir("{}/{}".format(parent_dir, dirname))]
    dirs = sorted(dirs)
    if limit is not None:
        dirs = dirs[0:limit]

    for img_name_dir in dirs:

        img_file_path = '{}/{}.jpg'.format(original_input_dir, img_name_dir)
        img = cv.imread(img_file_path, None)
        normals, normal_indices = read_img_normals_info(parent_dir, img_name_dir)
        get_rectified_keypoints(normals, normal_indices, img, K, K_inv, cv.SIFT_create(), True)


if __name__ == "__main__":

    start = time.time()

    show_rectifications("work/scene1/normals/simple_diff_mask_sigma_5", "original_dataset/scene1/images", limit=1)

    print("All done")
    end = time.time()
    print("Time elapsed: {}".format(end - start))
