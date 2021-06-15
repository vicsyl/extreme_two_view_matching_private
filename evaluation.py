import numpy as np

import argparse
from scene_info import *
from utils import quaternions_to_R
import cv2 as cv
import math
import time
import os
import matplotlib as plt
import glob
import pickle
from typing import List

from pathlib import Path

"""
DISCLAIMER: the following methods have been adopted from https://github.com/ducha-aiki/ransac-tutorial-2020-data:
- normalize_keypoints
- quaternion_from_matrix
- evaluate_R_t
"""


def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''

    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

    return keypoints


def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()
        raise

    return err_q, err_t


def split_points(tentative_matches, kps1, kps2):
    src_pts = np.float32([kps1[m.queryIdx].pt for m in tentative_matches]).reshape(-1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentative_matches]).reshape(-1, 2)
    return src_pts, dst_pts


def eval_essential_matrix(p1n, p2n, E, dR, dt):
    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return np.pi, np.pi / 2

    if E.size > 0:
        _, R, t, _ = cv.recoverPose(E, p1n, p2n)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            err_q = np.pi
            err_t = np.pi / 2

    else:
        err_q = np.pi
        err_t = np.pi / 2

    return err_q, err_t


def compare_poses(E, img_pair: ImagePairEntry, scene_info: SceneInfo, pts1, pts2):

    img_entry_1: ImageEntry = scene_info.img_info_map[img_pair.img1]
    T1 = img_entry_1.t
    R1 = img_entry_1.R

    img_entry_2: ImageEntry = scene_info.img_info_map[img_pair.img2]
    T2 = img_entry_2.t
    R2 = img_entry_2.R

    dR = R2 @ R1.T
    dT = T2 - dR @ T1

    K1 = scene_info.get_img_K(img_pair.img1)
    K2 = scene_info.get_img_K(img_pair.img2)

    # TODO Q: what is actually this? if I remove it, I can remove the call to scene_info.get_img_K
    p1n = normalize_keypoints(pts1, K1).astype(np.float64)
    p2n = normalize_keypoints(pts2, K2).astype(np.float64)
    # Q: this doesn't change the result!!!
    # p1n = pts1
    # p2n = pts2

    errors = eval_essential_matrix(p1n, p2n, E, dR, dT)
    errors_max = max(errors)

    print("errors: {}".format(errors))
    print("max error: {}".format(errors_max))

    return errors


# a HACK that enables pickling of cv2.KeyPoint - see
# https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror/48832618
import copyreg
import cv2


def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)


@dataclass
class ImageData:
    img: np.ndarray
    key_points: List[cv.KeyPoint]
    descriptions: object
    K: np.ndarray
    normals: np.ndarray
    components_indices: np.ndarray
    valid_components_dict: dict

    @staticmethod
    def from_serialized_data(img, K, img_serialized_data):
        return ImageData(img=img,
                         K=K,
                         key_points=img_serialized_data.kpts,
                         descriptions=img_serialized_data.descs,
                         normals=img_serialized_data.normals,
                         components_indices=img_serialized_data.components_indices,
                         valid_components_dict=img_serialized_data.valid_components_dict)

    def to_serialized_data(self):
        return ImageSerializedData(kpts=self.key_points,
                                   descs=self.descriptions,
                                   normals=self.normals,
                                   components_indices=self.components_indices,
                                   valid_components_dict=self.valid_components_dict)


@dataclass
class ImageSerializedData:
    kpts: list
    descs: list
    normals: np.ndarray
    components_indices: np.ndarray
    valid_components_dict: dict


@dataclass
class Stats:
    error_R: float
    error_T: float
    tentative_matches: int
    inliers: int
    all_features_1: int
    all_features_2: int
    E: np.ndarray
    normals1: np.ndarray
    normals2: np.ndarray

    # legacy
    def make_brief(self):
        self.src_pts_inliers = None
        self.dst_pts_inliers = None
        self.src_tentatives = None
        self.dst_tentatives = None
        self.kpts1 = None
        self.kpts2 = None


def evaluate_matching(scene_info,
                      E,
                      kps1,
                      kps2,
                      tentative_matches,
                      inlier_mask,
                      img_pair,
                      out_dir,
                      stats_map,
                      normals1,
                      normals2):

    save_suffix = "{}_{}".format(img_pair.img1, img_pair.img2)

    print("Image pair: {} <-> {}:".format(img_pair.img1, img_pair.img2))
    print("Number of inliers: {}".format(inlier_mask[inlier_mask == [1]].shape[0]))
    print("Number of outliers: {}".format(inlier_mask[inlier_mask == [0]].shape[0]))

    src_tentative, dst_tentative = split_points(tentative_matches, kps1, kps2)
    src_pts_inliers = src_tentative[inlier_mask[:, 0] == [1]]
    dst_pts_inliers = dst_tentative[inlier_mask[:, 0] == [1]]

    error_R, error_T = compare_poses(E, img_pair, scene_info, src_pts_inliers, dst_pts_inliers)
    inliers = np.sum(np.where(inlier_mask[:, 0] == [1], 1, 0))

    stats = Stats(error_R=error_R,
                  error_T=error_T,
                  tentative_matches=len(tentative_matches),
                  inliers=inliers,
                  all_features_1=len(kps1),
                  all_features_2=len(kps2),
                  E=E,
                  normals1=normals1,
                  normals2=normals2,
                  )

    key = "{}_{}".format(img_pair.img1, img_pair.img2)
    stats_map[key] = stats
    return stats


# def evaluate_all(scene_info: SceneInfo, input_dir, limit=None):
#
#     dirs = [dirname for dirname in sorted(os.listdir(input_dir)) if os.path.isdir("{}/{}".format(input_dir, dirname))]
#     dirs = sorted(dirs)
#     if limit is not None:
#         dirs = dirs[0:limit]
#
#     flattened_img_pairs = [pair for diff in scene_info.img_pairs_lists for pair in diff]
#     img_pair_map = {"{}_{}".format(img_pair.img1, img_pair.img2): img_pair for img_pair in flattened_img_pairs}
#
#     result_map = {}
#
#     for dir in dirs:
#
#         if not img_pair_map.__contains__(dir):
#             print("dir '{}' not recognized!!!".format(dir))
#             continue
#
#         whole_path = "{}/{}".format(input_dir, dir)
#
#         img_pair = img_pair_map[dir]
#         E = np.loadtxt("{}/essential_matrix.txt".format(whole_path), delimiter=',')
#         dst_pts = np.loadtxt("{}/dst_pts.txt".format(whole_path), delimiter=',')
#         src_pts = np.loadtxt("{}/src_pts.txt".format(whole_path), delimiter=',')
#         stats = np.loadtxt("{}/stats.txt".format(whole_path), delimiter=',')
#         tentative_matches = stats[0]
#         inliers = stats[1]
#         all_features_1 = stats[2]
#         all_features_2 = stats[3]
#
#         print("Evaluating: {}".format(dir))
#         errors = compare_poses(E, img_pair, scene_info, src_pts, dst_pts)
#         result_map[dir] = Stats(errors[0], errors[1], tentative_matches, inliers, all_features_1, all_features_2)
#
#     return result_map


def read_last():
    prefix = "pipeline_scene1"
    gl = glob.glob("work/{}*".format(prefix))
    gl.sort()
    last_file = "{}/{}".format(gl[-1], "all.stats.pkl")

    with open(last_file, "rb") as f:
        print("reading: {}".format(last_file))
        stats_map_read = pickle.load(f)

    return stats_map_read


def get_kps_gt_id(kps_matches_np, image_entry: ImageEntry, diff_threshold=2.0):

    # kps_matches_points = [list(kps[kps_index].pt) for kps_index in kps_indices]
    # kps_matches_np = np.array(kps_matches_points)

    image_data = image_entry.data
    data_ids = image_entry.data_point_idxs

    diff = np.ndarray(image_data.shape)
    mins = np.ndarray(kps_matches_np.shape[0])
    data_point_ids = -2 * np.ones(kps_matches_np.shape[0], dtype=np.int32)
    for p_idx, match_point in enumerate(kps_matches_np):
        diff[:, 0] = image_data[:, 0] - match_point[0]
        diff[:, 1] = image_data[:, 1] - match_point[1]
        diff_norm = np.linalg.norm(diff, axis=1)
        min_index = np.argmin(diff_norm)
        min_diff = diff_norm[min_index]
        mins[p_idx] = min_diff
        if min_diff < diff_threshold:
            data_point_ids[p_idx] = data_ids[min_index]
        # else:
        #     print()

    return data_point_ids, mins


# NOTE NOT USED, and probably won't be
def correctly_matched_point_for_image_pair(kps_inliers1, kps_inliers2, images_info, img_pair):

    data_point1_ids, mins1 = get_kps_gt_id(kps_inliers1, images_info[img_pair.img1], diff_threshold=2.0)
    data_point2_ids, mins2 = get_kps_gt_id(kps_inliers2, images_info[img_pair.img2], diff_threshold=2.0)

    # FIXME this is wrong
    data_point_ids_matches = data_point1_ids[data_point1_ids == data_point2_ids]
    unique = np.unique(data_point_ids_matches)
    unique = unique[unique != -1]
    unique = unique[unique != -2]
    return unique


def print_stats(stat_name: str, stat_in_list: list):
    np_ar = np.array(stat_in_list)
    print("average {}: {}".format(stat_name, np.sum(np_ar) / len(stat_in_list)))


def vector_product_matrix(vec: np.ndarray):
    return np.array([
        [    0.0, -vec[2],  vec[1]],
        [ vec[2],     0.0, -vec[0]],
        [-vec[1],  vec[0],       0],
    ])


def evaluate_tentatives_agains_ground_truth(scene_info: SceneInfo, img_pair: ImagePairEntry, src_tentatives_2d, dst_tentatives_2d, thresholds):

    # input: img pair -> imgs -> T1/2, R1/2 -> ground truth F
    # input: tentatives (src, dst)

    img_entry_1: ImageEntry = scene_info.img_info_map[img_pair.img1]
    T1 = np.array(img_entry_1.t)
    R1 = img_entry_1.R
    K1 = scene_info.get_img_K(img_pair.img1)
    # TODO we use the real K here, right?
    K1_inv = np.linalg.inv(K1)
    src_tentative = np.ndarray((src_tentatives_2d.shape[0], 3))
    src_tentative[:, :2] = src_tentatives_2d
    src_tentative[:, 2] = 1.0

    img_entry_2: ImageEntry = scene_info.img_info_map[img_pair.img2]
    T2 = np.array(img_entry_2.t)
    R2 = img_entry_2.R
    K2 = scene_info.get_img_K(img_pair.img2)
    K2_inv = np.linalg.inv(K2)
    dst_tentative = np.ndarray((dst_tentatives_2d.shape[0], 3))
    dst_tentative[:, :2] = dst_tentatives_2d
    dst_tentative[:, 2] = 1.0

    F_ground_truth = K2_inv.T @ R2 @ vector_product_matrix(T2 - T1) @ R1.T @ K1_inv
    F_x1 = F_ground_truth @ src_tentative.T
    x2_F_x1 = dst_tentative[:, 0] * F_x1[0] + dst_tentative[:, 1] * F_x1[1] + dst_tentative[:, 2] * F_x1[2]

    checks = np.zeros(3)
    checks[0] = np.sum(np.abs(x2_F_x1) < thresholds[0])
    checks[1] = np.sum(np.abs(x2_F_x1) < thresholds[1])
    checks[2] = np.sum(np.abs(x2_F_x1) < thresholds[2])

    #hist = np.histogram(np.abs(x2_F_x1), bins=100)

    return checks


def evaluate_all(stats_map_all: dict, n_worst_examples=None):
    print("Stats for all difficulties:")
    angle_thresholds = [5, 10]
    for angle_threshold in angle_thresholds:
        print("Group\tAcc.({}º)".format(angle_threshold))
        for diff, stats_map in stats_map_all.items():
            evaluate_percentage_correct(stats_map, diff, n_worst_examples=n_worst_examples, th_degrees=angle_threshold)


# def evaluate(stats_map: dict, scene_info: SceneInfo):
#
#     l_entries = []
#     n_entries = 0
#
#     error_R = []
#     error_T = []
#     tentative_matches = []
#     inliers = []
#     all_features_1 = []
#     all_features_2 = []
#     #matched_points = []
#     all_checks = []
#     x2_F_x1_thresholds = np.array([0.05, 0.01, 0.005])
#
#     for img_pair_str, stats in stats_map.items():
#
#         n_entries += 1
#         img_pair, diff = scene_info.find_img_pair_from_key(img_pair_str)
#
#         l_entries.append(str(img_pair))
#
#         error_R.append(stats.error_R)
#         error_T.append(stats.error_T)
#         tentative_matches.append(stats.tentative_matches)
#         inliers.append(stats.inliers)
#         all_features_1.append(stats.all_features_1)
#         all_features_2.append(stats.all_features_2)
#
#         checks = evaluate_tentatives_agains_ground_truth(scene_info, img_pair, stats.src_tentatives, stats.dst_tentatives, x2_F_x1_thresholds)
#         all_checks.append(checks)
#
#         # matched_points_local = correctly_matched_point_for_image_pair(stats.src_pts_inliers,
#         #                                                 stats.dst_pts_inliers,
#         #                                                 scene_info.img_info_map,
#         #                                                 img_pair)
#         # matched_points.append(matched_points_local.shape[0])
#
#     print("Image entries (img name, difficulty)")
#     print(",\n".join(l_entries))
#
#     print("Stats report")
#     print_stats("error_R", error_R)
#     print_stats("error_T", error_T)
#     print_stats("tentative_matches", tentative_matches)
#     print_stats("inliers", inliers)
#     print_stats("all_features_1", all_features_1)
#     print_stats("all_features_2", all_features_2)
#
#     all_checks = np.array(all_checks)
#     all_checks_avgs = np.sum(all_checks, axis=0) / all_checks.shape[0]
#     print("F ground truth checks averages (for thresholds: {}): {}".format(x2_F_x1_thresholds, all_checks_avgs))


# def evaluate_last(scene_name):
#
#     scene_info = SceneInfo.read_scene(scene_name)
#     stats_map = read_last()
#     evaluate(stats_map, scene_info)
#
#

def evaluate_percentage_correct(stats_map, difficulty, n_worst_examples=None, th_degrees=5):
    sorted_by_err_R = list(sorted(stats_map.items(), key=lambda key_value: -key_value[1].error_R))

    if n_worst_examples is not None:
        print("{} worst examples for diff={}".format(n_worst_examples, difficulty))
        for k, v in sorted_by_err_R[:n_worst_examples]:
            print("{}: {}".format(k, v.error_R))
        print("{} best examples for diff={}".format(n_worst_examples, difficulty))
        for k, v in sorted_by_err_R[-n_worst_examples:]:
            print("{}: {}".format(k, v.error_R))

    rad_th = th_degrees * math.pi / 180
    filtered = list(filter(lambda key_value: key_value[1].error_R < rad_th, stats_map.items()))
    filtered_len = len(filtered)
    all_len = len(stats_map.items())
    perc = filtered_len/all_len
    print("{}\t{:.03f}".format(difficulty, perc))
    return difficulty, perc


def evaluate_percentage_correct_from_file(file_name, difficulty, n_worst_examples=None, th_degrees=5):

    with open(file_name, "rb") as f:
        #print("reading: {}".format(file_name))
        stats_map = pickle.load(f)

    return evaluate_percentage_correct(stats_map, difficulty, n_worst_examples=n_worst_examples, th_degrees=th_degrees)


def make_light(file_name):

    with open(file_name, "rb") as f:
        print("reading: {}".format(file_name))
        stats_map = pickle.load(f)

    for key_value in stats_map.items():
        key_value[1].make_brief()

    with open("{}_light".format(file_name), "wb") as f:
        pickle.dump(stats_map, f)


def evaluate_file(scene_name, file_name):

    scene_info = SceneInfo.read_scene(scene_name)
    with open(file_name, "rb") as f:
        print("reading: {}".format(file_name))
        stats_map = pickle.load(f)

    degrees_th = 5
    rad_th = degrees_th * math.pi / 180
    filtered = list(filter(lambda key_value: key_value[1].error_R < rad_th, stats_map.items()))
    filtered_len = len(filtered)
    all_len = len(stats_map.items())
    print("percentage of correct {}/{} = {}".format(filtered_len, all_len, filtered_len/all_len))

    #items_in_list = list(stats_map.items())
    sorted_by_err_R = sorted(stats_map.items(), key=lambda key_value: key_value[1].error_R)
    sorted_by_diff_err_R = sorted(sorted_by_err_R, key=lambda key_value: scene_info.find_img_pair_from_key(key=key_value[0])[1])

    for (key, value) in sorted_by_diff_err_R:
        print("{} : {} : {}".format(key, value.error_R, scene_info.find_img_pair_from_key(key=key)[1]))

    vals = list(map(lambda v: v[1].error_R, sorted_by_err_R))
    vals1 = np.array(vals)
    hist1 = np.histogram(vals1)
    avg1 = np.average(vals1)

    vals2 = np.array(list(filter(lambda v: v != np.pi, vals)))
    hist2 = np.histogram(vals1)
    avg2 = np.average(vals2)

    print()
    #evaluate(stats_map_read, scene_info)


# def main():
#     start = time.time()
#
#     scene = "scene1"
#     scene_info = SceneInfo.read_scene(scene)
#     with_map = evaluate_all(scene_info, "work/{}/matching/with_rectification".format(scene), limit=None)
#     without_map = evaluate_all(scene_info, "work/{}/matching/without_rectification".format(scene), limit=None)
#
#     with_r_err = 0
#     with_t_err = 0
#     without_t_err = 0
#     without_r_err = 0
#
#     diff_r = []
#     diff_t = []
#
#     with_inlier_ratio = 0.0
#     without_inlier_ratio = 0.0
#
#     with_tentative_matches = 0
#     without_tentative_matches = 0
#
#     common_enties = 0
#
#     for key in with_map:
#         if not without_map.__contains__(key):
#             continue
#         common_enties += 1
#         with_r_err += with_map[key].error_R
#         with_t_err += with_map[key].error_T
#         without_r_err += without_map[key].error_R
#         without_t_err += without_map[key].error_T
#         diff_r.append((with_map[key].error_R - without_map[key].error_R, key))
#         diff_t.append((with_map[key].error_T - without_map[key].error_T, key))
#
#         with_tentative_matches_p = with_map[key].tentative_matches
#         without_tentative_matches_p = without_map[key].tentative_matches
#         print("with tentative matches for {}: {}".format(key, with_tentative_matches_p))
#         print("without tentative matches for {}: {}".format(key, without_tentative_matches_p))
#         with_tentative_matches += with_tentative_matches_p
#         without_tentative_matches += without_tentative_matches_p
#
#         with_inlier_ratio_p = (with_map[key].inliers / with_map[key].tentative_matches)
#         without_inlier_ratio_p = (without_map[key].inliers / without_map[key].tentative_matches)
#         print("with inlier ratio_p for {}: {}".format(key, with_inlier_ratio_p))
#         print("without inlier ratio_p for {}: {}".format(key, without_inlier_ratio_p))
#         with_inlier_ratio += with_inlier_ratio_p
#         without_inlier_ratio += without_inlier_ratio_p
#
#         #print("with: {}, without: {}".format(with_map[key], without_map[key]))
#
#
#     with_inlier_ratio /= float(common_enties)
#     without_inlier_ratio /= float(common_enties)
#     with_tentative_matches /= float(common_enties)
#     without_tentative_matches /= float(common_enties)
#
#
#     diff_r.sort(key=lambda x: x[0])
#     diff_t.sort(key=lambda x: x[0])
#
#     for r_diff in diff_r:
#         print("R diff: {} in {}".format(r_diff[0], r_diff[1]))
#
#     for t_diff in diff_t:
#         print("T diff: {} in {}".format(t_diff[0], t_diff[1]))
#
#
#     print("common entries: {}".format(common_enties))
#     print("with rectification errors. R: {}, T: {}".format(with_r_err, with_t_err))
#     print("without rectification errors. R: {}, T: {}".format(without_r_err, without_t_err))
#     print("with tentative matches: {}".format(with_tentative_matches))
#     print("without tentative_matches: {}".format(without_tentative_matches))
#     print("with inlier ratio: {}".format(with_inlier_ratio))
#     print("without inlier ratio: {}".format(without_inlier_ratio))
#
#
#     print("All done")
#     end = time.time()
#     print("Time elapsed: {}".format(end - start))


if __name__ == "__main__":

    print("Started")

    parser = argparse.ArgumentParser(prog='evaluation')
    parser.add_argument('--input_dir', help='input dir')
    parser.add_argument('--method', help='method')
    parser.add_argument('--n_worst', help='method')
    args = parser.parse_args()

    assert args.input_dir is not None

    # legacy
    if args.method == "make_light":
        for diff in range(18):
            file_path = "{}/stats_diff_{}.pkl".format(args.input_dir, diff)
            if os.path.isfile(file_path):
                make_light(file_path)
            else:
                print("{} not found".format(file_path))

    else:
        diff_percs = []
        for diff in range(18):
            file_path = "{}/stats_diff_{}.pkl".format(args.input_dir, diff)
            if os.path.isfile(file_path):
                n_worst = None if args.n_worst is None else int(args.n_worst)
                diff_perc = evaluate_percentage_correct_from_file(file_path, diff, n_worst_examples=n_worst, th_degrees=5)
                diff_percs.append(diff_perc)
            else:
                print("{} not found".format(file_path))

        print("Diff     Perc.")
        for diff, perc in diff_percs:
            print("{}    {}".format(diff, perc))
