import numpy as np

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
    R1 = quaternions_to_R(img_entry_1.qs)

    img_entry_2: ImageEntry = scene_info.img_info_map[img_pair.img2]
    T2 = img_entry_2.t
    R2 = quaternions_to_R(img_entry_2.qs)

    dR = R2 @ R1.T
    dT = T2 - dR @ T1

    camera_1_id = scene_info.img_info_map[img_pair.img1].camera_id
    K1 = scene_info.cameras[camera_1_id].get_K()
    camera_2_id = scene_info.img_info_map[img_pair.img2].camera_id
    K2 = scene_info.cameras[camera_2_id].get_K()

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
                                   components_indices=None,
                                   valid_components_dict=None)
                                   # components_indices=self.components_indices,
                                   # valid_components_dict=self.valid_components_dict)


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
    src_tentatives: np.ndarray
    dst_tentatives: np.ndarray
    tentative_matches: int
    inliers: int
    all_features_1: int
    all_features_2: int
    src_pts_inliers: np.ndarray
    dst_pts_inliers: np.ndarray
    E: np.ndarray
    normals1: np.ndarray
    normals2: np.ndarray
    kpts1: list
    kpts2: list


    # can be made to a constructor?
    # @staticmethod
    # def read_from_dict(d):
    #     stats = Stats.load_from_array(d["stats"])
    #     stats.src_pts_inliers = d["src_pts_inliers"]
    #     stats.dsrc_pts_inliers = d["dst_pts_inliers"]
    #     stats.E = d["E"]
    #     return stats

    @staticmethod
    def read_from_file(file_path):
        np_array = np.loadtxt(file_path, delimiter=";\n,")
        return Stats.load_from_array(np_array)

    @staticmethod
    def load_from_array(np_array: np.ndarray):
        return Stats(error_R=np_array[0],
                     error_T=np_array[1],
                     tentative_matches=int(np_array[2]),
                     inliers=int(np_array[3]),
                     all_features_1=int(np_array[4]),
                     all_features_2=int(np_array[5]),
                     src_pts_inliers=None,
                     dst_pts_inliers=None,
                     E=None,
                     normals=None
                     )

    @staticmethod
    def get_field_descs():
        l = ["error in R", "error in T", "tentative matches", "inliers", "all features in 1st", "all features in 2nd"]
        return ";\n".join(l)

    def to_numpy(self):
        return np.array([self.error_R, self.error_T, self.tentative_matches, self.inliers, self.all_features_1, self.all_features_2])

    def save_brief(self, file_path: str):
        val = self.to_numpy()
        np.savetxt(file_path, val, delimiter=';\n', fmt='%1.8f', header=Stats.get_field_descs())

    @staticmethod
    def save_parts(out_dir,
                   save_suffix,
                   E,
                   src_tentative,
                   dst_tentative,
                   src_inliers,
                   dst_inliers,
                   error_R,
                   error_T,
                   n_tentative_matches,
                   n_inliers,
                   n_all_features_1,
                   n_all_features_2,
                   normals1,
                   normals2,
                   whole_kpts1,
                   whole_kpts2,
                   ):

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # np.savetxt("{}/essential_matrix_{}.txt".format(out_dir, save_suffix), E, delimiter=',', fmt='%1.8f')
        # np.savetxt("{}/src_inliers_{}.txt".format(out_dir, save_suffix), src_inliers, delimiter=',', fmt='%1.8f')
        # np.savetxt("{}/dst_inliers_{}.txt".format(out_dir, save_suffix), dst_inliers, delimiter=',', fmt='%1.8f')
        # np.savetxt("{}/src_tentative_{}.txt".format(out_dir, save_suffix), src_tentative, delimiter=',', fmt='%1.8f')
        # np.savetxt("{}/dst_tentative_{}.txt".format(out_dir, save_suffix), dst_tentative, delimiter=',', fmt='%1.8f')
        # np.savetxt("{}/normals_1_{}.txt".format(out_dir, save_suffix), normals1, delimiter=',', fmt='%1.8f')
        # np.savetxt("{}/normals_2_{}.txt".format(out_dir, save_suffix), normals2, delimiter=',', fmt='%1.8f')

        stats = Stats(error_R=error_R,
                      error_T=error_T,
                      src_tentatives=src_tentative,
                      dst_tentatives=dst_tentative,
                      tentative_matches=n_tentative_matches,
                      inliers=n_inliers,
                      all_features_1=n_all_features_1,
                      all_features_2=n_all_features_2,
                      E=E,
                      src_pts_inliers=src_inliers,
                      dst_pts_inliers=dst_inliers,
                      normals1=normals1,
                      normals2=normals2,
                      kpts1=whole_kpts1,
                      kpts2=whole_kpts2
                      )

        #stats.save_brief("{}/stats_{}.txt".format(out_dir, save_suffix))

        return stats


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
    print("Number of correspondences: {}".format(inlier_mask[inlier_mask == [0]].shape[0]))
    print("Number of not-correspondences: {}".format(inlier_mask[inlier_mask == [1]].shape[0]))

    src_tentative, dst_tentative = split_points(tentative_matches, kps1, kps2)
    src_pts_inliers = src_tentative[inlier_mask[:, 0] == [1]]
    dst_pts_inliers = dst_tentative[inlier_mask[:, 0] == [1]]

    error_R, error_T = compare_poses(E, img_pair, scene_info, src_pts_inliers, dst_pts_inliers)
    inliers = np.sum(np.where(inlier_mask[:, 0] == [1], 1, 0))

    stats = Stats.save_parts(out_dir,
                             save_suffix,
                             E,
                             src_tentative,
                             dst_tentative,
                             src_pts_inliers,
                             dst_pts_inliers,
                             error_R,
                             error_T,
                             len(tentative_matches),
                             inliers,
                             len(kps1),
                             len(kps2),
                             normals1,
                             normals2,
                             kps1,
                             kps2
                             )

    key = "{}_{}".format(img_pair.img1, img_pair.img2)
    #stats_map[key] = inner_map
    stats_map[key] = stats
    return stats


def evaluate_all(scene_info: SceneInfo, input_dir, limit=None):

    dirs = [dirname for dirname in sorted(os.listdir(input_dir)) if os.path.isdir("{}/{}".format(input_dir, dirname))]
    dirs = sorted(dirs)
    if limit is not None:
        dirs = dirs[0:limit]

    flattened_img_pairs = [pair for diff in scene_info.img_pairs_lists for pair in diff]
    img_pair_map = {"{}_{}".format(img_pair.img1, img_pair.img2): img_pair for img_pair in flattened_img_pairs}

    result_map = {}

    for dir in dirs:

        if not img_pair_map.__contains__(dir):
            print("dir '{}' not recognized!!!".format(dir))
            continue

        whole_path = "{}/{}".format(input_dir, dir)

        img_pair = img_pair_map[dir]
        E = np.loadtxt("{}/essential_matrix.txt".format(whole_path), delimiter=',')
        dst_pts = np.loadtxt("{}/dst_pts.txt".format(whole_path), delimiter=',')
        src_pts = np.loadtxt("{}/src_pts.txt".format(whole_path), delimiter=',')
        stats = np.loadtxt("{}/stats.txt".format(whole_path), delimiter=',')
        tentative_matches = stats[0]
        inliers = stats[1]
        all_features_1 = stats[2]
        all_features_2 = stats[3]

        print("Evaluating: {}".format(dir))
        errors = compare_poses(E, img_pair, scene_info, src_pts, dst_pts)
        result_map[dir] = Stats(errors[0], errors[1], tentative_matches, inliers, all_features_1, all_features_2)

    return result_map


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
    R1 = quaternions_to_R(img_entry_1.qs)
    K1 = scene_info.get_img_K(img_pair.img1)
    K1_inv = np.linalg.inv(K1)
    src_tentative = np.ndarray((src_tentatives_2d.shape[0], 3))
    src_tentative[:, :2] = src_tentatives_2d
    src_tentative[:, 2] = 1.0

    img_entry_2: ImageEntry = scene_info.img_info_map[img_pair.img2]
    T2 = np.array(img_entry_2.t)
    R2 = quaternions_to_R(img_entry_2.qs)
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


def evaluate(stats_map: dict, scene_info: SceneInfo):

    l_entries = []
    n_entries = 0

    error_R = []
    error_T = []
    tentative_matches = []
    inliers = []
    all_features_1 = []
    all_features_2 = []
    #matched_points = []
    all_checks = []
    x2_F_x1_thresholds = np.array([0.05, 0.01, 0.005])

    for img_pair_str, stats in stats_map.items():

        n_entries += 1
        img_pair, diff = scene_info.find_img_pair_from_key(img_pair_str)

        l_entries.append(str(img_pair))

        error_R.append(stats.error_R)
        error_T.append(stats.error_T)
        tentative_matches.append(stats.tentative_matches)
        inliers.append(stats.inliers)
        all_features_1.append(stats.all_features_1)
        all_features_2.append(stats.all_features_2)

        checks = evaluate_tentatives_agains_ground_truth(scene_info, img_pair, stats.src_tentatives, stats.dst_tentatives, x2_F_x1_thresholds)
        all_checks.append(checks)

        # matched_points_local = correctly_matched_point_for_image_pair(stats.src_pts_inliers,
        #                                                 stats.dst_pts_inliers,
        #                                                 scene_info.img_info_map,
        #                                                 img_pair)
        # matched_points.append(matched_points_local.shape[0])

    print("Image entries (img name, difficulty)")
    print(",\n".join(l_entries))

    print("Stats report")
    print_stats("error_R", error_R)
    print_stats("error_T", error_T)
    print_stats("tentative_matches", tentative_matches)
    print_stats("inliers", inliers)
    print_stats("all_features_1", all_features_1)
    print_stats("all_features_2", all_features_2)

    all_checks = np.array(all_checks)
    all_checks_avgs = np.sum(all_checks, axis=0) / all_checks.shape[0]
    print("F ground truth checks averages (for thresholds: {}): {}".format(x2_F_x1_thresholds, all_checks_avgs))


def evaluate_last(scene_name):

    scene_info = SceneInfo.read_scene(scene_name)
    stats_map = read_last()
    evaluate(stats_map, scene_info)


def evaluate_file(scene_name, file_name):

    scene_info = SceneInfo.read_scene(scene_name)
    with open(file_name, "rb") as f:
        print("reading: {}".format(file_name))
        stats_map = pickle.load(f)


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


def main():
    start = time.time()

    scene = "scene1"
    scene_info = SceneInfo.read_scene(scene)
    with_map = evaluate_all(scene_info, "work/{}/matching/with_rectification".format(scene), limit=None)
    without_map = evaluate_all(scene_info, "work/{}/matching/without_rectification".format(scene), limit=None)

    with_r_err = 0
    with_t_err = 0
    without_t_err = 0
    without_r_err = 0

    diff_r = []
    diff_t = []

    with_inlier_ratio = 0.0
    without_inlier_ratio = 0.0

    with_tentative_matches = 0
    without_tentative_matches = 0

    common_enties = 0

    for key in with_map:
        if not without_map.__contains__(key):
            continue
        common_enties += 1
        with_r_err += with_map[key].error_R
        with_t_err += with_map[key].error_T
        without_r_err += without_map[key].error_R
        without_t_err += without_map[key].error_T
        diff_r.append((with_map[key].error_R - without_map[key].error_R, key))
        diff_t.append((with_map[key].error_T - without_map[key].error_T, key))

        with_tentative_matches_p = with_map[key].tentative_matches
        without_tentative_matches_p = without_map[key].tentative_matches
        print("with tentative matches for {}: {}".format(key, with_tentative_matches_p))
        print("without tentative matches for {}: {}".format(key, without_tentative_matches_p))
        with_tentative_matches += with_tentative_matches_p
        without_tentative_matches += without_tentative_matches_p

        with_inlier_ratio_p = (with_map[key].inliers / with_map[key].tentative_matches)
        without_inlier_ratio_p = (without_map[key].inliers / without_map[key].tentative_matches)
        print("with inlier ratio_p for {}: {}".format(key, with_inlier_ratio_p))
        print("without inlier ratio_p for {}: {}".format(key, without_inlier_ratio_p))
        with_inlier_ratio += with_inlier_ratio_p
        without_inlier_ratio += without_inlier_ratio_p

        #print("with: {}, without: {}".format(with_map[key], without_map[key]))


    with_inlier_ratio /= float(common_enties)
    without_inlier_ratio /= float(common_enties)
    with_tentative_matches /= float(common_enties)
    without_tentative_matches /= float(common_enties)


    diff_r.sort(key=lambda x: x[0])
    diff_t.sort(key=lambda x: x[0])

    for r_diff in diff_r:
        print("R diff: {} in {}".format(r_diff[0], r_diff[1]))

    for t_diff in diff_t:
        print("T diff: {} in {}".format(t_diff[0], t_diff[1]))


    print("common entries: {}".format(common_enties))
    print("with rectification errors. R: {}, T: {}".format(with_r_err, with_t_err))
    print("without rectification errors. R: {}, T: {}".format(without_r_err, without_t_err))
    print("with tentative matches: {}".format(with_tentative_matches))
    print("without tentative_matches: {}".format(without_tentative_matches))
    print("with inlier ratio: {}".format(with_inlier_ratio))
    print("without inlier ratio: {}".format(without_inlier_ratio))


    print("All done")
    end = time.time()
    print("Time elapsed: {}".format(end - start))


if __name__ == "__main__":

#    evaluate_file("scene1", "work/pipeline_scene1_2021_05_12_18_49_55_115656/all.stats.pkl")
    #evaluate_file("scene1", "all.stats.pkl")
    #evaluate_file("scene1", "all.stats_last_rect.pkl")
    evaluate_file("scene1", "work/pipeline_scene1_333/all.stats.pkl")

    #evaluate_last("scene1")
