import argparse
import pickle
import sys
import traceback
from typing import List

import torch

from scene_info import *
from utils import *

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


def evaulate_R_t_safe(dR, dt, R, t):

    try:
        err_q, err_t = evaluate_R_t(dR, dt, R, t)
    except:
        print("WARNING: evaulate_R_t_safe")
        print(traceback.format_exc(), file=sys.stdout)

        err_q = np.pi
        err_t = np.pi / 2

    return err_q, err_t


def eval_essential_matrix(p1n, p2n, E, dR, dt):
    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return np.pi, np.pi / 2

    if E.size > 0:
        _, R, t, _ = cv.recoverPose(E, p1n, p2n)
        err_q, err_t = evaulate_R_t_safe(dR, dt, R, t)

    else:
        err_q = np.pi
        err_t = np.pi / 2

    return err_q, err_t, R


def get_GT_R_t(img_pair: ImagePairEntry, scene_info: SceneInfo):

    img_entry_1: ImageEntry = scene_info.img_info_map[img_pair.img1]
    T1 = img_entry_1.t
    R1 = img_entry_1.R

    img_entry_2: ImageEntry = scene_info.img_info_map[img_pair.img2]
    T2 = img_entry_2.t
    R2 = img_entry_2.R

    dR = R2 @ R1.T
    dt = T2 - dR @ T1

    return dR, dt


def compare_R_to_GT(img_pair: ImagePairEntry, scene_info: SceneInfo, r):
    dR, dt = get_GT_R_t(img_pair, scene_info)
    err_q, _ = evaulate_R_t_safe(dR, dt, r, dt)
    return err_q


def compare_poses(E, img_pair: ImagePairEntry, scene_info: SceneInfo, pts1, pts2, img_data_list):

    img_entry_1: ImageEntry = scene_info.img_info_map[img_pair.img1]
    T1 = img_entry_1.t
    R1 = img_entry_1.R

    img_entry_2: ImageEntry = scene_info.img_info_map[img_pair.img2]
    T2 = img_entry_2.t
    R2 = img_entry_2.R

    dR = R2 @ R1.T
    dt = T2 - dR @ T1

    K1 = scene_info.get_img_K(img_pair.img1, img_data_list[0].img)
    K2 = scene_info.get_img_K(img_pair.img2, img_data_list[1].img)

    # TODO Q: what is actually this? if I remove it, I can remove the call to scene_info.get_img_K
    p1n = normalize_keypoints(pts1, K1).astype(np.float64)
    p2n = normalize_keypoints(pts2, K2).astype(np.float64)
    # Q: this doesn't change the result!!!
    # p1n = pts1
    # p2n = pts2

    err_q, err_t, dr_est = eval_essential_matrix(p1n, p2n, E, dR, dt)

    rot_vec_deg_est = get_rot_vec_deg(dr_est)
    rot_vec_deg_gt = get_rot_vec_deg(dR)

    print("rotation vector(GT): {}".format(rot_vec_deg_gt))
    print("rotation vector(est): {}".format(rot_vec_deg_est))
    print("errors (R, T): ({} degrees, {} (unscaled))".format(np.rad2deg(err_q), err_t))

    return err_q, err_t


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
    real_K: np.ndarray
    normals: np.ndarray
    ts_phis: torch.Tensor
    components_indices: np.ndarray
    valid_components_dict: dict

    @staticmethod
    def from_serialized_data(img, real_K, img_serialized_data):
        return ImageData(img=img,
                         real_K=real_K,
                         key_points=img_serialized_data.kpts,
                         descriptions=img_serialized_data.descs,
                         normals=img_serialized_data.normals,
                         ts_phis=None,
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

    inliers_against_gt: (int, int, int)
    tentatives_1: (float, float)
    tentatives_2: (float, float)
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


COMPLETE_IMAGE_PAIR_MATCHING_TAG = "complete_image_pair_matching"


@timer_label_decorator(tags=[COMPLETE_IMAGE_PAIR_MATCHING_TAG])
def evaluate_matching(scene_info,
                      E,
                      img_data_list,
                      tentative_matches,
                      inlier_mask,
                      img_pair,
                      stats_map,
                      ransac_th,
                      ):

    print("Image pair: {} <-> {}:".format(img_pair.img1, img_pair.img2))
    print("Number of inliers: {}".format(inlier_mask[inlier_mask == [1]].shape[0]))
    print("Number of outliers: {}".format(inlier_mask[inlier_mask == [0]].shape[0]))

    src_tentatives_2d, dst_tentatives_2d = split_points(tentative_matches, img_data_list[0].key_points, img_data_list[1].key_points)
    src_pts_inliers = src_tentatives_2d[inlier_mask[:, 0] == [1]]
    dst_pts_inliers = dst_tentatives_2d[inlier_mask[:, 0] == [1]]

    error_R, error_T = compare_poses(E, img_pair, scene_info, src_pts_inliers, dst_pts_inliers, img_data_list)
    inliers = np.sum(np.where(inlier_mask[:, 0] == [1], 1, 0))

    if is_rectified_condition(img_data_list[0]):
        _, unique, counts = get_normals_stats(img_data_list, src_tentatives_2d, dst_tentatives_2d)
        print("Matching stats in evaluation:")
        print("unique plane correspondence counts of tentatives:\n{}".format(np.vstack((unique.T, counts)).T))

    count_sampson_gt, \
    count_symmetrical_gt, \
    count_sampson_estimated, \
    count_symmetrical_estimated = evaluate_tentatives_agains_ground_truth(scene_info,
                                                                          img_pair,
                                                                          img_data_list,
                                                                          src_tentatives_2d,
                                                                          dst_tentatives_2d,
                                                                          [ransac_th, 0.1, 0.5, 1, 3],
                                                                          E, inliers)

    stats = Stats(inliers_against_gt=count_symmetrical_gt,
                  tentatives_1=src_tentatives_2d,
                  tentatives_2=dst_tentatives_2d,
                  error_R=error_R,
                  error_T=error_T,
                  tentative_matches=len(tentative_matches),
                  inliers=inliers,
                  all_features_1=len(img_data_list[0].key_points),
                  all_features_2=len(img_data_list[1].key_points),
                  E=E,
                  normals1=img_data_list[0].normals,
                  normals2=img_data_list[1].normals,
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
        [-vec[1],  vec[0],     0.0],
    ])


def evaluate_tentatives_agains_ground_truth(scene_info: SceneInfo,
                                            img_pair: ImagePairEntry,
                                            img_data_list,
                                            src_tentatives_2d,
                                            dst_tentatives_2d,
                                            thresholds,
                                            est_E,
                                            inliers_from_ransac):

    def get_T_R_K_inv(img_key, img):
        img_entry: ImageEntry = scene_info.img_info_map[img_key]
        T = np.array(img_entry.t)
        R = img_entry.R
        # TODO we use the real K here, right?
        K = scene_info.get_img_K(img_key, img)
        K_inv = np.linalg.inv(K)
        return T, R, K_inv, K

    T1, R1, K1_inv, K1 = get_T_R_K_inv(img_pair.img1, img_data_list[0].img)
    src_tentative_h = np.ndarray((src_tentatives_2d.shape[0], 3))
    src_tentative_h[:, :2] = src_tentatives_2d
    src_tentative_h[:, 2] = 1.0

    T2, R2, K2_inv, K2 = get_T_R_K_inv(img_pair.img2, img_data_list[1].img)
    dst_tentative_h = np.ndarray((dst_tentatives_2d.shape[0], 3))
    dst_tentative_h[:, :2] = dst_tentatives_2d
    dst_tentative_h[:, 2] = 1.0

    F_ground_truth = K2_inv.T @ R2 @ vector_product_matrix(T2 - T1) @ R1.T @ K1_inv
    F_ground_truth = torch.from_numpy(F_ground_truth).unsqueeze(0)

    src_pts_torch = torch.from_numpy(src_tentatives_2d).type(torch.DoubleTensor)
    dst_pts_torch = torch.from_numpy(dst_tentatives_2d).type(torch.DoubleTensor)

    kornia_sampson_gt = KG.epipolar.sampson_epipolar_distance(src_pts_torch, dst_pts_torch, F_ground_truth).numpy()
    kornia_symmetrical_gt = KG.epipolar.symmetrical_epipolar_distance(src_pts_torch, dst_pts_torch, F_ground_truth).numpy()

    computed_F = KG.fundamental_from_essential(torch.from_numpy(est_E), torch.from_numpy(K1), torch.from_numpy(K2))
    computed_F = computed_F.unsqueeze(0)

    kornia_sampson_estimated = KG.epipolar.sampson_epipolar_distance(src_pts_torch, dst_pts_torch, computed_F).numpy()
    kornia_symmetrical_estimated = KG.epipolar.symmetrical_epipolar_distance(src_pts_torch, dst_pts_torch, computed_F).numpy()

    count_sampson_gt = np.zeros(len(thresholds), dtype=int)
    count_symmetrical_gt = np.zeros(len(thresholds), dtype=int)
    count_sampson_estimated = np.zeros(len(thresholds), dtype=int)
    count_symmetrical_estimated = np.zeros(len(thresholds), dtype=int)

    print("Matching stats in inliers:")
    def evaluate_metric(metric, th, label):
        mask = (np.abs(metric) < th)[0]
        if is_rectified_condition(img_data_list[0]):
            _, unique, counts = get_normals_stats(img_data_list, src_tentatives_2d, dst_tentatives_2d, mask)
            print("{} < {}:".format(label, th))
            print("unique plane correspondence counts:\n{}".format(np.vstack((unique.T, counts)).T))
        return np.sum(mask)

    for i in range(len(thresholds)):
        count_sampson_gt[i] = evaluate_metric(kornia_sampson_gt, thresholds[i], "sampson gt")
        count_symmetrical_gt[i] = evaluate_metric(kornia_symmetrical_gt, thresholds[i], "symmetrical gt")
        count_sampson_estimated[i] = evaluate_metric(kornia_sampson_estimated, thresholds[i], "sampson estimated")
        count_symmetrical_estimated[i] = evaluate_metric(kornia_symmetrical_estimated, thresholds[i], "symmetrical estimated")

    print("inliers from ransac: {}".format(inliers_from_ransac))
    print("thresholds for inliers: {}".format(thresholds))
    print("inliers against GT - sampson:{}".format(count_sampson_gt))
    print("inliers against GT - symmetrical:{}".format(count_symmetrical_gt))
    print("inliers against estimated E/F - sampson:{}".format(count_sampson_estimated))
    print("inliers against estimated E/F - symmetrical:{}".format(count_symmetrical_estimated))

    return count_sampson_gt, count_symmetrical_gt, count_sampson_estimated, count_symmetrical_estimated


def evaluate_all_matching_stats_even_normalized(stats_map_all: dict, tex_save_path_prefix=None, n_examples=None, special_diff=None, scene_info: SceneInfo=None):
    evaluate_all_matching_stats(stats_map_all, tex_save_path_prefix, n_examples, special_diff)
    if scene_info is not None:
        tex_save_path_prefix = tex_save_path_prefix + "_normalized" if tex_save_path_prefix is not None else None
        evaluate_all_matching_stats(stats_map_all, tex_save_path_prefix, n_examples=None, special_diff=None, scene_info=scene_info)


def evaluate_all_matching_stats(stats_map_all: dict, tex_save_path_prefix=None, n_examples=None, special_diff=None, scene_info: SceneInfo=None):
    if scene_info is None:
        print("Stats for all difficulties unnormalized:")
    else:
        print("Stats for all difficulties normalized:")

    parameters_keys_list = list(stats_map_all.keys())

    all_diffs = set()
    for key in parameters_keys_list:
        all_diffs = all_diffs.union(set(stats_map_all[key].keys()))
    all_diffs = list(all_diffs)
    all_diffs.sort()

    if n_examples is not None and n_examples > 0:
        for diff in all_diffs:
            if special_diff is not None and special_diff != diff:
                continue
            for key in parameters_keys_list:
                if stats_map_all[key].__contains__(diff): # and len(stats_map_all[key][diff]) > 0:
                    print_significant_instances(stats_map_all[key][diff], key, diff, n_examples=n_examples)


    angle_thresholds = [5, 10, 20]
    accuracy_diff_acc_data_lists = [None] * len(angle_thresholds)
    for angle_threshold in angle_thresholds:

        diff_acc_data_lists = [[] for _ in parameters_keys_list]
        accuracy_diff_acc_data_lists.append(diff_acc_data_lists)

        print("Accuracy({}º) {}".format(angle_threshold, " ".join([str(k) for k in parameters_keys_list])))
        for diff in all_diffs:
            value_list = []
            for i_param_list, key in enumerate(parameters_keys_list):
                if stats_map_all[key].__contains__(diff) and len(stats_map_all[key][diff]) > 0:
                    all_len_const = len(scene_info.img_pairs_lists[diff]) if scene_info is not None else None
                    difficulty, perc = evaluate_percentage_correct(stats_map_all[key][diff], diff, th_degrees=angle_threshold, all_len_const=all_len_const)
                    value_list.append("{:.3f}".format(perc))
                    diff_acc_data_lists[i_param_list].append((float(diff), perc))
                else:
                    value_list.append("--")
            print(" ".join(value_list))

    # if tex_save_path_prefix is not None:
    #     for i, angle_threshold in enumerate(angle_thresholds):
    #         title = "accuracy < {} for: {}".format(angle_thresholds[i], parameters_keys_list)
    #         graph = convert_from_data(title, parameters_keys_list, diff_acc_data_lists)
    #         with open("{}_acc_{}.txt".format(tex_save_path_prefix, angle_threshold), "w") as f:
    #             f.write("%TEX for accuracy < {} degrees for {} \n".format(angle_thresholds[i], parameters_keys_list))
    #             f.write(graph)

    print("Counts {}".format(" ".join([str(k) for k in parameters_keys_list])))
    for diff in all_diffs:
        value_list = []
        for key in parameters_keys_list:
            if scene_info is not None:
                value_list.append(str(len(scene_info.img_pairs_lists[diff])))
            else:
                if stats_map_all[key].__contains__(diff):
                    value_list.append(str(len(stats_map_all[key][diff])))
                else:
                    value_list.append("0")
        print("{} {}".format(diff, " ".join(value_list)))

    # TODO scene_info/matching limit is not None: totally useless - just create the failed pairs as you go(?)
    if scene_info is not None:
        print("Failed pairs")
        failed_pairs = {}
        for key in parameters_keys_list:
            failed_pairs[key] = set()
        for diff in all_diffs:
            for key in parameters_keys_list:
                if stats_map_all[key].__contains__(diff):
                    relevant_set = set(stats_map_all[key][diff])
                else:
                    relevant_set = set()
                all_from_diff = set([SceneInfo.get_key_from_pair(p) for p in scene_info.img_pairs_lists[diff]])
                failed_pairs[key] = failed_pairs[key].union((all_from_diff - relevant_set))

        all_failed_pairs = set()
        for key in failed_pairs:
            print("Failed pairs for {}:".format(key))
            print("matching_pairs = [{}]".format(", ".join(failed_pairs[key])))
            all_failed_pairs = all_failed_pairs.union(failed_pairs[key])

        # NOTE I don't think there is a good use case for this
        # print("All failed pairs:")
        # print("matching_pairs = [{}]".format(", ".join(all_failed_pairs)))



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

def print_significant_instances(stats_map, difficulty, key, n_examples=10):

    sorted_by_err_R = list(sorted(stats_map.items(), key=lambda key_value: -key_value[1].error_R))
    print("{} worst examples for {} for diff={}".format(n_examples, key, difficulty))
    for k, v in sorted_by_err_R[:n_examples]:
        print("{}: {}".format(k, v.error_R))
    print("{} best examples for {} for diff={}".format(n_examples, key, difficulty))
    for k, v in sorted_by_err_R[-n_examples:]:
        print("{}: {}".format(k, v.error_R))


def evaluate_percentage_correct(stats_map, difficulty, th_degrees=5, all_len_const=None):

    rad_th = th_degrees * math.pi / 180
    filtered = list(filter(lambda key_value: key_value[1].error_R < rad_th, stats_map.items()))
    filtered_len = len(filtered)
    all_len = all_len_const if all_len_const is not None else len(stats_map.items())
    if all_len == 0:
        return difficulty, 0.0
    else:
        perc = filtered_len/all_len
        return difficulty, perc


def compare_stats_maps(stats_map1: dict, stats_map2: dict, difficulty, n_worst_examples=None, th_degrees=5):

    keys1: set = set(stats_map1.keys())
    keys2: set = set(stats_map2.keys())

    both = keys1.intersection(keys2)
    only1 = keys1 - keys2
    only2 = keys2 - keys1

    print("Stats maps: {} common keys".format(len(both)))
    print("Stats maps: {} keys only in 1st map".format(len(only1)))
    print("Stats maps: {} keys only in 2nd map".format(len(only2)))

    rad_th = th_degrees * math.pi / 180

    def get_sat_keys(stats_map):
        filtered = list(filter(lambda key_value: key_value[1].error_R < rad_th, stats_map.items()))
        keys = set([i[0] for i in filtered])
        return keys

    keys1 = get_sat_keys(stats_map1)
    perc1 = len(keys1) / len(stats_map1)

    keys2 = get_sat_keys(stats_map2)
    perc2 = len(keys2) / len(stats_map2)

    stats = [(key, stats_map1[key], stats_map2[key]) for key in both]
    sorted_by_err_R_diff = list(sorted(stats, key=lambda tuple: tuple[1].error_R - tuple[2].error_R))

    def print_out(l):
        for k, r1, r2 in l:
            print("{}: R: {} vs. {}, inliers: {} vs. {}".format(k, r1.error_R, r2.error_R, r1.inliers, r2.inliers))
        print("Repeating the keys:")
        print(", ".join([k[0] for k in l]))

    if n_worst_examples is not None:

        print("{} best examples for 1st map for diff={}".format(n_worst_examples, difficulty))
        l = sorted_by_err_R_diff[:n_worst_examples]
        print_out(l)

        filtered1 = list(filter(lambda key_value: key_value[0] in keys1, sorted_by_err_R_diff))
        print("{} best satisfying examples for 1st map for diff={}".format(n_worst_examples, difficulty))
        print_out(filtered1[:n_worst_examples])

        print("{} best examples for 2nd map for diff={}".format(n_worst_examples, difficulty))
        print_out(sorted_by_err_R_diff[-n_worst_examples:])

        filtered2 = list(filter(lambda key_value: key_value[0] in keys2, sorted_by_err_R_diff))
        print("{} best satisfying examples for 2nd map for diff={}".format(n_worst_examples, difficulty))
        print_out(filtered2[-n_worst_examples:])

    print("{}\t{:.03f}\t{:.03f}".format(difficulty, perc1, perc2))
    return difficulty, perc1, perc2


def evaluate_percentage_correct_from_file(file_name, difficulty, n_worst_examples=None, th_degrees=5):

    with open(file_name, "rb") as f:
        stats_map = pickle.load(f)

    print_significant_instances(stats_map, difficulty, key="default", n_examples=n_worst_examples)
    return evaluate_percentage_correct(stats_map, difficulty, th_degrees=th_degrees)


def evaluate_percentage_correct_from_files(file_name1, file_name2, difficulty, n_worst_examples=None, th_degrees=5):

    with open(file_name1, "rb") as f:
        stats_map1 = pickle.load(f)

    with open(file_name2, "rb") as f:
        stats_map2 = pickle.load(f)

    return compare_stats_maps(stats_map1, stats_map2, difficulty, n_worst_examples=n_worst_examples, th_degrees=th_degrees)


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

def evaluate_stats(stats_map):
    evaluate_normals_stats(stats_map)
    evaluate_matching_stats(stats_map)
    evaluate_per_img_stats(stats_map)


def excel_friendly_format(keys, stats):
    print("excel friendly:")
    print(" ".join(keys))
    print(" ".join(stats))


def evaluate_per_img_stats(stats_map):

    def get_sum(m, key):
        return sum(m.get(key, []))

    if not stats_map.__contains__("per_img_stats"):
        print("WARNING: 'per_img_stats' not found, skipping")
    else:
        print("Per img stats:")
        m = stats_map["per_img_stats"]
        for configuration in m:
            print("configuration: {}".format(configuration))
            sum_area = 0
            sum_warps = 0
            sum_components = 0
            sum_kpts = 0
            sum_kpts_test = 0
            sum_warped_kpts = 0
            sum_kpts_mask_size = 0
            sum_corner_components = 0
            conf_stat = m[configuration]
            for img in conf_stat:
                img_key = conf_stat[img]
                affnet_warps_per_component = img_key.get("affnet_warps_per_component", [])
                sum_components += len(affnet_warps_per_component)

                sum_warps += get_sum(img_key, "affnet_warps_per_component")
                sum_area += get_sum(img_key, "affnet_warped_img_size")
                sum_kpts += get_sum(img_key, "affnet_added_kpts")
                sum_warped_kpts += get_sum(img_key, "affnet_warped_added_kpts")
                sum_kpts_mask_size += get_sum(img_key, "affnet_effective_kpts_mask_size")
                sum_corner_components += get_sum(img_key, "rect_components_without_keypoints")

                all_kpts = img_key.get("affnet_added_kpts", [])
                for kpts in all_kpts:
                    sum_kpts_test += kpts
                assert sum_kpts == sum_kpts_test

            img_count = len(m[configuration])

            avg_components = sum_components / img_count
            print("avg number of components (per image): {}".format(avg_components))

            avg_kpts_mask_size = sum_kpts_mask_size / img_count
            print("avg size of kpts mask (per image): {}".format(avg_kpts_mask_size))

            avg_area = sum_area / img_count
            print("avg rectified warped region area (per image): {}".format(avg_area))

            avg_kpts = sum_kpts / img_count
            print("avg kpts (per image): {}".format(avg_kpts))

            avg_kpts_warped = sum_warped_kpts / img_count
            print("avg kpts in warped regions (per image): {}".format(avg_kpts_warped))

            avg_corner_components = sum_corner_components / img_count
            print("avg corner components (per image): {}".format(avg_corner_components))

            if sum_components != 0:
                avg_warps_per_component = sum_warps / sum_components
                print("avg number of warps per component: {}".format(avg_warps_per_component))
                avg_area_component = sum_area / sum_components
                print("avg rectified warped region area (per component): {}".format(avg_area_component))
            else:
                avg_warps_per_component = 0.0
                print("avg number of warps per component: N/A 'sum_components == 0'")
                print("avg rectified warped region area (per component): N/A 'sum_components == 0'")
            excel_friendly_format(["avg_components, avg_area", "avg_kpts", "avg_warps_per_component", "avg_kpts_mask_size"],
                                  [str(i) for i in [avg_components, avg_area, avg_kpts, avg_warps_per_component, avg_kpts_mask_size]])


def get_all_diffs(maps_all_params):
    keys_list = maps_all_params.keys()
    all_diffs = set()
    for key in keys_list:
        all_diffs = all_diffs.union(set(maps_all_params[key].keys()))
    all_diffs = list(all_diffs)
    all_diffs.sort()
    return all_diffs


def evaluate_matching_stats(stats_map):

    if not stats_map.__contains__("matching"):
        print("WARNING: 'matching key' not found, skipping")
        return
    matching_map_all = stats_map["matching"]

    stats_local = {"all_keypoints": {}, "tentatives": {}, "inliers": {}, "inlier_ratio": {}}

    all_diffs = get_all_diffs(matching_map_all)
    keys_list = matching_map_all.keys()

    for difficulty in all_diffs:

        stats_local["all_keypoints"][difficulty] = []
        stats_local["tentatives"][difficulty] = []
        stats_local["inliers"][difficulty] = []
        stats_local["inlier_ratio"][difficulty] = []

        for param_key in keys_list:
            matching_map_per_key = matching_map_all[param_key]

            if matching_map_per_key.__contains__(difficulty):
                matching_map = matching_map_per_key[difficulty]

                kps = 0
                tentatives = 0
                inliers = 0
                for pair_name in matching_map:
                    kps = kps + matching_map[pair_name]["kps1"]
                    kps = kps + matching_map[pair_name]["kps2"]
                    tentatives = tentatives + matching_map[pair_name]["tentatives"]
                    inliers = inliers + matching_map[pair_name]["inliers"]

                all_value = kps / (2 * len(matching_map))
                tentatives_value = tentatives / len(matching_map)
                inliers_value = inliers / len(matching_map)
                stats_local["all_keypoints"][difficulty].append("{:.3f}".format(all_value))
                stats_local["tentatives"][difficulty].append("{:.3f}".format(tentatives_value))
                stats_local["inliers"][difficulty].append("{:.3f}".format(inliers_value))
                stats_local["inlier_ratio"][difficulty].append("{:.3f}".format(inliers_value / tentatives_value))
            else:
                stats_local["all_keypoints"][difficulty].append("--")
                stats_local["tentatives"][difficulty].append("--")
                stats_local["inliers"][difficulty].append("--")
                stats_local["inlier_ratio"][difficulty].append("--")


    #print("difficulties: [{}]: ".format(", ".join([str(difficulty) for difficulty in matching_map_per_key])))
    for key in ["all_keypoints", "tentatives", "inliers", "inlier_ratio"]:
        print("{} across difficulties: ".format(key))
        print("\t".join(keys_list))
        for difficulty in all_diffs:
            print("{}".format("\t".join(stats_local[key][difficulty])))


def evaluate_normals_stats(stats_map):

    normals_degrees = stats_map.get('normals_degrees', None)
    valid_normals = stats_map.get('valid_normals', None)
    # normals = stats_map.get('normals', None)
    if normals_degrees is None or valid_normals is None:
        print("normals will not be evaluated, probably all has been cached")
        return

    def shared_pairs(keys):

        at_least_two_sets = {}
        for k in keys:
            at_least_two_sets[k] = set()
            for img in normals_degrees[k]:
                deg_list = normals_degrees[k][img]
                if len(deg_list) > 0: #if valid_normals.get(img, 0) >= 2:
                    at_least_two_sets[k].add(img)

        first_key = list(at_least_two_sets.keys())[0]
        shared_keys = at_least_two_sets[first_key]
        for k in at_least_two_sets:
            shared_keys = shared_keys.intersection(at_least_two_sets[k])

        return shared_keys, at_least_two_sets

    shared_at_least_two, _ = shared_pairs(normals_degrees.keys())

    ms_None_1_0_25_0_8_list = ["frame_0000000165_2", "frame_0000000175_4", "frame_0000000220_4", "frame_0000000050_4", "frame_0000000095_1", "frame_0000000130_3", "frame_0000000105_1", "frame_0000000170_4", "frame_0000000130_2", "frame_0000000030_3", "frame_0000000230_1", "frame_0000000125_1", "frame_0000000035_4", "frame_0000000080_4", "frame_0000000220_3", "frame_0000000260_4", "frame_0000000125_3", "frame_0000000040_1", "frame_0000000100_4", "frame_0000000215_4", "frame_0000000235_4", "frame_0000000180_3", "frame_0000000145_3", "frame_0000000240_3", "frame_0000000085_2", "frame_0000000130_4", "frame_0000000045_2", "frame_0000000060_2", "frame_0000000180_4", "frame_0000000080_1", "frame_0000000110_1", "frame_0000000270_1", "frame_0000000210_1", "frame_0000000150_1", "frame_0000000140_3", "frame_0000000165_4", "frame_0000000045_1", "frame_0000000120_4", "frame_0000000205_4", "frame_0000000090_4", "frame_0000000165_1", "frame_0000000080_3", "frame_0000000160_4", "frame_0000000115_4", "frame_0000000190_1", "frame_0000000185_1", "frame_0000000190_4", "frame_0000000235_3", "frame_0000000245_3", "frame_0000000245_4", "frame_0000000050_2", "frame_0000000055_4", "frame_0000000250_3", "frame_0000000155_3", "frame_0000000040_2", "frame_0000000050_3", "frame_0000000220_1", "frame_0000000155_4", "frame_0000000060_3", "frame_0000000260_3", "frame_0000000105_3", "frame_0000000250_4", "frame_0000000255_4", "frame_0000000175_3", "frame_0000000060_4", "frame_0000000125_4", "frame_0000000150_4", "frame_0000000145_4", "frame_0000000145_2", "frame_0000000175_1", "frame_0000000225_1", "frame_0000000110_3", "frame_0000000095_3", "frame_0000000040_3", "frame_0000000035_3", "frame_0000000015_3", "frame_0000000090_2", "frame_0000000085_4", "frame_0000000065_4", "frame_0000000010_3", "frame_0000000100_1", "frame_0000000085_3", "frame_0000000240_4", "frame_0000000070_4", "frame_0000000115_3", "frame_0000000140_1", "frame_0000000085_1", "frame_0000000050_1", "frame_0000000070_2", "frame_0000000110_2", "frame_0000000150_3", "frame_0000000025_3", "frame_0000000075_1", "frame_0000000165_3", "frame_0000000045_3", "frame_0000000065_1", "frame_0000000195_1", "frame_0000000080_2", "frame_0000000115_2", "frame_0000000225_4", "frame_0000000020_3", "frame_0000000140_2", "frame_0000000155_1", "frame_0000000185_4", "frame_0000000170_2", "frame_0000000115_1", "frame_0000000065_2", "frame_0000000095_2", "frame_0000000210_4", "frame_0000000170_1", "frame_0000000185_3", "frame_0000000100_2", "frame_0000000215_3", "frame_0000000265_3", "frame_0000000135_2", "frame_0000000160_3", "frame_0000000075_2", "frame_0000000065_3", "frame_0000000095_4", "frame_0000000090_3", "frame_0000000265_4", "frame_0000000225_3", "frame_0000000120_1", "frame_0000000120_2", "frame_0000000130_1", "frame_0000000230_3", "frame_0000000100_3", "frame_0000000020_4", "frame_0000000195_4", "frame_0000000200_1", "frame_0000000255_3", "frame_0000000195_3", "frame_0000000190_3", "frame_0000000135_3", "frame_0000000180_1", "frame_0000000070_3", "frame_0000000055_1", "frame_0000000055_3", "frame_0000000260_1", "frame_0000000135_4", "frame_0000000125_2", "frame_0000000200_3", "frame_0000000060_1", "frame_0000000035_2", "frame_0000000205_1", "frame_0000000015_4", "frame_0000000210_3", "frame_0000000230_4", "frame_0000000120_3", "frame_0000000205_3", "frame_0000000075_3", "frame_0000000110_4", "frame_0000000140_4", "frame_0000000170_3", "frame_0000000145_1", "frame_0000000040_4", "frame_0000000075_4", "frame_0000000045_4", "frame_0000000160_1", "frame_0000000215_1", "frame_0000000135_1", "frame_0000000105_4", "frame_0000000105_2", "frame_0000000030_4", "frame_0000000265_1", "frame_0000000025_4", "frame_0000000070_1", "frame_0000000030_2", "frame_0000000090_1", "frame_0000000055_2"]
    ms_mean_0_s8_35_0_8_list = ["frame_0000000165_2", "frame_0000000175_2", "frame_0000000175_4", "frame_0000000220_4", "frame_0000000050_4", "frame_0000000095_1", "frame_0000000130_3", "frame_0000000105_1", "frame_0000000170_4", "frame_0000000130_2", "frame_0000000030_3", "frame_0000000125_1", "frame_0000000035_4", "frame_0000000080_4", "frame_0000000220_3", "frame_0000000260_4", "frame_0000000125_3", "frame_0000000040_1", "frame_0000000100_4", "frame_0000000215_4", "frame_0000000235_4", "frame_0000000145_3", "frame_0000000240_3", "frame_0000000085_2", "frame_0000000130_4", "frame_0000000045_2", "frame_0000000060_2", "frame_0000000180_4", "frame_0000000080_1", "frame_0000000110_1", "frame_0000000270_1", "frame_0000000210_1", "frame_0000000150_1", "frame_0000000140_3", "frame_0000000165_4", "frame_0000000045_1", "frame_0000000120_4", "frame_0000000205_4", "frame_0000000090_4", "frame_0000000165_1", "frame_0000000080_3", "frame_0000000160_4", "frame_0000000115_4", "frame_0000000190_1", "frame_0000000185_1", "frame_0000000190_4", "frame_0000000235_3", "frame_0000000245_3", "frame_0000000245_4", "frame_0000000050_2", "frame_0000000055_4", "frame_0000000250_3", "frame_0000000155_3", "frame_0000000040_2", "frame_0000000050_3", "frame_0000000220_1", "frame_0000000155_4", "frame_0000000060_3", "frame_0000000260_3", "frame_0000000105_3", "frame_0000000250_4", "frame_0000000060_4", "frame_0000000125_4", "frame_0000000150_4", "frame_0000000145_4", "frame_0000000145_2", "frame_0000000175_1", "frame_0000000225_1", "frame_0000000110_3", "frame_0000000150_2", "frame_0000000095_3", "frame_0000000040_3", "frame_0000000035_3", "frame_0000000015_3", "frame_0000000090_2", "frame_0000000085_4", "frame_0000000065_4", "frame_0000000010_3", "frame_0000000100_1", "frame_0000000085_3", "frame_0000000240_4", "frame_0000000070_4", "frame_0000000115_3", "frame_0000000140_1", "frame_0000000085_1", "frame_0000000050_1", "frame_0000000070_2", "frame_0000000110_2", "frame_0000000150_3", "frame_0000000025_3", "frame_0000000075_1", "frame_0000000165_3", "frame_0000000045_3", "frame_0000000065_1", "frame_0000000195_1", "frame_0000000080_2", "frame_0000000115_2", "frame_0000000225_4", "frame_0000000020_3", "frame_0000000140_2", "frame_0000000155_1", "frame_0000000185_4", "frame_0000000170_2", "frame_0000000115_1", "frame_0000000065_2", "frame_0000000095_2", "frame_0000000210_4", "frame_0000000170_1", "frame_0000000100_2", "frame_0000000215_3", "frame_0000000265_3", "frame_0000000135_2", "frame_0000000160_3", "frame_0000000075_2", "frame_0000000065_3", "frame_0000000095_4", "frame_0000000090_3", "frame_0000000265_4", "frame_0000000225_3", "frame_0000000160_2", "frame_0000000120_1", "frame_0000000120_2", "frame_0000000130_1", "frame_0000000230_3", "frame_0000000100_3", "frame_0000000195_4", "frame_0000000200_1", "frame_0000000195_3", "frame_0000000255_3", "frame_0000000135_3", "frame_0000000180_1", "frame_0000000070_3", "frame_0000000055_1", "frame_0000000055_3", "frame_0000000260_1", "frame_0000000135_4", "frame_0000000125_2", "frame_0000000200_3", "frame_0000000060_1", "frame_0000000035_2", "frame_0000000205_1", "frame_0000000210_3", "frame_0000000230_4", "frame_0000000120_3", "frame_0000000205_3", "frame_0000000075_3", "frame_0000000110_4", "frame_0000000140_4", "frame_0000000170_3", "frame_0000000145_1", "frame_0000000040_4", "frame_0000000155_2", "frame_0000000075_4", "frame_0000000045_4", "frame_0000000160_1", "frame_0000000215_1", "frame_0000000135_1", "frame_0000000105_4", "frame_0000000105_2", "frame_0000000030_4", "frame_0000000265_1", "frame_0000000025_4", "frame_0000000030_2", "frame_0000000090_1", "frame_0000000055_2"]
    shared_at_least_two = ms_None_1_0_25_0_8_list

    print("{} imgs are common to all keys".format(len(shared_at_least_two)))

    # key1 = "ms_mean_0.8_35_0.8" #_0.0_False"
    # key2 = "ms_None_1.0_25_0.8" #_0.0_False"
    # stats1 = deg[key1]
    # stats2 = deg[key2]
    #
    # shared_for_2, at_least_2_planes = shared_pairs([key1, key2])
    #
    # stats1_extra_keys = at_least_2_planes[key1] - shared_for_2
    # stats2_extra_keys = at_least_2_planes[key2] - shared_for_2
    # print("extra: {},\n {}".format(stats1_extra_keys, stats2_extra_keys))

    table = " parameters.... avg_l1_shared count_shared avg_l1_valid count_valid"
    for param_key in normals_degrees:
        print(f"Key: {param_key}")
        # out = False
        # if k.startswith("ms_None_1.0_25_0.8"):
        #     out = True
        count = 0
        count_shared = 0
        count_valid = 0
        avg_l2 = 0.0
        avg_l1 = 0.0
        avg_l2_shared = 0.0
        avg_l1_shared = 0.0
        avg_l1_valid = 0.0
        avg_l2_valid = 0.0
        for img in normals_degrees[param_key]:
            deg_list = normals_degrees[param_key][img]
            if valid_normals[param_key][img] > 1:
                count_valid = count_valid + 1
                avg_l1_valid = avg_l1_valid + math.fabs(90.0 - deg_list[0])
                avg_l2_valid = avg_l2_valid + (90.0 - deg_list[0]) ** 2
                assert len(deg_list) > 0
            if len(deg_list) > 0:
                assert valid_normals[param_key][img] > 1
                avg_l2 = avg_l2 + (90.0 - deg_list[0]) ** 2
                avg_l1 = avg_l1 + math.fabs(90.0 - deg_list[0])
                count = count + 1
                if shared_at_least_two.__contains__(img):
                    # if out:
                    #     print("img: {}: {}".format(img, deg_list))
                    #     print("normals: {}".format(normals[k][img]))
                    avg_l2_shared = avg_l2_shared + (90.0 - deg_list[0]) ** 2
                    avg_l1_shared = avg_l1_shared + math.fabs(90.0 - deg_list[0])
                    count_shared = count_shared + 1

        if count_valid > 0:
            avg_l1_valid = avg_l1_valid / count_valid
            avg_l2_valid = avg_l2_valid / count_valid
        if count > 0:
            avg_l2 = avg_l2 / count
            avg_l1 = avg_l1 / count
        if count_shared > 0:
            avg_l2_shared = avg_l2_shared / count_shared
            avg_l1_shared = avg_l1_shared / count_shared
        print("L1")
        print("L1: {} {:.3f} {} / {}".format(param_key, avg_l1, count, count_valid))
        # print("L2: {} {:.3f} {} / {}".format(param_key, avg_l2, count, count_valid))
        print("L1:{} - shared: {:.3f}/{} valid: {:.3f}/{}".format(param_key, avg_l1_shared, count_shared, avg_l1_valid, count_valid))
        table += f"{' '.join(param_key.split(','))} {avg_l1_shared} {count_shared} {avg_l1_valid} {count_valid} \n"
        # print("L2:{} - shared: {:.3f}/{} valid: {:.3f}/{}".format(param_key, avg_l2_shared, count_shared, avg_l2_valid, count_valid))
    print(table)


def old_main():

    print("Started")

    parser = argparse.ArgumentParser(prog='evaluation')
    parser.add_argument('--input_dir', help='input dir')
    parser.add_argument('--input_dir2', help='input dir2')
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

    elif args.method == "compare":
        diff_percs = []
        n_worst = None if args.n_worst is None else int(args.n_worst)
        for diff in range(18):
            file_path1 = "{}/stats_diff_{}.pkl".format(args.input_dir, diff)
            file_path2 = "{}/stats_diff_{}.pkl".format(args.input_dir2, diff)
            if os.path.isfile(file_path1) and os.path.isfile(file_path2):
                diff_perc = evaluate_percentage_correct_from_files(file_path1, file_path2, diff, n_worst_examples=n_worst, th_degrees=5)
                diff_percs.append(diff_perc)
            else:
                print("{} or {} not found".format(file_path1, file_path2))
        print("Diff     Perc. 1   Perc. 2")
        for diff, perc1, perc2 in diff_percs:
            print("{}    {}     {}".format(diff, perc1, perc2))

    else:
        diff_percs = []
        n_worst = None if args.n_worst is None else int(args.n_worst)
        for diff in range(18):
            file_path = "{}/stats_diff_{}.pkl_light".format(args.input_dir, diff)
            if os.path.isfile(file_path):
                diff_perc = evaluate_percentage_correct_from_file(file_path, diff, n_worst_examples=n_worst, th_degrees=5)
                diff_percs.append(diff_perc)
            else:
                print("{} not found".format(file_path))

        print("Diff     Perc.")
        for diff, perc in diff_percs:
            print("{}    {}".format(diff, perc))


def quantil_stats():
    file_name = "work/stats_diff_all.pkl"
    file_name = "work/stats_superpoint_all.pkl"
    file_name = "./work/stats_matching_baseline.pkl"; just_root = False
    file_name = "./work/all_stats_old_baseline.pkl"; just_root = True
    file_name = "./persist_do_not_commit/stats_matching_baseline.pkl"; just_root = False
    file_name = "./persist_do_not_commit/stats_diff_all_False.pkl"; just_root = False
    file_name = "./persist_do_not_commit/stats_diff_all_True.pkl"; just_root = False

    with open(file_name, "rb") as f:
        stats_map = pickle.load(f)
        print("data loaded from: {}".format(file_name))

    if just_root:
        stats_map_new = {}
        stats_map_new["default"] = stats_map
        stats_map = stats_map_new

    keys_list = stats_map.keys()

    all_diffs = set()
    for key in keys_list:
        all_diffs = all_diffs.union(set(stats_map[key].keys()))
    all_diffs = list(all_diffs)
    all_diffs.sort()

    for key in keys_list:
        for diff in all_diffs:
            if stats_map[key].__contains__(diff):
                print("key: {}, diff: {}".format(key, diff))
                # TODO #visualization
                #sorted_foo = list(sorted(stats_map[key][diff].items(), key=lambda key_value: -max(len(key_value[1].normals1), len(key_value[1].normals2))))
                #sorted_foo = list(map(lambda kv: (kv[0], [len(kv[1].normals1), len(kv[1].normals2)]), sorted_foo))
                #DEBUG max(sorted_foo[0][1]) > 2
                sorted_by_err_R = list(sorted(stats_map[key][diff].items(), key=lambda key_value: -key_value[1].error_R))
                err_R_only_list = list(map(lambda kv: (kv[0], kv[1].error_R * 180 / math.pi, kv[1]), sorted_by_err_R))
                count = len(err_R_only_list)
                for i, err_R in enumerate(err_R_only_list):
                    o = err_R[2]
                    rich_info = "{:.03f} = {}/{} inliers/tents, all_kps: ({},{}), normals:({},{})".format(o.inliers/o.tentative_matches, o.inliers, o.tentative_matches, o.all_features_1, o.all_features_2, len(o.normals1), len(o.normals2))
                    print("{}: {:.03f}: {:.03f} = {}/{} \t {}".format(err_R[0], err_R[1], float(count-i) / count, count-i, count, rich_info))


if __name__ == "__main__":
    quantil_stats()
