import itertools
import math

import kornia.geometry as KG
import numpy as np
import scipy.optimize

from evaluation import *
from scene_info import SceneInfo
from utils import get_rotation_matrix_safe


@dataclass
class RotationSolution:

    objective_fnc: float
    rotation_vector_normal: float
    rotation_vector: np.ndarray
    permutation: list


def get_distance(one, two, angular_distance=True, squared=True):

    if angular_distance:
        scalar_products = one[:, 0] * two[:, 0] + one[:, 1] * two[:, 1] + one[:, 2] * two[:, 2]
        scalar_products = np.clip(scalar_products, a_min=-1.0, a_max=1.0)
        angular_distances = np.arccos(scalar_products)
        if squared:
            dist = np.sum(angular_distances ** 2) / angular_distances.shape[0]
        else:
            dist = np.sum(np.abs(angular_distances)) / angular_distances.shape[0]
    else:
        norms = np.linalg.norm(one - two, axis=1)
        if squared:
            dist = np.sum(norms ** 2) / norms.shape[0]
        else:
            dist = np.sum(np.abs(norms)) / norms.shape[0]

    return dist


def rotation_sort_value(solution: RotationSolution):
    return math.sqrt(solution.objective_fnc) * 10 + solution.rotation_vector_normal


def find_sorted_rotations(normals1, normals2, zero_around_z):

    assert len(normals1.shape) == 2
    assert len(normals2.shape) == 2

    assert np.all(np.abs(np.linalg.norm(normals1, axis=1) - 1.0) < 10 ** -5)
    assert np.all(np.abs(np.linalg.norm(normals2, axis=1) - 1.0) < 10 ** -5)

    solutions = find_rotations_info(normals1, normals2, zero_around_z)
    solutions.sort(key=lambda solution: rotation_sort_value(solution))
    return solutions


class ObjFunction:

    def __init__(self, src_normals, dst_normals, zero_around_z):
        self.src_normals = src_normals
        self.dst_normals = dst_normals
        self.zero_around_z = zero_around_z

    def function_value(self, r):

        if self.zero_around_z:
            r = np.array([r[0], r[1], 0.0])

        R = get_rotation_matrix_safe(r)
        rotated = (R @ self.src_normals.T).T

        return get_distance(rotated, self.dst_normals)


def find_rotations_info(normals1, normals2, zero_around_z):

    swapped = False
    if len(normals1) > len(normals2):
        swapped = True
        normals1, normals2 = normals2, normals1

    len1 = len(normals1)
    len2 = len(normals2)

    solutions = []

    for cur_permutation in itertools.permutations(range(len2), len1):

        def min_function(r):
            target = normals2[list(cur_permutation)]
            return ObjFunction(normals1, target, zero_around_z).function_value(r)

        use_shgo = False
        if use_shgo:
            sf = 1.0
            if zero_around_z:
                bounds = [(-math.pi * sf, math.pi * sf), (-math.pi * sf, math.pi * sf)]
            else:
                bounds = [(-math.pi * sf, math.pi * sf), (-math.pi * sf, math.pi * sf), (-math.pi * sf, math.pi * sf)]
            solution = scipy.optimize.shgo(min_function, bounds)
            if solution.success:

                for i in range(solution.xl.shape[0]):
                    r_star = solution.xl[i]
                    minimazer = ObjFunction()
                    value = min_function(r_star)

                if swapped:
                    r_star = -r_star
                if zero_around_z:
                    r_star = np.array([r_star[0], r_star[1], 0])
                solutions.append(RotationSolution(rotation_vector=r_star, objective_fnc=value, rotation_vector_normal=np.linalg.norm(r_star), permutation=list(cur_permutation)))
        else:
            if zero_around_z:
                initial = np.zeros(2)
            else:
                initial = np.zeros(3)
            r_star = scipy.optimize.fmin(min_function, initial, disp=False)
            value = min_function(r_star)
            if swapped:
                r_star = -r_star
            if zero_around_z:
                r_star = np.array([r_star[0], r_star[1], 0])
            solutions.append(RotationSolution(rotation_vector=r_star, objective_fnc=value, rotation_vector_normal=np.linalg.norm(r_star), permutation=list(cur_permutation)))

    return solutions


# section - testing


def test_rotations():
    # experiment with sf = 1.0 x sf = 3.0
    # 0.29980072084789616	3.9500983239989957	[-1.69245218 -2.52898877 -2.51855081] [1, 2, 0] (sf=1.0)
    # 0.3022565769340434	14.892615668454276	[6.64357752 9.42477796 9.42477796] [1, 2, 0] (sf=3.0
    # normals1 = np.array([
    #     [1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.007, 0.007, 1.0],
    # ])
    #
    # normals2 = np.array([
    #     [0.71, 0.71, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.001, 1.0],
    # ])


    # dist = np.sum(angular_distances ** 2) vs.  dist = np.sum(angular_distances) affects the result drastically!
    normals1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])

    # normals2 = np.array([
    #     [1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.0, 1.0],
    # ])

    normals2 = np.array([
        [0.71, 0.71, 0.0],
        [-0.71, 0.71, 0.0],
        [0.0, 0.0, 1.0],
    ])

    normals1 = normals1 / np.expand_dims(np.linalg.norm(normals1, axis=1), axis=1)
    normals2 = normals2 / np.expand_dims(np.linalg.norm(normals2, axis=1), axis=1)

    solutions = find_sorted_rotations(normals1, normals2)

    print("solutions")
    for solution in solutions:
        print("{}\t{}\t{}".format(solution.objective_fnc, solution.rotation_vector_normal, solution.rotation_vector, solution.permutation))


def possibly_expand_normals(normals):
    if len(normals.shape) == 1:
        normals = np.expand_dims(normals, axis=0)
    return normals


def single_function_value_for_GT(GT_r, normals1, normals2, zero_around_z):

    len1 = len(normals1)
    len2 = len(normals2)

    min = None
    arg_min = None

    for i in range(len1):
        for j in range(len2):
            value = ObjFunction(normals1[i:i + 1], normals2[j:j + 1], zero_around_z).function_value(GT_r)
            if min is None or value < min:
                min = value
                arg_min = (i, j)

    return min, arg_min


def function_value_for_GT(GT_r, normals1, normals2, zero_around_z):

    swapped = False
    if len(normals1) > len(normals2):
        swapped = True
        normals1, normals2 = normals2, normals1

    len1 = len(normals1)
    len2 = len(normals2)

    min = None
    arg_min = None

    for cur_permutation in itertools.permutations(range(len2), len1):
        target = normals2[list(cur_permutation)]
        if swapped:
            src, dst = target, normals1
        else:
            src, dst = normals1, target

        value = ObjFunction(src, dst, zero_around_z).function_value(GT_r)
        if min is None or value < min:
            min = value
            arg_min = list(cur_permutation)

    return min, arg_min


class MC:
    gt = "GT"
    best = "best"
    first = "first"

    permutation = "permutation"
    f_value = "f_value"
    single_permutation = "single_permutation"
    single_f_value = "single_f_value"
    angle_to_GT_I = "angle_to_GT_I"
    rot_vector = "rot_vector"


def analyze_rotation_via_normals(normals1, normals2, img_pair, scene_info):

    ret = {}

    normals1 = possibly_expand_normals(normals1)
    normals2 = possibly_expand_normals(normals2)
    #print("normals counts: ({}, {})".format(normals1.shape[0], normals2.shape[0]))

    GT_angle = compare_R_to_GT(img_pair, scene_info, np.eye(3)) * 180 / math.pi
    GT_mat, _ = get_GT_R_t(img_pair, scene_info)
    GT_vec = KG.rotation_matrix_to_angle_axis(torch.from_numpy(GT_mat)[None])[0].numpy()
    GT_vec_deg = np.rad2deg(GT_vec)
    f_value, f_arg = function_value_for_GT(GT_vec, normals1, normals2, zero_around_z=False)
    single_f_value, single_f_arg = single_function_value_for_GT(GT_vec, normals1, normals2, zero_around_z=False)

    # print("GT angle: {} degrees".format(GT_angle))
    # print("GT rotation vector(deg): {}".format(GT_vec_deg))
    # print("GT rotation vector normal: {}".format(np.linalg.norm(GT_vec)))
    # print("GT f value for {}: {}".format(f_arg, f_value))

    ret[MC.gt] = {}
    ret[MC.gt][MC.angle_to_GT_I] = GT_angle
    ret[MC.gt][MC.rot_vector] = GT_vec_deg
    ret[MC.gt][MC.permutation] = f_arg
    ret[MC.gt][MC.f_value] = f_value
    ret[MC.gt][MC.single_permutation] = single_f_arg
    ret[MC.gt][MC.single_f_value] = single_f_value

    def analyze_solution(label, solution, map):

        r_vec_first = solution.rotation_vector
        r_matrix_first = get_rotation_matrix_safe(r_vec_first)
        r_vec_first_deg = np.rad2deg(r_vec_first)
        first_err = compare_R_to_GT(img_pair, scene_info, r_matrix_first)
        first_err = np.rad2deg(first_err)

        # print("{} solution rotation angle relative to GT: {} degrees".format(label, first_err))
        # print("{} solution rotation vector (deg): {}".format(label, r_vec_first_deg))
        # print("{} solution norm: {}".format(label, solution.rotation_vector_normal))
        # print("{} solution permutation: {}".format(label, solution.permutation))
        # print("{} solution objective function value: {}".format(label, solution.objective_fnc))

        map[MC.angle_to_GT_I] = first_err
        map[MC.rot_vector] = r_vec_first_deg
        map[MC.permutation] = solution.permutation
        map[MC.f_value] = solution.objective_fnc

    solutions = find_sorted_rotations(normals1, normals2, zero_around_z=False)

    ret[MC.first] = {}
    analyze_solution("1st", solutions[0], ret[MC.first])

    min = None
    arg_min = None
    for idx, solution in enumerate(solutions):
        r = get_rotation_matrix_safe(solution.rotation_vector)
        err_q = compare_R_to_GT(img_pair, scene_info, r)
        if min is None or err_q < min:
            min = err_q
            arg_min = idx

    ret[MC.best] = {}
    analyze_solution("{}th".format(arg_min), solutions[arg_min], ret[MC.best])

    return ret


def analyze_data():

    # CONTINUE: even single GT values are pretty high -> investigate!!
    # also just try to investigate around the alpha (0.5) - maybe combination with the normals (and ortogonality with z-axis)

    fn = "work/stats_baseline.pkl"

    scene_info = SceneInfo.read_scene("scene1")

    with open(fn, "rb") as f:
        print("reading: {}".format(fn))
        stats_map = pickle.load(f)
        print()

    normals = stats_map['normals']['fginn_False_n_features_None_use_hardnet_False']

    stats = {}
    stats["avg_G_angle"] = {}
    stats["avg_G_value"] = {}
    stats["avg_single_G_value"] = {}
    stats["avg_best_G_angle"] = {}
    stats["avg_best_G_value"] = {}

    print("avg_G_angle, avg_G_value, avg_single_G_value, avg_best_G_angle, avg_best_G_value")
    for difficulty in range(10):
        #print("Diff: {}".format(difficulty))
        stats[difficulty] = {}
        stats["avg_G_angle"][difficulty] = 0.0
        stats["avg_G_value"][difficulty] = 0.0
        stats["avg_single_G_value"][difficulty] = 0.0
        stats["avg_best_G_angle"][difficulty] = 0.0
        stats["avg_best_G_value"][difficulty] = 0.0
        for img_pair in scene_info.img_pairs_lists[difficulty]:
            key = scene_info.get_key(img_pair.img1, img_pair.img2)
            normals1 = normals[img_pair.img1]
            normals2 = normals[img_pair.img2]
            stats[difficulty][key] = analyze_rotation_via_normals(normals1, normals2, img_pair, scene_info)
            stats["avg_G_angle"][difficulty] += stats[difficulty][key][MC.gt][MC.angle_to_GT_I]
            stats["avg_G_value"][difficulty] += stats[difficulty][key][MC.gt][MC.f_value]
            stats["avg_single_G_value"][difficulty] += stats[difficulty][key][MC.gt][MC.single_f_value]
            stats["avg_best_G_angle"][difficulty] += stats[difficulty][key][MC.best][MC.angle_to_GT_I]
            stats["avg_best_G_value"][difficulty] += stats[difficulty][key][MC.best][MC.f_value]

        for stat_key in ["avg_G_angle", "avg_G_value", "avg_single_G_value", "avg_best_G_angle", "avg_best_G_value"]:
            stats[stat_key][difficulty] /= len(scene_info.img_pairs_lists[difficulty])
        #for difficulty in range(18):
        print("{}: {}\t{}\t{}\t{}\t{}".format(difficulty,
                                          stats["avg_G_angle"][difficulty],
                                          stats["avg_G_value"][difficulty],
                                          stats["avg_single_G_value"][difficulty],
                                          stats["avg_best_G_angle"][difficulty],
                                          stats["avg_best_G_value"][difficulty]))


if __name__ == "__main__":
    #test_rotations()
    analyze_data()
