import itertools
import math

import numpy as np
import scipy.optimize

from utils import get_rotation_matrix


def get_safe_r(r):
    r_norm = np.linalg.norm(r)
    if r_norm == 0.0:
        R = np.eye(3)
    else:
        r = r / r_norm
        R = get_rotation_matrix(r, r_norm)
    return R


def get_distance(one, two):

    angular_distance = True
    if angular_distance:
        scalar_products = one[:, 0] * two[:, 0] + one[:, 1] * two[:, 1] + one[:, 2] * two[:, 2]
        angular_distances = np.arccos(scalar_products)
        dist = np.sum(angular_distances ** 2)
    else:
        norms_squared = np.linalg.norm(one - two, axis=1) ** 2
        dist = np.sum(norms_squared)

    return dist


def rotation_sort_value(objective_function_value, rotation_vector_norm):
    return math.sqrt(objective_function_value) * 10 + rotation_vector_norm


def find_sorted_rotations(normals1, normals2):
    solutions = find_rotations_info(normals1, normals2)
    solutions.sort(key=lambda r_v_n: rotation_sort_value(r_v_n[1], r_v_n[2]))
    return solutions


def find_rotations_info(normals1, normals2):

    assert len(normals1.shape) == 2
    assert len(normals2.shape) == 2

    swapped = False
    if len(normals1) > len(normals2):
        swapped = True
        normals1, normals2 = normals2, normals1

    len1 = len(normals1)
    len2 = len(normals2)

    solutions = []

    for cur_permutation in itertools.permutations(range(len2), len1):

        def min_function(r):

            R = get_safe_r(r)
            rotated = (R @ normals1.T).T
            target = normals2[list(cur_permutation)]

            return get_distance(rotated, target)

        use_shgo = False
        if use_shgo:
            sf = 1.0
            solution = scipy.optimize.shgo(min_function, [(-math.pi * sf, math.pi * sf), (-math.pi * sf, math.pi * sf), (-math.pi * sf, math.pi * sf)])
            if solution.success:

                for i in range(solution.xl.shape[0]):
                    r_star = solution.xl[i]
                    value = min_function(r_star)

                if swapped:
                    r_star = -r_star
                solutions.append((r_star, value, np.linalg.norm(r_star), list(cur_permutation)))
        else:
            r_star = scipy.optimize.fmin(min_function, np.zeros(3), disp=False)
            value = min_function(r_star)
            if swapped:
                r_star = -r_star
            solutions.append((r_star, value, np.linalg.norm(r_star), list(cur_permutation)))

    return solutions


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

    solutions = find_sorted_rotations(normals1, normals2)

    print("solutions")
    for solution in solutions:
        print("{}\t{}\t{}".format(solution[1], solution[2], solution[0]), solution[3])


if __name__ == "__main__":
    test_rotations()
