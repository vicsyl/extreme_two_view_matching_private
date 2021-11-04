import itertools
import math

import numpy as np
import scipy.optimize

from utils import get_rotation_matrix


def get_safe_R(r):
    r_norm = np.linalg.norm(r)
    if r_norm == 0.0:
        R = np.eye(3)
    else:
        r = r / r_norm
        R = get_rotation_matrix(r, r_norm)
    return R


def get_distance(one, two):
    norms_squared = np.linalg.norm(one - two, axis=1) ** 2
    dist = np.sum(norms_squared)
    return dist


def find_rotations(normals1, normals2):

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

            R = get_safe_R(r)
            rotated = (R @ normals1.T).T
            target = normals2[list(cur_permutation)]

            return get_distance(rotated, target)

        r_star = scipy.optimize.fmin(min_function, np.zeros(3))
        value = min_function(r_star)
        if swapped:
            r_star = -r_star
        solutions.append((get_safe_R(r_star), value, np.linalg.norm(r_star)))

    solutions.sort(key=lambda r_v_n: math.sqrt(r_v_n[1]) * 10 + r_v_n[2])

    return solutions


def test_rotations():

    normals1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])

    normals2 = np.array([
        [0.71, 0.71, 0.0],
        [-0.71, 0.71, 0.0],
        [0.0, 0.0, 1.0],
    ])

    solutions = find_rotations(normals1, normals2)
    print("solutions")
    for solution in solutions:
        print("{}\t{}".format(solution[1], solution[2]))


if __name__ == "__main__":
    test_rotations()
