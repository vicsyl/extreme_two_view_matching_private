import torch
import numpy as np
import os
from resize import upsample

def get_files(dir, suffix, limit=None):
    filenames = [filename for filename in sorted(os.listdir(dir)) if filename.endswith(suffix)]
    if limit is not None:
        filenames = filenames[0:limit]
    return filenames


def read_depth_data_np(directory, limit=None):

    data_map = {}

    filenames = get_files(directory, ".npy", limit)

    for filename in filenames:
        np_depth = np.load('{}/{}'.format(directory, filename))
        depth_data = torch.from_numpy(np_depth)
        data_map[filename[:-4]] = depth_data

    return data_map


def read_depth_data(filename, directory, height, width):

    depth_data_np = np.load('{}/{}'.format(directory, filename))
    depth_data = torch.from_numpy(depth_data_np)
    depth_data = upsample(depth_data, height, width)
    return depth_data


def quaternions_to_R(qs):

    q00 = qs[0] * qs[0]
    q11 = qs[1] * qs[1]
    q22 = qs[2] * qs[2]
    q33 = qs[3] * qs[3]

    q01 = qs[0] * qs[1]
    q02 = qs[0] * qs[2]
    q03 = qs[0] * qs[3]

    q12 = qs[1] * qs[2]
    q13 = qs[1] * qs[3]

    q23 = qs[2] * qs[3]

    rot_matrix = np.array([
        [2 * (q00 + q11) - 1, 2 * (q12 - q03), 2 * (q13 + q02)],
        [2 * (q12 + q03), 2 * (q00 + q22) - 1, 2 * (q23 - q01)],
        [2 * (q13 - q02), 2 * (q23 + q01), 2 * (q00 + q33) - 1],
    ])

    return rot_matrix


import math

def test_quaternions():

    sqrt_2_d_2 = math.sqrt(2) / 2

    inputs = [
        np.array([1, 0, 0, 0]),
        np.array([sqrt_2_d_2, sqrt_2_d_2, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([sqrt_2_d_2, 0, sqrt_2_d_2, 0]),
        np.array([0, 0, 1, 0]),
        np.array([sqrt_2_d_2, 0, 0, sqrt_2_d_2]),
        np.array([0, 0, 0, 1]),
    ]

    for input in inputs:
        R = quaternions_to_R(input)
        print("input:\n{}".format(input))
        print("R:\n{}".format(R))


if __name__ == "__main__":
    test_quaternions()