import torch
import numpy as np
import os
import math
import time
import cv2 as cv
from resize import upsample_bilinear
import torch.nn.functional as F


def comma_float(f):
    return str("{:.03f}".format(f)).replace(".", ",")


def pad_normals(normals, window_size, mode="replicate"):
    """
    :param normals: (h, w, 3)
    :return:
    """
    normals = normals.unsqueeze(dim=0)
    normals = normals.permute(0, 3, 1, 2)

    pad = (window_size//2, window_size//2, window_size//2, window_size//2)  # pad last dim by 1 on each side
    normals = F.pad(normals, pad, mode=mode)

    normals = normals.squeeze(dim=0)
    normals = normals.permute(1, 2, 0)

    return normals


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

    R = a + b + c
    return R


def get_rotation_matrices(unit_rotation_vector, theta):

    h, w, three = unit_rotation_vector.shape
    h2, w2 = theta.shape
    assert three == 3
    assert h == h2
    assert w == w2

    K = np.zeros((h, w, 3, 3))
    a = np.ndarray((h, w, 3, 3))
    #b = np.ndarray((h, w, 3, 3))
    c = np.ndarray((h, w, 3, 3))

    zer = np.zeros((h, w))

    # Rodrigues formula
    # R = I + sin(theta) . K + (1 - cos(theta)).K**2

    K[:, :, 0, 0] = 0
    K[:, :, 0, 1] = -unit_rotation_vector[:, :, 2]
    K[:, :, 0, 2] = unit_rotation_vector[:, :, 1]
    K[:, :, 1, 0] = unit_rotation_vector[:, :, 2]
    K[:, :, 1, 1] = 0
    K[:, :, 1, 2] = -unit_rotation_vector[:, :, 0]
    K[:, :, 2, 0] = -unit_rotation_vector[:, :, 1]
    K[:, :, 2, 1] = unit_rotation_vector[:, :, 0]
    K[:, :, 2, 2] = 0

    # K[:, :] = np.array([
    #     [zer, -unit_rotation_vector[:, :, 2], unit_rotation_vector[:, :, 1]],
    #     [unit_rotation_vector[:, :, 2], zer, -unit_rotation_vector[:, :, 0]],
    #     [-unit_rotation_vector[:, :, 1], unit_rotation_vector[:, :, 0], zer],
    # ])
    a[:, :] = np.eye(3)
    sins = np.sin(theta[:, :])
    sins = np.expand_dims(sins, axis=(2, 3))
    b = sins * K
    one_m_cos_theta = np.expand_dims(-(np.cos(theta) - 1.0), axis=(2, 3))
    c[:, :] = one_m_cos_theta * K[:, :] @ K[:, :]

    # R = get_rotation_matrix(unit_rotation_vector[0, 0], theta[0, 0])
    # det = np.linalg.det(R)

    # a_0 = a[0, 0]
    # b_0 = b[0, 0]
    # c_0 = c[0, 0]
    # print("{},\n {},\n {}".format(a_0, b_0, c_0))

    Rs = a + b + c
    return Rs


def identity_map(_iterable):
    return {i: i for i in _iterable}


def identity_map_from_range_of_iter(_iterable):
    return identity_map(range(len(_iterable)))


def merge_keys_for_same_value(d: dict):
    inverted_dict = {}
    for k, v in d.items():
        l = inverted_dict.get(v, [])
        l.append(k)
        inverted_dict[v] = l

    merged_dict = {}
    for k, v in inverted_dict.items():
        merged_dict[tuple(v)] = k

    return merged_dict


def get_file_names(dir, suffix, limit=None):
    filenames = [filename for filename in sorted(os.listdir(dir)) if filename.endswith(suffix)]
    filenames = sorted(filenames)
    if limit is not None:
        filenames = filenames[0:limit]
    return filenames


def read_depth_data_np(directory, limit=None):

    data_map = {}

    filenames = get_file_names(directory, ".npy", limit)

    for filename in filenames:
        np_depth = np.load('{}/{}'.format(directory, filename))
        depth_data = torch.from_numpy(np_depth)
        data_map[filename[:-4]] = depth_data

    return data_map


def read_depth_data_from_path(file_path, height=None, width=None, device=torch.device("cpu")):
    depth_data_np = np.load(file_path).astype(np.float64)
    depth_data = torch.from_numpy(depth_data_np).to(device)
    depth_data = depth_data.view(1, 1, depth_data.shape[0], depth_data.shape[1])
    if height is not None and width is not None:
        depth_data = upsample_bilinear(depth_data, height, width)
    return depth_data


def read_depth_data(filename, directory, height=None, width=None, device=torch.device("cpu")):
    file_path = '{}/{}'.format(directory, filename)
    if not os.path.isfile(file_path):
        raise Exception("ERROR: {} doesn't exist, skipping".format(file_path))
    return read_depth_data_from_path(file_path, height, width, device)


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


def save_img_with_timestamp_png(path_prefix, np_img):

    t = time.time()
    # TODO add trailing zeros
    timestamp = str(round(t * 1000) / 1000).replace(".", "_")
    cv.imwrite("{}_{}.png".format(path_prefix, timestamp), np_img)


def save_img_with_timestamp_jpg(path_prefix, np_img):
    t = time.time()
    timestamp = str(round(t * 1000) / 1000).replace(".", "_")
    cv.imwrite("{}_{}.jpg".format(path_prefix, timestamp), np_img)


class Timer:

    log_enabled = False
    start_time = None

    stats_times = {}
    stats_counts = {}
    stats_start_times = {}

    @staticmethod
    def log(message):
        if Timer.log_enabled:
            print(message)

    @staticmethod
    def start():
        Timer.log("Starting the timer")
        Timer.start_time = time.time()

    @staticmethod
    def start_check_point(label, parameter=None):
        assert label is not None
        Timer.log("{} starting: {}".format(label, parameter))
        start = Timer.stats_start_times.get(label)
        if start is not None:
            Timer.log("WARNING: missing call of end_check_point for label '{}'".format(label))
        Timer.stats_start_times[label] = time.time()

    @staticmethod
    def end_check_point(label):
        assert label is not None
        end = time.time()
        start = Timer.stats_start_times.get(label)
        if start is None:
            Timer.log("WARNING: missing call of start_check_point for label '{}'".format(label))
        else:
            duration = end - start
            Timer.log("{} finished. It took {}".format(label, duration))
            Timer.stats_start_times[label] = None
            if Timer.stats_counts.get(label) is None:
                Timer.stats_counts[label] = 0
            Timer.stats_counts[label] += 1
            if Timer.stats_times.get(label) is None:
                Timer.stats_times[label] = 0
            Timer.stats_times[label] += duration

    @staticmethod
    def end():
        end = time.time()
        print("Time elapsed from start: {:.4f}., ".format(end - Timer.start_time))
        print("Statistics: ")
        for key in Timer.stats_times:
            print("{} called {} times and it took {:.4f} secs. on average".format(key, Timer.stats_counts[key], Timer.stats_times[key]/Timer.stats_counts[key]))


if __name__ == "__main__":
    test_quaternions()