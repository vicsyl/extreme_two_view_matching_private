import torch
import numpy as np
import os
import math
import time
import cv2 as cv
from resize import upsample_bilinear


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


def read_depth_data(filename, directory, height=None, width=None):

    depth_data_np = np.load('{}/{}'.format(directory, filename))
    depth_data = torch.from_numpy(depth_data_np)
    depth_data = depth_data.view(1, 1, depth_data.shape[0], depth_data.shape[1])
    if height is not None and width is not None:
        depth_data = upsample_bilinear(depth_data, height, width)
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

    start_time = None

    stats_times = {}
    stats_counts = {}
    stats_start_times = {}

    @staticmethod
    def start():
        print("Starting the timer")
        Timer.start_time = time.time()

    @staticmethod
    def start_check_point(label):
        assert label is not None
        print("{} starting".format(label))
        start = Timer.stats_start_times.get(label)
        if start is not None:
            print("WARNING: missing call of end_check_point for label '{}'".format(label))
        Timer.stats_start_times[label] = time.time()

    @staticmethod
    def end_check_point(label):
        assert label is not None
        end = time.time()
        start = Timer.stats_start_times.get(label)
        if start is None:
            print("WARNING: missing call of start_check_point for label '{}'".format(label))
        else:
            duration = end - start
            print("{} finished. It took {}".format(label, duration))
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
        print("Done. Time elapsed from start: {:.4f}., ".format(end - Timer.start_time))
        print("Statistics: ")
        for key in Timer.stats_times:
            print("{} called {} times and it took {:.4f} secs. on average".format(key, Timer.stats_counts[key], Timer.stats_times[key]/Timer.stats_counts[key]))


if __name__ == "__main__":
    test_quaternions()