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


def get_depth_data_file_names(directory, limit=None):
    return [filename for filename in get_files(directory, ".npy", limit)]


def read_depth_data(filename, directory, height, width):

    depth_data_np = np.load('{}/{}'.format(directory, filename))
    depth_data = torch.from_numpy(depth_data_np)
    depth_data = upsample(depth_data, height, width)
    return depth_data
