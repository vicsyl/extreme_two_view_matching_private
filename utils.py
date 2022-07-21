import itertools
import torch
import numpy as np
import os
import math
import time
import cv2 as cv
from resize import upsample_bilinear
import torch.nn.functional as F
import kornia.geometry as KG


def parse_device(config):
    dev_s = config.get("device", "cpu")
    if dev_s == "cpu":
        device = torch.device("cpu")
    elif dev_s == "cuda":
        device = torch.device("cuda")
    else:
        raise Exception("Unknown param value for 'device': {}".format(dev_s))
    return device


def use_cuda_from_device(device):
    return device == torch.device("cuda")


class Timer:

    TRANSFORMS_TAG = "transforms_tag"
    NOT_NESTED_TAG = "not_nested"

    log_enabled = False
    start_time = None
    global_start_time = None

    stats_times = {}
    stats_counts = {}
    stats_start_times = {}
    tags_labels_map = {}

    @staticmethod
    def log(message):
        if Timer.log_enabled:
            print(message)

    @staticmethod
    def start():
        Timer.stats_times = {}
        Timer.stats_counts = {}
        Timer.stats_start_times = {}
        Timer.tags_labels_map = {}
        Timer.log("Starting the timer")
        Timer.start_time = time.time()
        if Timer.global_start_time is None:
            Timer.global_start_time = time.time()

    @staticmethod
    def start_check_point(label, parameter=None, tags=None):
        assert label is not None
        Timer.log("{} starting: {}".format(label, parameter))
        start = Timer.stats_start_times.get(label)
        if start is not None:
            Timer.log("WARNING: missing call of end_check_point for label '{}'".format(label))
        Timer.stats_start_times[label] = time.time()
        if tags is not None:
            for tag in tags:
                if not Timer.tags_labels_map.__contains__(tag):
                    Timer.tags_labels_map[tag] = set()
                Timer.tags_labels_map[tag].add(label)

        return label

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
    def log_stats():
        end = time.time()
        config_time = end - Timer.start_time
        print("Time elapsed from start: {:.4f}., ".format(config_time))
        print("Global time elapsed from start: {:.4f}., ".format(end - Timer.global_start_time))
        print("Statistics: ")
        print("key calls avg_time")
        print("all_time 1 {:.4f}".format(config_time))

        def print_keys(tag_param=None):
            keys = list(Timer.stats_times.keys())
            if tag_param is not None:
                keys = list(filter(lambda k: Timer.tags_labels_map[tag_param].__contains__(k), keys))
                if tag_param == "main":
                    main_keys = []
                    keys = keys
                else:
                    main_keys = list(filter(lambda k: Timer.tags_labels_map["main"].__contains__(k), keys))
                    keys = list(filter(lambda k: not Timer.tags_labels_map["main"].__contains__(k), keys))
                main_key_count = None
                if len(main_keys) == 1:
                    main_key_count = Timer.stats_counts[main_keys[0]]
                    main_avg_time = Timer.stats_times[main_keys[0]] / main_key_count
                elif len(main_keys) > 1:
                    raise "too many main keys for {}: {}".format(tag_param, main_keys)
                title = "'{}'".format(tag_param)
            else:
                title = "All"
                main_key_count = None
            print("\n{} keys:".format(title))
            if main_key_count is None:
                sorted_keys = list(sorted(keys, key=lambda key: -Timer.stats_times[key] / Timer.stats_counts[key]))
            else:
                sorted_keys = list(sorted(keys, key=lambda key: -Timer.stats_times[key] / main_key_count))

            sum = 0.0
            for key in sorted_keys:
                counts = Timer.stats_counts[key]
                real_avg = Timer.stats_times[key] / Timer.stats_counts[key]
                if main_key_count is None:
                    print("{} {} {:.4f}".format(key.replace(" ", "_"), counts, real_avg))
                    sum += real_avg
                else:
                    main_avg = Timer.stats_times[key] / main_key_count
                    print("{} {} {:.4f} ({:.4f})".format(key.replace(" ", "_"), counts, main_avg, real_avg))
                    sum += main_avg

            if main_key_count is None:
                print("whole runtime: {:.4f}".format(sum))
            else:
                print("whole runtime: {:.4f}, main key runtime: {:.4f}, difference(remains): {:.4f}".format(sum, main_avg_time, main_avg_time - sum))

        print_keys()
        for tag in Timer.tags_labels_map:
            print_keys(tag)


def timer_label_decorator(label=None, tags=[]):
    def timer_decorator(func):
        label_i = label if label is not None else ""
        label_u = "{}:{}".format(label_i, func.__name__)
        def run(*args, **kwargs):
            Timer.start_check_point(label_u, tags=tags)
            ret = func(*args, **kwargs)
            Timer.end_check_point(label_u)
            return ret
        return run
    return timer_decorator


# TODO option: frame or mask sensitive
def adjust_affine_transform(components_indices, index, A):
    def add_third_row(column_vecs):
        return np.vstack((column_vecs, np.ones(column_vecs.shape[1])))

    coords = np.float32([[0.0, 0.0, 1.0],
                         [components_indices.shape[1] + 1, 0.0, 1.0],
                         [components_indices.shape[1] + 1, components_indices.shape[0] + 1, 1.0],
                         [0.0, components_indices.shape[0] + 1, 1.0]])
    coords = np.transpose(coords)

    new_coords = A @ coords
    new_coords = new_coords / new_coords[2, :]

    min_row = min(new_coords[1])
    max_row = max(new_coords[1])
    min_col = min(new_coords[0])
    max_col = max(new_coords[0])

    dst = np.float32([[min_col, min_row], [min_col, max_row - 1], [max_col - 1, max_row - 1], [max_col - 1, min_row]])
    dst = np.transpose(dst)
    dst = add_third_row(dst)

    print("coords: {}".format(coords))
    print("dst: {}".format(dst))

    translate_vec_new = (-np.min(dst[0]), -np.min(dst[1]))
    translate_matrix_new = np.array([
        [1, 0, translate_vec_new[0]],
        [0, 1, translate_vec_new[1]],
        [0, 0, 1],
    ])

    bounding_box = (math.ceil(max_col - min_col), math.ceil(max_row - min_row))

    A = translate_matrix_new @ A
    return A, bounding_box


def ensure_key(map, key):
    if not map.keys().__contains__(key):
        map[key] = {}


def ensure_keys(map, keys_list):
    for i in range(len(keys_list)):
        ensure_key(map, keys_list[i])
        map = map[keys_list[i]]
    return map


def update_stats_map_static(key_list, obj, map_in):
    if map_in is None:
        return
    map = ensure_keys(map_in, key_list[:-1])
    map[key_list[-1]] = obj


def append_update_stats_map_static(key_list, obj, map_in):
    if map_in is None:
        return
    map = ensure_keys(map_in, key_list[:-1])
    if not map.__contains__(key_list[-1]):
        map[key_list[-1]] = []
    map[key_list[-1]].append(obj)


def is_rectified_condition(img_data):
    return img_data.valid_components_dict is not None


def split_points(tentative_matches, kps1, kps2):
    src_pts = np.float32([kps1[m.queryIdx].pt for m in tentative_matches]).reshape(-1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentative_matches]).reshape(-1, 2)
    return src_pts, dst_pts


def get_normals_stats(img_data_list, src_tentatives_2d, dst_tentatives_2d, mask=None):

    src_kpts_normals = get_kpts_normals(img_data_list[0].components_indices, img_data_list[0].valid_components_dict, src_tentatives_2d)
    dst_kpts_normals = get_kpts_normals(img_data_list[1].components_indices, img_data_list[1].valid_components_dict, dst_tentatives_2d)

    if mask is not None:
        src_kpts_normals = src_kpts_normals[mask]
        dst_kpts_normals = dst_kpts_normals[mask]

    stats = np.vstack((src_kpts_normals, dst_kpts_normals))
    unique, counts = np.unique(stats, axis=1, return_counts=True)
    unique = unique.T
    return stats.T, unique, counts


def get_filter(stats, unique, counts, src_normals: int, dst_normals: int):

    mp = {}
    for i, unq_key in enumerate(unique):
        if unq_key[0] != -1 and unq_key[1] != -1:
            mp[(unq_key[0], unq_key[1])] = counts[i]

    swap = src_normals > dst_normals
    if swap:
        permutation_items = range(src_normals)
        perm_length = dst_normals
    else:
        permutation_items = range(dst_normals)
        perm_length = src_normals

    all_counts = []
    max_count = None
    max_src_i = max_dst_i = None

    for cur_permutation in itertools.permutations(permutation_items, perm_length):
        cur_c = 0
        if swap:
            src_indices = list(cur_permutation)
            dst_indices = list(range(perm_length))
        else:
            src_indices = list(range(perm_length))
            dst_indices = list(cur_permutation)

        for i in range(len(src_indices)):
            c = mp.get((src_indices[i], dst_indices[i]), 0)
            cur_c += c

        all_counts.append(cur_c)
        if max_count is None or cur_c > max_count:
            max_src_i = src_indices
            max_dst_i = dst_indices
            max_count = cur_c

    # print("all counts: {}".format(all_counts))
    # print("max_src_i: {}".format(max_src_i))
    # print("max_dst_i: {}".format(max_dst_i))

    max_set = set()
    for i in range(len(max_src_i)):
        max_set.add((max_src_i[i], max_dst_i[i]))

    return max_set #, stats, fltr


def get_kpts_normals(components_indices, valid_components_dict, kpts_2d):

    kpts_2d = np.array(kpts_2d)

    # 'np.int32' avoids warning
    kpts_ints = np.round(kpts_2d).astype(np.int32)

    np.clip(kpts_ints[:, 0], 0, components_indices.shape[1] - 1, out=kpts_ints[:, 0])
    np.clip(kpts_ints[:, 1], 0, components_indices.shape[0] - 1, out=kpts_ints[:, 1])

    keypoints_components = components_indices[kpts_ints[:, 1], kpts_ints[:, 0]]
    keypoints_normals = np.array([valid_components_dict.get(component, -1) for component in keypoints_components])
    return keypoints_normals


def get_rot_vec_deg(np_r):
    rot_vec = KG.rotation_matrix_to_angle_axis(torch.from_numpy(np_r)[None])[0].numpy()
    return np.rad2deg(rot_vec)


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


def get_rotation_matrix_safe(r):
    r_norm = np.linalg.norm(r)
    if r_norm == 0.0:
        R = np.eye(3)
    else:
        r = r / r_norm
        R = get_rotation_matrix(r, r_norm)
    return R


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


# TODO duplicated - remove
def read_depth_data_from_path(file_path, height=None, width=None, device=torch.device("cpu")):
    depth_data_np = np.load(file_path).astype(np.float64)
    depth_data = torch.from_numpy(depth_data_np).to(device)
    depth_data = depth_data.view(1, 1, depth_data.shape[0], depth_data.shape[1])
    if height is not None and width is not None:
        depth_data = upsample_bilinear(depth_data, height, width)
    return depth_data


class DepthReader:

    def __init__(self, input_dir):
        self.input_dir = input_dir

    @timer_label_decorator("DepthReader")
    def read_depth_data(self, img_name, height=None, width=None, device=torch.device("cpu")):
        depth_data_file_name = "{}.npy".format(img_name)
        file_path = '{}/{}'.format(self.input_dir, depth_data_file_name)
        if not os.path.isfile(file_path):
            raise Exception("ERROR: {} doesn't exist, skipping".format(file_path))
        return DepthReader.read_depth_data_from_path(file_path, height, width, device)

    @staticmethod
    def read_depth_data_from_path(file_path, height=None, width=None, device=torch.device("cpu")):
        depth_data_np = np.load(file_path).astype(np.float64)
        depth_data = torch.from_numpy(depth_data_np).to(device)
        depth_data = depth_data.view(1, 1, depth_data.shape[0], depth_data.shape[1])
        if height is not None and width is not None:
            depth_data = upsample_bilinear(depth_data, height, width)
        return depth_data


# TODO - duplicated - remove
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


if __name__ == "__main__":
    test_quaternions()