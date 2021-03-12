import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import time
from utils import Timer


# original_input_dir - to scene info
def read_img_normals_info(parent_dir, img_name_dir):

    if not os.path.isdir("{}/{}".format(parent_dir, img_name_dir)):
        return None, None

    paths_png = glob.glob("{}/{}/*.png".format(parent_dir, img_name_dir))
    paths_txt = glob.glob("{}/{}/*.txt".format(parent_dir, img_name_dir))

    if paths_png is None or paths_txt is None:
        print(".txt or .png file doesn't exist in {}!".format(img_name_dir))
        raise

    normals = np.loadtxt(paths_txt[0], delimiter=',')
    greyscale_const = 0
    normal_indices = cv.imread(paths_png[0], greyscale_const)
    return normals, normal_indices


def show_components(cluster_indices, valid_indices, title=None):

    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [127, 0, 0],
        [0, 127, 0],
        [0, 0, 127],
    ]

    cluster_colors = np.zeros((cluster_indices.shape[0], cluster_indices.shape[1], 3), dtype=np.int32)
    for i, c_index in enumerate(valid_indices):
        #print("i, colors[c_index % 3]: {}".format(c_index, colors[c_index % 9]))
        cluster_colors[np.where(cluster_indices == c_index)] = colors[i % 9]

    if title is not None:
        plt.title(title)
    plt.imshow(cluster_colors)
    plt.show()
    return cluster_colors


def get_connected_components(normal_indices, valid_indices, show=False, fraction_threshold=0.03):

    component_size_threshold = normal_indices.shape[0] * normal_indices.shape[1] * fraction_threshold

    out = np.zeros((normal_indices.shape[0], normal_indices.shape[1]), dtype=np.int32)
    out_valid_indices = {}
    out_valid_indices_counter = 0

    for v_i in valid_indices:
        input = np.where(normal_indices == v_i, 1, 0).astype(dtype=np.uint8)
        # if show:
        #     show_components(input, [1], "input for {}".format(v_i))
        ret, labels = cv.connectedComponents(input, connectivity=4)
        unique, counts = np.unique(labels, return_counts=True)
        valid_labels = np.where(counts > component_size_threshold)[0]
        if valid_labels[0] == 0:
            valid_labels = valid_labels[1:]
        if len(valid_labels) != 0:
            max_valid_labels = np.max(valid_labels)
            valid_labels = valid_labels + out_valid_indices_counter
            labels = labels + out_valid_indices_counter

            for v_i_i in valid_labels:
                out = np.where(labels == v_i_i, labels, out)

            out_valid_indices.update({v_i_i: v_i for v_i_i in valid_labels})
            out_valid_indices_counter = out_valid_indices_counter + max_valid_labels

    if show:
        show_components(out, out_valid_indices.keys(), "out after v_i={}".format(v_i))

    return out, out_valid_indices


def get_connected_components_own_impl(normal_indices, dummy_valid_indices=None, show=False, fraction_threshold=0.03):
    # NOTE 'dummy_valid_indices' is there only not to clash with the other implementation (so that fraction_threshold is not given the value of True)

    visited = np.ndarray(normal_indices.shape, dtype=bool)
    visited[:] = False

    clusters = np.ndarray(normal_indices.shape, dtype=np.int32)
    cluster_counter = 1

    component_size_threshold = normal_indices.shape[0] * normal_indices.shape[1] * fraction_threshold
    valid_components = {}

    is_free = True
    while is_free:
        is_free = False
        for x in range(visited.shape[0]):
            for y in range(visited.shape[1]):
                if not visited[x, y]:
                    if normal_indices[x, y] == 3: # background
                        visited[x, y] = True
                    else:
                        is_free = True
                        normal_index = normal_indices[x, y]
                        component_size = get_connected_components_for_index(normal_indices, clusters, visited, cluster_counter, x, y, normal_index)
                        print("found {} with size of {}".format(cluster_counter, component_size))
                        if component_size >= component_size_threshold:
                            valid_components[cluster_counter] = normal_index
                        cluster_counter += 1

    # print("all components: {}".format(cluster_counter))
    # print("valid components".format(len(valid_components)))

    if show:
        show_components(clusters, valid_components.keys())

    return clusters, valid_components


def get_connected_components_for_index(normal_indices, clusters, visited, component_index, x, y, normal_index):

    point_counter = 0

    offsets = [
        [ 1,  0],
        [ 0, -1],
        [-1,  0],
        [ 0,  1],
    ]

    to_be_visited = [(x, y)]

    while len(to_be_visited) != 0:
        visiting = to_be_visited[0]
        to_be_visited = to_be_visited[1:]
        x = visiting[0]
        y = visiting[1]
        visited[x, y] = True
        clusters[x, y] = component_index
        point_counter += 1

        for offset in offsets:
            new_x = x + offset[0]
            new_y = y + offset[1]
            if 0 <= new_x < normal_indices.shape[0] and 0 <= new_y < normal_indices.shape[1] and not visited[new_x][new_y] and normal_indices[new_x, new_y] == normal_index:
                visited[new_x, new_y] = True
                to_be_visited.append((new_x, new_y))

    return point_counter


def find_and_show_clusters(parent_dir, limit, interesting_dirs=None):

    if interesting_dirs is not None:
        dirs = interesting_dirs
    else:
        dirs = [dirname for dirname in sorted(os.listdir(parent_dir)) if os.path.isdir("{}/{}".format(parent_dir, dirname))]
        dirs = sorted(dirs)
        if limit is not None:
            dirs = dirs[0:limit]

    for img_name in dirs:

        print("Reading normals for img: {}".format(img_name))
        normals, normal_indices = read_img_normals_info(parent_dir, img_name)
        show_components(normal_indices, range(len(normals)))

    Timer.start_check_point("get_connected_components")
    clusters, valid_components_dict = get_connected_components(normal_indices, range(len(normals)), True)
    print("valid components mapping: {}".format(valid_components_dict))
    Timer.end_check_point("get_connected_components")


if __name__ == "__main__":

    Timer.start()

    interesting_dirs = ["frame_0000000145_2"]
    find_and_show_clusters("work/scene1/normals/simple_diff_mask_sigma_5", limit=1, interesting_dirs=interesting_dirs)

    Timer.end()
