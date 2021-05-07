import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import glob
from utils import Timer, identity_map_from_range_of_iter, merge_keys_for_same_value

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


def show_components(cluster_indices, valid_component_dict, title=None, normals=None, show=True):

    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
    ]

    color_names = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "maroon",
        "dark green",
        "navy"
    ]

    cluster_colors = np.zeros((cluster_indices.shape[0], cluster_indices.shape[1], 3), dtype=np.int32)
    for i, c_index in enumerate(valid_component_dict.keys()):
        cluster_colors[np.where(cluster_indices == c_index)] = colors[i % 9]

    if show:
        if title is not None:
            plt.title(title)
        elif normals is not None:
            desc = ""
            new_component_dict = {}
            for i, c_index in enumerate(valid_component_dict.keys()):
                new_component_dict[i] = valid_component_dict[c_index]
            merged_dict = merge_keys_for_same_value(new_component_dict)
            for merged_values in merged_dict:
                cur_colors_names = ", ".join([color_names[val % 9] for val in merged_values])
                desc = "{}[{}]={},\n".format(desc, cur_colors_names, normals[merged_dict[merged_values]])
            plt.title(desc)
        plt.imshow(cluster_colors)
        plt.show(block=False)
    return cluster_colors


def get_connected_components(normal_indices, valid_indices, show=False, fraction_threshold=0.03):

    component_size_threshold = normal_indices.shape[0] * normal_indices.shape[1] * fraction_threshold

    out = np.zeros((normal_indices.shape[0], normal_indices.shape[1]), dtype=np.int32)
    out_valid_indices_dict = {}
    out_valid_indices_counter = 0

    for v_i in valid_indices:
        input = np.where(normal_indices == v_i, 1, 0).astype(dtype=np.uint8)
        ret, labels = cv.connectedComponents(input, connectivity=4)
        unique, counts = np.unique(labels, return_counts=True)
        valid_labels = np.where(counts > component_size_threshold)[0]
        # TODO index of? - anyway the goal is to filter out label value of 0
        if valid_labels[0] == 0:
            valid_labels = valid_labels[1:]
        if len(valid_labels) != 0:
            max_valid_labels = np.max(valid_labels)
            valid_labels = valid_labels + out_valid_indices_counter
            labels = labels + out_valid_indices_counter

            for v_i_i in valid_labels:
                out = np.where(labels == v_i_i, labels, out)

            out_valid_indices_dict.update({v_i_i: v_i for v_i_i in valid_labels})
            out_valid_indices_counter = out_valid_indices_counter + max_valid_labels

        if show:
            show_components(out, out_valid_indices_dict, "out after normal index={}".format(v_i))

    return out, out_valid_indices_dict


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
    clusters, valid_components_dict = get_connected_components(normal_indices, identity_map_from_range_of_iter(normals), True)
    print("valid components mapping: {}".format(valid_components_dict))
    Timer.end_check_point("get_connected_components")


if __name__ == "__main__":

    Timer.start()

    interesting_dirs = ["frame_0000000145_2"]
    find_and_show_clusters("work/scene1/normals/simple_diff_mask_sigma_5", limit=1, interesting_dirs=interesting_dirs)

    Timer.end()
