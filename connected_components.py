import math
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import glob

import torch

from utils import Timer, identity_map_from_range_of_iter, merge_keys_for_same_value, timer_label_decorator
from img_utils import *
import torch.nn as nn

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


# def get_and_show_components_inner(cluster_indices, valid_component_dict, title=None, normals=None, show=True, save=False, path=None, file_name=None, filter=None, img=None, non_sky_mask=None):
#
#     colors = [
#         [255, 255, 0],
#         [255, 0, 0],
#         [0, 255, 0],
#         [0, 0, 255],
#         [255, 0, 255],
#         [0, 255, 255],
#         [128, 0, 0],
#         [0, 128, 0],
#         [0, 0, 128],
#     ]
#
#     # TODO instead of listing the colors lets add them to the legend
#     color_names = [
#         "red",
#         "green",
#         "blue",
#         "yellow",
#         "magenta",
#         "cyan",
#         "maroon",
#         "dark green",
#         "navy"
#     ]
#
#     if img is None:
#         cluster_colors = np.zeros((cluster_indices.shape[0], cluster_indices.shape[1], 3), dtype=np.int32)
#
#         if non_sky_mask is not None and filter.__contains__(-1):
#             cluster_colors[non_sky_mask == 1] = [0, 0, 128]
#         else:
#             for i, c_index in enumerate(valid_component_dict.keys()):
#                 if filter is None or filter.__contains__(c_index):
#                     cluster_colors[cluster_indices == c_index] = colors[i % 9]
#
#         mask = 1 - np_rgb_mask(cluster_colors)
#         indices = np.where(mask)
#         min0 = indices[0].min()
#         max0 = indices[0].max() + 1
#         min1 = indices[1].min()
#         max1 = indices[1].max() + 1
#         cropped = np.zeros((max0 - min0, max1 - min1, 3))
#         cropped[:, :] = cluster_colors[min0:max0, min1:max1]
#         cluster_colors = cropped
#
#     else:
#         cluster_colors = np.copy(img)
#         for i, c_index in enumerate(valid_component_dict.keys()):
#             if filter is None or filter.__contains__(c_index):
#                 for rgb_index in range(3):
#                     if colors[i % 9][rgb_index] == 0:
#                         cluster_colors[cluster_indices == c_index, rgb_index] = 0
#
#         if non_sky_mask is not None:
#             cluster_colors[non_sky_mask == 0, :2] = 0
#             cluster_colors[non_sky_mask == 0, 2] = 128 # cluster_colors[non_sky_mask == 0, 2] / 2
#
#     if show or save:
#
#         if img is not None:
#             plt.figure()
#             plt.imshow(img)
#             plt.show(block=False)
#
#         if title is None:
#             title = "{} - (connected) components: \n".format(file_name)
#             new_component_dict = {}
#             for i, c_index in enumerate(valid_component_dict.keys()):
#                 new_component_dict[i] = valid_component_dict[c_index]
#             merged_dict = merge_keys_for_same_value(new_component_dict)
#             for merged_values in merged_dict:
#                 cur_colors_names = ", ".join([color_names[val % 9] for val in merged_values])
#                 if normals is not None:
#                     title = "{}[{}]={}={},\n".format(title, cur_colors_names, normals[merged_dict[merged_values]],
#                                                      merged_dict[merged_values])
#                 else:
#                     title = "{}[{}]={},\n".format(title, cur_colors_names, merged_dict[merged_values])
#
#         create_plot_only_img(title, cluster_colors, 10, transparent=img is None)
#
#         save = True
#         if save:
#             # plt.savefig("foo.png", dpi=24, transparent=True)
#             # 'work/pipeline_scene1_333/imgs//frame_0000001350_2_cluster_connected_components'
#             plt.savefig(path, dpi=24, transparent=True, facecolor=(0.0, 0.0, 0.0, 0.0))
#             # plt.savefig(path, dpi=24, transparent=True)
#         show_or_close(show)
#
#     return cluster_colors


@timer_label_decorator()
def get_and_show_components(cluster_indices, valid_component_dict, title=None, normals=None, show=True, save=False, path=None, file_name=None):

    if not show and not save:
        return

    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 0, 0], # [255, 0, 255],
        [0, 255, 255],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
    ]

    # TODO instead of listing the colors let's add them to the legend
    color_names = [
        "red",
        "green",
        "blue",
        "yellow",
        "black", #"magenta",
        "cyan",
        "maroon",
        "dark green",
        "navy"
    ]

    cluster_colors = np.zeros((cluster_indices.shape[0], cluster_indices.shape[1], 3), dtype=np.int32)
    for i, c_index in enumerate(valid_component_dict.keys()):
        cluster_colors[np.where(cluster_indices == c_index)] = colors[i % 9]

    # TODO clean up
    size_h = 10
    fig = plt.figure(figsize=(size_h, size_h * cluster_colors.shape[0] / cluster_colors.shape[1]))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if title is not None:
        plt.title(title)
    # TODO this was commented out to suppress setting the title...
    else:
        title = "{} - (connected) components: \n".format(file_name)
        new_component_dict = {}
        for i, c_index in enumerate(valid_component_dict.keys()):
            new_component_dict[i] = valid_component_dict[c_index]
        merged_dict = merge_keys_for_same_value(new_component_dict)
        for merged_values in merged_dict:
            cur_colors_names = ", ".join([color_names[val % 9] for val in merged_values])
            if normals is not None:
                title = "{}[{}]={}={},\n".format(title, cur_colors_names, normals[merged_dict[merged_values]], merged_dict[merged_values])
            else:
                title = "{}[{}]={},\n".format(title, cur_colors_names, merged_dict[merged_values])

        plt.title(title)

    plt.imshow(cluster_colors)
    if save:
        # FIXME
        full_path = "{}_{}".format(path, file_name)
        print("saving to {}".format(full_path))
        plt.savefig(full_path)

    show_or_close(show)


# def get_and_show_components_new(cluster_indices, valid_component_dict, title=None, normals=None, show=True, save=False, path=None, file_name=None, iterate_through_all=False, img=None, non_sky_mask=None):
#
#     def non_valid_mask(cluster_indices, valid_component_dict):
#         print("non_valid_mask cluster_indices.shape: {}".format(cluster_indices.shape))
#
#         mask = np.zeros(cluster_indices.shape)
#         for k in valid_component_dict.keys():
#             mask = np.logical_or(mask, cluster_indices == k)
#         ret = 1 - mask
#         print("non_valid_mask shape: {}".format(ret.shape))
#         print("non_valid_mask sum: {}".format(ret.sum()))
#         return ret
#
#     magic = 1000000
#     mask = non_valid_mask(cluster_indices, valid_component_dict)
#     cluster_indices[mask] = magic
#     #valid_component_dict[magic] = magic
#
#     upsample = nn.Upsample(size=cluster_indices.shape, mode='nearest')
#     t = torch.from_numpy(non_sky_mask)
#     non_sky_mask = upsample(t[None, None].to(float))[0, 0].to(bool).numpy()
#
#     cluster_colors = get_and_show_components_inner(cluster_indices, valid_component_dict, title, normals, show, save, path, file_name, filter=None, img=img, non_sky_mask=non_sky_mask)
#     iterate_through_all = True
#     if iterate_through_all:
#         for k in valid_component_dict.keys():
#             get_and_show_components_inner(cluster_indices, valid_component_dict, title, normals, show, save, "{}_{}".format(path, k), file_name, filter={k}, img=None, non_sky_mask=non_sky_mask)
#         get_and_show_components_inner(cluster_indices, valid_component_dict, title, normals, show, save, "{}_{}".format(path, "sky"), file_name, filter={-1}, img=None, non_sky_mask=1-non_sky_mask)
#     return cluster_colors
#
#
def circle_like_ones(size):
    ret = np.ones((size, size), np.uint8)
    r_check = (size / 2 - 0.4) ** 2
    for i in range(size):
        for j in range(size):
            r = (size / 2 - (i + 0.5)) ** 2 + (size / 2 - (j + 0.5)) ** 2
            if r > r_check:
                ret[i, j] = 0
    return ret


def flood_fill(input_img):

    flood_filled = input_img.copy()
    flood_filled[0, :] = 0
    flood_filled[flood_filled.shape[0] - 1, :] = 0
    flood_filled[:, flood_filled.shape[1] - 1] = 0
    flood_filled[:, 0] = 0

    mask = np.zeros((flood_filled.shape[0] + 2, flood_filled.shape[1] + 2), np.uint8)
    cv.floodFill(flood_filled, mask, (0, 0), 2)
    flood_filled = np.where(flood_filled == 2, 0, 1).astype(dtype=np.uint8)
    flood_filled = flood_filled | input_img
    return flood_filled


@timer_label_decorator()
def get_connected_components(normal_indices, valid_indices, show=False,
                             fraction_threshold=0.03, closing_size=None, flood_filling=False, connectivity=4):

    component_size_threshold = normal_indices.shape[0] * normal_indices.shape[1] * fraction_threshold

    out = np.zeros((normal_indices.shape[0], normal_indices.shape[1]), dtype=np.int32)
    out_valid_indices_dict = {}
    out_valid_indices_counter = 0

    for v_i in valid_indices:
        input = np.where(normal_indices == v_i, 1, 0).astype(dtype=np.uint8)

        assert closing_size is None
        if closing_size is not None:
            kernel = circle_like_ones(size=closing_size) # np.ones((closing_size, closing_size) np.uint8)
            input = cv.morphologyEx(input, cv.MORPH_CLOSE, kernel)

        assert not flood_filling
        if flood_filling:
            input = flood_fill(input)

        _, labels = cv.connectedComponents(input, connectivity=connectivity)

        unique, counts = np.unique(labels, return_counts=True)
        valid_labels = np.where(counts > component_size_threshold)[0]
        # Docs: RETURNS: The sorted unique values. - see https://numpy.org/doc/stable/reference/generated/numpy.unique.html
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
            # NOTE not very revealing btw.
            get_and_show_components(out, out_valid_indices_dict, "out after normal index={}".format(v_i))

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
        get_and_show_components(normal_indices, range(len(normals)))

    Timer.start_check_point("get_connected_components")
    clusters, valid_components_dict = get_connected_components(normal_indices, identity_map_from_range_of_iter(normals), True)
    print("valid components mapping: {}".format(valid_components_dict))
    Timer.end_check_point("get_connected_components")


if __name__ == "__main__":

    Timer.start()

    interesting_dirs = ["frame_0000000145_2"]
    find_and_show_clusters("work/scene1/normals/simple_diff_mask_sigma_5", limit=1, interesting_dirs=interesting_dirs)

    Timer.log_stats()
