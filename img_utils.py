import imp
import math
import time

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def np_rgb_mask(np_img):
    mask = np.logical_and(np_img[:, :, 0] == 0, np_img[:, :, 1] == 0)
    mask = np.logical_and(mask, np_img[:, :, 2] == 0)
    return mask


# TODO definitely clean this up!!
# TODO the title doesn't really work!!
def create_plot_only_img(title, img_np, h_size_inches=10, transparent=False, show_axis=False):

    fig = plt.figure(frameon=False, figsize=(h_size_inches, h_size_inches * img_np.shape[0] / img_np.shape[1]))

    if transparent:
        data_to_show = np.ones((*img_np.shape[:2], 4), dtype=int) * 255
        data_to_show[:, :, :3] = img_np
        mask = np_rgb_mask(data_to_show)
        data_to_show[mask, 3] = 0
    else:
        data_to_show = img_np

    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0.0)
    if not show_axis:
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
    fig.add_axes(ax)
    if title is not None:
        plt.title(title)
    plt.imshow(data_to_show)
    return fig


# does not look work so well
def simple_mask_to_colors(np_mask):
    img = np_mask.astype(np.int) * 255
    img = img[:, :, None].repeat(3, axis=2)
    img[:, :, :] = 0
    img[:, :, 1][np_mask] = 255
    img[:, :, 0][np.logical_not(np_mask)] = 255
    return img


def show_point_cloud(points_x, points_y, points_z):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Point cloud at {} sec. ".format(str(int(time.time()))))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.plot(0, 0, 0, 'o', color="black", markersize=2.0)

    ax.plot((points_x), (points_y), (points_z), 'o', color="black", markersize=0.5)

    ax.view_init(elev=10.0, azim=None)

    show_or_close(True)


def show_or_close(show):
    if show:
        plt.show(block=False)
    else:
        plt.close()


def show_imgs(img_paths):
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        img = cv.imread(img_path, None)
        plt.figure(figsize=(9, 9))
        plt.title(img_name)
        plt.imshow(img)
        show_or_close(True)


def get_degrees_between_normals(normals):

    size = normals.shape[0]
    degrees = []
    for i in range(size - 1):
        for j in range(i + 1, size):
            x = normals[i] / np.linalg.norm(normals[i])
            y = normals[j] / np.linalg.norm(normals[j])
            degrees.append(math.acos(np.dot(x, y)) * 180 / math.pi)
    return degrees


def show_and_save_normal_clusters_3d(normals, clustered_normals, normal_indices, show, save, out_dir, img_name):

    if not show and not save:
        return

    cluster_color_names = ["red", "green", "blue"]

    #fig = plt.figure(figsize=(9, 9))
    #plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    degrees_list = get_degrees_between_normals(clustered_normals)
    title = "Normals clustering: {}".format(img_name)
    if len(degrees_list) > 0:
        title = title + "\n degrees: {}".format(",\n".join([str(s) for s in degrees_list]))
    plt.title(title)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")

    ax.plot(0, 0, 0, 'o', color="black", markersize=2.0)

    rel_normals = normals[normal_indices == 3]
    ax.plot((rel_normals[::10, 0]), (rel_normals[::10, 2]), (rel_normals[::10, 1]), '.', color="yellow", markersize=0.5)
    ax.plot((-rel_normals[::10, 0]), (-rel_normals[::10, 2]), (-rel_normals[::10, 1]), '.', color="yellow", markersize=0.5)

    for i in range(len(clustered_normals)):
        rel_normals = normals[normal_indices == i]
        ax.plot((rel_normals[::10, 0]), (rel_normals[::10, 2]), (rel_normals[::10, 1]), '.', color=cluster_color_names[i], markersize=0.5)
        # NOTE comment this for better # visualizations
        #ax.plot((-rel_normals[::10, 0]), (-rel_normals[::10, 2]), (-rel_normals[::10, 1]), '.', color=cluster_color_names[i], markersize=0.5)

    if len(clustered_normals.shape) == 1:
        clustered_normals = np.expand_dims(clustered_normals, axis=0)

    for i in range(len(clustered_normals)):
        ax.plot((clustered_normals[i, 0]), (clustered_normals[i, 2]), (clustered_normals[i, 1]), 'o', color="black", markersize=5.0)

    ax.view_init(elev=10.0, azim=None)

    x_lim = [-1, 1]
    y_lim = [-1, 1]
    z_lim = [-1, 1]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    if show:
        for i in range(clustered_normals.shape[0] - 1):
            for j in range(i + 1, clustered_normals.shape[0]):
                angle = np.arccos(np.dot(clustered_normals[i], clustered_normals[j]))
                angle_degrees = 180 / math.pi * angle
                print("angle between normal {} and {}: {} degrees".format(i, j, angle_degrees))

    # NOTE: first save, then show!!!
    if save:
        out_path = '{}/{}_point_cloud.jpg'.format(out_dir, img_name[:-4])
        plt.savefig(out_path)

    show_or_close(show)


def show_normals_components(normals, title, figsize=None):

    if len(normals.shape) == 5:
        normals = normals.squeeze(dim=0).squeeze(dim=0)

    img = normals.numpy()
    fig = plt.figure()
    plt.title(title)
    for index in range(3):
        # row, columns, index
        ax = fig.add_subplot(131 + index)
        ax.imshow(img[:, :, index])
    show_or_close(True)


# previously used in pipeline.process_img
# NOTE this is just to get the #visualization
# 'frame_0000001535_4' - just to first img from scene1
# self.counter += 1
# normals_np = normals.numpy()
# normals_np[:, :, 2] *= -1
# plt.imshow(normals)
# plt.show()
# #cv.imwrite("thesis_work/normals_{}.png".format(self.counter), normals * 255)
# plt.imshow(img)
# plt.show()
# #cv.imwrite("thesis_work/normals_original{}.png".format(self.counter), img)

