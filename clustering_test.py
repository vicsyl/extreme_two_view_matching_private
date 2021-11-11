# import cv2 as cv
# import numpy as np
# import torch
import math

import matplotlib.pyplot as plt
import torch

from config import Config

from scene_info import SceneInfo
from depth_to_normals import compute_normals_from_svd, show_or_save_clusters, compute_normals_all
from utils import *
import clustering
from sky_filter import get_nonsky_mask


#https://web.archive.org/web/20120107030109/http://cgafaq.info/wiki/Evenly_distributed_points_on_sphere#Spirals

def inter_intra(normals, filter_mask, depth_data_file_name):

    for n_clusters in range(1, 2):

        Timer.start_check_point("clustering normals")
        # TODO consider to return clustered_normals.numpy()
        cluster_repr_normal, normal_indices = clustering.cluster(normals, filter_mask, n_clusters)

        print("normals: {}".format(cluster_repr_normal))

        matrix_normals = normals.reshape(-1, 3)

        # silhouette - this is infeasible as I would have to compute distances between the normals with each other, i.e.
        # the following (not shortened, in full) : torch.mm(matrix_normals[:100], matrix_normals[:100].transpose(0, 1))

        # TODO again: Euclidian distance
        cluster_normals_mean = cluster_repr_normal.sum(dim=0) / cluster_repr_normal.shape[0]
        cluster_normals_mean = cluster_normals_mean.reshape(3, -1)
        cos_distance_squared = (1 - torch.mm(cluster_repr_normal, cluster_normals_mean)) ** 2
        intra_variance = cos_distance_squared.sum() / cos_distance_squared.shape[0]
        print("number of clusters: {}".format(n_clusters))
        print("intra_variance: {}".format(intra_variance))

        variances = []
        for cluster in range(n_clusters):
            rel_normals = normals[filter_mask == 1]

            # torch.logical_and(normal_indices == cluster and filter_mask == 1)

            cluster_normals = normals[torch.logical_and(normal_indices == cluster, filter_mask == 1)]
            cluster_normals = cluster_normals.reshape(-1, 3)

            cluster_repr = cluster_repr_normal[cluster]
            cluster_repr_m = cluster_repr.reshape(3, -1)
            cos_distance_squared = (1 - torch.mm(cluster_normals, cluster_repr_m)) ** 2
            variance = cos_distance_squared.sum() / cos_distance_squared.shape[0]
            variances.append(variance)

            # cluster_normals_complement = normals[torch.logical_and(normal_indices != cluster, filter_mask == 1)]
            # cluster_normals_complement = cluster_normals_complement.reshape(-1, 3)
            # cos_distance_complement_squared = torch.mm(cluster_normals_complement, cluster_repr_m) ** 2

            print()

        print("inter variances: {}".format(variances))
        print("rho_is = {}".format([intra_variance / (intra_variance + v) for v in variances]))

        torch.mm(matrix_normals[:100], matrix_normals[:100].transpose(0, 1))

        # TODO (see above: consider to return clustered_normals.numpy())
        normal_indices_np = normal_indices.numpy().astype(dtype=np.uint8)
        cluster_repr_normal_np = cluster_repr_normal.numpy()

        # sklearn.metrics.silhouette_score(X, labels()

        Timer.end_check_point("clustering normals")

        # print(sklearn.metrics.silhouette_score(X, labels,)
        show_or_save_clusters(normals, normal_indices_np, cluster_repr_normal_np, out_dir=None,
                              img_name=depth_data_file_name, save=False)


def iterative(normals, filter_mask, depth_data_file_name):

    max_clusters = 4
    variances = []
    norm_variances = []
    norm_variances2 = []
    points_list = []
    for cluster_0_based in range(max_clusters):

        Timer.start_check_point("clustering normals")
        # TODO consider to return clustered_normals.numpy()
        cluster_repr_normal, normal_indices = clustering.kmeans(normals, filter_mask, 1, max_iter=100)

        cluster_normals = normals[torch.logical_and(normal_indices == 0, filter_mask == 1)]
        cluster_normals = cluster_normals.reshape(-1, 3)

        cluster_repr = cluster_repr_normal[0]
        cluster_repr_m = cluster_repr.reshape(3, -1)
        cos_distance_squared = (1 - torch.mm(cluster_normals, cluster_repr_m)) ** 2
        variance = cos_distance_squared.sum() / cos_distance_squared.shape[0]
        points = cluster_normals.shape[0]
        norm_variance = variance / math.sqrt(points)
        norm_variance2 = variance / points
        variances.append(variance)
        norm_variances.append(norm_variance)
        norm_variances2.append(norm_variance2)
        points_list.append(points)

        normal_indices_np = normal_indices.numpy().astype(dtype=np.uint8)
        cluster_repr_normal_np = cluster_repr_normal.numpy()

        Timer.end_check_point("clustering normals")

        # print(sklearn.metrics.silhouette_score(X, labels,)
        show_or_save_clusters(normals, normal_indices_np, cluster_repr_normal_np, out_dir=None,
                              img_name=depth_data_file_name, save=False)

        filter_mask = torch.logical_and(filter_mask, normal_indices != 0)


    print("variances = {}".format(variances))
    print("variances * sqrt(|points|)  = {}".format(norm_variances))
    print("variances * |points|  = {}".format(norm_variances2))
    print("points = {}".format(points_list))


def cluster_and_save_normals_test(normals,
                             depth_data_file_name,
                             angle_threshold=4*math.pi/9,
                             filter_mask=None,
                             ):


    # TODO just confirm if this happens for monodepth
    if len(normals.shape) == 5:
        normals = normals.squeeze(dim=0).squeeze(dim=0)

    minus_z_direction = torch.zeros(normals.shape)
    minus_z_direction[:, :, 2] = -1.0

    # dot_product = torch.sum(normals * minus_z_direction, dim=-1)
    # threshold = math.cos(angle_threshold)
    # filtered = dot_product >= threshold #, True, False)
    # TWEAK - enable all

    if filter_mask is None:
        # only ones
        filter_mask = torch.ones(normals.shape[:2])
    elif isinstance(filter_mask, np.ndarray):
        filter_mask = torch.from_numpy(filter_mask)

    inter_intra(normals, filter_mask, depth_data_file_name)
    #iterative(normals, filter_mask, depth_data_file_name)


def show_3d_points(points):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Approximately equidistant {} points on sphere".format(points.shape[0]))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.plot(0, 0, 0, 'o', color="black", markersize=2.0)

    ax.plot((points[:, 0]), (points[:, 1]), (points[:, 2]), '.', color="red", markersize=2.0)
    ax.view_init(elev=10.0, azim=None)

    x_lim = [-1, 1]
    y_lim = [-1, 1]
    z_lim = [-1, 1]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    plt.show(block=False)


def main():

    Timer.start()
    Config.log()

    interesting_files = ["frame_0000000650_2.npy"]
    #interesting_files = ["frame_0000001285_2.npy"]

    scene_name = "scene1"

    scene_info = SceneInfo.read_scene(scene_name, lazy=True)
    file_names, depth_input_directory = scene_info.get_megadepth_file_names_and_dir(limit=20, interesting_files=interesting_files)

    for depth_input_file_name in file_names:

        img_name = depth_input_file_name[0:-4]
        camera = scene_info.get_camera_from_img(img_name)

        depth_data = read_depth_data(depth_input_file_name, depth_input_directory)

        normals, _ = compute_normals_from_svd(camera, depth_data)

        img_file_path = scene_info.get_img_file_path(img_name)
        img = cv.imread(img_file_path)
        filter_mask = get_nonsky_mask(img, normals.shape[0], normals.shape[1])

        plt.title("sky mask for {}".format(img_name))
        plt.imshow(filter_mask)
        plt.show()

        cluster_and_save_normals_test(normals,
                                 depth_input_file_name,
                                 filter_mask=filter_mask)


    Timer.log_stats()


def points():
    n = 20
    points = clustering.n_points_across_half_sphere(n)
    show_3d_points(points)


def visualize_normals(impl, old_impl=False):
    Timer.start()
    Config.log()

    scene_name = "scene1"
    scene_info = SceneInfo.read_scene(scene_name, lazy=True)

    file_names = ["frame_0000001285_2.npy",
                  # 'frame_0000000010_3.npy', # 'frame_0000000015_3.npy',
                  # 'frame_0000000015_4.npy', #'frame_0000000020_3.npy',
                  # 'frame_0000000020_4.npy', #'frame_0000000025_3.npy',
                  #                   'frame_0000000025_4.npy', 'frame_0000000030_1.npy',
                  #                   'frame_0000000030_2.npy', 'frame_0000000030_3.npy',
                  ]
    input_directory = "depth_data/mega_depth/{}".format(scene_name)

    #     def compute_normals_all(scene: SceneInfo,
    #                         file_names,
    #                         read_directory,
    #                         output_parent_dir,
    #                         skip_existing=True,
    #                         impl="svd",
    #                         old_impl=False):

    compute_normals_all(scene_info,
                        file_names,
                        input_directory,
                        output_parent_dir=None,
                        skip_existing=False,
                        impl=impl,
                        old_impl=old_impl)

    Timer.log_stats()


if __name__ == "__main__":
    points()
    #main()

    # visualize_normals(impl="svd")
