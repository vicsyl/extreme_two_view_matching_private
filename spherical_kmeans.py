import torch
import math

initial_cluster_centers = torch.Tensor([
        [+math.sqrt(3) / 2,  0.00, -0.5],
        [-math.sqrt(3) / 4, +0.75, -0.5],
        [-math.sqrt(3) / 4, -0.75, -0.5]
    ])

# TWEAK
distance_threshold = 0.6
angle_distance = 2 * math.asin(distance_threshold / 2)

# just an informative message about angle distance threshold for keeping a normal in a cluster
#print("angle distance in rad used for k_means in radians: {}".format(angle_distance))
#print("the same in degrees: {}".format(angle_distance / math.pi * 180.0))


def kmeans(normals: torch.Tensor, filter_mask, clusters, max_iter=20):
    """
    :param normals: torch: w,h,3 (may add b)
    :return:
    """

    shape = tuple([clusters]) + tuple(normals.shape)
    cluster_centers = torch.zeros(shape)
    for i in range(clusters):
        cluster_centers[i] = initial_cluster_centers[i]

    old_arg_mins = None
    for iter in range(max_iter):

        diffs = cluster_centers[:] - normals

        diff_norm = torch.norm(diffs, dim=3)
        mins = torch.min(diff_norm, dim=0, keepdim=True)
        arg_mins = mins[1].squeeze(0)
        filtered_arg_mins = torch.where(filter_mask, arg_mins, 3)

        if old_arg_mins is not None:
            changes = old_arg_mins[old_arg_mins != filtered_arg_mins].shape[0]
            if changes == 0:
                break

        old_arg_mins = filtered_arg_mins

        for i in range(clusters):
            cluster_i_points = normals[filtered_arg_mins == i]
            new_cluster = torch.sum(cluster_i_points, 0) / cluster_i_points.shape[0]
            new_cluster = new_cluster / torch.norm(new_cluster)
            cluster_centers[i, :, :, :] = new_cluster

    diffs = cluster_centers[:] - normals
    diff_norm = torch.norm(diffs, dim=3)
    mins = torch.min(diff_norm, dim=0, keepdim=True)
    arg_mins = mins[1].squeeze(0)
    filtered_arg_mins = torch.where(filter_mask == 1, arg_mins, 3)
    mins = mins[0].squeeze(0)

    arg_mins = torch.where(mins < distance_threshold, filtered_arg_mins, 3)
    #print_and_get_stats(arg_mins)

    ret = cluster_centers[:, 0, 0, :], arg_mins
    return ret


def print_and_get_stats(arg_mins):
    stats = [None] * 4
    for i in range(4):
        stats[i] = torch.where(arg_mins == i, 1, 0).sum().item()
    print("stats: {}".format(stats))
    return stats
