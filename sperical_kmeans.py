import torch
import math


def kmeans(normals: torch.Tensor, max_iter=20):
    """
    :param normals: torch: w,h,3 (may add b)
    :return:
    """

    old_arg_mins = None
    quit = False

    cluster_center_1_vec = torch.Tensor([+math.sqrt(3) / 2, 0.00, -0.5])
    cluster_center_2_vec = torch.Tensor([-math.sqrt(3) / 4, +0.75, -0.5])
    cluster_center_3_vec = torch.Tensor([-math.sqrt(3) / 4, -0.75, -0.5])

    assert torch.norm(cluster_center_1_vec).item() - 1.0 < 000.1
    assert torch.norm(cluster_center_2_vec).item() - 1.0 < 000.1
    assert torch.norm(cluster_center_3_vec).item() - 1.0 < 000.1
    shape = tuple([3]) + tuple(normals.shape)

    # diffs = torch.zeros(shape)
    cluster_centers = torch.zeros(shape)
    cluster_centers[0, :, :, :] = cluster_center_1_vec
    cluster_centers[1, :, :, :] = cluster_center_2_vec
    cluster_centers[2, :, :, :] = cluster_center_3_vec

    for iter in range(max_iter):

        diffs = cluster_centers[:] - normals

        diff_norm = torch.norm(diffs, dim=3)
        #arg_mins = torch.argmin(diff_norm, dim=0, keepdim=True).squeeze(0)
        mins = torch.min(diff_norm, dim=0, keepdim=True)
        arg_mins = mins[1].squeeze(0)
        mins = mins[0].squeeze(0)
        arg_mins = torch.where(mins < 0.6, arg_mins, 3)

        if old_arg_mins is not None:
            print("changes: {}".format(old_arg_mins[old_arg_mins != arg_mins].shape[0]))
            if torch.all(torch.eq(old_arg_mins, arg_mins)):
                quit = True

        old_arg_mins = arg_mins

        for i in range(3):
            cluster_i_points = normals[arg_mins == i]
            new_cluster = torch.sum(cluster_i_points, 0) / cluster_i_points.shape[0]
            new_cluster = new_cluster / torch.norm(new_cluster)
            cluster_centers[i, :, :, :] = new_cluster
        if quit:
            break

    for i in range(3):
        diffs = cluster_centers[:] - normals
        diff_norm = torch.norm(diffs, dim=3)
        mins = torch.min(diff_norm, dim=0, keepdim=True)
        arg_mins = mins[1].squeeze(0)
        mins = mins[0].squeeze(0)
        arg_mins = torch.where(mins < 0.45, arg_mins, 3)
        print()

    ret = cluster_centers[:, 0, 0, :], arg_mins
    return ret
