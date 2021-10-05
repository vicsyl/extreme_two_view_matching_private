import torch
import math
from utils import Timer


def assert_almost_equal(one, two):
    assert math.fabs(one - two) < 0.000001


def recompute_points_threshold_ratio(angle_distance_threshold_degrees, points_threshold_ratio_factor=1.0):
    return 0.13 * (angle_distance_threshold_degrees / 30) * points_threshold_ratio_factor


def from_degrees_to_dist(degrees, log_key, factor=1.0):
    rads = factor * degrees * math.pi / 180
    distance = math.sin(rads / 2) * 2
    print("{}: degrees: {}, distance: {}".format(log_key, degrees, distance))
    return distance


class Clustering:

    # primary params
    N_points = 300
    angle_distance_threshold_degrees = 30
    distance_threshold = from_degrees_to_dist(angle_distance_threshold_degrees, "bin angle")
    distance_inter_cluster_threshold_factor = 2.5
    distance_inter_cluster_threshold = from_degrees_to_dist(angle_distance_threshold_degrees, "seed inter cluster angle", distance_inter_cluster_threshold_factor)

    ms_kernel_max_distance = distance_threshold
    ms_adjustment_th = 0.1
    ms_max_iter = 100
    ms_bandwidth = ms_kernel_max_distance / 2
    ms_distance_inter_cluster_threshold_factor = 2
    ms_distance_inter_cluster_threshold = from_degrees_to_dist(angle_distance_threshold_degrees, "mean shift seed inter cluster angle", ms_distance_inter_cluster_threshold_factor)

    # previous hard-coded value: 0.13
    points_threshold_ratio = recompute_points_threshold_ratio(angle_distance_threshold_degrees, points_threshold_ratio_factor=1.0)

    @staticmethod
    def recompute(points_threshold_ratio_factor):

        Clustering.distance_threshold = from_degrees_to_dist(Clustering.angle_distance_threshold_degrees, "bin angle")
        Clustering.distance_inter_cluster_threshold = from_degrees_to_dist(Clustering.angle_distance_threshold_degrees, "seed inter cluster angle", Clustering.distance_inter_cluster_threshold_factor)
        Clustering.ms_distance_inter_cluster_threshold = from_degrees_to_dist(Clustering.angle_distance_threshold_degrees, "mean shift seed inter cluster angle", Clustering.ms_distance_inter_cluster_threshold_factor)

        # magic formula
        Clustering.points_threshold_ratio = recompute_points_threshold_ratio(Clustering.angle_distance_threshold_degrees, points_threshold_ratio_factor)
        print("Recomputed")
        Clustering.log()

    @staticmethod
    def log():
        print("Clustering:")
        print("\tN_points\t{}".format(Clustering.N_points))
        print("\tangle_distance_threshold\t{} degrees".format(Clustering.angle_distance_threshold_degrees))
        print("\tdistance_threshold\t{}".format(Clustering.distance_threshold))
        print("\tdistance_inter_cluster_threshold_factor\t{}".format(Clustering.distance_inter_cluster_threshold_factor))
        print("\tdistance_inter_cluster_threshold\t{}".format(Clustering.distance_inter_cluster_threshold))
        print("\tms_kernel_max_distance\t{}".format(Clustering.ms_kernel_max_distance))
        print("\tms_adjustment_th\t{}".format(Clustering.ms_adjustment_th))
        print("\tms_max_iter\t{}".format(Clustering.ms_max_iter))
        print("\tms_bandwidth\t{}".format(Clustering.ms_bandwidth))
        print("\tms_distance_inter_cluster_threshold_factor\t{}".format(Clustering.ms_distance_inter_cluster_threshold_factor))
        print("\tms_distance_inter_cluster_threshold\t{}".format(Clustering.ms_distance_inter_cluster_threshold))
        print("\tpoints_threshold_ratio\t{}".format(Clustering.points_threshold_ratio))


# https://web.archive.org/web/20120107030109/http://cgafaq.info/wiki/Evenly_distributed_points_on_sphere#Spirals
def n_points_across_half_sphere(N):
    """
    :param N: number of points to distribute across a hemisphere
    :return:
    """
    s = 3.6 / math.sqrt(N)
    dz = 1.0 / N
    longitude = 0
    z = 1 - dz / 2

    points = torch.zeros((N, 3))
    for k in range(N):
        r = math.sqrt(1 - z * z)
        points[k] = torch.tensor([math.cos(longitude) * r, math.sin(longitude) * r, -z])
        z = z - dz
        longitude = longitude + s / r

    return points


def angle_2_unit_vectors(v1, v2):
    return math.acos(v1.T @ v2) / math.pi * 180


def cluster(normals: torch.Tensor, filter_mask, mean_shift=None):

    points_threshold = torch.prod(torch.tensor(normals.shape[:2])) * Clustering.points_threshold_ratio

    timer_label = "clustering for N={}".format(Clustering.N_points)
    Timer.start_check_point(timer_label)
    n_centers = n_points_across_half_sphere(Clustering.N_points)

    n_centers = n_centers.expand(normals.shape[0], normals.shape[1], -1, -1)
    n_centers = n_centers.permute(2, 0, 1, 3)

    diffs = n_centers[:] - normals
    diff_norm = torch.norm(diffs, dim=3)

    near_ones_per_cluster_center = torch.where(diff_norm < Clustering.distance_threshold, 1, 0)
    near_ones_per_cluster_center = torch.logical_and(near_ones_per_cluster_center, filter_mask)

    sums = near_ones_per_cluster_center.sum(dim=(1, 2))

    sortd = torch.sort(sums, descending=True)

    cluster_centers = []
    points_list = []

    arg_mins = torch.ones(normals.shape[:2]) * 3
    arg_mins = arg_mins.to(torch.int)

    max_clusters = 3
    for center_index, points in zip(sortd[1], sortd[0]):
        if len(cluster_centers) >= max_clusters:
            break
        if points < points_threshold:
            break

        def is_distance_ok(new_center, threshold):
            for cluster_center in cluster_centers:
                diff = new_center - cluster_center
                diff_norm = torch.norm(diff)
                if diff_norm < threshold:
                    return False
            return True

        if mean_shift == "full":
            th = Clustering.ms_distance_inter_cluster_threshold
        else:
            th = Clustering.distance_inter_cluster_threshold

        distance_ok = is_distance_ok(n_centers[center_index, 0, 0], th)

        if distance_ok:
            if mean_shift is None:
                cluster_center = n_centers[center_index, 0, 0]
                cluster_centers.append(cluster_center)
                arg_mins[near_ones_per_cluster_center[center_index]] = len(cluster_centers) - 1
            elif mean_shift == "mean":
                # near_ones_per_cluster_center needs recomputing
                coords = torch.where(near_ones_per_cluster_center[center_index, :, :])
                normals_to_mean = normals[coords[0], coords[1]]

                # normals_to_mean conv with kernel + normalization (to norm == 1)
                cluster_center = normals_to_mean.sum(dim=0) / normals_to_mean.shape[0]
                cluster_center = cluster_center / torch.norm(cluster_center)

                # just printing the shift
                angle = angle_2_unit_vectors(cluster_center, n_centers[center_index, 0, 0])
                print("delta (mean vs. cluster center): {} degrees".format(angle))

                cluster_centers.append(cluster_center)
                distances = torch.norm(cluster_center - normals, dim=2).expand(normals.shape[0], normals.shape[1])
                neighborhood = torch.where(distances < Clustering.distance_threshold, 1, 0)
                neighborhood = torch.logical_and(neighborhood, filter_mask)
                coords = torch.where(neighborhood)
                arg_mins[coords[0], coords[1]] = len(cluster_centers) - 1

            elif mean_shift == "full":

                cluster_center = n_centers[center_index, 0, 0]
                orig_center = cluster_center

                angle_diff = Clustering.ms_adjustment_th
                for _ in range(Clustering.ms_max_iter):
                    if angle_diff < Clustering.ms_adjustment_th:
                        break

                    distances = torch.norm(cluster_center - normals, dim=2).expand(normals.shape[0], normals.shape[1])

                    neighborhood = torch.where(distances < Clustering.ms_kernel_max_distance, 1, 0)
                    neighborhood = torch.logical_and(neighborhood, filter_mask)

                    coords = torch.where(neighborhood)
                    normals_for_shift = normals[coords[0], coords[1]]
                    distances_squared = (distances[coords[0], coords[1]] / Clustering.ms_bandwidth) ** 2

                    # normalization const. ignored (should be very close to 1 anyway)
                    kernel_values = torch.exp(distances_squared * -0.5) * 0.5 / math.pi
                    new_center = (normals_for_shift * kernel_values.expand(3, -1).permute(1, 0)).sum(dim=0) / kernel_values.sum()
                    new_center = new_center / torch.norm(new_center)

                    angle_diff = angle_2_unit_vectors(cluster_center, new_center)
                    print("mode adjustment (iteration): {} degrees".format(angle_diff))
                    angle_diff_overall = angle_2_unit_vectors(orig_center, new_center)
                    print("mode adjustment (overall): {} degrees".format(angle_diff_overall))
                    print("orig: {}, old: {}, new: {}".format(orig_center, cluster_center, new_center))
                    cluster_center = new_center

                distance_ok = is_distance_ok(cluster_center, Clustering.distance_inter_cluster_threshold)
                if distance_ok:
                    cluster_centers.append(cluster_center)
                    distances = torch.norm(cluster_center - normals, dim=2).expand(normals.shape[0], normals.shape[1])
                    neighborhood = torch.where(distances < Clustering.distance_threshold, 1, 0)
                    neighborhood = torch.logical_and(neighborhood, filter_mask)
                    coords = torch.where(neighborhood)
                    arg_mins[coords[0], coords[1]] = len(cluster_centers) - 1

    if len(cluster_centers) == 1:
        cluster_centers = cluster_centers[0]
        cluster_centers = torch.unsqueeze(cluster_centers, dim=0)
    elif len(cluster_centers) > 1:
        cluster_centers = torch.vstack(cluster_centers)
    else:
        # NOTE corner case - no clusters found
        cluster_centers = torch.zeros((0, 3))

    Timer.end_check_point(timer_label)

    return cluster_centers, arg_mins

    # comparing the cluster_centers found by this method with the results of kmeans taking the cluster_centers as initial guesses

    # def expand_centers_get_sums(centers):
    #
    #     centers = centers.expand(normals.shape[0], normals.shape[1], -1, -1)
    #     centers = centers.permute(2, 0, 1, 3)
    #
    #     diffs = centers[:] - normals
    #     diff_norm = torch.norm(diffs, dim=3)
    #
    #     near_ones_per_cluster_center_inner = torch.where(diff_norm < Clustering.distance_threshold, 1, 0)
    #     near_ones_per_cluster_center_inner = torch.logical_and(near_ones_per_cluster_center_inner, filter_mask)
    #
    #     sums = near_ones_per_cluster_center_inner.sum(dim=(1, 2))
    #     return centers, sums
    #
    # cluster_centers_old = cluster_centers.clone()
    # for i in range(cluster_centers_old.shape[0] - 1):
    #     for j in range(i + 1, cluster_centers_old.shape[0]):
    #         angle = math.acos(cluster_centers_old[i, 0, 0] @ cluster_centers_old[j, 0, 0].T)
    #         angle_degrees = 180 / math.pi * angle
    #         print("angle between normal {} and {}: {} degrees".format(i, j, angle_degrees))
    #
    # timer_label2 = "2nd phase of clustering - kmeans"
    # Timer.start_check_point(timer_label2)
    # cluster_centers_new, arg_mins = kmeans(normals, filter_mask, clusters=None, max_iter=max_iter, cluster_centers=cluster_centers)
    # Timer.end_check_point(timer_label2)
    #
    # angles = []
    # for i in range(cluster_centers_new.shape[0]):
    #     angle = math.acos(cluster_centers_new[i] @ cluster_centers_old[i, 0, 0].T)
    #     angle_degrees = 180 / math.pi * angle
    #     angles.append(angle_degrees)
    # print("Angles differences: {}".format(angles))
    #
    # _, sums = expand_centers_get_sums(cluster_centers_new)
    # for i in range(cluster_centers_new.shape[0]):
    #     print("{}th cluster: from {} to {} points".format(i, points_list[i], sums[i]))
    #
    # return cluster_centers_new, arg_mins


# for kmeans
initial_cluster_centers = torch.Tensor([
        [+math.sqrt(3) / 2,  0.00, -0.5],
        [-math.sqrt(3) / 4, +0.75, -0.5],
        [-math.sqrt(3) / 4, -0.75, -0.5]
    ])


# NOT USED
def kmeans(normals: torch.Tensor, filter_mask, clusters=None, max_iter=20, cluster_centers=None):
    """
    :param normals: torch: w,h,3 (may add b)
    :return:
    """

    if clusters is None:
        assert cluster_centers is not None
        clusters = cluster_centers.shape[0]

    if cluster_centers is None:
        assert clusters <= 3
        shape = tuple([clusters]) + tuple(normals.shape)
        cluster_centers = torch.zeros(shape)
        for i in range(clusters):
            cluster_centers[i] = initial_cluster_centers[i]


    old_arg_mins = None
    iter = 0 # so that max_iter == 0 works
    for iter in range(max_iter):

        diffs = cluster_centers[:] - normals

        diff_norm = torch.norm(diffs, dim=3)
        mins = torch.min(diff_norm, dim=0, keepdim=True)
        arg_mins = mins[1].squeeze(0)
        filtered_arg_mins = torch.where(filter_mask, arg_mins, 3)
        filtered_arg_mins = torch.where(mins[0].squeeze(0) < Clustering.distance_threshold, filtered_arg_mins, 3)

        if old_arg_mins is not None:
            changes = old_arg_mins[old_arg_mins != filtered_arg_mins].shape[0]
            if changes == 0:
                break

        old_arg_mins = filtered_arg_mins

        for i in range(clusters):
        #for i in range(1):
            cluster_i_points = normals[filtered_arg_mins == i]
            # TODO REALLY??!! - this is using just the euclidian distance!!!
            new_cluster = torch.sum(cluster_i_points, 0) / cluster_i_points.shape[0]
            new_cluster = new_cluster / torch.norm(new_cluster)
            cluster_centers[i, :, :, :] = new_cluster

    print("iter: {}".format(iter))

    diffs = cluster_centers[:] - normals
    diff_norm = torch.norm(diffs, dim=3)
    mins = torch.min(diff_norm, dim=0, keepdim=True)
    arg_mins = mins[1].squeeze(0)
    filtered_arg_mins = torch.where(filter_mask == 1, arg_mins, 3)
    mins = mins[0].squeeze(0)

    arg_mins = torch.where(mins < Clustering.distance_threshold, filtered_arg_mins, 3)
    #print_and_get_stats(arg_mins)

    ret = cluster_centers[:, 0, 0, :], arg_mins
    return ret


def print_and_get_stats(arg_mins):
    stats = [None] * 4
    for i in range(4):
        stats[i] = torch.where(arg_mins == i, 1, 0).sum().item()
    print("stats: {}".format(stats))
    return stats
