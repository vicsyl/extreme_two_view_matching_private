import math

class Alg:
    mean = None
    mean_shift = None

def equidistant_points_on_hemisphere(N):
    return

def rodriguez_formula(N):
    return

def warp_perspective(a, b, c):
    return

def translation(a, b):
    return

def inverse(N):
    return

def bounding_box(N):
    return

def closest(a, b, c):
    return

def cross_product(a, b):
    return

def cluster_mean(a, b, c):
    return

def max_distance(a, b, c):
    return

def mean_shift(a, b, c):
    return

def closest(a, b, c):
    return

DESCENDING = None

# Parameters and their default values

# number of initial clusters
N = 300
# angle defining the extent of the cluster
alpha_th = 35
# quantile used for filtering based on singular value ratios
alpha_q = 1.0
# factor defining the inhibition area around detected clusters
c = 2.5
# Alg.no_refinement, Alg.mean, Alg.mean_shift
algorithm = Alg.mean


def compute_clusters(normals, mean_shift_type, max_clusters):

    size = normals.height * normals.width
    # minimal number of normals in the cluster
    p_th = (0.13 / 30) * alpha_th * alpha_q * size

    candidates = equidistant_points_on_hemisphere(N)

    clusters = {}

    for c in candidates:
        c.count = closest(normals, c, candidates)

    candidates = candidates.sort(key=lambda c: c.count, order=DESCENDING)
    for c in candidates:
        if c < p_th:
            break
        elif algorithm == Alg.mean:
            c = cluster_mean(c, normals, alpha_th)
        elif algorithm == Alg.mean_shift:
            c = mean_shift(c, normals, alpha_th)

        if max_distance(c, clusters) < c * alpha_th:
            continue
        else:
            clusters.add(c)

    return clusters


def rectify_img(K, normal, img, patch_mask):
    """
    :param K: calibration matrix
    ...
    ...
    :return: rectified image, rectified patch mask
    """

    # check the sign
    rot = cross_product([0, 0, 1], normal)
    R = rodriguez_formula(rot)
    H = K @ R @ inverse(K)
    bb_size = bounding_box(H @ patch_mask).size
    scale = math.sqrt(2 * patch_mask.size / bb_size)
    H[2] = H[2] / scale
    (t_x, ty) = -min(H @ patch_mask)
    T = translation(t_x, t_y)
    H = T @ H
    bb = bounding_box(H @ patch_mask)
    rect_img = warp_perspective(img, H, bb)
    rect_patch = H @ patch_mask
    return rect_img, rect_patch


# def aff_net_and_depth(img, depth_map):
#
#     patches = detect_planar_surfaces(img, depth_map)
#
#     kpts, descs, shapes = HardNet.detectAndCompute(img)
#
#     for patch in patches():
#         to_cover = restrict(shapes, patch)
#         kpts_p, descs_p = greedy_cover_selection(patch, to_cover)
#         kpts = kpts.union(kpts_p)
#         descs = descs.union(descs_p)
#
#     return kpts, descs