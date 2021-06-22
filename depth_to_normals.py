import cv2 as cv
import numpy as np
import torch

from config import Config

from scene_info import read_cameras, read_images, SceneInfo, CameraEntry
from image_processing import *
from utils import *
from img_utils import show_and_save_normal_clusters_3d, show_point_cloud, show_or_close
import matplotlib.pyplot as plt

import torch.nn.functional as F

import clustering
from pathlib import Path
from clusters_map import clusters_map
from sky_filter import get_nonsky_mask


"""
Functions in this file are supposed to compute normals from the
depth maps. I tried two different approaches
a) as the original paper describes, it reprojects points in 5x5 window to 3D and a plane was fitted
   to these points. I used SVD decomposition for this, which I think is very inefficient, also because I cannot easily parallelize it  
   Using directly psedoinverse matrixces to solve MSE shouldn't be much more efficient.
   The backprojected points within the window will be spread by different distances depending on the projected ray angle and - more importantly - on the depth    
b) I used differential convolution masks to estimate the normal. This is much faster and easier. The normals are computed as if the projecting ray is in the 
   direction of the z axis - i.e. the angle of the projection ray is not accounted for

Problems:
  - I thought I would see clear clusters of the normals corresponding to the dominant planes. but...
  - megadepth seems to capture various irregularities (bricks on the wall) and the normals are not clustered that smoothly
  - in some functions I use the spherical k-means, which I implemented in 'clustering.py'
  - in the original paper they also say that they look for "connected regions of points whose normals belong to the same cluster" - not sure how this was done 
    but obviously some effort can be done in this respect to further smooth out the clusters (enlarge them)
  - I am actually confused, but still convinced that I cannot predict normals correctly if I have only unscaled depth info - I would need to have scaled depth info
    and then normalize it according to the focal point length (which I have)
  - there are other details that may influence the result, as upsampling the depth map to the original size of the image before doing anything else, 
    however I don't think this particular one would have any significant effect
  - actually maybe I should have tested with other (e.g. by the real monodepth) CNN's depth maps         
"""


def get_gauss_weighted_coeffs_for_window(window_size=5, sigma=1.33):

    x = torch.linspace(-float(window_size//2), float(window_size//2), window_size)
    x, y = torch.meshgrid(x, x)

    normalizing_gauss_coeffs = 1.0 / (2.0 * math.pi * sigma ** 2)
    gauss_coeffs = normalizing_gauss_coeffs * torch.exp(-(x ** 2 + y ** 2) / (2.0 * sigma**2))

    gauss_weighted_coeffs = gauss_coeffs.flatten()
    gauss_weighted_coeffs_normalized = window_size ** 2 * gauss_weighted_coeffs / gauss_weighted_coeffs.sum()

    assert math.fabs(gauss_weighted_coeffs_normalized.sum() - window_size ** 2) < 0.0001

    return gauss_weighted_coeffs_normalized


def get_rotation_matrices_across_img(camera: CameraEntry, depth_data: np.ndarray):

    # TODO remove, camera.get_K() is obsolete
    K = camera.get_K()
    down_sample_factor_x = depth_data.shape[3] / camera.height_width[1]
    down_sample_factor_y = depth_data.shape[2] / camera.height_width[0]

    K[0] = K[0] * down_sample_factor_x
    K[1] = K[1] * down_sample_factor_y

    Q_inv = np.linalg.inv(K)

    height = depth_data.shape[2]
    width = depth_data.shape[3]

    m = np.mgrid[0:width, 0:height]
    m = np.moveaxis(m, 0, -1)
    m_line = np.ndarray((width, height, 3, 1))
    m_line[:, :, :2, 0] = m
    m_line[:, :, 2, 0] = 1.0

    d = Q_inv @ m_line

    f_norms = np.expand_dims(np.linalg.norm(d, axis=2), axis=2)
    d_unit = d / f_norms

    # OK - checked
    d_unit_row_vec = np.moveaxis(d_unit, -1, -2)

    z = np.array([0.0, 0.0, 1.0])

    rotation_vector = np.cross(z, d_unit_row_vec)

    rotation_vector_norm = np.linalg.norm(rotation_vector, axis=3)
    sin_theta = np.linalg.norm(rotation_vector, axis=3)
    factor = 1.0
    sin_theta = sin_theta.copy() * factor
    unit_rotation_vector = rotation_vector[:, :, 0, :] / rotation_vector_norm
    unit_rotation_vector = np.where(rotation_vector_norm >= 0.0001, unit_rotation_vector, [0.0, 0.0, 0.0])
    theta = np.arcsin(sin_theta[:, :, 0])

    Rs = get_rotation_matrices(unit_rotation_vector, theta)
    det = np.linalg.det(Rs)
    assert np.all(np.abs(det - 1.0) < 0.0001)
    # Rs[w, h], (x, y) coord OK
    return Rs


def diff_normal_from_depth_data(camera: CameraEntry,
                                depth_data,
                                mask,
                                smoothed: bool=False,
                                sigma: float=1.0):

    focal_length = camera.focal_length

    # FIXME !!! test this
    down_sample_factor_x = depth_data.shape[3] / camera.height_width[1]
    down_sample_factor_y = depth_data.shape[2] / camera.height_width[0]
    down_sample_factor = (down_sample_factor_x + down_sample_factor_y) / 2.0
    # TWEAK - this fix seems to work (default value is 0.26666)
    real_focal_length = focal_length * down_sample_factor
    #real_focal_length = focal_length

    Rs = get_rotation_matrices_across_img(camera, depth_data)
    Rs = torch.from_numpy(Rs)
    # (x, y) -> (y, x)
    Rs = Rs.permute((1, 0, 2, 3))

    # Could be also done from reprojected data, but this seems to be correct and more straightforward

    gradient_dzdx, gradient_dzdy = spatial_gradient_first_order(depth_data, mask=mask, smoothed=smoothed, sigma=sigma)
    gradient_dzdx = (gradient_dzdx * real_focal_length / depth_data).unsqueeze(dim=4)
    gradient_dzdy = (gradient_dzdy * real_focal_length / depth_data).unsqueeze(dim=4)
    z_ones = torch.ones(gradient_dzdy.shape)
    normals = torch.cat((-gradient_dzdx, -gradient_dzdy, -z_ones), dim=4)
    normals_norms = torch.norm(normals, dim=4).unsqueeze(dim=4)
    normals = normals / normals_norms
    normals = normals.squeeze(dim=0).squeeze(dim=0)

    normals = normals.unsqueeze(3)
    normals = Rs @ normals
    normals = normals.squeeze()
    normals_norms = torch.norm(normals, dim=2).unsqueeze(dim=2)
    normals = normals / normals_norms
    return normals


def diff_normal_from_depth_data_old(focal_length,
                                depth_data,
                                mask,
                                smoothed: bool=False,
                                sigma: float=1.0,
                                depth_factor=1/30):

    # Could be also done from reprojected data, but this seems to be correct and more straightforward
    gradient_dzdx, gradient_dzdy = spatial_gradient_first_order(depth_data, mask=mask, smoothed=smoothed, sigma=sigma)
    gradient_dzdx = (gradient_dzdx * focal_length * depth_factor).unsqueeze(dim=4)
    gradient_dzdy = (gradient_dzdy * focal_length * depth_factor).unsqueeze(dim=4)
    z_ones = torch.ones(gradient_dzdy.shape)
    normals = torch.cat((-gradient_dzdx, -gradient_dzdy, -z_ones), dim=4)
    normals_norms = torch.norm(normals, dim=4).unsqueeze(dim=4)
    normals = normals / normals_norms
    normals = normals.squeeze(0).squeeze(0)

    return normals


def show_or_save_clusters(normals, normal_indices_np, cluster_repr_normal_np, out_dir, img_name, show=False, save=False):

    if show or save:
        img = np.ndarray(normal_indices_np.shape + (3,))
        img[:, :, 0][normal_indices_np == 0] = 255
        img[:, :, 0][normal_indices_np != 0] = 0
        img[:, :, 1][normal_indices_np == 1] = 255
        img[:, :, 1][normal_indices_np != 1] = 0
        img[:, :, 2][normal_indices_np == 2] = 255
        img[:, :, 2][normal_indices_np != 2] = 0

        plt.figure(figsize=(9, 9))
        color_names = ["red", "green", "blue"]
        title = "{}:\n".format(img_name)
        np.set_printoptions(suppress=True, precision=3)
        for i in range(cluster_repr_normal_np.shape[0]):
            degrees = np.array([math.acos(np.dot(np.array([0, 0, -1]), cluster_repr_normal_np[i])) * 180 / math.pi])
            title = "{}{}={} - {} deg.,\n".format(title, color_names[i], cluster_repr_normal_np[i], degrees)
        plt.title(title)
        plt.imshow(img)
        if save:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path = '{}/{}'.format(out_dir, img_name[:-4])
            plt.savefig("{}_clusters.jpg".format(out_path))
        show_or_close(show)

    if save:
        cv.imwrite("{}_clusters_indices_unused.png".format(out_path), normal_indices_np)
        np.savetxt('{}_clusters_normals_unused.txt'.format(out_path), cluster_repr_normal_np, delimiter=',', fmt='%1.8f')

    if show:
        show_and_save_normal_clusters_3d(normals, cluster_repr_normal_np, normal_indices_np, show, save, out_dir, img_name)


def cluster_normals(normals, filter_mask=None):

    # TODO just confirm if this happens for monodepth
    if len(normals.shape) == 5:
        normals = normals.squeeze(dim=0).squeeze(dim=0)
        raise Exception("should not happen")

    if filter_mask is None:
        # only ones
        filter_mask = torch.ones(normals.shape[:2])
    elif isinstance(filter_mask, np.ndarray):
        filter_mask = torch.from_numpy(filter_mask)

    Timer.start_check_point("clustering normals")
    # TODO consider to return clustered_normals.numpy()
    cluster_repr_normal, normal_indices = clustering.cluster(normals, filter_mask)

    normal_indices_np = normal_indices.numpy().astype(dtype=np.uint8)
    cluster_repr_normal_np = cluster_repr_normal.numpy()

    Timer.end_check_point("clustering normals")

    #show_or_save_clusters(normals, normal_indices_np, cluster_repr_normal_np, output_directory, depth_data_file_name)

    return cluster_repr_normal_np, normal_indices_np


def get_file_names_from_dir(input_dir: str, limit: int, interesting_files: list, suffix: str):
    if interesting_files is not None:
        return interesting_files
    else:
        return get_file_names(input_dir, suffix, limit)


def show_sky_mask(img, filter_mask, img_name, show, save=False, path=None):
    if not save and not show:
        return
    fig = plt.figure()
    plt.title("sky mask for {}".format(img_name))
    plt.axis('off')
    ax = fig.add_subplot(121)
    ax.imshow(img)
    ax = fig.add_subplot(122)
    ax.imshow(filter_mask)
    if save:
        plt.savefig(path)
    show_or_close(show)


def compute_only_normals(
        focal_length,
        orig_height,
        orig_width,
        depth_data_read_directory,
        depth_data_file_name):

    depth_data = read_depth_data(depth_data_file_name, depth_data_read_directory)
    normals = compute_normals_from_svd(focal_length, orig_height, orig_width, depth_data)
    return normals


def compute_normals(scene: SceneInfo,
                    depth_data_read_directory,
                    depth_data_file_name,
                    output_directory
                    ):
    """
    Currently the standard way to compute the normals and cluster them. This is called from other modules
    (extension point) on a per file basis

    :param scene:
    :param depth_data_read_directory:
    :param depth_data_file_name:
    :param output_directory:
    :return: (normals, normal_indices)
    """

    if True:
        raise Exception("not implemented - need 'img_name' as a parameter as in 'K = scene.get_img_K(img_name)'")

    K = scene.get_img_K("img_name")

    focal_length = K[0, 0]
    orig_height = img.shape[0]
    orig_width = img.shape[1]

    normals = compute_only_normals(focal_length,
                                   orig_height,
                                   orig_width,
                                   depth_data_read_directory,
                                   depth_data_file_name)

    img_name = depth_data_file_name[0:-4]
    img_file_path = scene.get_img_file_path(img_name)
    img = cv.imread(img_file_path)
    filter_mask = get_nonsky_mask(img, normals.shape[0], normals.shape[1])
    show_sky_mask(img, filter_mask, img_name, show=True)

    clustered_normals_np, normal_indices_np = cluster_normals(normals, filter_mask=filter_mask)
    return clustered_normals_np, normal_indices_np


def compute_normals_convolution(camera,
                                depth_data,
                                output_directory=None,
                                img_name=None,
                                override_mask=None,
                                old_implementation=False):

    default_mask = torch.tensor([[0.25, 0.25, 0, -0.25, -0.25]]).float()
    #mask = torch.tensor([[0.5, 0, -0.5]]).float()

    if override_mask is not None:
        mask = override_mask
    else:
        mask = default_mask
    smoothed = True

    # TWEAK
    sigma = 1.33
    #sigma = 3

    if old_implementation:
        normals = diff_normal_from_depth_data_old(camera.focal_length, depth_data, mask=mask, smoothed=smoothed, sigma=sigma, depth_factor=1/6)
    else:
        normals = diff_normal_from_depth_data(camera, depth_data, mask=mask, smoothed=smoothed, sigma=sigma)

    return normals


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


def compute_normals_from_svd(
        focal_length,
        orig_height,
        orig_width,
        depth_data,
):

    window_size = 5

    if Config.svd_smoothing:
        depth_data = gaussian_filter2d(depth_data, Config.svd_smoothing_sigma)

    depth_height = depth_data.shape[2]
    depth_width = depth_data.shape[3]

    # depth_data shapes
    f_factor_x = depth_width / orig_width
    f_factor_y = depth_height / orig_height
    if abs(f_factor_y - f_factor_x) > 0.001:
        print("WARNING: downsampled anisotropically")
    f_factor = (f_factor_x + f_factor_y) / 2
    real_focal_length_x = focal_length * f_factor_x
    real_focal_length_y = focal_length * f_factor_y

    # or I need to handle odd numbers (see linspace)
    assert depth_height % 2 == 0
    assert depth_width % 2 == 0

    # TODO this can be done only once #performance
    width_linspace = torch.linspace(-depth_width/2, depth_width/2 - 1, steps=depth_width) # / real_focal_length_x
    height_linspace = torch.linspace(-depth_height/2, depth_height/2 - 1, steps=depth_height) # / real_focal_length_y

    grid_y, grid_x = torch.meshgrid(height_linspace, width_linspace)

    origin_to_z1 = torch.sqrt(1 + (grid_x / real_focal_length_x) ** 2 + (grid_y / real_focal_length_y) ** 2)

    # (1, h, w, 3)
    point_cloud = torch.Tensor(depth_data.shape[1:] + (3,))
    point_cloud[:, :, :, 2] = depth_data[0] / origin_to_z1
    point_cloud[:, :, :, 0] = point_cloud[:, :, :, 2] * grid_x / real_focal_length_x
    point_cloud[:, :, :, 1] = point_cloud[:, :, :, 2] * grid_y / real_focal_length_y

    show = False
    if show:
        x = point_cloud[::5, ::, 0].flatten()
        y = point_cloud[::5, ::, 1].flatten()
        z = point_cloud[::5, ::, 2].flatten()
        show_point_cloud(x, y, z)

    #point_cloud = torch.squeeze(point_cloud, dim=0)

    # (1, h, w, 3) -> (3, 1, h, w)
    point_cloud = point_cloud.permute(3, 0, 1, 2)

    unfold = torch.nn.Unfold(kernel_size=(window_size, window_size))

    # (3, window_size ** 2, (h - window_size//2) * (w - window_size//2))
    unfolded = unfold(point_cloud)

    new_depth_height = depth_height - (window_size//2 * 2)
    new_depth_width = depth_width - (window_size//2 * 2)
    assert unfolded.shape[2] == new_depth_height * new_depth_width

    window_pixels = window_size ** 2
    centered = unfolded - (torch.sum(unfolded, dim=1) / window_pixels).unsqueeze(dim=1)

    # (-1, -1, -1) -> ((h - window_size // 2) * (w - window_size // 2), window_size ** 2, 3)
    centered = centered.permute(2, 1, 0)

    if Config.svd_weighted:
        # the understanding of how the input to SVD becomes 3x3 instead of 25x3
        # https://www.cs.auckland.ac.nz/courses/compsci369s1c/lectures/GG-notes/CS369-LeastSquares.pdf
        # slides 29 and 36
        w_diag = torch.diag_embed(get_gauss_weighted_coeffs_for_window(window_size=window_size, sigma=Config.svd_weighted_sigma))
        c2 = centered.transpose(-2, -1) @ w_diag @ centered
        U, S, V = torch.svd(c2)
    else:
        U, S, V = torch.svd(centered)

    normals = V[:, :, 2]

    normals = normals.reshape(new_depth_height, new_depth_width, 3)

    # flip if z > 0
    where = torch.where(normals[:, :, 2] > 0)
    normals[where[0], where[1]] = -normals[where[0], where[1]]

    # is this necessary?
    normals = normals / torch.norm(normals, dim=2).unsqueeze(dim=2)

    normals = pad_normals(normals, window_size=window_size)
    assert normals.shape[0] == depth_height
    assert normals.shape[1] == depth_width

    return normals


def compute_normals_all(scene: SceneInfo,
                        file_names,
                        read_directory,
                        output_parent_dir,
                        skip_existing=True,
                        impl="svd",
                        old_impl=False):


    print("file names:\n{}".format(file_names))
    print("input dir:\n{}".format(read_directory))

    for depth_data_file_name in file_names:

        print("Processing: {}".format(depth_data_file_name))

        output_directory = "{}/{}".format(output_parent_dir, depth_data_file_name[:-4])
        if skip_existing and os.path.isdir(output_parent_dir):
            print("{} already exists, skipping".format(output_directory))
            continue

        compute_normals(scene,
                        read_directory,
                        depth_data_file_name,
                        output_directory,
                        impl=impl,
                        old_impl=old_impl)


def main():

    Timer.start()
    Config.log()

    #interesting_files = scene_info.imgs_for_comparing_difficulty(0)
    interesting_files = ["frame_0000000015_4.npy"]

    scene_name = "scene1"
    scene_info = SceneInfo.read_scene(scene_name, lazy=True)
    file_names, input_directory = scene_info.get_megadepth_file_names_and_dir(limit=20, interesting_files=interesting_files)

    impl = "svd"
    #impl = "not svd"
    if impl == "svd":
        output_parent_dir = "work/{}/normals/svd".format(scene_name)
    else:
        output_parent_dir = "work/{}/normals/simple_diff_mask".format(scene_name)

    compute_normals_all(scene_info, file_names, input_directory, output_parent_dir=output_parent_dir, skip_existing=False, impl=impl)

    Timer.end()


if __name__ == "__main__":
    main()
