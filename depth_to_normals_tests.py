import torch
import numpy as np
import cv2 as cv

from config import Config

from scene_info import read_cameras, CameraEntry
from dataclasses import dataclass
from depth_to_normals import *

from img_utils import show_normals_components
from utils import read_depth_data

@dataclass
class DepthSyntheticData:
    plane: np.ndarray
    camera: CameraEntry
    file_dir_and_name: tuple


def depth_map_of_plane(dsd: DepthSyntheticData, allow_and_nullify_negative_depths, save=True):

    # plane, Q, height_width, save_path,

    # height = height_width[0]
    # width = height_width[1]

    # TODO count with C != 0
    C = np.array([0, 0, 0, 1])

    height = 512
    width = 288

    K = dsd.camera.get_K()
    down_sample_factor_x = width / dsd.camera.height_width[1]
    down_sample_factor_y = height / dsd.camera.height_width[0]
    K[0] = K[0] * down_sample_factor_x
    K[1] = K[1] * down_sample_factor_y

    Q_inv = np.linalg.inv(K)

    m = np.mgrid[0:width, 0:height]
    m = np.moveaxis(m, 0, -1)
    m_line = np.ndarray((width, height, 3, 1))
    m_line[:, :, :2, 0] = m
    m_line[:, :, 2, 0] = 1.0

    d = Q_inv @ m_line

    f_norms = np.expand_dims(np.linalg.norm(d, axis=2), axis=2)
    d_unit = d / f_norms

    d_unit_t = np.moveaxis(d_unit, -1, -2)

    # depth = (C^T, 1).w / (Q^(-1).d_unit)^T.w[:3]
    depth = -np.dot(C, dsd.plane) / np.dot(d_unit_t, dsd.plane[:3])

    min_d = np.min(depth)
    if min_d < 0:
        if allow_and_nullify_negative_depths:
            depth = np.where(depth > 0, depth, 0)
        else:
            raise Exception("depth < 0")

    #depth = depth / np.max(depth) * 255.0
    depth = np.swapaxes(depth, 0, 1).astype(np.float32)

    if save:
        np.save("{}/{}".format(dsd.file_dir_and_name[0], dsd.file_dir_and_name[1]), depth)

    return depth


def get_file_dir_and_name(plane):
    #plane = plane[:3]
    file_name_fuffix = "_".join(map(str, plane.tolist())).replace("-", "m")
    return "work/tests/", "depth_map_{}.npy".format(file_name_fuffix)


def test_depth_to_normals(old_implementation=True, impl="svd"):

    Config.config_map[Config.show_normals_in_img] = False

    planes_coeffs = np.array([
        [0, 0, 1, -1],
        [0, 0, 1, -100],
        [1, 0, 1, -1],
        [1, 0, 1, -100000],
        [0, 1, 1, -1],
        [1, 2, 2, -1],
        [1, 3, 3, -1],
        [1, 1, 1, -1],
        [2, 1, 2, -1],
        [3, 1, 3, -1],
    ])

    cameras = read_cameras("scene1")
    first_camera = next(iter(cameras.values()))

    for plane in planes_coeffs:

        #plane = -plane

        exact = plane[:3].copy()
        exact = -exact / np.linalg.norm(exact)
        print("\n\n\n\nTesting synthetic depth map for plane coeffs (ax + by + cz + d = 0): {}".format(plane))

        dsd = DepthSyntheticData(plane=plane, camera=first_camera, file_dir_and_name=get_file_dir_and_name(plane))
        depth_map_of_plane(dsd, allow_and_nullify_negative_depths=False, save=True)

        mask = torch.tensor([[0.5, 0, -0.5]]).float()

        depth_data = read_depth_data(dsd.file_dir_and_name[1], dsd.file_dir_and_name[0])

        if impl == "svd":
            normals = compute_normals_from_svd(dsd.camera,
                                               depth_data,
                                               output_directory=dsd.file_dir_and_name[0],
                                               img_name=dsd.file_dir_and_name[1],
                                               )

        else:
            normals = compute_normals_convolution(dsd.camera,
                                                  depth_data,
                                                  output_directory=dsd.file_dir_and_name[0],
                                                  img_name=dsd.file_dir_and_name[1],
                                                  override_mask=mask,
                                                  old_implementation=old_implementation
                                                  )

        clustered_normals, normal_indices = \
            cluster_and_save_normals(normals,
                                     depth_data_file_name=dsd.file_dir_and_name[1],
                                     output_directory=dsd.file_dir_and_name[0])

        normals_diff = normals - dsd.plane[:3]
        show_normals_components(normals_diff, "difference from exact result", (30.0, 20.0))

        if len(normals.shape) == 5:
            normals = normals.squeeze(dim=0).squeeze(dim=0)

        normals_np = normals.numpy()[5:-5, 5:-5]
        diff = normals_np - exact

        maxima = np.max(normals_np, axis=(0, 1))
        minima = np.min(normals_np, axis=(0, 1))
        max_dev = maxima - exact
        min_dev = minima - exact

        print("exact normal of the plane (x, y, y): {}".format(exact))
        print("normal component maxima (x, y, y): {}".format(maxima))
        print("normal component minima (x, y, y): {}".format(minima))
        print("difference of normal component maxima from the exact values (x, y, y): {}".format(max_dev))
        print("difference of normal component minima from the exact values (x, y, y): {}".format(min_dev))

        # assert np.all(normal_indices == 0.0)
        # assert normals.shape[0] == 1
        # assert np.equal(normals[0], dsd.plane[:3])


if __name__ == "__main__":
    test_depth_to_normals(old_implementation=False)
    #test_depth_to_normals(old_implementation=False)
