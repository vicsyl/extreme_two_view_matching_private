import numpy as np
import cv2 as cv

from scene_info import read_cameras, CameraEntry
from dataclasses import dataclass
from depth_to_normals import compute_normals_simple_diff_convolution_simple


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

    Q_inv = np.linalg.inv(dsd.camera.get_K())

    height = dsd.camera.height_width[0]
    width = dsd.camera.height_width[1]

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
    depth = np.swapaxes(depth, 0, 1)

    if save:
        np.save("{}/{}".format(dsd.file_dir_and_name[0], dsd.file_dir_and_name[1]), depth)


def get_file_dir_and_name(plane):
    #plane = plane[:3]
    file_name_fuffix = "_".join(map(str, plane.tolist())).replace("-", "m")
    return "work/tests/", "depth_map_{}.npy".format(file_name_fuffix)


def get_depth_synthetic_data():

    planes = np.array([
        [0, 0, 1, -100],
        [0, 0, 1, -1],
        # [1, 0, 1, -1],
        # [1, 0, 1, -100000],
        # [0, 1, 1, -1],
        # [1, 2, 2, -1],
        # [1, 3, 3, -1],
        # [1, 1, 1, -1],
        # [2, 1, 2, -1],
        # [3, 1, 3, -1],
    ])

    cameras = read_cameras("scene1")
    first_camera = next(iter(cameras.values()))

    return [DepthSyntheticData(plane=plane,
                               camera=first_camera,
                               file_dir_and_name=get_file_dir_and_name(plane))
            for plane in planes]


def generate_depth_info():
    return [depth_map_of_plane(dsd, False) for dsd in get_depth_synthetic_data()]


def test_depth_to_normals():

    normals = None
    normals_old = None
    for dsd in get_depth_synthetic_data():
        depth_map_of_plane(dsd, False)

        h = dsd.camera.height_width[0]
        w = dsd.camera.height_width[1]
        f = dsd.camera.focal_length

        if normals is not None:
            normals_old = normals
        depth, normals, clustered_normals, normal_indices = compute_normals_simple_diff_convolution_simple(h, w, f, dsd.file_dir_and_name[0], dsd.file_dir_and_name[1], save=True, output_directory=dsd.file_dir_and_name[0])

        if normals_old is not None:
            diff = normals - normals_old
            print()
        print()
        # assert np.all(normal_indices == 0.0)
        # assert normals.shape[0] == 1
        # assert np.equal(normals[0], dsd.plane[:3])


if __name__ == "__main__":
    test_depth_to_normals()
