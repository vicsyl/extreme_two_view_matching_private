import numpy as np
import cv2 as cv

from scene_info import read_cameras
from dataclasses import dataclass


@dataclass
class DepthSyntheticData:
    plane: np.ndarray
    Q: np.ndarray
    height: int
    width: int
    file_path: str


def save_depth_map_of_plane(dsd: DepthSyntheticData, allow_and_nullify_negative_depths):

    # plane, Q, height_width, save_path,

    # height = height_width[0]
    # width = height_width[1]

    # TODO count with C != 0
    C = np.array([0, 0, 0, 1])

    Q_inv = np.linalg.inv(dsd.Q)

    m = np.mgrid[0:dsd.width, 0:dsd.height]
    m = np.moveaxis(m, 0, -1)
    m_line = np.ndarray((dsd.width, dsd.height, 3, 1))
    m_line[:, :, :2, 0] = m
    m_line[:, :, 2, 0] = 1.0

    d = Q_inv @ m_line

    f_norms = np.expand_dims(np.linalg.norm(d, axis=2), axis=2)
    f_normed = d * f_norms

    f_t = np.moveaxis(f_normed, -1, -2)

    # depth = Q^(-1).d_unit
    depth = np.dot(f_t, dsd.plane[:3]) / -np.dot(C, dsd.plane)

    min_d = np.min(depth)
    if min_d < 0:
        if allow_and_nullify_negative_depths:
            depth = np.where(depth > 0, depth, 0)
        else:
            raise Exception("depth < 0")

    depth = depth / np.max(depth) * 255.0
    depth = np.swapaxes(depth, 0, 1)

    cv.imwrite(dsd.file_path, depth)


def get_file_name(plane):
    plane = plane[:3]
    file_name_fuffix = "_".join(map(str, plane.tolist()))
    return "work/tests/normals_{}.png".format(file_name_fuffix)


def get_depth_synthetic_data():

    planes = np.array([
        [1, 0, 1, -1],
        [0, 1, 1, -1],
        [1, 2, 2, -1],
        [1, 3, 3, -1],
        [1, 1, 1, -1],
        [2, 1, 2, -1],
        [3, 1, 3, -1],
    ])

    cameras = read_cameras("scene1")
    first_camera = next(iter(cameras.values()))
    first_K = first_camera.get_K()
    height, width = first_camera.height_width
    Q = first_K

    return [DepthSyntheticData(plane=plane,
                               width=width,
                               height=height,
                               Q=Q,
                               file_path=get_file_name(plane))
            for plane in planes]


def test_depth():
    for dsd in get_depth_synthetic_data():
        save_depth_map_of_plane(dsd, False)


if __name__ == "__main__":
    test_depth()
