import math
import numpy as np
import os
import cv2 as cv
from pipeline import decolorize
from utils import get_files
import matplotlib.pyplot as plt
from scene_info import read_cameras

def get_rotation_matrix(unit_rotation_vector, theta):

    # Rodrigues formula
    # R = I + sin(theta) . K + (1 - cos(theta)).K**2

    K = np.array([
        [0.0, -unit_rotation_vector[2], unit_rotation_vector[1]],
        [unit_rotation_vector[2], 0.0, -unit_rotation_vector[0]],
        [-unit_rotation_vector[1], unit_rotation_vector[0], 0.0],
    ])
    a = np.eye(3)
    b = math.sin(theta) * K
    c = (1.0 - math.cos(theta)) * K @ K
    return a + b + c


def get_rectification_rotations(normals):

    # now the normals will be "from" me, "inside" the surfaces
    normals = -normals

    z = np.array([0.0, 0.0, 1.0])
    Rs = []

    for i, normal in enumerate(normals):
        assert normal[2] > 0
        rotation_vector = np.cross(normal, z)
        rotation_vector_norm = sin_theta = np.linalg.norm(rotation_vector)
        unit_rotation_vector = rotation_vector / rotation_vector_norm
        theta = math.asin(sin_theta)

        R = get_rotation_matrix(unit_rotation_vector, theta)
        det = np.linalg.det(R)
        assert math.fabs(det - 1.0) < 0.0001
        Rs.append(R)

    return Rs


def show_rectifications(directory, limit):

    cameras = read_cameras("scene1")
    K = cameras[1801].get_K()
    K_inv = np.linalg.inv(K)

    file_names = get_files(directory, ".jpg", limit)

    for file_name in file_names:

        #TODO the original suffix is not stripped
        img_file = '{}/{}'.format(directory, file_name)
        normals_file = '{}/{}_normals.txt'.format(directory, file_name[:-4])

        if not os.path.isfile(normals_file):
            print("{} doesn't exist!".format(normals_file))
            continue

        normals = np.loadtxt(normals_file, delimiter=',')
        img = cv.imread(img_file, None)
        h, w, _ = img.shape
        src = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        Rs = get_rectification_rotations(normals)

        for i, R in enumerate(Rs):

            T = K @ R @ K_inv

            dst = cv.perspectiveTransform(src, T)
            mins = (np.min(dst[:, 0, 0]), np.min(dst[:, 0, 1]))
            if mins[0] < 0 or mins[1] < 0:
                translate = np.array([
                    [1, 0, -mins[0]],
                    [0, 1, -mins[1]],
                    [0, 0, 1],
                ])
                T = translate @ T
                dst = cv.perspectiveTransform(src, T)

            print("rotation: \n {}".format(R))
            print("transformation: \n {}".format(T))
            print("src: \n {}".format(src))
            print("dst: \n {}".format(dst))

            maxs_precise = (np.max(dst[:, 0, 0]), np.max(dst[:, 0, 1]))
            rotated = cv.warpPerspective(img, T, maxs_precise)
            img_rectified = cv.polylines(decolorize(img), [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)

            plt.figure()
            plt.title("normal {}".format(normals[i]))
            plt.imshow(rotated)
            plt.show()

            plt.imshow(img_rectified)
            plt.show()


if __name__ == "__main__":

    show_rectifications("work/cluster_transformations", limit=1)
