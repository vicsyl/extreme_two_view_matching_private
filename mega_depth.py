import math
import numpy as np
import os
import torch
import cv2 as cv
import time
from scene_info import read_cameras, read_images

def read_depth_data(directory, limit=None):

    data_map = {}

    filenames = [filename for filename in sorted(os.listdir(directory)) if filename.endswith(".txt")]
    if limit is not None:
        filenames = filenames[0:limit]

    for filename in filenames:
        np_depth = np.loadtxt('{}/{}'.format(directory, filename), delimiter=',')
        depth_data = torch.from_numpy(np_depth)
        data_map[filename[:-4]] = depth_data

    return data_map


def normal_from_reprojected(reprojected_data, coord, window_size=5):

    (u, v) = coord

    assert window_size % 2 == 1

    pixel_count = window_size ** 2
    from_minus = int((window_size - 1) / 2)
    to_plus = int((window_size - 1) / 2 + 1)
    xs = reprojected_data[0, u - from_minus:u + to_plus, v - from_minus:v + to_plus].reshape(pixel_count)
    ys = reprojected_data[1, u - from_minus:u + to_plus, v - from_minus:v + to_plus].reshape(pixel_count)
    zs = reprojected_data[2, u - from_minus:u + to_plus, v - from_minus:v + to_plus].reshape(pixel_count)

    centered_xs = xs - torch.sum(xs) / float(pixel_count)
    centered_ys = ys - torch.sum(ys) / float(pixel_count)
    centered_zs = zs - torch.sum(zs) / float(pixel_count)
    to_decompose = torch.stack((centered_xs, centered_ys, centered_zs), dim=1)
    U, S, V = torch.svd(to_decompose)
    normal = V[:, 2]

    norm = torch.norm(normal)
    #print("norm: {}".format(norm))
    normal = normal / norm

    return normal

def test_reproject_project(depth_data_map, cameras, images, reprojected_data):

    for dict_idx, depth_data_file in enumerate(depth_data_map):

        camera_id = images[depth_data_file]["camera_id"]
        camera = cameras[camera_id]
        focal_point_length = camera['focal_length']
        width = camera["width"]
        height = camera["height"]
        principal_point_x = camera["principal_point_x"]
        principal_point_y = camera["principal_point_y"]

        # TODO centralize
        depth_data = depth_data_map[depth_data_file]
        depth_data = depth_data.view(1, 1, depth_data.shape[0], depth_data.shape[1])
        upsampling = torch.nn.Upsample(size=(height, width), mode='bilinear')
        depth_data = upsampling(depth_data)

        xs = reprojected_data[dict_idx, 0] / reprojected_data[dict_idx, 2]
        ys = reprojected_data[dict_idx, 1] / reprojected_data[dict_idx, 2]

        width_linspace = torch.linspace(0 - principal_point_x, width - 1 - principal_point_x, steps=width) / focal_point_length
        height_linspace = torch.linspace(0 - principal_point_y, height - 1 - principal_point_y, steps=height) / focal_point_length

        grid_y, grid_x = torch.meshgrid(height_linspace, width_linspace)

        diff_y = ys - grid_y
        norm_y = torch.norm(diff_y)
        diff_x = xs - grid_x
        norm_x = torch.norm(diff_x)

        assert norm_x.item() < 0.1
        assert norm_y.item() < 0.1

    print("test_reproject_project ok")

def reproject(depth_data_map, cameras, images):
    """
    :param depth_data_map:
    :param cameras:
    :param images:
    :return: torch.tensor (B, 3, H, W) - beware it in the order of x, y, z
    """

    # "model": model,
    # "width": width,
    # "height": height,
    # "focal_length": focal_length,
    # "principal_point_x": principal_point_x,
    # "principal_point_y": principal_point_y,
    # "distortion": distortion,

    ret = None

    for dict_idx, depth_data_file in enumerate(depth_data_map):

        camera_id = images[depth_data_file]["camera_id"]
        camera = cameras[camera_id]
        focal_point_length = camera['focal_length']
        width = camera["width"]
        height = camera["height"]
        principal_point_x = camera["principal_point_x"]
        principal_point_y = camera["principal_point_y"]

        if ret is None:
            ret = torch.zeros(len(depth_data_map), 3, height, width)

        depth_data = depth_data_map[depth_data_file]
        depth_data = depth_data.view(1, 1, depth_data.shape[0], depth_data.shape[1])
        upsampling = torch.nn.Upsample(size=(height, width), mode='bilinear')
        depth_data = upsampling(depth_data)

        width_linspace = torch.linspace(0 - principal_point_x, width - 1 - principal_point_x, steps=width)
        height_linspace = torch.linspace(0 - principal_point_y, height - 1 - principal_point_y, steps=height)

        grid_y, grid_x = torch.meshgrid(height_linspace, width_linspace)

        projection_distances_from_origin = torch.sqrt(1 + torch.sqrt((grid_x / focal_point_length) ** 2 + (grid_y / focal_point_length) ** 2))
        zs = depth_data / projection_distances_from_origin
        xs = grid_x * zs / focal_point_length
        ys = grid_y * zs / focal_point_length

        ret[dict_idx, 0] = xs
        ret[dict_idx, 1] = ys
        ret[dict_idx, 2] = zs

    return ret


if __name__ == "__main__":

    depth_data_map = read_depth_data("depth_data/mega_depth/scene1", limit=1)
    cameras = read_cameras("scene1")
    images = read_images("scene1")
    reprojected_data = reproject(depth_data_map, cameras, images)

    test_reproject_project(depth_data_map, cameras, images, reprojected_data)

    single_file = next(iter(depth_data_map))
    camera_id = images[single_file]["camera_id"]
    camera = cameras[camera_id]
    focal_length = camera['focal_length']
    principal_point_x = camera["principal_point_x"]
    principal_point_y = camera["principal_point_y"]

    #image = cv.line(img, (0, 0), (1080, 1920), (255, 255, 255), thickness=5)

    start_time = time.time()
    print("clock started")

    counter = 0
    normals = set()

    window_sizes = [5, 7, 9, 11, 13]

    for window_size in window_sizes:
        img = cv.imread('original_dataset/scene1/images/{}.jpg'.format(single_file))
        for y in range(window_size, 1920 - window_size, 1):
            for x in range(window_size, 1080 - window_size, 1):

                counter = counter + 1

                normal = normal_from_reprojected(reprojected_data[0], (y, x), window_size=window_size)

                X = reprojected_data[0, :, y, x]
                to_project = X + normal / focal_length * 10

                u = (to_project[0] / to_project[2]).item() * focal_length + principal_point_x
                v = (to_project[1] / to_project[2]).item() * focal_length + principal_point_y

                start = (x, y)
                end = (round(u), round(v))
                rgb = normal.tolist()
                if counter % 1000 == 0:
                    print("Drawing {}, {} for window size: {}".format(y, x, window_size))
                #image = cv.line(img, start, end, (255, 255, 255), thickness=1)

                point = (x, y)
                norm = torch.norm(normal)
                rgb_from_normal = [
                    int(255 * (normal[0] / norm).item()),
                    int(255 * (normal[1] / norm).item()),
                    int(255 * (normal[2] / norm).item()),
                ]
                img[y, x] = rgb_from_normal

                normals.add(tuple(rgb_from_normal))


        cv.imwrite('work/normals_window_size_{}.jpg'.format(window_size), img)

    end_time = time.time()
    print("done. Elapsed time: {}".format(end_time - start_time))
