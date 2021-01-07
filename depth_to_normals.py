import numpy as np
import os
import torch
import cv2 as cv
import time
from scene_info import read_cameras, read_images
from image_processing import spatial_gradient_first_order


def get_files(dir, suffix, limit=None):
    filenames = [filename for filename in sorted(os.listdir(dir)) if filename.endswith(suffix)]
    if limit is not None:
        filenames = filenames[0:limit]
    return filenames


def read_depth_data_np(directory, limit=None):

    data_map = {}

    filenames = get_files(directory, ".npy", limit)

    for filename in filenames:
        np_depth = np.load('{}/{}'.format(directory, filename))
        depth_data = torch.from_numpy(np_depth)
        data_map[filename[:-4]] = depth_data

    return data_map


def read_depth_data_txt(directory, limit=None):

    data_map = {}

    filenames = get_files(directory, ".txt", limit)
    # filenames = [filename for filename in sorted(os.listdir(directory)) if filename.endswith(".txt")]
    # if limit is not None:
    #     filenames = filenames[0:limit]

    for filename in filenames:
        np_depth = np.loadtxt('{}/{}'.format(directory, filename), delimiter=',')
        depth_data = torch.from_numpy(np_depth)
        data_map[filename[:-4]] = depth_data

    return data_map


def svd_normal_from_reprojected_test(reprojected_data, coord, window_size=5):

    assert window_size % 2 == 1

    pixel_count = window_size ** 2

    xs2 = torch.zeros((window_size, window_size))
    ys2 = torch.zeros((window_size, window_size))
    zs2 = torch.zeros((window_size, window_size))

    for x_i in range(window_size):
        for y_i in range(window_size):
            xs2[x_i, y_i] = x_i
            ys2[x_i, y_i] = y_i
            zs2[x_i, y_i] = x_i

    xs = xs2.reshape(pixel_count)
    ys = ys2.reshape(pixel_count)
    zs = zs2.reshape(pixel_count)

    centered_xs = xs - torch.sum(xs) / float(pixel_count)
    centered_ys = ys - torch.sum(ys) / float(pixel_count)
    centered_zs = zs - torch.sum(zs) / float(pixel_count)
    to_decompose = torch.stack((centered_xs, centered_ys, centered_zs), dim=1)
    U, S, V = torch.svd(to_decompose)
    normal = -V[2, :]

    norm = torch.norm(normal)
    #print("norm: {}".format(norm))
    normal = normal / norm

    return normal


def svd_normal_from_reprojected(reprojected_data, coord, window_size=5):

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
    normal = -V[2, :]

    norm = torch.norm(normal)
    #print("norm: {}".format(norm))
    normal = normal / norm

    return normal

def test_reproject_project(depth_data_map, cameras, images, reprojected_data):

    for dict_idx, depth_data_file in enumerate(depth_data_map):

        camera_id = images[depth_data_file].camera_id
        camera = cameras[camera_id]
        focal_point_length = camera.focal_length
        width = camera.height_width[1]
        height = camera.height_width[0]
        principal_point_x = camera.principal_point_x_y[0]
        principal_point_y = camera.principal_point_x_y[1]

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

def upsample_depth_data(depth_data, shape_h_w):

    (height, width) = shape_h_w
    depth_data = depth_data.view(1, 1, depth_data.shape[0], depth_data.shape[1])
    upsampling = torch.nn.Upsample(size=(height, width), mode='bilinear')
    depth_data = upsampling(depth_data)
    return depth_data


'''
THIS is actually not that simple
'''
def reproject_test_simple_planes(depth_data_map, cameras, images):

    """
    :param depth_data_map:
    :param cameras:
    :param images:
    :return: torch.tensor (B, 3, H, W) - beware it in the order of x, y, z
    """
    ret = None
    zs_c = 1

    for dict_idx, depth_data_file in enumerate(depth_data_map):

        camera_id = images[depth_data_file].camera_id
        camera = cameras[camera_id]
        focal_point_length = camera.focal_length
        width = camera.height_width[1]
        height = camera.height_width[0]
        principal_point_x = camera.principal_point_x_y[0]
        principal_point_y = camera.principal_point_x_y[1]

        if ret is None:
            ret = torch.zeros(len(depth_data_map), 3, height, width)

        width_linspace = torch.linspace(0 - principal_point_x, width - 1 - principal_point_x, steps=width)
        height_linspace = torch.linspace(0 - principal_point_y, height - 1 - principal_point_y, steps=height)

        grid_y, grid_x = torch.meshgrid(height_linspace, width_linspace)

        projection_distances_from_origin = torch.sqrt(1 + torch.sqrt((grid_x / focal_point_length) ** 2 + (grid_y / focal_point_length) ** 2))
        zs = (1 / (1 - 0.1 * grid_x / focal_point_length))
        xs = grid_x * zs_c / focal_point_length
        ys = grid_y * zs_c / focal_point_length

        ret[dict_idx, 0] = xs
        ret[dict_idx, 1] = ys
        ret[dict_idx, 2] = zs

    return ret



def reproject(depth_data_map, cameras, images):
    """
    :param depth_data_map:
    :param cameras:
    :param images:
    :return: torch.tensor (B, 3, H, W) - be aware it is in the order of x, y, z
    """
    ret = None

    for dict_idx, depth_data_file in enumerate(depth_data_map):

        camera_id = images[depth_data_file].camera_id
        camera = cameras[camera_id]
        focal_point_length = camera.focal_length
        width = camera.height_width[1]
        height = camera.height_width[0]
        principal_point_x = camera.principal_point_x_y[0]
        principal_point_y = camera.principal_point_x_y[1]

        if ret is None:
            ret = torch.zeros(len(depth_data_map), 3, height, width)

        depth_data = depth_data_map[depth_data_file]
        depth_data = upsample_depth_data(depth_data, (height, width))
        # depth_data = depth_data.view(1, 1, depth_data.shape[0], depth_data.shape[1])
        # upsampling = torch.nn.Upsample(size=(height, width), mode='bilinear')
        # depth_data = upsampling(depth_data)

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


def svd_normals():

    depth_data_map = read_depth_data_np("depth_data/mega_depth/scene1", limit=3)
    cameras = read_cameras("scene1")
    images = read_images("scene1")
    reprojected_data = reproject(depth_data_map, cameras, images)
    #reprojected_data = reproject_test_simple_planes(depth_data_map, cameras, images)
    test_reproject_project(depth_data_map, cameras, images, reprojected_data)

    for file_name in depth_data_map:

        #single_file = next(iter(depth_data_map))
        camera_id = images[file_name].camera_id
        camera = cameras[camera_id]
        focal_length = camera.focal_length
        principal_point_x = camera.principal_point_x_y[0]
        principal_point_y = camera.principal_point_x_y[1]

        window_sizes = [5] #, 7, 9, 11, 13]
        counter = 0

        for window_size in window_sizes:
            img = cv.imread('original_dataset/scene1/images/{}.jpg'.format(file_name))
            for y in range(window_size + 2, 1920 - window_size, 1):
                for x in range(window_size + 3, 1080 - window_size, 1):

                    counter = counter + 1
                    normal = svd_normal_from_reprojected(reprojected_data[0], (y, x), window_size=window_size)

                    X = reprojected_data[0, :, y, x]
                    #print(X)
                    to_project = X + normal / focal_length * 35
                    u = (to_project[0] / to_project[2]).item() * focal_length + principal_point_x
                    v = (to_project[1] / to_project[2]).item() * focal_length + principal_point_y
                    #cv.line(img, (x, y), (int(u), int(v)), color=(255, 255, 255), thickness=2)

                    if counter % 1000 == 0:
                        print("Drawing {}, {} for window size: {}".format(y, x, window_size))

                    norm = torch.norm(normal)
                    rgb_from_normal = [
                        int(255 * (normal[0] / norm).item()),
                        int(255 * (normal[1] / norm).item()),
                        int(255 * (normal[2] / norm).item()),
                    ]
                    img[y, x] = rgb_from_normal

            cv.imwrite('work/{}_normals_svd_window_size_{}.jpg'.format(file_name, window_size), img)


def diff_normal_from_depth_data(focal_length, depth_data_map, smoothed: bool=False, sigma: float=1.0):

    # Could be also done from reprojected data, but this seems to be correct and more straghtforward
    to_grad = depth_data_map
    gradient_dzdx, gradient_dzdy = spatial_gradient_first_order(to_grad, smoothed=smoothed, sigma=sigma)
    gradient_dzdx = (gradient_dzdx * 30000).unsqueeze(dim=4)
    gradient_dzdy = (gradient_dzdy * 30000).unsqueeze(dim=4)
    z_ones = torch.ones(gradient_dzdy.shape)
    normals = torch.cat((-gradient_dzdx, -gradient_dzdy, -z_ones), dim=4)
    normals_norms = torch.norm(normals, dim=4).unsqueeze(dim=4)
    normals = normals / normals_norms

    return normals


def save_diff_normals(normals, img_file_name, start_time, camera, reprojected_data, out_suffix):

    focal_length = camera.focal_length
    principal_point_x = camera.principal_point_x_y[0]
    principal_point_y = camera.principal_point_x_y[0]

    end_time = time.time()
    print("done. Elapsed time: {}".format(end_time - start_time))
    img = normals[0, 0].numpy() * 255
    img[:, :, 2] = -img[:, :, 2]
    cv.imwrite('work/normals_diff_normals_colors_fixed_{}.png'.format(out_suffix), img)

    # img = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_file_name))

    # TODO remove the loop and do it with the help of O(1) matrix operations (and the filter in a loop)
    counter = 0
    for y in range(0, 1920, 20):
        for x in range(0, 1080, 20):
            counter = counter + 1
            X = reprojected_data[0, :, y, x]

            to_project = X # + normals[0, 0, y, x] / focal_length
            u = (to_project[0] / to_project[2]).item() * focal_length + principal_point_x
            v = (to_project[1] / to_project[2]).item() * focal_length + principal_point_y
            color = normals[0, 0, y, x].tolist()
            if counter % 100 == 0:
                print("Drawing {}, {}".format(y, x))
            cv.line(img, (x, y), (int(u), int(v)), color=(255, 255, 255), thickness=1)

    cv.imwrite('work/normals_diff_normals_fixed_{}.png'.format(out_suffix), img)

    end_time = time.time()
    print("Elapsed time: {}".format(end_time - start_time))


def save_diff_normals_different_windows():

    #depth_data_map = read_depth_data_txt("depth_data/mega_depth/scene1", limit=1)
    depth_data_map = read_depth_data_np("depth_data/mega_depth/scene1", limit=1)

    end_time = time.time()
    print("done. Elapsed time: {}".format(end_time - start_time))

    cameras = read_cameras("scene1")
    images = read_images("scene1")

    end_time = time.time()
    print("done. Elapsed time: {}".format(end_time - start_time))

    reprojected_data = reproject(depth_data_map, cameras, images)
    test_reproject_project(depth_data_map, cameras, images, reprojected_data)

    end_time = time.time()
    print("done. Elapsed time: {}".format(end_time - start_time))

    single_file = next(iter(depth_data_map))
    camera_id = images[single_file].camera_id
    camera = cameras[camera_id]
    focal_length = camera.focal_length
    width = camera.height_width[1]
    height = camera.height_width[0]

    end_time = time.time()
    print("done. Elapsed time: {}".format(end_time - start_time))

    depth_data_single_file = depth_data_map[single_file]
    depth_data_single_file = upsample_depth_data(depth_data_single_file, (height, width))

    end_time = time.time()
    print("done. Elapsed time: {}".format(end_time - start_time))


    normals_modes = [
        diff_normal_from_depth_data(focal_length, depth_data_single_file),
        diff_normal_from_depth_data(focal_length, depth_data_single_file, True, 1.0),
        diff_normal_from_depth_data(focal_length, depth_data_single_file, True, 3.0),
        diff_normal_from_depth_data(focal_length, depth_data_single_file, True, 5.0),
        diff_normal_from_depth_data(focal_length, depth_data_single_file, True, 7.0),
        diff_normal_from_depth_data(focal_length, depth_data_single_file, True, 9.0),
        diff_normal_from_depth_data(focal_length, depth_data_single_file, True, 11.0),
        ]
    out_suffixes = [
        "unsmoothed",
        "sigma_1",
        "sigma_3",
        "sigma_5",
        "sigma_7",
        "sigma_9",
        "sigma_11",
    ]

    for idx, normals in enumerate(normals_modes):
        save_diff_normals(normals, single_file, start_time, camera, reprojected_data, out_suffixes[idx])


def sobel_normals_5x5():

    depth_data_map = read_depth_data_np("depth_data/mega_depth/scene1", limit=10)

    cameras = read_cameras("scene1")
    images = read_images("scene1")

    # reprojected_data = reproject(depth_data_map, cameras, images)
    # test_reproject_project(depth_data_map, cameras, images, reprojected_data)

    for file_name in depth_data_map:
        #single_file = next(iter(depth_data_map))
        camera_id = images[file_name].camera_id
        camera = cameras[camera_id]
        focal_length = camera.focal_length
        width = camera.height_width[1]
        height = camera.height_width[0]

        depth_data = depth_data_map[file_name]
        depth_data = upsample_depth_data(depth_data, (height, width))

        cv_img = depth_data.squeeze(dim=0).squeeze(0).unsqueeze(2).numpy()

        sobelx = cv.Sobel(cv_img, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(cv_img, cv.CV_64F, 0, 1, ksize=5)

        sobelx = (torch.from_numpy(sobelx) * 50).unsqueeze(2)
        sobely = (torch.from_numpy(sobely) * 50).unsqueeze(2)
        z_ones = torch.ones(sobelx.shape)
        normals = torch.cat((-sobelx, -sobely, -z_ones), dim=2)
        normals_norms = torch.norm(normals, dim=2).unsqueeze(dim=2)
        normals = normals / normals_norms

        img = normals.numpy() * 255
        img[:, :, 2] = -img[:, :, 2]
        cv.imwrite('work/normals_sobel_normals_colors_fixed.png', img)

        # def diff_normal_from_depth_data(focal_length, depth_data_map, smoothed: bool = False, sigma: float = 1.0):
        #     # Could be also done from reprojected data, but this seems to be correct and more straghtforward
        #     to_grad = depth_data_map
        #     gradient_dzdx, gradient_dzdy = spatial_gradient_first_order(to_grad, smoothed=smoothed, sigma=sigma)
        #     gradient_dzdx = (gradient_dzdx * 30000).unsqueeze(dim=4)
        #     gradient_dzdy = (gradient_dzdy * 30000).unsqueeze(dim=4)
        #     z_ones = torch.ones(gradient_dzdy.shape)
        #     normals = torch.cat((-gradient_dzdx, -gradient_dzdy, -z_ones), dim=4)
        #     normals_norms = torch.norm(normals, dim=4).unsqueeze(dim=4)
        #     normals = normals / normals_norms
        #
        #     return normals


if __name__ == "__main__":

    start_time = time.time()
    print("clock started")

    #svd_normals()
    #save_diff_normals_different_windows()
    sobel_normals_5x5()

    end_time = time.time()
    print("done. Elapsed time: {}".format(end_time - start_time))
