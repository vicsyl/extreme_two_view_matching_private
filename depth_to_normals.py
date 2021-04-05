import cv2 as cv
from scene_info import read_cameras, read_images, SceneInfo
from image_processing import spatial_gradient_first_order
from tests import *
from utils import *
import matplotlib.pyplot as plt

import spherical_kmeans
from pathlib import Path
from clusters_map import clusters_map


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
  - in some functions I use the spherical k-means, which I implemented in 'spherical_kmeans.py' 
  - in the original paper they also say that they look for "connected regions of points whose normals belong to the same cluster" - not sure how this was done 
    but obviously some effort can be done in this respect to further smooth out the clusters (enlarge them)
  - I am actually confused, but still convinced that I cannot predict normals correctly if I have only unscaled depth info - I would need to have scaled depth info
    and then normalize it according to the focal point length (which I have)
  - there are other details that may influence the result, as upsampling the depth map to the original size of the image before doing anything else, 
    however I don't think this particular one would have any significant effect
  - actually maybe I should have tested with other (e.g. by the real monodepth) CNN's depth maps         
    
 
"""
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


#TODO remove - logic moved to resize.upsample
def upsample_depth_data(depth_data, shape_h_w):

    (height, width) = shape_h_w
    depth_data = depth_data.view(1, 1, depth_data.shape[0], depth_data.shape[1])
    upsampling = torch.nn.Upsample(size=(height, width), mode='bilinear')
    depth_data = upsampling(depth_data)
    return depth_data


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
    test_reproject_project_old(depth_data_map, cameras, images, reprojected_data)

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


# TODO centralize the logic around the "factor"
def diff_normal_from_depth_data(focal_length,
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

    return normals


# TODO centralize the logic around the "factor"
def normal_from_sobel_and_depth_data(depth_data, size):

    # sobel size x size - imput numpy, output numpy, mask unknown
    # normals: (u, v, n_x, n_y, n_z)
    cv_img = depth_data.squeeze(dim=0).squeeze(0).unsqueeze(2).numpy()
    sobelx = cv.Sobel(cv_img, cv.CV_64F, 1, 0, ksize=size)
    sobely = cv.Sobel(cv_img, cv.CV_64F, 0, 1, ksize=size)
    sobelx = (torch.from_numpy(sobelx) * 50).unsqueeze(2)
    sobely = (torch.from_numpy(sobely) * 50).unsqueeze(2)
    z_ones = torch.ones(sobelx.shape)
    normals = torch.cat((-sobelx, -sobely, -z_ones), dim=2)
    normals_norms = torch.norm(normals, dim=2).unsqueeze(dim=2)
    normals = normals / normals_norms
    return normals


def cluster_and_save_normals(normals,
                             depth_data_file_name,
                             output_directory,
                             show=False,
                             title=None,
                             save=False,
                             angle_threshold=4*math.pi/9):

    file_name_prefix = '{}/{}'.format(output_directory, depth_data_file_name[:-4])
    if clusters_map.__contains__(depth_data_file_name[:-4]):
        clusters = clusters_map[depth_data_file_name[:-4]]
    else:
        clusters = 2
        print("WARNING: number of clusters unknown for {}, defaulting to 2 ...".format(depth_data_file_name))

    # TODO just confirm if this happens for monodepth
    if len(normals.shape) == 5:
        normals = normals.squeeze(dim=0).squeeze(dim=0)
        img = normals.numpy() * 255
    img[:, :, 2] = -img[:, :, 2]

    if show:
        plt.figure()
        plt.title(title)
        plt.imshow(img / 255)
        plt.show()
    if save:
        cv.imwrite("{}.jpg".format(file_name_prefix), img)

    minus_z_direction = torch.zeros(normals.shape)
    minus_z_direction[:, :, 2] = -1.0
    dot_product = torch.sum(normals * minus_z_direction, dim=-1)
    threshold = math.cos(angle_threshold)
    filtered = torch.where(dot_product >= threshold, 1, 0)

    Timer.start_check_point("clustering normals")
    # TODO consider to return clustered_normals.numpy()
    clustered_normals, normal_indices = spherical_kmeans.kmeans(normals, filtered, clusters)
    Timer.end_check_point("clustering normals")

    img[:, :, 0][normal_indices == 0] = 255
    img[:, :, 0][normal_indices != 0] = 0
    img[:, :, 1][normal_indices == 1] = 255
    img[:, :, 1][normal_indices != 1] = 0
    img[:, :, 2][normal_indices == 2] = 255
    img[:, :, 2][normal_indices != 2] = 0

    if show or save:
        #plt.figure()
        color_names = ["red", "green", "blue"]
        desc = ""
        for i in range(clusters):
            desc = "{}{}={},\n".format(desc, color_names[i], clustered_normals[i].numpy())
        plt.title(desc)
        plt.imshow(img)
        if save:
            plt.savefig("{}_clusters.jpg".format(file_name_prefix))
        if show:
            plt.show()

    normal_indices_np = normal_indices.numpy().astype(dtype=np.uint8)
    clustered_normals_np = clustered_normals.numpy()
    if save:
        cv.imwrite("{}_clusters_indices.png".format(file_name_prefix), normal_indices_np)
        np.savetxt('{}_clusters_normals.txt'.format(file_name_prefix), clustered_normals_np, delimiter=',', fmt='%1.8f')

    return clustered_normals_np, normal_indices_np


def megadepth_input_dir(scene_name: str):
    return "depth_data/mega_depth/{}".format(scene_name)


def get_file_names_from_dir(input_dir: str, limit: int, interesting_files: list, suffix: str):
    if interesting_files is not None:
        return interesting_files
    else:
        return get_file_names(input_dir, suffix, limit)


def get_megadepth_file_names_and_dir(scene_name, limit, interesting_files):
    directory = megadepth_input_dir(scene_name)
    file_names = get_file_names_from_dir(directory, limit, interesting_files, ".npy")
    return file_names, directory


def compute_normals_simple_diff_convolution(scene: SceneInfo, depth_data_read_directory, depth_data_file_name, save, output_directory, upsample=False):
    """
    Currently the standard way to compute the normals and cluster them. This is called from other modules
    (extension point) on a per file basis

    :param scene:
    :param depth_data_read_directory:
    :param depth_data_file_name:
    :param save:
    :param output_directory:
    :param upsample:
    :return: (normals, normal_indices)
    """

    cameras = scene.cameras
    images = scene.img_info_map
    camera_id = images[depth_data_file_name[0:-4]].camera_id
    camera = cameras[camera_id]
    h = camera.height_width[0]
    w = camera.height_width[1]
    f = camera.focal_length

    return compute_normals_simple_diff_convolution_simple(h, w, f, depth_data_read_directory, depth_data_file_name, save, output_directory, upsample)


def compute_normals_simple_diff_convolution_simple(height, width, focal_length, depth_data_read_directory, depth_data_file_name, save, output_directory, upsample=False):
    """see :func:`compute_normals_simple_diff_convolution`"""

    # 3.75 upsample
    # -> sigma (5.0 -> 1.33)
    # -> mask_21 -> mask_5 (not exact)
    # -> depth_factor (1/30 -> 1/6) (should be exactly 1/8 actually)

    mask_5 = torch.tensor([[0.5, 0.5, 0, -0.5, -0.5]]).float()
    # this is interesting
    # mask_7 = torch.tensor([[0.5, 0.5, 0.5, 0, -0.5, -0.5, -0.5]]).float()

    mask_21 = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5,
                             0.5, 0.5, 0.5, 0.5, 0.5,
                             0,
                             -0.5, -0.5, -0.5, -0.5, -0.5,
                             -0.5, -0.5, -0.5, -0.5, -0.5,
                             ]]).float()

    if upsample:
        mask = mask_21
        sigma = 5.0
        depth_factor = 1 / 30
        # normals_params_list = [
        #     (True, sigma, "sigma_5_{}".format(1/30), 1/30)
        # ]
    else:
        mask = mask_5
        sigma = 1.33
        depth_factor = 1 / 6
        # normals_params_list = []
        # for depth_factor_inv in range(6, 7, 2):
        #     normals_params_list.append((True, sigma, "sigma_5_{}".format(1/depth_factor_inv), 1/depth_factor_inv))

    if not upsample:
        width = None
        height = None

    depth_data = read_depth_data(depth_data_file_name, depth_data_read_directory, height, width)

    normals = diff_normal_from_depth_data(focal_length, depth_data, mask=mask, smoothed=True, sigma=sigma,
                                          depth_factor=depth_factor)
    print("Creating dir (if not exists already): {}".format(output_directory))
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    title = "normals with plain diff mask - {}".format(depth_data_file_name)
    clustered_normals_np, normal_indices_np = cluster_and_save_normals(normals, depth_data_file_name, output_directory, show=True, title=title, save=save)
    return clustered_normals_np, normal_indices_np


def compute_normals_simple_diff_convolution_all(scene: SceneInfo, file_names, read_directory, save, output_parent_dir, skip_existing=True):

    for depth_data_file_name in file_names:

        print("Processing: {}".format(depth_data_file_name))

        output_directory = "{}/{}".format(output_parent_dir, depth_data_file_name[:-4])
        if skip_existing and os.path.isdir(output_parent_dir):
            print("{} already exists, skipping".format(output_directory))
            continue

        compute_normals_simple_diff_convolution(scene, read_directory, depth_data_file_name, save, output_directory)


# TODO test this
def sobel_normals_5x5(scene: str, limit, save):

    file_names, directory = get_megadepth_file_names_and_dir(scene, limit, None)
    cameras = read_cameras(scene)
    images = read_images(scene)

    for file_name in file_names:
        camera_id = images[file_name[:-4]].camera_id
        camera = cameras[camera_id]
        width = camera.height_width[1]
        height = camera.height_width[0]

        depth_data = read_depth_data(file_name, directory, height, width)

        mask_size = 5
        normals = normal_from_sobel_and_depth_data(depth_data, size=mask_size)

        file_name ='work/{}_normals_sobel_normals_colors_fixed.png'.format(file_name)
        title = "normals sobel {}x{} - {}".format(mask_size, mask_size, file_name[:-4])

        cluster_and_save_normals(normals, title, clusters=2, file_name_prefix=file_name, save=save)


def main():

    Timer.start()

    scene_name = "scene1"
    scene_info = SceneInfo.read_scene(scene_name)
    interesting_imgs = scene_info.imgs_for_comparing_difficulty(0)
    #interesting_imgs = ["frame_0000000025_3.npy"]

    file_names, input_directory = get_megadepth_file_names_and_dir(scene_name, limit=2, interesting_files=None)

    output_parent_dir = "work/{}/normals/simple_diff_mask".format(scene_name)

    compute_normals_simple_diff_convolution_all(scene_info, file_names, input_directory, save=True, output_parent_dir=output_parent_dir, skip_existing=False)

    Timer.end()


if __name__ == "__main__":
    main()
