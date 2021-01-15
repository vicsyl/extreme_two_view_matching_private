import torch
from resize import upsample

'''
This was to test svd_normal_from_reprojected - this would return the reprojected data points
so that svd_normal_from_reprojected would return expected normals,
but this data is actually not that simple to craft  
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


def test_reproject_project(depth_data_file, cameras, images, reprojected_data):

    camera_id = images[depth_data_file].camera_id
    camera = cameras[camera_id]
    focal_point_length = camera.focal_length
    width = camera.height_width[1]
    height = camera.height_width[0]
    principal_point_x = camera.principal_point_x_y[0]
    principal_point_y = camera.principal_point_x_y[1]

    xs = reprojected_data[0] / reprojected_data[2]
    ys = reprojected_data[1] / reprojected_data[2]

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


def test_reproject_project_old(depth_data_map, cameras, images, reprojected_data):

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
