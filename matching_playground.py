import numpy as np

from matching import *
from depth_to_normals import compute_only_normals, show_sky_mask, cluster_normals
from sky_filter import *
from connected_components import get_and_show_components


def prepare_to_rectify(scene_info, img_name):

    img_file_path = scene_info.get_img_file_path(img_name)
    img = cv.imread(img_file_path, None)
    plt.figure()
    plt.title(img_name)
    plt.imshow(img)
    show_original_image = False
    show_or_close(show_original_image)

    K = scene_info.get_img_K(img_name)

    # depth => indices

    depth_data_file_name = "{}.npy".format(img_name)
    depth_input_dir = "depth_data/mega_depth/{}".format(scene_info.name)

    # normals, normal_indices = compute_normals(scene_info, depth_input_dir, depth_data_file_name, output_directory=None)

    focal_length = K[0, 0]
    orig_height = img.shape[0]
    orig_width = img.shape[1]

    normals, _ = compute_only_normals(focal_length, orig_height, orig_width, depth_input_dir, depth_data_file_name)
    filter_mask = get_nonsky_mask(img, normals.shape[0], normals.shape[1])
    show_sky_mask(img, filter_mask, img_name, show=True, save=False)

    normals_clusters_repr, normal_indices, _ = cluster_normals(normals, filter_mask=filter_mask)

    # filtering out the clusters ... not here though

    # normal indices => cluster indices (maybe safe here?)
    normal_indices = possibly_upsample_normals(img, normal_indices)
    components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals_clusters_repr)))

    show_clustered_components = True
    if show_clustered_components:
        get_and_show_components(components_indices, valid_components_dict, normals=normals_clusters_repr)

    return components_indices, valid_components_dict, normals_clusters_repr, img, K


def rectify_play(scene_info, img_name, rectify, use_default_dict=True):

    img_file_path = scene_info.get_img_file_path(img_name)
    img = cv.imread(img_file_path, None)
    plt.figure()
    plt.title(img_name)
    plt.imshow(img)
    show_original_image = False
    show_or_close(show_original_image)

    K = scene_info.get_img_K(img_name)

    # depth => indices

    depth_data_file_name = "{}.npy".format(img_name)
    depth_input_dir = "depth_data/mega_depth/{}".format(scene_info.name)

    # normals, normal_indices = compute_normals(scene_info, depth_input_dir, depth_data_file_name, output_directory=None)

    focal_length = K[0, 0]
    orig_height = img.shape[0]
    orig_width = img.shape[1]

    normals, _ = compute_only_normals(focal_length, orig_height, orig_width, depth_input_dir, depth_data_file_name)
    filter_mask = get_nonsky_mask(img, normals.shape[0], normals.shape[1])
    show_sky_mask(img, filter_mask, img_name, show=True, save=False)

    normals_clusters_repr, normal_indices, _ = cluster_normals(normals, filter_mask=filter_mask)

    # filtering out the clusters ... not here though

    # normal indices => cluster indices (maybe safe here?)
    normal_indices = possibly_upsample_normals(img, normal_indices)
    components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals_clusters_repr)))

    show_clustered_components = False
    if show_clustered_components:
        get_and_show_components(components_indices, valid_components_dict, normals=normals_clusters_repr)

    feature_descriptor = cv.SIFT_create()

    valid_components_dicts = {
        "frame_0000000535_3": {3: 0},
        "frame_0000000450_3": {1: 0, 126: 0, 169: 0} #, 175: 1}
    }

    if valid_components_dicts.__contains__(img_name) and not use_default_dict:
        valid_components_dict_new = valid_components_dicts[img_name]
    else:
        valid_components_dict_new = valid_components_dict

    plt.figure(figsize=(5, 5))
    img_filtered = img.copy()

    if not use_default_dict:
        valid_key = list(valid_components_dict_new.items())[0][0]
        coords = np.where(components_indices != valid_key)
        img_filtered[coords[0], coords[1]] = [0, 0, 0]

    if rectify:
        kps, descs = show_rectification_play(normals_clusters_repr, valid_components_dict_new, img_name, img, K, components_indices, feature_descriptor,
                                             show_all=True, show_all_regions=False)

    else:

        kps, descs = feature_descriptor.detectAndCompute(img_filtered, None)
        cv.drawKeypoints(img, kps, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        show_unrectified = True
        if show_unrectified:
            plt.figure(figsize=(10, 10))
            plt.imshow(img_filtered)
            size = img.shape[0] * img.shape[1]
            plt.title("unrectified {}, size={}, \nkps={}".format(img_name, size, len(kps)))
        show_or_close(show_unrectified)

    return img_filtered, kps, descs


def get_valid_component_dict(img_name, valid_components_dict):

    valid_components_dicts = {
        "frame_0000000450_3": {5: 1},
        "frame_0000000535_3": {2: 0}
    }

    if valid_components_dicts.__contains__(img_name):
        return valid_components_dicts[img_name]
    else:
        return valid_components_dict


def rectify_iterate_play(scene_info: SceneInfo, files_to_match=None):

    # baseline
    img1, kps1, descs1 = rectify_play(scene_info, rectify=False, img_name=files_to_match[0][:-4])
    img2, kps2, descs2 = rectify_play(scene_info, rectify=False, img_name=files_to_match[1][:-4])

    H, tentative_matches, inliers, src_kps, src_dsc, dst_kps, dst_dsc = \
        match_images_and_keypoints_play(img1, kps1, descs1, img2, kps2, descs2, ratio_thresh=0.75, show=True, title="without rectification")

    # rectified
    img1, kps1, descs1 = rectify_play(scene_info, rectify=True, img_name=files_to_match[0][:-4])
    img2, kps2, descs2 = rectify_play(scene_info, rectify=True, img_name=files_to_match[1][:-4])

    H, tentative_matches, inliers, src_kps, src_dsc, dst_kps, dst_dsc = \
        match_images_and_keypoints_play(img1, kps1, descs1, img2, kps2, descs2, ratio_thresh=0.75, show=True, title="with rectification")

    print()

    # for idx, img_name_full in enumerate(files_to_match):
    #     img_name = img_name_full[:-4]
    #     print("Processing: {}".format(img_name))
    #
    #     Timer.start_check_point("processing img")
    #     print("Processing: {}".format(img_name))
    #
    #     # input image
    #     img_file_path = scene_info.get_img_file_path(img_name)
    #     img = cv.imread(img_file_path, None)
    #     plt.figure()
    #     plt.title(img_name_full)
    #     plt.imshow(img)
    #     show_original_image = False
    #     show_or_close(show_original_image)
    #
    #     K = scene_info.get_img_K(img_name)
    #
    #     # depth => indices
    #     depth_data_file_name = "{}.npy".format(img_name)
    #     depth_input_dir = "depth_data/mega_depth/{}".format(scene_info.name)
    #     normals, normal_indices = compute_normals(scene_info, depth_input_dir, depth_data_file_name,
    #                                               output_directory=None)
    #
    #     # normal indices => cluster indices (maybe safe here?)
    #     normal_indices = possibly_upsample_normals(img, normal_indices)
    #     components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals)))
    #
    #     show_clustered_components = False
    #     if show_clustered_components:
    #         show_components(components_indices, valid_components_dict, normals=normals)
    #
    #     feature_descriptor = cv.SIFT_create()
    #
    #     rectify = True
    #     if rectify:
    #         kps, descs = show_rectification_foo(normals, img_name, img, K, components_indices, feature_descriptor, show_all=False)
    #
    #     else:
    #         kps, descs = feature_descriptor.detectAndCompute(img, None)
    #
    #     Timer.end_check_point("processing img")
    #     foo = (img, kps, descs, normals, components_indices, valid_components_dict)
    #     print("all gathered: {}".format(foo))


def show_rectification_play(normals,
                            valid_components_dict,
                            img_name,
                            img,
                            K,
                            components_indices,
                            feature_descriptor,
                            show_all,
                            show_all_regions,
                            rotation_factor=1.0):
    """
    Wrapper around get_rectified_keypoints, which provides some extra visualizations
    :param normals:
    :param valid_components_dict:
    :param img_name:
    :param img:
    :param K:
    :param components_indices:
    :param feature_descriptor:
    :param show_all:
    :param show_all_regions:
    :return:
    """

    # valid comp dict1: {2: 0}
    # valid comp dict2: {5, 118: 1}

    # valid_components_dicts = {
    #     "frame_0000000450_3": {5: 1},
    #     "frame_0000000535_3": {2: 0}
    # }
    #
    # valid_components_dict = get_valid_component_dict(img_name, ) valid_components_dicts[img_name]

    plt.figure(figsize=(5, 5))
    img_unrect = img.copy()

    if show_all:
        if not show_all_regions:
            valid_key = list(valid_components_dict.items())[0][0]
            coords = np.where(components_indices != valid_key)
            img_unrect[coords[0], coords[1]] = [0, 0, 0]
        kps_unrect, descs_unrect = feature_descriptor.detectAndCompute(img_unrect, None)
        cv.drawKeypoints(img_unrect, kps_unrect, img_unrect, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        size = img_unrect.shape[0] * img_unrect.shape[1]
        plt.figure(figsize=(10, 10))
        plt.title("unrectified yet {}, size={}, \nkps={}".format(img_name, size, len(kps_unrect)))
        plt.imshow(img_unrect)
        show_or_close(True)

    # get rectification
    kps, descs = get_rectified_keypoints(normals,
                                         components_indices,
                                         valid_components_dict,
                                         img,
                                         K,
                                         descriptor=feature_descriptor,
                                         img_name=img_name,
                                         out_prefix=None,
                                         show=show_all,
                                         rotation_factor=rotation_factor)

    if show_all:
        plt.figure(figsize=(5, 5))
        img_rect = img.copy()
        cv.drawKeypoints(img_rect, kps, img_rect, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        size = img_rect.shape[0] * img_rect.shape[1]
        plt.title("recified already {}, size={}, \nkps={}".format(img_name, size, len(kps)))
        plt.imshow(img_rect)
        show_or_close(True)
        #
        # print("unrect: {}={}".format(len(kps_unrect), len(descs_unrect)))
        # print("rect: {}={}".format(len(kps), len(descs)))
        # print()

    return kps, descs


def match_images_and_keypoints_play(img1, kps1, descs1, img2, kps2, descs2, ratio_thresh=0.75, show=True, title=""):

    Timer.start_check_point("matching")

    tentative_matches = find_correspondences(img1, kps1, descs1, img2, kps2, descs2, show=True, save=False, ratio_thresh=ratio_thresh)
    src_pts, src_kps, src_dsc, dst_pts, dst_kps, dst_dsc = rich_split_points(tentative_matches, kps1, descs1, kps2, descs2)

    H, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=2.0, confidence=0.9999)

    inlier_count = np.sum(inlier_mask)

    if show:
        plt.figure(figsize=(8, 8))
        tentatives = len(tentative_matches)
        long_title = "{}:\n tentatives: {} inliers: {}, ratio: {}".format(title, tentatives, inlier_count, inlier_count/tentatives)
        plt.title(long_title)
        img = draw_matches(kps1, kps2, tentative_matches, H, inlier_mask, img1, img2)
        plt.imshow(img)
        plt.savefig("work/combinatorics_{}.png".format(title))
        plt.show(block=False)

    src_kps = apply_inliers_on_list(src_kps, inlier_mask)
    src_dsc = apply_inliers_on_list(src_dsc, inlier_mask)
    dst_kps = apply_inliers_on_list(dst_kps, inlier_mask)
    dst_dsc = apply_inliers_on_list(dst_dsc, inlier_mask)
    #tentative_matches = apply_inliers_on_list(tentative_matches, inlier_mask)

    Timer.end_check_point("matching")

    return H, tentative_matches, inlier_count, src_kps, src_dsc, dst_kps, dst_dsc


def show_combinatorics(X, Y, Z, title):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title(title)

    # Make data.
    # X = np.arange(0, 5, 1)
    # Y = np.arange(0, 5, 1)
    # X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X ** 2 + Y ** 2)
    # Z = np.sin(R)

    # Plot the surface.
    from matplotlib import cm
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    #Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig("work/combinatorics_{}.png".format(title))
    plt.show()


def rect_play2(use_original_valid_components_dict=True):


    # ar = np.array([[0.000e+00, 0.000e+00, 2.973e+03, 2.226e+03],
    #                [0.000e+00, 1.500e+00, 1.109e+03, 6.180e+02],
    #                [1.500e+00, 0.000e+00, 2.316e+03, 1.579e+03],
    #                [1.500e+00, 1.500e+00, 1.072e+03, 5.390e+02]])
    #
    # show_combinatorics(ar[:, 0], ar[:, 1], ar[:, 2])
    # if True:
    #     return


    # files_to_match = [
    #             # "frame_0000001670_1.jpg",
    #             # "frame_0000000705_3.jpg",
    #             "frame_0000000535_3.jpg",
    #             "frame_0000000450_3.jpg",
    #     # "frame_0000001465_4.jpg",
    #     # "frame_0000001220_3.jpg",
    # ]

    # first (acute angle) 0.5-0.625, second 1.00
    files_to_match = [
        "frame_0000000995_1.jpg",
        "frame_0000001610_3.jpg",
    ]

    scene_info = SceneInfo.read_scene("scene1")

    components_indices_l = []
    valid_components_dict_l = []
    normals_clusters_repr_l = []
    imgs_l = []
    Ks = []
    for index in range(2):
        img_name = files_to_match[index][:-4]
        components_indices, valid_components_dict, normals_clusters_repr, img, K = prepare_to_rectify(scene_info, img_name=img_name)
        components_indices_l.append(components_indices)
        valid_components_dict_l.append(valid_components_dict)
        normals_clusters_repr_l.append(normals_clusters_repr)
        imgs_l.append(img)
        Ks.append(K)

    valid_components_dicts = {
        "frame_0000000535_3": {3: 0},
        "frame_0000000450_3": {1: 0, 126: 0, 169: 0},  # , 175: 1} # how do I do it for all keys mapped to a given value ?
        "frame_0000000995_1": {12: 0},
        "frame_0000001610_3": {7: 0}, # 8: 1}
    }

    feature_descriptor = cv.SIFT_create()

    rotation_factor_min = 0.5
    rotation_factor_max = 1.0
    show_each = 1
    show_each_counter = 0
    rotations = 4

    measurements = np.ndarray((4, rotations + 1, rotations + 1))

    for rotation_factor_counter1 in range(0, rotations + 1):
        for rotation_factor_counter2 in range(0, rotations + 1):

            rotation_factor1 = rotation_factor_min + (rotation_factor_max - rotation_factor_min) * rotation_factor_counter1 / rotations
            rotation_factor2 = rotation_factor_min + (rotation_factor_max - rotation_factor_min) * rotation_factor_counter2 / rotations
            rotation_factors = [rotation_factor1, rotation_factor2]
            kps_l = []
            descs_l = []

            show_each_counter = show_each_counter + 1
            show = (show_each_counter % show_each == 0)
            if show:
                print("Will show for rotation factors {}".format(rotation_factors))

            for index in range(2):
                img_name = files_to_match[index][:-4]

                # sth like use the ith normal
                if valid_components_dicts.__contains__(img_name):
                    valid_components_dict_new = valid_components_dicts[img_name]
                else:
                    valid_components_dict_new = valid_components_dict


                kps, descs = show_rectification_play(normals_clusters_repr_l[index],
                                                     valid_components_dict_new,
                                                     img_name,
                                                     imgs_l[index],
                                                     Ks[index],
                                                     components_indices_l[index],
                                                     feature_descriptor,
                                                     show_all=True,
                                                     show_all_regions=False,
                                                     rotation_factor=rotation_factors[index])
                kps_l.append(kps)
                descs_l.append(descs)


            H, tentative_matches, inlier_count, src_kps, src_dsc, dst_kps, dst_dsc = \
                match_images_and_keypoints_play(imgs_l[0], kps_l[0], descs_l[0],
                                                imgs_l[1], kps_l[1], descs_l[1],
                                                ratio_thresh=0.75,
                                                show=show,
                                                title="rectification_{}_{}".format(rotation_factor1, rotation_factor2))

            measurements[:, rotation_factor_counter1, rotation_factor_counter2] = np.array([rotation_factor1,
                                                                                            rotation_factor2,
                                                                                            len(tentative_matches),
                                                                                            inlier_count])

            #measurements.append([rotation_factor1, rotation_factor2, len(tentative_matches), inlier_count])

    # print("tentatives:\n {}".format(measurements[2]))
    # print("inliers:\n {}".format(measurements[3]))

    show_combinatorics(measurements[0], measurements[1], measurements[2], "tentatives")
    show_combinatorics(measurements[0], measurements[1], measurements[3], "inlier")
    show_combinatorics(measurements[0], measurements[1], measurements[3] / measurements[2], "inlier_divided_by_tentatives")


    # img1, kps1, descs1 = rectify_play(scene_info, rectify=rectify, img_name=files_to_match[0][:-4],
    #                                   use_default_dict=use_original_valid_components_dict)
    # img2, kps2, descs2 = rectify_play(scene_info, rectify=rectify, img_name=files_to_match[1][:-4],
    #                                   use_default_dict=use_original_valid_components_dict)
    #
    # H, tentative_matches, inliers, src_kps, src_dsc, dst_kps, dst_dsc = \
    #     match_images_and_keypoints_play(img1, kps1, descs1, img2, kps2, descs2, ratio_thresh=0.75, show=True,
    #                                     title="without rectification")


def rect_play(rectify, use_original_valid_components_dict=True):
    files_to_match = [
                # "frame_0000001670_1.jpg",
                # "frame_0000000705_3.jpg",
                "frame_0000000535_3.jpg",
                "frame_0000000450_3.jpg",
        # "frame_0000001465_4.jpg",
        # "frame_0000001220_3.jpg",
    ]

    scene_info = SceneInfo.read_scene("scene1")

    # rectify_iterate_play(scene_info, files_to_match=interesting_files)

    #     Config.config_map[Config.save_normals_in_img] = False
    #     Config.config_map[Config.show_normals_in_img] = False

    title = "rectified" if rectify else "not rectified"

    img1, kps1, descs1 = rectify_play(scene_info, rectify=rectify, img_name=files_to_match[0][:-4],
                                      use_default_dict=use_original_valid_components_dict)
    img2, kps2, descs2 = rectify_play(scene_info, rectify=rectify, img_name=files_to_match[1][:-4],
                                      use_default_dict=use_original_valid_components_dict)

    H, tentative_matches, inliers, src_kps, src_dsc, dst_kps, dst_dsc = \
        match_images_and_keypoints_play(img1, kps1, descs1, img2, kps2, descs2, ratio_thresh=0.75, show=True,
                                        title="without rectification")

#     # rectified
#     img1, kps1, descs1 = rectify_play(scene_info, rectify=True, img_name=files_to_match[0][:-4])
#     img2, kps2, descs2 = rectify_play(scene_info, rectify=True, img_name=files_to_match[1][:-4])

#     H, tentative_matches, inliers, src_kps, src_dsc, dst_kps, dst_dsc = \
#         match_images_and_keypoints_foo(img1, kps1, descs1, img2, kps2, descs2, ratio_thresh=0.75, show=True, title="with rectification")

#     print()


def play_main():

    Timer.start()

    interesting_files = [
        # "frame_0000001465_4.jpg",
        "frame_0000000535_3.jpg",
        "frame_0000000450_3.jpg",
    ]

    scene_info = SceneInfo.read_scene("scene1")

    #"original_dataset/scene1/images", limit = 20,
    rectify_iterate_play(scene_info, files_to_match=interesting_files)

    Timer.end()


if __name__ == "__main__":
    Config.config_map[Config.key_do_flann] = False
    Config.config_map[Config.rectification_interpolation_key] = cv.INTER_LINEAR
    #rect_play(rectify=True, use_original_valid_components_dict=False)
    rect_play2(use_original_valid_components_dict=False)

    #play_main()
