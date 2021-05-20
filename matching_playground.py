from matching import *


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
    normals, normal_indices = compute_normals(scene_info, depth_input_dir, depth_data_file_name,
                                              output_directory=None)

    # normal indices => cluster indices (maybe safe here?)
    normal_indices = possibly_upsample_normals(img, normal_indices)
    components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals)))

    show_clustered_components = False
    if show_clustered_components:
        get_and_show_components(components_indices, valid_components_dict, normals=normals)

    feature_descriptor = cv.SIFT_create()

    valid_components_dicts = {
        "frame_0000000450_3": {5: 1},
        "frame_0000000535_3": {2: 0}
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
        kps, descs = show_rectification_play(normals, valid_components_dict_new, img_name, img, K, components_indices, feature_descriptor,
                                             show_all=True, show_all_regions=use_default_dict)

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

    H, tentative_matches, src_kps, src_dsc, dst_kps, dst_dsc = \
        match_images_and_keypoints_foo(img1, kps1, descs1, img2, kps2, descs2, ratio_thresh=0.75, show=True, title="without rectification")

    # rectified
    img1, kps1, descs1 = rectify_play(scene_info, rectify=True, img_name=files_to_match[0][:-4])
    img2, kps2, descs2 = rectify_play(scene_info, rectify=True, img_name=files_to_match[1][:-4])

    H, tentative_matches, src_kps, src_dsc, dst_kps, dst_dsc = \
        match_images_and_keypoints_foo(img1, kps1, descs1, img2, kps2, descs2, ratio_thresh=0.75, show=True, title="with rectification")

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


def show_rectification_play(normals, valid_components_dict, img_name, img, K, components_indices, feature_descriptor, show_all,
                            show_all_regions):

    show_rectification = show_all

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

    show_unrectified = show_all
    if show_unrectified:
        if not show_all_regions:
            valid_key = list(valid_components_dict.items())[0][0]
            coords = np.where(components_indices != valid_key)
            img_unrect[coords[0], coords[1]] = [0, 0, 0]
        kps_unrect, descs_unrect = feature_descriptor.detectAndCompute(img_unrect, None)
        cv.drawKeypoints(img_unrect, kps_unrect, img_unrect, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        size = img_unrect.shape[0] * img_unrect.shape[1]
        plt.figure(figsize=(10, 10))
        plt.title("unrectified {}, size={}, \nkps={}".format(img_name, size, len(kps_unrect)))
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
                                         show=show_rectification)


    show_rectified = show_all
    if show_rectified:
        plt.figure(figsize=(5, 5))
        img_rect = img.copy()
        cv.drawKeypoints(img_rect, kps, img_rect, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        size = img_rect.shape[0] * img_rect.shape[1]
        plt.title("recified {}, size={}, \nkps={}".format(img_name, size, len(kps)))
        plt.imshow(img_rect)
        show_or_close(True)
        #
        # print("unrect: {}={}".format(len(kps_unrect), len(descs_unrect)))
        # print("rect: {}={}".format(len(kps), len(descs)))
        # print()

    return kps, descs


def match_images_and_keypoints_foo(img1, kps1, descs1, img2, kps2, descs2, ratio_thresh=0.75, show=True, title= ""):

    Timer.start_check_point("matching")

    tentative_matches = find_correspondences(img1, kps1, descs1, img2, kps2, descs2, show=True, save=False, ratio_thresh=ratio_thresh)
    src_pts, src_kps, src_dsc, dst_pts, dst_kps, dst_dsc = rich_split_points(tentative_matches, kps1, descs1, kps2, descs2)

    H, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=2.0, confidence=0.9999)

    if show:
        plt.figure(figsize=(8, 8))
        tentatives = len(tentative_matches)
        inliers = np.sum(inlier_mask)
        title = "{}:\n tentatives: {} inliers: {}, ratio: {}".format(title, tentatives, inliers, inliers/tentatives)
        plt.title(title)
        img = draw_matches(kps1, kps2, tentative_matches, H, inlier_mask, img1, img2)
        plt.imshow(img)
        plt.show(block=False)

    src_kps = apply_inliers_on_list(src_kps, inlier_mask)
    src_dsc = apply_inliers_on_list(src_dsc, inlier_mask)
    dst_kps = apply_inliers_on_list(dst_kps, inlier_mask)
    dst_dsc = apply_inliers_on_list(dst_dsc, inlier_mask)
    #tentative_matches = apply_inliers_on_list(tentative_matches, inlier_mask)

    Timer.end_check_point("matching")

    return H, tentative_matches, src_kps, src_dsc, dst_kps, dst_dsc


def play_main():

    Timer.start()

    interesting_files = [
        "frame_0000001465_4.jpg",
        # "frame_0000000535_3.jpg",
        # "frame_0000000450_3.jpg",
    ]

    scene_info = SceneInfo.read_scene("scene1", lazy=True)

    #"original_dataset/scene1/images", limit = 20,
    rectify_iterate_play(scene_info, files_to_match=interesting_files)

    Timer.end()


if __name__ == "__main__":
    play_main()
