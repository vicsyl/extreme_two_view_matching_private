from scene_info import *
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from connected_components import get_connected_components
from utils import Timer, merge_keys_for_same_value
from config import Config
from image_data import ImageData
from copy import deepcopy, copy
import itertools

from rectification import read_img_normals_info, get_rectified_keypoints, possibly_upsample_normals
from pathlib import Path

"""
These functions find correspondences using various feature descriptors
and then by finding a transformation homography using RANSAC.
When found, the correspondences are the examined and compared with the dataset info
- it checks whether some of the matched features actually correspond to same identified
points in both images
- I also want to check whether the homography correspond to the correct relative posed 
   - this also can be checked against the dataset info and it's still to be done
   
Problems:
- during the implementation I actually realized that homographies would typically only 
find matches between corresponding planes. I can either match using multiple homographies 
or employ some means to estimate epipolar geometry       
"""


"""
DISCLAIMER: some of these functions were implemented by me (Vaclav Vavra)
during the MPV course in spring semester 2020, mostly with the help
of the provided template.
"""


def decolorize(img):
    return cv.cvtColor(cv.cvtColor(img, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)


def draw_matches(kps1, kps2, tentative_matches, H, inlier_mask, img1, img2):
    h, w, ch = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    if H is not None:
        dst = cv.perspectiveTransform(pts, H)
        img2_tr = cv.polylines(decolorize(img2), [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
    else:
        img2_tr = decolorize(img2)

    matches_mask = inlier_mask.ravel().tolist()

    # Blue is estimated homography
    draw_params = dict(matchColor=(255, 255, 0),  # draw matches in yellow color
                       singlePointColor=None,
                       matchesMask=matches_mask,  # draw only inliers
                       flags=20)
    img_out = cv.drawMatches(decolorize(img1), kps1, img2_tr, kps2, tentative_matches, None, **draw_params)
    return img_out


def split_points(tentative_matches, kps1, kps2):
    src_pts = np.float32([kps1[m.queryIdx].pt for m in tentative_matches]).reshape(-1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentative_matches]).reshape(-1, 2)
    return src_pts, dst_pts


def rich_split_points(tentative_matches, kps1, dsc1, kps2, dsc2):

    src_pts = np.float32([kps1[m.queryIdx].pt for m in tentative_matches]).reshape(-1, 2)
    src_kps = [kps1[m.queryIdx] for m in tentative_matches]
    src_dsc = [dsc1[m.queryIdx] for m in tentative_matches]

    dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentative_matches]).reshape(-1, 2)
    dst_kps = [kps2[m.trainIdx] for m in tentative_matches]
    dst_dsc = [dsc2[m.trainIdx] for m in tentative_matches]

    return src_pts, src_kps, src_dsc, dst_pts, dst_kps, dst_dsc


def find_correspondences(img1, kps1, descs1, img2, kps2, descs2, out_dir, ratio_thresh=0.8, show=True, save=True):

    if Config.do_flann():
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=128)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv.BFMatcher()

    knn_matches = matcher.knnMatch(descs1, descs2, 2)

    tentative_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            tentative_matches.append(m)

    if show or save:
        tentative_matches_in_singleton_list = [[m] for m in tentative_matches]
        img3 = cv.drawMatchesKnn(img1, kps1, img2, kps2, tentative_matches_in_singleton_list, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3)
        if save:
            plt.savefig("{}/tentative.jpg".format(out_dir))
        if show:
            plt.show()

    return tentative_matches


def correctly_matched_point_for_image_pair(inlier_mask, tentative_matches, kps1, kps2, images_info, img_pair):
    matches = [m for (idx, m) in enumerate(tentative_matches) if inlier_mask[idx ,0] == 1]
    kps1_indices = [m.queryIdx for m in matches]
    kps2_indices = [m.trainIdx for m in matches]

    data_point1_ids, mins1 = get_kps_gt_id(kps1, kps1_indices, images_info[img_pair.img1], diff_threshold=100.0)
    data_point2_ids, mins2 = get_kps_gt_id(kps2, kps2_indices, images_info[img_pair.img2], diff_threshold=100.0)

    data_point_ids_matches = data_point1_ids[data_point1_ids == data_point2_ids]
    unique = np.unique(data_point_ids_matches)
    return unique


def get_kps_gt_id(kps, kps_indices, image_entry: ImageEntry, diff_threshold=2.0):

    kps_matches_points = [list(kps[kps_index].pt) for kps_index in kps_indices]
    kps_matches_np = np.array(kps_matches_points)

    image_data = image_entry.data
    data_ids = image_entry.data_point_idxs

    diff = np.ndarray(image_data.shape)
    mins = np.ndarray(kps_matches_np.shape[0])
    data_point_ids = -2 * np.ones(kps_matches_np.shape[0], dtype=np.int32)
    for p_idx, match_point in enumerate(kps_matches_np):
        diff[:, 0] = image_data[:, 0] - match_point[0]
        diff[:, 1] = image_data[:, 1] - match_point[1]
        diff_norm = np.linalg.norm(diff, axis=1)
        min_index = np.argmin(diff_norm)
        min_diff = diff_norm[min_index]
        mins[p_idx] = min_diff
        if min_diff < diff_threshold:
            data_point_ids[p_idx] = data_ids[min_index]

    return data_point_ids, mins


def find_keypoints(scene_name, image_entry: ImageEntry, descriptor):

    img_path = 'original_dataset/{}/images/{}.jpg'.format(scene_name, image_entry.image_name)
    img = cv.imread(img_path)
    if img is None:
        return None, None
    kps, descs = descriptor.detectAndCompute(img, None)
    return kps, descs


def find_keypoints_match_with_data(scene_name, image_entry: ImageEntry, descriptor, diff_threshold):

    kps, descs = find_keypoints(scene_name, image_entry, descriptor)
    if kps is None:
        return None
    kps_indices = list(range(len(kps)))

    data_point_ids, mins = get_kps_gt_id(kps, kps_indices, image_entry, diff_threshold=diff_threshold)
    # FIXME - still need to distinguish between indices -1!!!
    data_point_ids = data_point_ids[data_point_ids != -2]
    return data_point_ids


def keypoints_match_with_data(scene_name, diff_threshold, descriptor=cv.SIFT_create(), limit=None):

    images_info = read_images(scene_name)

    existent_ids = 0
    for idx, image_entry_key in enumerate(images_info):
        image_entry = images_info[image_entry_key]
        # FIXME - still need to distinguish between indices -1!!!
        data_point_ids = find_keypoints_match_with_data(scene_name, image_entry, descriptor, diff_threshold)
        if data_point_ids is None:
            print("Image: {} doesn't exist!!!".format(image_entry.image_name))
        else:
            all_points = len(image_entry.data)
            print("Image: {}, points matches:{}/{}".format(image_entry.image_name, len(data_point_ids), all_points))
            existent_ids = existent_ids + 1
            if limit is not None and existent_ids == limit:
                break


def get_kts_desc_normal_list(image_data: ImageData, merge_components: bool):
    r"""
    Function that splits the keypoints and their description according to
    their corresponding components (merge_component=False) or normals (merge_components=True)

    Arguments:
        image_data: ImageData (see image_data.py)
        merge_components (whether to merge connected components with the same normal
        - i.e. split only according to the normal)

    Returns:
        list of tuples in the form of:
         (list of keypoints, ndarray of keypoint descriptions, tuple of component indices, normal index]
    """

    if merge_components:
        # valid_components_dict({k1: v1, k2: v1}) => valid_components_dict({[k1, k2: v1]})
        valid_components_dict = merge_keys_for_same_value(image_data.valid_components_dict)
    else:
        # valid_components_dict(k, v) => valid_components_dict([k], v)
        valid_components_dict = {(k, ): v for k, v in image_data.valid_components_dict.items()}

    int_raw_kps = [[round(kpt.pt[0]), round(kpt.pt[1])] for kpt in image_data.key_points]

    kpts_desc_list = []
    sh0 = image_data.components_indices.shape[0]
    sh1 = image_data.components_indices.shape[1]
    for cur_components, normal_index in valid_components_dict.items():
        kps = []
        dscs = []
        for ix, int_raw_kp in enumerate(int_raw_kps):
            x = int_raw_kp[0]
            y = int_raw_kp[1]
            if 0 <= x <= sh1 and 0 <= y <= sh0 and image_data.components_indices[int_raw_kp[1], int_raw_kp[0]] in cur_components:
                kps.append(image_data.key_points[ix])
                dscs.append(image_data.descriptions[ix])
        kpts_desc_list.append((kps, np.array(dscs), cur_components, normal_index))

    return kpts_desc_list


def apply_inliers_on_list(l: list, inlier_mask):
    return [i for idx, i in enumerate(l) if inlier_mask[idx, 0] == 0]


def find_and_draw_homography(img1, kps1, descs1, img2, kps2, descs2):

    tentative_matches = find_correspondences(img1, kps1, descs1, img2, kps2, descs2, None, show=False, save=False)
    src_pts, src_kps, src_dsc, dst_pts, dst_kps, dst_dsc = rich_split_points(tentative_matches, kps1, descs1, kps2, descs2)

    H, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=3.0, confidence=0.999)

    if True:
        img = draw_matches(kps1, kps2, tentative_matches, H, inlier_mask, img1, img2)
        plt.imshow(img)
        plt.show()

    src_kps = apply_inliers_on_list(src_kps, inlier_mask)
    src_dsc = apply_inliers_on_list(src_dsc, inlier_mask)
    dst_kps = apply_inliers_on_list(dst_kps, inlier_mask)
    dst_dsc = apply_inliers_on_list(dst_dsc, inlier_mask)
    tentative_matches = apply_inliers_on_list(tentative_matches, inlier_mask)

    return H, tentative_matches, src_kps, src_dsc, dst_kps, dst_dsc


def match_images_with_dominant_planes(image_data1: ImageData, image_data2: ImageData, images_info, img_pair, out_dir, show: bool, save: bool):

    merge_components = Config.config_map[Config.key_planes_based_matching_merge_components]
    kpts_desc_list1 = get_kts_desc_normal_list(image_data1, merge_components)
    kpts_desc_list2 = get_kts_desc_normal_list(image_data2, merge_components)

    # (id1, id2) => (homography, inlier_kps1, inlier_dsc1, inlier_kps2, inlier_dsc2)
    homography_matching_dict = {}

    for ix1 in range(len(kpts_desc_list1)):
        for ix2 in range(len(kpts_desc_list2)):
            kps1, desc1, components1, normal_index1 = kpts_desc_list1[ix1]
            kps2, desc2, components2, normal_index2 = kpts_desc_list2[ix2]
            print("matching component/normal {} from 1st image against component/normal from 2nd image {}".format(ix1, ix2))
            H, tentative_matches, src_kps, src_dsc, dst_kps, dst_dsc = find_and_draw_homography(image_data1.img, kps1, desc1, image_data2.img, kps2, desc2)
            homography_matching_dict[(ix1, ix2)] = (H, tentative_matches, src_kps, src_dsc, dst_kps, dst_dsc)

    if len(kpts_desc_list1) <= len(kpts_desc_list2):
        less_planes = 1
        more_planes = 2
        perm_length = len(kpts_desc_list1)
        permutation_items = range(len(kpts_desc_list2))
        swap = False
    else:
        less_planes = 2
        more_planes = 1
        perm_length = len(kpts_desc_list2)
        permutation_items = range(len(kpts_desc_list1))
        swap = True

    max_inliers = None
    best_idxs_1 = None
    best_idxs_2 = None
    for cur_permutation in itertools.permutations(permutation_items, perm_length):

        print("matching permutation of {} against {}: {} <=> {}".format(cur_permutation, list(range(perm_length)), more_planes, less_planes))

        kps1_l = []
        dsc1_l = []
        kps2_l = []
        dsc2_l = []
        #tentative_matches_l = []

        if swap:
            all_idxs_1 = cur_permutation
            all_idxs_2 = range(perm_length)
        else:
            all_idxs_1 = range(perm_length)
            all_idxs_2 = cur_permutation

        for i in range(perm_length):
            if swap:
                ix1 = cur_permutation[i]
                ix2 = i
            else:
                ix1 = i
                ix2 = cur_permutation[i]

            _, tentative_matches, cur_kps1, cur_dsc1, cur_kps2, cur_dsc2 = homography_matching_dict[(ix1, ix2)]
            kps1_l.extend(cur_kps1)
            kps2_l.extend(cur_kps2)
            dsc1_l.extend(cur_dsc1)
            dsc2_l.extend(cur_dsc2)
            #tentative_matches_l.extend(tentative_matches)

        cur_tentative_matches = find_correspondences(image_data1.img,
                                                     kps1_l,
                                                     np.array(dsc1_l),
                                                     image_data2.img,
                                                     kps2_l,
                                                     np.array(dsc2_l),
                                                     out_dir,
                                                     ratio_thresh=0.75,
                                                     show=True,
                                                     save=save)

        src_pts, dst_pts = split_points(cur_tentative_matches, kps1_l, kps2_l)
        # TODO use the same parameters as with the normal matching
        #ransacReprojThreshold = 3.0, confidence = 0.999
        E, inlier_mask = cv.findEssentialMat(src_pts, dst_pts, image_data1.K, None, image_data2.K, None, cv.RANSAC, prob=0.999, threshold=3.0)
        inliers = inlier_mask[inlier_mask == [0]].shape[0]

        if max_inliers is None or max_inliers < inliers:
            best_idxs_1 = all_idxs_1
            best_idxs_2 = all_idxs_2

        if show or save:
            # tentative_matches = []
            img_matches = draw_matches(kps1_l, kps2_l, cur_tentative_matches, None, inlier_mask, image_data1.img, image_data2.img)
            plt.figure()
            plt.title("({} <=> {}) - {} inliers".format(all_idxs_1, all_idxs_2, inliers))
            plt.imshow(img_matches)
            # if save:
            #     plt.savefig("{}/matches.jpg".format(out_dir))
            if show:
                plt.show()

    print("best indices: {} <=> {}".format(best_idxs_1, best_idxs_2))
    return None #E, inlier_mask, src_pts, dst_pts, kps1, kps2, len(tentative_matches)


def match_images_and_keypoints(img1, kps1, descs1, K_1, img2, kps2, descs2, K_2, images_info, img_pair, out_dir, show, save):

    Timer.start_check_point("matching")

    tentative_matches = find_correspondences(img1, kps1, descs1, img2, kps2, descs2, out_dir, ratio_thresh=0.75, show=show, save=save)

    src_pts, dst_pts = split_points(tentative_matches, kps1, kps2)

    # TODO threshold and prob params left to default values
    E, inlier_mask = cv.findEssentialMat(src_pts, dst_pts, K_1, None, K_2, None, cv.RANSAC)

    if show or save:
        img_matches = draw_matches(kps1, kps2, tentative_matches, None, inlier_mask, img1, img2)
        plt.figure()
        plt.title("Matches in line with the model")
        plt.imshow(img_matches)
        if save:
            plt.savefig("{}/matches.jpg".format(out_dir))
        if show:
            plt.show()

    unique = correctly_matched_point_for_image_pair(inlier_mask, tentative_matches, kps1, kps2, images_info, img_pair)

    print("Image pair: {}x{}:".format(img_pair.img1, img_pair.img2))
    print("Number of correspondences: {}".format(inlier_mask[inlier_mask == [0]].shape[0]))
    print("Number of not-correspondences: {}".format(inlier_mask[inlier_mask == [1]].shape[0]))
    print("correctly_matched_point_for_image_pair: unique = {}".format(unique.shape[0]))

    src_tentative, dst_tentative = split_points(tentative_matches, kps1, kps2)
    src_pts_inliers = src_tentative[inlier_mask[:, 0] == [1]]
    dst_pts_inliers = dst_tentative[inlier_mask[:, 0] == [1]]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt("{}/essential_matrix.txt".format(out_dir), E, delimiter=',', fmt='%1.8f')
    np.savetxt("{}/src_pts.txt".format(out_dir), src_pts_inliers, delimiter=',', fmt='%1.8f')
    np.savetxt("{}/dst_pts.txt".format(out_dir), dst_pts_inliers, delimiter=',', fmt='%1.8f')

    Timer.end_check_point("matching")
    return E, inlier_mask, src_pts, dst_pts, kps1, kps2, len(tentative_matches)


def prepare_data_for_keypoints_and_desc(scene_info, img_name, normal_indices, normals, descriptor, out_dir):

    K = scene_info.get_img_K(img_name)
    img = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_name))

    if Config.rectify():
        normal_indices = possibly_upsample_normals(img, normal_indices)
        components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals)), True)
        kps, descs = get_rectified_keypoints(normals, components_indices, valid_components_dict, img, K, descriptor, img_name, out_dir=out_dir)
    else:
        kps, descs = descriptor.detectAndCompute(img, None)

    return img, K, kps, descs


def img_correspondences(scene_info: SceneInfo, output_dir, descriptor, normals_dir, difficulties=range(18), rectify=True, limit=None, override_existing=True):

    processed_pairs = 0
    for difficulty in difficulties:

        print("Difficulty: {}".format(difficulty))

        max_limit = len(scene_info.img_pairs[difficulty])
        for i in range(max_limit):

            if limit is not None and processed_pairs >= limit:
                break

            img_pair: ImagePairEntry = scene_info.img_pairs[difficulty][i]

            out_dir = "work/{}/matching/{}/{}_{}".format(scene_info.name, output_dir, img_pair.img1, img_pair.img2)
            if os.path.isdir(out_dir) and not override_existing:
                print("{} already exists, skipping".format(out_dir))
                continue

            normals1, normal_indices1 = read_img_normals_info(normals_dir, img_pair.img1)
            if normals1 is None:
                print("first img's normals not found: {}".format(img_pair.img1))
                continue
            normals2, normal_indices2 = read_img_normals_info(normals_dir, img_pair.img2)
            if normals2 is None:
                print("second img's normals not found: {}".format(img_pair.img2))
                continue
            processed_pairs = processed_pairs + 1

            Path(out_dir).mkdir(parents=True, exist_ok=True)

            img1, K_1, kps1, descs1 = prepare_data_for_keypoints_and_desc(scene_info, img_pair.img1, normal_indices1, normals1, descriptor, out_dir)
            img2, K_2, kps2, descs2 = prepare_data_for_keypoints_and_desc(scene_info, img_pair.img2, normal_indices2, normals2, descriptor, out_dir)

            return match_images_and_keypoints(img1, kps1, descs1, K_1, img2, kps2, descs2, K_2, scene_info.img_info_map, img_pair, out_dir, show=True, save=True)


def main():

    normals_dir ="work/scene1/normals/simple_diff_mask_sigma_5"

    Timer.start()

    scene_info = SceneInfo.read_scene("scene1")

    sift_descriptor = cv.SIFT_create()

    limit = 1
    difficulties = [1]
    img_correspondences(scene_info, "with_rectification_foo", sift_descriptor, normals_dir, difficulties, rectify=True, limit=limit, override_existing=False)

    Timer.end()


if __name__ == "__main__":
    main()
