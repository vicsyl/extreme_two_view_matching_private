from scene_info import *
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from connected_components import get_connected_components
from utils import Timer, merge_keys_for_same_value
from config import Config
from evaluation import ImageData
import itertools
import pydegensac

from cv2 import DMatch

from rectification import read_img_normals_info, get_rectified_keypoints, possibly_upsample_normals
from pathlib import Path
from img_utils import show_or_close

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


def get_cross_checked_tentatives(matcher, descs1, descs2, ratio_threshold, k):

    knn_matches = matcher.knnMatch(descs1, descs2, k=k)
    # For cross-check - TODO does is work for flann?
    matches2 = matcher.match(descs2, descs1)
    tentative_matches = []
    # TODO what about this?
    # if len(matches) < 10:
    #     return None, [], []
    for m, n in knn_matches:
        if matches2[m.trainIdx].trainIdx != m.queryIdx:
            continue
        if m.distance < ratio_threshold * n.distance: # ratio_threshold was 0.85
            tentative_matches.append(m)

    return tentative_matches


def filter_fginn_matches(matches, desc1, desc2, num_nn, cfg):
    """
    :param matches:
    :param desc1:
    :param desc2:
    :param num_nn:
    :param cfg:
    :return:

    FGINN — 1st geometrically inconsistent nearest neighbor ratio
    https://ducha-aiki.medium.com/how-to-match-to-learn-or-not-to-learn-part-2-1ab52ede2022
    """

    valid_matches = []

    fginn_spatial_th = cfg["fginn_spatial_th"]
    ratio_th = cfg["ratio_th"]

    if fginn_spatial_th < 0 or fginn_spatial_th > 500:
        raise ValueError('FGINN radius outside expected range')

    if ratio_th < 0.1 or ratio_th > 1.01:
        raise ValueError('Ratio test threshold outside expected range')

    flann = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    not_fginns = flann.radiusMatch(desc2.astype(np.float32),
                                   desc2.astype(np.float32),
                                   fginn_spatial_th)

    for m_idx, cur_match in enumerate(matches):
        for mii in range(num_nn):
            cur_non_fginns = [
                x.trainIdx for x in not_fginns[cur_match[mii].trainIdx]
            ]
            for nn_2 in cur_match[mii + 1:]:
                if cur_match[mii].distance <= ratio_th * nn_2.distance:
                    valid_matches.append(cur_match[mii])
                    break
                if nn_2.trainIdx not in cur_non_fginns:
                    break

    return valid_matches


def find_correspondences(img1, kps1, descs1, img2, kps2, descs2, cfg, out_dir=None, save_suffix=None, ratio_thresh=None, show=True, save=True):

    fginn = cfg["fginn"]
    num_nn = cfg["num_nn"]

    # TODO aren't we doing cross check down below?
    crossCheck = False
    if Config.do_flann():
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=128, crossCheck=crossCheck)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv.BFMatcher(crossCheck=crossCheck)

    assert descs1 is not None and len(descs1) != 0
    assert descs2 is not None and len(descs2) != 0

    if fginn:
        k = 10 + num_nn # TODO why?
        knn_matches = matcher.knnMatch(descs1, descs2, k=k)
        tentative_matches = filter_fginn_matches(knn_matches, descs1, descs2, num_nn, cfg)
    else:
        k = 2   # max(2, num_nn + 1) # TODO resolve
        tentative_matches = get_cross_checked_tentatives(matcher, descs1, descs2, ratio_thresh, k=k)

    if show or save:
        tentative_matches_in_singleton_list = [[m] for m in tentative_matches]
        img3 = cv.drawMatchesKnn(img1, kps1, img2, kps2, tentative_matches_in_singleton_list, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("auto")
        plt.title("{} tentative matches".format(len(tentative_matches)))
        plt.imshow(img3)
        if save:
            assert out_dir is not None
            assert save_suffix is not None
            plt.savefig("{}/tentative_{}.jpg".format(out_dir, save_suffix))
        show_or_close(show)

    return tentative_matches


def find_keypoints(scene_name, image_entry: ImageEntry, descriptor):

    img_path = 'original_dataset/{}/images/{}.jpg'.format(scene_name, image_entry.image_name)
    img = cv.imread(img_path)
    if img is None:
        return None, None
    kps, descs = descriptor.detectAndCompute(img, None)
    return kps, descs


def get_kts_desc_normal_list(image_data: ImageData, merge_components: bool):
    r"""
    Function that splits the keypoints and their description according to
    their corresponding components (merge_component=False) or normals (merge_components=True)

    Arguments:
        image_data: ImageData (see image_data.py)
        merge_components (whether to merge connected components with the same normal
        - i.e. split only according to the normal)

    Returns:
        (kpts_desc_list, rest_kpts, rest_descs)
        kpts_desc_list:
         (list of keypoints, ndarray of keypoint descriptions, tuple of component indices, normal index]
         rest_kpts - keypoints not belonging to any plane
         rest_descs - their descriptions
    """

    if merge_components:
        # valid_components_dict({k1: v1, k2: v1}) => valid_components_dict({(k1, k2): v1})
        valid_components_dict = merge_keys_for_same_value(image_data.valid_components_dict)
    else:
        # valid_components_dict(k, v) => valid_components_dict((k,), v)
        valid_components_dict = {(k, ): v for k, v in image_data.valid_components_dict.items()}

    sh0 = image_data.components_indices.shape[0]
    sh1 = image_data.components_indices.shape[1]

    all_kpts_map = {}
    all_desc_map = {}
    for keys, v in valid_components_dict.items():
        empty_kpts = (v, [])
        empty_descs = (v, [])
        for key in keys:
            all_kpts_map[key] = empty_kpts
            all_desc_map[key] = empty_descs

    rest_kpts = []
    rest_descs = []

    for kpts_ix, kpt in enumerate(image_data.key_points):
        x = round(kpt.pt[0])
        y = round(kpt.pt[1])
        if 0 <= x < sh1 and 0 <= y < sh0:
            component = image_data.components_indices[y, x]
            if all_kpts_map.__contains__(component):
                all_kpts_map[component][1].append(image_data.key_points[kpts_ix])
                all_desc_map[component][1].append(image_data.descriptions[kpts_ix])
            else:
                rest_kpts.append(image_data.key_points[kpts_ix])
                rest_descs.append(image_data.descriptions[kpts_ix])

    kpts_desc_list = []
    for keys in valid_components_dict.keys():
        key = keys[0]
        kptss = all_kpts_map[key][1]
        descss = np.array(all_desc_map[key][1])
        cur_componentss = keys
        normall_index = all_kpts_map[key][0]
        print("adding {} for key {}".format(len(kptss), cur_componentss))
        kpts_desc_list.append((kptss, descss, cur_componentss, normall_index))

    return kpts_desc_list, rest_kpts, rest_descs


def apply_inliers_on_list(l: list, inlier_mask):
    return [i for idx, i in enumerate(l) if inlier_mask[idx, 0] == 1]


def find_and_draw_homography_or_fallback(img1, kps1, descs1, img2, kps2, descs2,
                                         ratio_thresh, ransac_thresh, ransac_confidence, ransac_iters, save_suffix, show, save, out_dir):

    tentative_matches = find_correspondences(img1, kps1, descs1, img2, kps2, descs2, out_dir=out_dir, save_suffix=save_suffix, show=show, save=save, ratio_thresh=ratio_thresh)
    src_pts, src_kps, src_dsc, dst_pts, dst_kps, dst_dsc = rich_split_points(tentative_matches, kps1, descs1, kps2, descs2)

    points = len(src_pts)
    if points >= 4:
        H, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, maxIters=ransac_iters, ransacReprojThreshold=ransac_thresh, confidence=ransac_confidence)
    else:
        # TODO CONTINUE OR DROP EVERYTHING
        print("WARNING: not enough matches ({}) for finding a homography".format(points))
        H = None
        inlier_mask = np.ones((points, 1))

    if show or save:
        img = draw_matches(kps1, kps2, tentative_matches, H, inlier_mask, img1, img2)
        plt.figure(figsize=(10, 10))
        plt.title("{} - {} inliers".format(save_suffix, np.sum(inlier_mask)))
        plt.imshow(img)
        if save:
            plt.savefig("{}/{}.jpg".format(out_dir, save_suffix))
        show_or_close(show)

    src_kps = apply_inliers_on_list(src_kps, inlier_mask)
    src_dsc = apply_inliers_on_list(src_dsc, inlier_mask)
    dst_kps = apply_inliers_on_list(dst_kps, inlier_mask)
    dst_dsc = apply_inliers_on_list(dst_dsc, inlier_mask)
    matches = apply_inliers_on_list(tentative_matches, inlier_mask)

    return H, matches, src_kps, src_dsc, dst_kps, dst_dsc


def get_synthetic_DMatches(up_to):
    return [get_synthetic_DMatch(index) for index in range(up_to)]


def get_synthetic_DMatch(index):
    dm = DMatch()
    dm.trainIdx = index
    dm.queryIdx = index
    dm.distance = 200.0
    dm.imgIdx = 0
    return dm


def match_images_with_dominant_planes(image_data1: ImageData, image_data2: ImageData,
                                      use_degensac, find_fundamental, img_pair, out_dir, show: bool, save: bool,
                                      ratio_thresh: float, ransac_th, ransac_conf, ransac_iters):

    img_pair_key = "{}_{}".format(img_pair.img1, img_pair.img2)

    merge_components = Config.config_map[Config.key_planes_based_matching_merge_components]
    kpts_desc_list1, rest_kpts1, rest_descs1 = get_kts_desc_normal_list(image_data1, merge_components)
    kpts_desc_list2, rest_kpts2, rest_descs2 = get_kts_desc_normal_list(image_data2, merge_components)

    if len(kpts_desc_list1) == 0 or len(kpts_desc_list2) == 0:
        print("WARNING: {}: NO VALID CLUSTER - fallback to plain degensac".format(img_pair_key))
        if use_degensac:
            return match_find_F_degensac(image_data1.img,
                                  image_data1.descriptions,
                                  image_data1.real_K,
                                  image_data2.img,
                                  image_data2.descriptions,
                                  image_data2.real_K,
                                  img_pair,
                                  out_dir,
                                  show, save,
                                  ratio_thresh=ratio_thresh,
                                  ransac_th=ransac_th,
                                  ransac_conf=ransac_conf,
                                  ransac_iters=ransac_iters)

        else:
            return match_epipolar(image_data1.img,
                                  image_data1.descriptions,
                                  image_data1.real_K,
                                  image_data2.img,
                                  image_data2.descriptions,
                                  image_data2.real_K,
                                  find_fundamental,
                                  img_pair,
                                  out_dir,
                                  show, save,
                                  ratio_thresh=ratio_thresh,
                                  ransac_th=ransac_th,
                                  ransac_conf=ransac_conf,
                                  ransac_iters=ransac_iters)


    # (id1, id2) => (homography, inlier_kps1, inlier_dsc1, inlier_kps2, inlier_dsc2)
    homography_matching_dict = {}

    for ix1 in range(len(kpts_desc_list1)):
        for ix2 in range(len(kpts_desc_list2)):
            kps1, desc1, _, _ = kpts_desc_list1[ix1]
            kps2, desc2, _, _ = kpts_desc_list2[ix2]
            print("matching component/normal {} from 1st image against {} component/normal from 2nd image".format(ix1, ix2))
            H, matches, src_kps, src_dsc, dst_kps, dst_dsc = find_and_draw_homography_or_fallback(image_data1.img,
                                                                                                  kps1,
                                                                                                  desc1,
                                                                                                  image_data2.img,
                                                                                                  kps2,
                                                                                                  desc2,
                                                                                                  ratio_thresh=ratio_thresh,
                                                                                                  ransac_thresh=ransac_th,
                                                                                                  ransac_confidence=ransac_conf,
                                                                                                  ransac_iters=ransac_iters,
                                                                                                  save_suffix="{}_homography_{}_{}".format(img_pair_key, ix1, ix2),
                                                                                                  show=show,
                                                                                                  save=save,
                                                                                                  out_dir=out_dir)

            homography_matching_dict[(ix1, ix2)] = (H, matches, src_kps, src_dsc, dst_kps, dst_dsc)

    if len(kpts_desc_list1) <= len(kpts_desc_list2):
        perm_length = len(kpts_desc_list1)
        all_items_length = len(kpts_desc_list2)
        permutation_items = range(all_items_length)
        swap = False
    else:
        perm_length = len(kpts_desc_list2)
        all_items_length = len(kpts_desc_list1)
        permutation_items = range(all_items_length)
        swap = True

    max_inliers = None
    for cur_permutation in itertools.permutations(permutation_items, perm_length):

        rest_kpts1_l = rest_kpts1
        rest_kpts2_l = rest_kpts2
        rest_descs1_l = rest_descs1
        rest_descs2_l = rest_descs2

        kps1_l = []
        kps2_l = []

        if swap:
            all_idxs_1 = list(cur_permutation)
            all_idxs_2 = list(range(perm_length))
        else:
            all_idxs_1 = list(range(perm_length))
            all_idxs_2 = list(cur_permutation)

        for i in range(perm_length):
            if swap:
                ix1 = cur_permutation[i]
                ix2 = i
            else:
                ix1 = i
                ix2 = cur_permutation[i]

            _, _, cur_kps1, _, cur_kps2, _ = homography_matching_dict[(ix1, ix2)]
            kps1_l.extend(cur_kps1)
            kps2_l.extend(cur_kps2)

        for i in range(all_items_length):
            if i not in cur_permutation:
                if swap:
                    kps1, desc1, _, _ = kpts_desc_list1[i]
                    rest_kpts1_l.extend(kps1)
                    rest_descs1_l.extend(desc1)
                else:
                    kps2, desc2, _, _ = kpts_desc_list2[i]
                    rest_kpts2_l.extend(kps2)
                    rest_descs2_l.extend(desc2)

        pairing_str = "{}x{}".format(str(all_idxs_1).replace(" ", "").replace(",", "_"), str(all_idxs_2).replace(" ", "").replace(",", "_"))
        save_suffix = "{}_{}".format(img_pair_key, pairing_str)
        tentative_matches_rest = find_correspondences(image_data1.img,
                                                      rest_kpts1_l,
                                                      np.array(rest_descs1_l),
                                                      image_data2.img,
                                                      rest_kpts2_l,
                                                      np.array(rest_descs2_l),
                                                      out_dir=out_dir,
                                                      save_suffix=save_suffix,
                                                      show=show,
                                                      save=save,
                                                      ratio_thresh=ratio_thresh)

        src_pts_rest, dst_pts_rest = split_points(tentative_matches_rest, rest_kpts1_l, rest_kpts2_l)

        src_pts = np.float32([kps1.pt for kps1 in kps1_l]).reshape(-1, 2)
        dst_pts = np.float32([kps2.pt for kps2 in kps2_l]).reshape(-1, 2)

        src_pts = np.vstack((src_pts, src_pts_rest))
        dst_pts = np.vstack((dst_pts, dst_pts_rest))

        tentative_matches = get_synthetic_DMatches(len(kps1_l))
        for match in tentative_matches_rest:
            match.queryIdx += len(kps1_l)
            match.trainIdx += len(kps2_l)
        tentative_matches.extend(tentative_matches_rest)

        kps1_l.extend(rest_kpts1_l)
        kps2_l.extend(rest_kpts2_l)

        if use_degensac:
            F, inlier_mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, px_th=ransac_th, conf=ransac_conf,
                                                             max_iters=ransac_iters, enable_degeneracy_check=True)
            inlier_mask = np.expand_dims(inlier_mask, axis=1)
            E = image_data2.real_K.T @ F @ image_data1.real_K
        elif find_fundamental:
            F, inlier_mask = cv.findFundamentalMat(src_pts, dst_pts,
                                                   method=cv.USAC_MAGSAC,
                                                   ransacReprojThreshold=ransac_th,
                                                   confidence=ransac_conf,
                                                   maxIters=ransac_iters)
            E = image_data2.real_K.T @ F @ image_data1.real_K
        else:
            # NOTE previously default RANSAC params used here; no iters param
            E, inlier_mask = cv.findEssentialMat(src_pts, dst_pts, image_data1.real_K, None, image_data2.real_K, None, cv.RANSAC, prob=ransac_conf, threshold=ransac_th)

        inliers_count = np.sum(inlier_mask)

        if max_inliers is None or max_inliers < inliers_count:
            best_E = E
            best_inlier_mask = inlier_mask
            best_src_pts = src_pts
            best_dst_pts = dst_pts
            max_inliers = inliers_count
            best_kps1_l = kps1_l
            best_kps2_l = kps2_l
            best_tentative_matches = tentative_matches
            best_idxs_1 = all_idxs_1
            best_idxs_2 = all_idxs_2

        if show or save:
            img_matches = draw_matches(kps1_l, kps2_l, tentative_matches, None, inlier_mask, image_data1.img, image_data2.img)
            plt.figure(figsize=(10, 10))
            plt.title("{}({} <=> {}) - {} inliers".format(img_pair_key, all_idxs_1, all_idxs_2, inliers_count))
            plt.imshow(img_matches)
            if save:
                plt.savefig("{}/matches_{}.jpg".format(out_dir, save_suffix))
            show_or_close(show)


    best_pairing_str = "{}x{}".format(str(best_idxs_1).replace(" ", "").replace(",", "_"),
                                 str(best_idxs_2).replace(" ", "").replace(",", "_"))
    save_suffix = "{}_best_matching_{}".format(img_pair_key, best_pairing_str)

    show_save_matching(image_data1.img,
                       best_kps1_l,
                       image_data2.img,
                       best_kps2_l,
                       best_tentative_matches,
                       best_inlier_mask,
                       out_dir,
                       save_suffix,
                       show,
                       save)

    image_data1.key_points = best_kps1_l
    image_data2.key_points = best_kps2_l
    return best_E, best_inlier_mask, best_src_pts, best_dst_pts, best_tentative_matches


def show_save_matching(img1,
                       kps1,
                       img2,
                       kps2,
                       tentative_matches,
                       inlier_mask,
                       out_dir,
                       save_suffix,
                       show,
                       save):

    if show or save:
        img_matches = draw_matches(kps1, kps2, tentative_matches, None, inlier_mask, img1, img2)
        plt.figure()
        inliers_count = np.sum(inlier_mask)
        plt.title("Matches in line with the model - {} inliers".format(inliers_count))
        plt.imshow(img_matches)
        if save:
            plt.savefig("{}/matches_{}.jpg".format(out_dir, save_suffix))
        show_or_close(show)


def match_epipolar(img1, kps1, descs1, real_K_1,
                   img2, kps2, descs2, real_K_2,
                   find_fundamental, img_pair, out_dir, show, save, ratio_thresh,
                   ransac_th, ransac_conf, ransac_iters, cfg):

    Timer.start_check_point("matching")

    save_suffix = "{}_{}".format(img_pair.img1, img_pair.img2)

    tentative_matches = find_correspondences(img1, kps1, descs1, img2, kps2, descs2, cfg, out_dir, save_suffix, ratio_thresh=ratio_thresh, show=show, save=save)

    src_pts, dst_pts = split_points(tentative_matches, kps1, kps2)

    # TODO threshold and prob params left to default values
    if find_fundamental:
        F, inlier_mask = cv.findFundamentalMat(src_pts, dst_pts, method=cv.USAC_MAGSAC, ransacReprojThreshold=ransac_th, confidence=ransac_conf, maxIters=ransac_iters)
        if F is None or inlier_mask is None:
            print("WARNING: F:{} or inlier mask:{} are None".format(F, inlier_mask))
            return None, None, None, None, None
        print("F:\n{}".format(F))
        E = real_K_2.T @ F @ real_K_1
    else:
        # NOTE previously default RANSAC params used here; no iters param
        E, inlier_mask = cv.findEssentialMat(src_pts, dst_pts, real_K_1, None, real_K_2, None, cv.RANSAC, prob=ransac_conf, threshold=ransac_th)
        if E is None or inlier_mask is None:
            print("WARNING: E:{} or inlier mask:{} are None".format(F, inlier_mask))
            return None, None, None, None, None

    Timer.end_check_point("matching")

    show_save_matching(img1,
                       kps1,
                       img2,
                       kps2,
                       tentative_matches,
                       inlier_mask,
                       out_dir,
                       save_suffix,
                       show,
                       save)

    return E, inlier_mask, src_pts, dst_pts, tentative_matches


def match_find_F_degensac(img1, kps1, descs1, real_K_1, img2, kps2, descs2, real_K_2, img_pair, out_dir, show, save, ratio_thresh, ransac_th, ransac_conf, ransac_iters):

    Timer.start_check_point("matching")

    save_suffix = "{}_{}".format(img_pair.img1, img_pair.img2)

    tentative_matches = find_correspondences(img1, kps1, descs1, img2, kps2, descs2, out_dir, save_suffix, ratio_thresh=ratio_thresh, show=show, save=save)

    src_pts, dst_pts = split_points(tentative_matches, kps1, kps2)

    F, inlier_mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, px_th=ransac_th, conf=ransac_conf, max_iters=ransac_iters, enable_degeneracy_check=True)
    inlier_mask = np.expand_dims(inlier_mask, axis=1)

    E = real_K_2.T @ F @ real_K_1

    Timer.end_check_point("matching")

    show_save_matching(img1,
                       kps1,
                       img2,
                       kps2,
                       tentative_matches,
                       inlier_mask,
                       out_dir,
                       save_suffix,
                       show,
                       save)

    return E, inlier_mask, src_pts, dst_pts, tentative_matches


def prepare_data_for_keypoints_and_desc(scene_info, img_name, normal_indices, normals, descriptor, out_dir, rectify):

    K = scene_info.get_img_K(img_name)
    img = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_name))

    if rectify:
        normal_indices = possibly_upsample_normals(img, normal_indices)
        components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals)), True)
        kps, descs, _ = get_rectified_keypoints(normals, components_indices, valid_components_dict, img, K, descriptor, img_name, out_prefix=out_dir)
    else:
        kps, descs = descriptor.detectAndCompute(img, None)

    return img, K, kps, descs


def img_correspondences(scene_info: SceneInfo, output_dir, descriptor, normals_dir, difficulties=range(18), rectify=True, limit=None, override_existing=True):

    processed_pairs = 0
    for difficulty in difficulties:

        print("Difficulty: {}".format(difficulty))

        max_limit = len(scene_info.img_pairs_lists[difficulty])
        for i in range(max_limit):

            if limit is not None and processed_pairs >= limit:
                break

            img_pair: ImagePairEntry = scene_info.img_pairs_lists[difficulty][i]

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

            img1, K_1, kps1, descs1 = prepare_data_for_keypoints_and_desc(scene_info, img_pair.img1, normal_indices1, normals1, descriptor, out_dir, rectify)
            img2, K_2, kps2, descs2 = prepare_data_for_keypoints_and_desc(scene_info, img_pair.img2, normal_indices2, normals2, descriptor, out_dir, rectify)

            return match_epipolar(img1, kps1, descs1, K_1, img2, kps2, descs2, K_2, scene_info.img_info_map, img_pair, out_dir, show=True, save=True, ratio_thresh=0.85)


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
